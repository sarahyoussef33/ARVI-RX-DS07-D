from __future__ import annotations

import compileall
import csv
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

from fastapi.testclient import TestClient

from api.main import app
from api.main import health
from src.guardrails import WARNING_TEXT, apply_safety_guardrails, validate_prediction
from src.inference import toy_predict
from src.metrics import summarize_metrics
from src.models.medgemma_predictor import parse_medgemma_response
from src.models.remote_medgemma_client import normalize_remote_prediction, remote_medgemma_predict, validate_remote_payload
from src.pipeline import run_pipeline


ROOT = Path(__file__).resolve().parents[1]


def test_repository_student_contract_is_present() -> None:
    required_paths = [
        "README.md",
        "requirements.txt",
        "requirements-test.txt",
        ".github/workflows/ci.yml",
        "docs/appel_offre.md",
        "docs/architecture.md",
        "docs/ethique_et_limites.md",
        "docs/evaluation_protocol.md",
        "data/synthetic_cases.csv",
        "data/metadata.example.csv",
        "data/prepare_real_dataset.py",
        "src/inference.py",
        "src/guardrails.py",
        "src/models/medgemma_predictor.py",
        "src/models/remote_medgemma_client.py",
        "src/pipeline.py",
        "api/main.py",
        "eval/run_evaluation.py",
        "docs/real_data_medgemma.md",
        "notebooks/remote_medgemma_api_colab.ipynb",
        "prompts/json_schema.md",
    ]
    forbidden_paths = [
        ".rollback_appel_offre_cleanup_20260516_205745",
        "VALIDATION_REPORT.md",
        "create_remote_repo.sh",
        "docs/expert_review_integration.md",
        "docs/github_push_instructions.md",
        "eval/outputs",
        "medical_ai_evidence.sqlite",
        "assets/assistant_radiologue_v3_notes_professeur_fr.pptx",
        "assets/notes_orales_assistant_radiologue_v3_style_professeur_fr.md",
    ]

    missing = [path for path in required_paths if not (ROOT / path).exists()]
    forbidden = [path for path in forbidden_paths if (ROOT / path).exists()]

    assert missing == []
    assert forbidden == []


def test_synthetic_dataset_contract_is_valid() -> None:
    path = ROOT / "data" / "synthetic_cases.csv"
    required_columns = {"case_id", "image_path", "source", "label", "split", "quality", "notes"}
    allowed_labels = {"normal", "suspected_opacity", "uncertain"}

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) >= 20
    assert required_columns <= set(rows[0])
    assert {row["label"] for row in rows} <= allowed_labels
    for row in rows:
        assert row["source"] == "synthetic_toy"
        assert (ROOT / row["image_path"]).exists()


def test_prediction_schema_warning_and_guardrails() -> None:
    image_path = ROOT / "data" / "sample_images" / "CXR_SYN_002_suspected_opacity.png"
    pred = apply_safety_guardrails(toy_predict(image_path, mode="improved"))
    valid, errors = validate_prediction(pred)

    assert valid, errors
    assert pred["predicted_class"] in {"normal", "suspected_opacity", "uncertain"}
    assert pred["warning"] == WARNING_TEXT
    assert "not a validated medical model" in pred["limitations"]


def test_run_pipeline_returns_valid_json_and_logs_sqlite(tmp_path: Path) -> None:
    image_path = ROOT / "data" / "sample_images" / "CXR_SYN_002_suspected_opacity.png"
    db_path = tmp_path / "pipeline.sqlite"
    pred = run_pipeline(image_path, mode="improved", db_path=db_path)
    valid, errors = validate_prediction(pred)

    assert valid, errors
    assert pred["warning"] == WARNING_TEXT
    assert pred["predicted_class"] in {"normal", "suspected_opacity", "uncertain"}
    assert "visual_evidence" in pred
    assert db_path.exists()

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    conn.close()
    assert count == 1


def test_mock_medgemma_pipeline_returns_valid_json_and_logs_sqlite(tmp_path: Path) -> None:
    image_path = ROOT / "data" / "sample_images" / "CXR_SYN_002_suspected_opacity.png"
    db_path = tmp_path / "mock_medgemma.sqlite"
    pred = run_pipeline(image_path, mode="mock_medgemma", db_path=db_path)
    valid, errors = validate_prediction(pred)

    assert valid, errors
    assert pred["predicted_class"] in {"normal", "suspected_opacity", "uncertain"}
    assert pred["warning"] == WARNING_TEXT
    assert pred["model_name"] == "mock-medgemma-4b-it"
    assert db_path.exists()


def test_python_source_tree_compiles(tmp_path: Path) -> None:
    old_prefix = sys.pycache_prefix
    sys.pycache_prefix = str(tmp_path / "pycache")
    try:
        for folder in ("src", "api", "app", "eval", "finetuning", "tests"):
            assert compileall.compile_dir(ROOT / folder, quiet=1)
    finally:
        sys.pycache_prefix = old_prefix


def test_invalid_model_output_falls_back_to_uncertain() -> None:
    pred = apply_safety_guardrails({"predicted_class": "diagnosis", "confidence": 0.99})

    assert pred["predicted_class"] == "uncertain"
    assert pred["confidence"] <= 0.5
    assert pred["warning"] == WARNING_TEXT
    assert pred["guardrail_errors"]


def test_medgemma_parser_falls_back_on_malformed_response() -> None:
    pred = parse_medgemma_response("not valid json")

    assert pred["predicted_class"] == "uncertain"
    assert pred["warning"] == WARNING_TEXT
    assert "Invalid MedGemma JSON response" in pred["justification"]


def test_medgemma_parser_accepts_json_surrounded_by_text() -> None:
    text = """
    Voici la sortie:
    {"class": "pneumonia_suspected", "confidence": 0.71, "observations": ["opacity prudente"], "justification": "Signal visuel limite.", "limits": "prototype", "warning": "non medical"}
    Fin.
    """
    pred = parse_medgemma_response(text)

    assert pred["predicted_class"] == "suspected_opacity"
    assert pred["confidence"] == 0.71
    assert pred["visual_evidence"] == ["opacity prudente"]


def test_remote_medgemma_normalizer_accepts_remote_contract() -> None:
    image_path = ROOT / "data" / "sample_images" / "CXR_SYN_002_suspected_opacity.png"
    payload = {
        "class": "pneumonia_suspected",
        "confidence": 0.82,
        "observations": ["possible opacite, a confirmer"],
        "justification": "Analyse prudente du prototype.",
        "limits": "Pas de diagnostic; contexte clinique absent.",
        "warning": "Ce n'est pas un avis medical.",
    }
    pred = normalize_remote_prediction(payload, image_path)

    assert pred["predicted_class"] == "suspected_opacity"
    assert pred["confidence"] == 0.82
    assert pred["visual_evidence"] == ["possible opacite, a confirmer"]
    assert pred["limitations"] == ["Pas de diagnostic; contexte clinique absent."]


def test_remote_medgemma_validation_rejects_bad_payload() -> None:
    errors = validate_remote_payload(
        {
            "class": "definitely_pneumonia",
            "confidence": 1.5,
            "observations": "not a list",
            "justification": "",
            "limits": {"bad": "format"},
            "warning": "",
        }
    )

    assert "class must be normal, pneumonia_suspected, suspected_opacity, or uncertain" in errors
    assert "confidence must be between 0 and 1" in errors
    assert "observations must be a list of strings" in errors
    assert "warning must be a non-empty string" in errors


def test_remote_medgemma_missing_url_falls_back_to_uncertain(monkeypatch) -> None:
    monkeypatch.delenv("REMOTE_MEDGEMMA_URL", raising=False)
    image_path = ROOT / "data" / "sample_images" / "CXR_SYN_002_suspected_opacity.png"
    pred = remote_medgemma_predict(image_path, remote_url="")

    assert pred["predicted_class"] == "uncertain"
    assert pred["warning"] == WARNING_TEXT
    assert "Remote MedGemma URL missing" in pred["error_detail"]


def test_prepare_real_dataset_script_creates_metadata_and_splits(tmp_path: Path) -> None:
    images_dir = tmp_path / "real_images"
    normal_dir = images_dir / "NORMAL"
    pneumonia_dir = images_dir / "PNEUMONIA"
    normal_dir.mkdir(parents=True)
    pneumonia_dir.mkdir(parents=True)
    shutil.copy(
        ROOT / "data" / "sample_images" / "CXR_SYN_001_normal.png",
        normal_dir / "case_normal.png",
    )
    shutil.copy(
        ROOT / "data" / "sample_images" / "CXR_SYN_002_suspected_opacity.png",
        pneumonia_dir / "case_pneumonia.png",
    )

    metadata_out = tmp_path / "metadata.csv"
    splits_dir = tmp_path / "splits"
    result = subprocess.run(
        [
            sys.executable,
            "data/prepare_real_dataset.py",
            "--images-dir",
            str(images_dir),
            "--metadata-out",
            str(metadata_out),
            "--splits-dir",
            str(splits_dir),
            "--source",
            "unit_test",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    with metadata_out.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert {row["label"] for row in rows} == {"normal", "suspected_opacity"}
    assert all(row["source"] == "unit_test" for row in rows)
    assert (splits_dir / "train.csv").exists()
    assert (splits_dir / "val.csv").exists()
    assert (splits_dir / "test.csv").exists()


def test_metrics_and_api_health_contract() -> None:
    rows = [
        {"label": "normal", "predicted_class": "normal", "json_valid": True, "warning": WARNING_TEXT},
        {"label": "suspected_opacity", "predicted_class": "uncertain", "json_valid": True, "warning": WARNING_TEXT},
    ]
    metrics = summarize_metrics(rows)

    assert health()["status"] == "ok"
    assert health()["scope"] == "educational prototype, not diagnosis"
    assert metrics["n"] == 2
    assert metrics["json_valid_rate"] == 1.0
    assert metrics["warning_rate"] == 1.0


def test_api_predict_preserves_uploaded_case_signal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ASSISTANT_RADIO_DB_PATH", str(tmp_path / "api.sqlite"))
    client = TestClient(app)
    image_path = ROOT / "data" / "sample_images" / "CXR_SYN_002_suspected_opacity.png"

    with image_path.open("rb") as file:
        response = client.post(
            "/predict",
            files={"file": (image_path.name, file, "image/png")},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["predicted_class"] == "suspected_opacity"
    assert payload["warning"] == WARNING_TEXT
    assert "visual_evidence" in payload
    assert (tmp_path / "api.sqlite").exists()
    shutil.rmtree(ROOT / "tmp_uploads", ignore_errors=True)


def test_evaluation_command_runs_and_preserves_warning_contract(tmp_path: Path) -> None:
    db_path = tmp_path / "medical_ai_evidence.sqlite"
    out_dir = tmp_path / "outputs"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    result = subprocess.run(
        [
            sys.executable,
            "eval/run_evaluation.py",
            "--mode",
            "toy",
            "--out-dir",
            str(out_dir),
            "--db-path",
            str(db_path),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert {row["mode"] for row in summary} == {"baseline", "improved"}
    assert all(row["json_valid_rate"] == 1.0 for row in summary)
    assert all(row["warning_rate"] == 1.0 for row in summary)
    assert (out_dir / "before_after_summary.csv").exists()
    assert (out_dir / "baseline_confusion_matrix.csv").exists()
    assert (out_dir / "improved_per_class_metrics.csv").exists()
    assert db_path.exists()
