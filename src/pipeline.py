from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

from .database import insert_run
from .guardrails import WARNING_TEXT, apply_safety_guardrails, validate_prediction
from .inference import toy_predict
from .preprocessing import load_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "outputs" / "assistant_radio.sqlite"


def resolve_db_path(db_path: str | Path | None = None) -> Path:
    """Resolve the SQLite path used for evidence logs."""
    candidate = db_path or os.environ.get("ASSISTANT_RADIO_DB_PATH") or DEFAULT_DB_PATH
    path = Path(candidate)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _case_id_from_path(image_path: str | Path) -> str:
    return Path(image_path).stem


def _apply_improved_uncertainty_rule(prediction: dict[str, Any]) -> dict[str, Any]:
    confidence = float(prediction.get("confidence", 0.0) or 0.0)
    quality = prediction.get("image_quality")
    if confidence < 0.60 or quality in {"limited", "poor"}:
        prediction["predicted_class"] = "uncertain"
        prediction["confidence"] = min(confidence, 0.60)
        prediction.setdefault("limitations", []).append(
            "uncertainty forced because confidence or image quality was insufficient"
        )
    return prediction


def run_pipeline(image_path: str | Path, mode: str = "toy", db_path: str | Path | None = None) -> dict[str, Any]:
    """Run the educational prediction pipeline and persist one SQLite log.

    This is a non-clinical software pipeline: preprocessing, toy inference,
    safety guardrails, SQLite evidence logging, then a structured JSON result.
    """
    start = time.perf_counter()
    image_path = Path(image_path)
    resolved_db_path = resolve_db_path(db_path)

    # Validate that the file is loadable while preserving the current toy model behavior.
    load_image(image_path)

    inference_mode = "baseline" if mode == "toy" else mode
    if inference_mode not in {"baseline", "improved"}:
        inference_mode = "baseline"

    prediction = toy_predict(image_path, mode=inference_mode)
    if inference_mode == "improved":
        prediction = _apply_improved_uncertainty_rule(prediction)

    prediction.setdefault("visual_evidence", [])
    prediction["warning"] = WARNING_TEXT
    prediction = apply_safety_guardrails(prediction)
    prediction["warning"] = WARNING_TEXT
    prediction.setdefault("visual_evidence", [])
    prediction["latency_ms"] = int((time.perf_counter() - start) * 1000)
    prediction["pipeline_mode"] = mode

    valid, errors = validate_prediction(prediction)
    prediction["json_valid"] = valid
    if errors:
        existing = prediction.get("guardrail_errors", [])
        prediction["guardrail_errors"] = list(dict.fromkeys([*existing, *errors]))

    insert_run(
        resolved_db_path,
        case_id=_case_id_from_path(image_path),
        image_path=str(image_path),
        prediction=prediction,
    )
    prediction["db_path"] = str(resolved_db_path)
    return prediction
