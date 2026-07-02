from __future__ import annotations

import os
import time
import json
import mimetypes
from pathlib import Path
from typing import Any

import requests

from src.guardrails import WARNING_TEXT, apply_safety_guardrails
from src.models.medgemma_predictor import MEDGEMMA_PROMPT, _extract_json_object
from src.preprocessing import basic_quality_flag


REMOTE_MEDGEMMA_URL_ENV = "REMOTE_MEDGEMMA_URL"
REMOTE_REQUIRED_KEYS = {"class", "confidence", "observations", "justification", "limits", "warning"}
REMOTE_ALLOWED_CLASSES = {"normal", "pneumonia_suspected", "suspected_opacity", "uncertain"}
NGROK_HEADERS = {"ngrok-skip-browser-warning": "true"}


def _request_session() -> requests.Session:
    session = requests.Session()
    # Streamlit/Python can inherit sandbox or corporate proxy variables.
    # The ngrok demo endpoint is public HTTPS and should be called directly.
    session.trust_env = False
    return session


def _fallback_prediction(image_path: str | Path, reason: str, latency_ms: int = 0) -> dict[str, Any]:
    return apply_safety_guardrails(
        {
            "image_quality": basic_quality_flag(image_path),
            "predicted_class": "uncertain",
            "confidence": 0.0,
            "visual_evidence": [],
            "justification": reason,
            "limitations": [
                "remote MedGemma inference unavailable",
                "no definitive medical diagnosis",
                "professional review required",
            ],
            "warning": WARNING_TEXT,
            "model_name": "remote-google/medgemma-4b-it",
            "prompt_version": "remote_medgemma_v1",
            "latency_ms": latency_ms,
            "error_detail": reason,
        }
    )


def _request_error_detail(exc: Exception) -> str:
    if isinstance(exc, requests.Timeout):
        return "Remote MedGemma timeout: the Colab API did not answer before the timeout."
    if isinstance(exc, requests.ConnectionError):
        return "Remote MedGemma API unavailable: connection failed. Check the Colab/ngrok URL."
    if isinstance(exc, requests.HTTPError):
        response = exc.response
        status = response.status_code if response is not None else "unknown"
        text = (response.text if response is not None else "").strip()
        if len(text) > 500:
            text = text[:500] + "..."
        return f"Remote MedGemma API returned HTTP {status}: {text or exc}"
    if isinstance(exc, json.JSONDecodeError):
        return "Remote MedGemma response was not valid JSON."
    return f"Remote MedGemma request failed: {exc}"


def _load_remote_payload(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        payload = _extract_json_object(response.text)
    if not isinstance(payload, dict):
        raise ValueError("Remote MedGemma response JSON is not an object")
    return payload


def remote_medgemma_health(remote_url: str | None = None, timeout_s: int = 30) -> dict[str, Any]:
    url = (remote_url or os.environ.get(REMOTE_MEDGEMMA_URL_ENV) or "").strip()
    if not url:
        return {"status": "error", "error_detail": f"Remote MedGemma URL missing. Set {REMOTE_MEDGEMMA_URL_ENV}."}

    errors: list[str] = []
    session = _request_session()
    for path in ("/health", "/"):
        endpoint = url.rstrip("/") + path
        try:
            response = session.get(endpoint, headers=NGROK_HEADERS, timeout=timeout_s)
            response.raise_for_status()
            payload = _load_remote_payload(response)
            payload["health_endpoint"] = path
            return payload
        except Exception as exc:
            errors.append(f"{path}: {_request_error_detail(exc)}")
    return {"status": "error", "error_detail": " ; ".join(errors)}


def _missing_remote_keys(payload: dict[str, Any]) -> list[str]:
    missing = sorted(REMOTE_REQUIRED_KEYS - set(payload))
    if "class" in missing and "predicted_class" in payload:
        missing.remove("class")
    if "observations" in missing and "visual_evidence" in payload:
        missing.remove("observations")
    if "limits" in missing and "limitations" in payload:
        missing.remove("limits")
    return missing


def validate_remote_payload(payload: dict[str, Any]) -> list[str]:
    errors = _missing_remote_keys(payload)
    raw_class = payload.get("class", payload.get("predicted_class"))
    if raw_class not in REMOTE_ALLOWED_CLASSES:
        errors.append("class must be normal, pneumonia_suspected, suspected_opacity, or uncertain")
    try:
        confidence = float(payload.get("confidence", -1))
        if not 0 <= confidence <= 1:
            errors.append("confidence must be between 0 and 1")
    except Exception:
        errors.append("confidence must be numeric")
    observations = payload.get("observations", payload.get("visual_evidence"))
    if not isinstance(observations, list) or not all(isinstance(item, str) for item in observations):
        errors.append("observations must be a list of strings")
    if not isinstance(payload.get("justification"), str) or not payload.get("justification", "").strip():
        errors.append("justification must be a non-empty string")
    limits = payload.get("limits", payload.get("limitations"))
    if not isinstance(limits, (str, list)):
        errors.append("limits must be a string or a list of strings")
    if not isinstance(payload.get("warning"), str) or not payload.get("warning", "").strip():
        errors.append("warning must be a non-empty string")
    return list(dict.fromkeys(errors))


def normalize_remote_prediction(payload: dict[str, Any], image_path: str | Path) -> dict[str, Any]:
    predicted_class = str(payload.get("predicted_class") or payload.get("class") or "uncertain")
    if predicted_class == "pneumonia_suspected":
        predicted_class = "suspected_opacity"
    if predicted_class not in {"normal", "suspected_opacity", "uncertain"}:
        predicted_class = "uncertain"

    try:
        confidence = float(payload.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))

    evidence = payload.get("visual_evidence") or payload.get("observations") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        evidence = []

    limitations = payload.get("limitations") or payload.get("limits") or []
    if isinstance(limitations, str):
        limitations = [limitations]
    if not isinstance(limitations, list):
        limitations = []

    prediction = {
        "image_quality": str(payload.get("image_quality") or basic_quality_flag(image_path)),
        "predicted_class": predicted_class,
        "confidence": round(confidence, 3),
        "visual_evidence": [str(item) for item in evidence],
        "justification": str(payload.get("justification") or "Remote MedGemma response without justification."),
        "limitations": [str(item) for item in limitations] or ["remote model output requires professional review"],
        "warning": WARNING_TEXT,
        "model_name": str(payload.get("model_name") or "remote-google/medgemma-4b-it"),
        "prompt_version": str(payload.get("prompt_version") or "remote_medgemma_v1"),
        "latency_ms": int(payload.get("latency_ms", 0) or 0),
    }
    if payload.get("raw_model_response"):
        prediction["raw_model_response"] = payload["raw_model_response"]
    return apply_safety_guardrails(prediction)


def remote_medgemma_predict(
    image_path: str | Path,
    remote_url: str | None = None,
    timeout_s: int = 300,
) -> dict[str, Any]:
    start = time.perf_counter()
    url = (remote_url or os.environ.get(REMOTE_MEDGEMMA_URL_ENV) or "").strip()
    if not url:
        return _fallback_prediction(
            image_path,
            f"Remote MedGemma URL missing. Set {REMOTE_MEDGEMMA_URL_ENV} or provide it in Streamlit.",
        )

    endpoint = url.rstrip("/") + "/predict"
    try:
        session = _request_session()
        with Path(image_path).open("rb") as file:
            content_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
            response = session.post(
                endpoint,
                files={"file": (Path(image_path).name, file, content_type)},
                data={"prompt": MEDGEMMA_PROMPT},
                headers=NGROK_HEADERS,
                timeout=timeout_s,
            )
        response.raise_for_status()
        payload = _load_remote_payload(response)
    except Exception as exc:
        return _fallback_prediction(
            image_path,
            _request_error_detail(exc),
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    validation_errors = validate_remote_payload(payload)
    if validation_errors:
        prediction = _fallback_prediction(
            image_path,
            f"Remote MedGemma response failed JSON validation: {validation_errors}",
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
        prediction["raw_remote_response"] = payload
        prediction["remote_url"] = url
        prediction["remote_endpoint"] = endpoint
        return prediction

    prediction = normalize_remote_prediction(payload, image_path)
    prediction["latency_ms"] = int((time.perf_counter() - start) * 1000)
    prediction["remote_url"] = url
    prediction["remote_endpoint"] = endpoint
    return prediction
