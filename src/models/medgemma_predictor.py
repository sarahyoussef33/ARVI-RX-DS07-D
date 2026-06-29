from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

from src.guardrails import ALLOWED_CLASSES, WARNING_TEXT, apply_safety_guardrails
from src.preprocessing import basic_quality_flag


MODEL_ID = "google/medgemma-4b-it"
PROMPT_VERSION = "medgemma_strict_json_v1"
MEDGEMMA_PROMPT = (
    "Tu es un assistant radiologue virtuel pedagogique. Analyse cette radiographie "
    "thoracique frontale. Reponds uniquement en JSON valide avec les champs : "
    "class, confidence, observations, justification, limits, warning. Les seules "
    "classes autorisees sont normal, suspected_opacity, uncertain. Si l'image est "
    "ambigue, de mauvaise qualite ou si tu n'es pas sur, reponds uncertain. "
    "Ne donne pas de diagnostic medical definitif."
)


def _fallback_prediction(image_path: str | Path, reason: str, latency_ms: int = 0) -> dict[str, Any]:
    return apply_safety_guardrails(
        {
            "image_quality": basic_quality_flag(image_path),
            "predicted_class": "uncertain",
            "confidence": 0.0,
            "visual_evidence": [],
            "justification": reason,
            "limitations": [
                "MedGemma inference unavailable or invalid",
                "no definitive medical diagnosis",
                "professional review required",
            ],
            "warning": WARNING_TEXT,
            "model_name": MODEL_ID,
            "prompt_version": PROMPT_VERSION,
            "latency_ms": latency_ms,
            "error_detail": reason,
        }
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))

    if not isinstance(payload, dict):
        raise ValueError("MedGemma response JSON is not an object")
    return payload


def parse_medgemma_response(text: str, image_path: str | Path = "") -> dict[str, Any]:
    """Parse strict MedGemma JSON into the repository prediction schema."""
    try:
        payload = _extract_json_object(text)
    except Exception as exc:
        return _fallback_prediction(image_path, f"Invalid MedGemma JSON response: {exc}")

    predicted_class = str(payload.get("class") or payload.get("predicted_class") or "uncertain")
    if predicted_class not in ALLOWED_CLASSES:
        predicted_class = "uncertain"

    try:
        confidence = float(payload.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))

    observations = payload.get("observations") or payload.get("visual_evidence") or []
    if isinstance(observations, str):
        observations = [observations]
    if not isinstance(observations, list):
        observations = []

    limits = payload.get("limits") or payload.get("limitations") or []
    if isinstance(limits, str):
        limits = [limits]
    if not isinstance(limits, list):
        limits = []

    prediction = {
        "image_quality": basic_quality_flag(image_path),
        "predicted_class": predicted_class,
        "confidence": round(confidence, 3),
        "visual_evidence": [str(item) for item in observations],
        "justification": str(payload.get("justification") or "No justification provided."),
        "limitations": [str(item) for item in limits],
        "warning": WARNING_TEXT,
        "model_name": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "latency_ms": 0,
    }
    if not prediction["limitations"]:
        prediction["limitations"] = ["model output requires professional review"]
    return apply_safety_guardrails(prediction)


def mock_medgemma_predict(image_path: str | Path) -> dict[str, Any]:
    """Offline MedGemma-shaped predictor for tests and demos without HF/GPU access."""
    start = time.perf_counter()
    name = Path(image_path).name.lower()
    if "normal" in name:
        raw_class = "normal"
        confidence = 0.66
        observations = ["mock review found no synthetic opacity marker"]
    elif "suspected_opacity" in name or "pneumonia" in name or "opacity" in name:
        raw_class = "suspected_opacity"
        confidence = 0.64
        observations = ["mock review found an opacity-like filename signal"]
    else:
        raw_class = "uncertain"
        confidence = 0.45
        observations = ["mock review could not identify a reliable class signal"]

    text = json.dumps(
        {
            "class": raw_class,
            "confidence": confidence,
            "observations": observations,
            "justification": "Mock MedGemma output for pipeline validation only.",
            "limits": ["mock backend", "not a medical model output"],
            "warning": WARNING_TEXT,
        }
    )
    prediction = parse_medgemma_response(text, image_path=image_path)
    prediction["model_name"] = "mock-medgemma-4b-it"
    prediction["latency_ms"] = int((time.perf_counter() - start) * 1000)
    return prediction


def _model_load_kwargs(torch_module: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"device_map": "auto"}
    if torch_module.cuda.is_available():
        kwargs["torch_dtype"] = torch_module.bfloat16
    else:
        kwargs["torch_dtype"] = torch_module.float32
    return kwargs


def _load_model_class() -> Any:
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText


@lru_cache(maxsize=1)
def load_medgemma_resources(model_id: str = MODEL_ID) -> tuple[Any, Any, Any]:
    """Load and cache the real MedGemma processor/model resources."""
    import torch
    from transformers import AutoProcessor

    model_class = _load_model_class()
    processor = AutoProcessor.from_pretrained(model_id)
    model = model_class.from_pretrained(model_id, **_model_load_kwargs(torch))
    model.eval()
    return processor, model, torch


def _tensor_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except Exception:
        return getattr(model, "device", "cpu")


def _move_inputs_to_device(inputs: Any, model: Any) -> Any:
    device = _tensor_device(model)
    try:
        return inputs.to(device)
    except Exception:
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def generate_medgemma_text(
    image_path: str | Path,
    processor: Any,
    model: Any,
    torch_module: Any,
    prompt: str = MEDGEMMA_PROMPT,
) -> str:
    """Generate text from the real image + strict text prompt."""
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(text=prompt_text, images=image, return_tensors="pt")
    inputs = _move_inputs_to_device(inputs, model)
    input_length = int(inputs["input_ids"].shape[-1])
    with torch_module.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated = output[0][input_length:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def medgemma_predict_with_resources(
    image_path: str | Path,
    processor: Any,
    model: Any,
    torch_module: Any,
    model_id: str = MODEL_ID,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        decoded = generate_medgemma_text(image_path, processor, model, torch_module)
    except Exception as exc:
        return _fallback_prediction(
            image_path,
            f"MedGemma inference could not run locally: {exc}",
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    prediction = parse_medgemma_response(decoded, image_path=image_path)
    prediction["model_name"] = model_id
    prediction["latency_ms"] = int((time.perf_counter() - start) * 1000)
    prediction["raw_model_response"] = decoded
    return prediction


def medgemma_predict(image_path: str | Path, model_id: str = MODEL_ID) -> dict[str, Any]:
    """Run the real multimodal MedGemma path: image pixels + text prompt."""
    start = time.perf_counter()
    try:
        processor, model, torch_module = load_medgemma_resources(model_id)
    except Exception as exc:
        return _fallback_prediction(
            image_path,
            f"MedGemma model could not be loaded: {exc}",
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
    prediction = medgemma_predict_with_resources(image_path, processor, model, torch_module, model_id=model_id)
    prediction["latency_ms"] = int((time.perf_counter() - start) * 1000)
    return prediction
