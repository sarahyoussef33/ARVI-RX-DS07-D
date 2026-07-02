from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

from src.guardrails import ALLOWED_CLASSES, WARNING_TEXT, apply_safety_guardrails
from src.preprocessing import basic_quality_flag


logger = logging.getLogger(__name__)

MODEL_ID = "google/medgemma-4b-it"
PROMPT_VERSION = "medgemma_strict_json_v1"
MEDGEMMA_PROMPT = (
    "Tu es un assistant radiologue virtuel pedagogique pour le projet ARVI-RX. "
    "Analyse prudemment cette radiographie thoracique frontale. Le modele est un "
    "prototype non valide cliniquement et ne doit pas produire d'avis medical. "
    "Reponds uniquement avec un JSON valide, sans texte avant ni apres. "
    "Schema obligatoire: {"
    '"class": "normal | pneumonia_suspected | uncertain", '
    '"confidence": 0.0, '
    '"observations": ["observation courte et prudente"], '
    '"justification": "justification courte et prudente", '
    '"limits": "limites de l analyse", '
    '"warning": "message rappelant que ce n est pas un avis medical"'
    "}. Si l'image est ambigue, de mauvaise qualite ou si tu n'es pas sur, "
    "utilise uncertain."
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


def _format_exception(context: str, exc: BaseException) -> str:
    return f"{context}: {exc}\n{traceback.format_exc()}"


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
    if predicted_class == "pneumonia_suspected":
        predicted_class = "suspected_opacity"
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


def _runtime_device(torch_module: Any) -> tuple[str, Any]:
    cuda_available = bool(torch_module.cuda.is_available())
    if not cuda_available:
        return "cpu", torch_module.float32
    if hasattr(torch_module.cuda, "is_bf16_supported") and torch_module.cuda.is_bf16_supported():
        return "cuda", torch_module.bfloat16
    return "cuda", torch_module.float16


def _available_memory_gb() -> float | None:
    if sys.platform.startswith("win"):
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.ullAvailPhys / (1024**3)
        except Exception:
            return None
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) / (1024**3)
    except Exception:
        return None


def _check_cpu_memory_for_float32(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        return
    available_gb = _available_memory_gb()
    required_gb = 24.0
    logger.info("MedGemma CPU memory preflight: available_gb=%s required_gb=%s", available_gb, required_gb)
    if available_gb is not None and available_gb < required_gb:
        raise RuntimeError(
            "MedGemma 4B CPU float32 loading requires roughly 24 GB of free RAM. "
            f"Only {available_gb:.1f} GB appears available. Install/use a CUDA build of torch "
            "with a suitable GPU, or run on a machine with more RAM."
        )


def _model_load_kwargs(torch_module: Any, local_files_only: bool) -> dict[str, Any]:
    device, dtype = _runtime_device(torch_module)
    # Do not use device_map="auto": on this project it can leave parameters on meta tensors.
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "local_files_only": local_files_only,
        "low_cpu_mem_usage": True,
    }
    if device == "cuda":
        kwargs["device_map"] = {"": "cuda:0"}
    else:
        kwargs["device_map"] = {"": "cpu"}
    logger.info(
        "MedGemma load config: model_id=%s cuda_available=%s device=%s dtype=%s local_files_only=%s device_map=%s",
        MODEL_ID,
        torch_module.cuda.is_available(),
        device,
        dtype,
        local_files_only,
        kwargs["device_map"],
    )
    return kwargs


def _load_model_class() -> Any:
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText


def _has_meta_tensors(model: Any) -> bool:
    for tensor in list(model.parameters()) + list(model.buffers()):
        if bool(getattr(tensor, "is_meta", False)):
            return True
    return False


def _load_medgemma_once(model_id: str, local_files_only: bool) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoProcessor

    device, _ = _runtime_device(torch)
    model_class = _load_model_class()
    _check_cpu_memory_for_float32(torch)
    logger.info(
        "Loading MedGemma resources: model_id=%s cuda_available=%s target_device=%s local_files_only=%s",
        model_id,
        torch.cuda.is_available(),
        device,
        local_files_only,
    )
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    model = model_class.from_pretrained(model_id, **_model_load_kwargs(torch, local_files_only))
    if _has_meta_tensors(model):
        raise RuntimeError(
            "MedGemma loaded with meta tensors. Refusing to run inference; "
            "use explicit CPU/CUDA loading instead of device_map='auto'."
        )
    model.eval()
    model._arvi_rx_device = device
    logger.info("MedGemma loaded successfully on device=%s", device)
    return processor, model, torch


@lru_cache(maxsize=1)
def load_medgemma_resources(model_id: str = MODEL_ID) -> tuple[Any, Any, Any]:
    """Load and cache the real MedGemma processor/model resources."""
    try:
        return _load_medgemma_once(model_id, local_files_only=True)
    except Exception as local_exc:
        logger.warning("MedGemma local cache load failed: %s", local_exc, exc_info=True)
        try:
            return _load_medgemma_once(model_id, local_files_only=False)
        except Exception:
            logger.exception("MedGemma remote/cache load failed")
            raise


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
    logger.info(
        "Running MedGemma multimodal generation: image=%s size=%s mode=%s device=%s",
        image_path,
        image.size,
        image.mode,
        getattr(model, "_arvi_rx_device", _tensor_device(model)),
    )
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
            _format_exception("MedGemma inference could not run locally", exc),
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
            _format_exception("MedGemma model could not be loaded", exc),
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
    prediction = medgemma_predict_with_resources(image_path, processor, model, torch_module, model_id=model_id)
    prediction["latency_ms"] = int((time.perf_counter() - start) * 1000)
    return prediction
