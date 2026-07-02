from __future__ import annotations

import re
import shutil
from pathlib import Path
from fastapi import FastAPI, File, Form, UploadFile

from src.pipeline import run_pipeline

app = FastAPI(title="Assistant radiologue virtuel EFREI", version="0.1.0")
UPLOAD_DIR = Path("tmp_uploads")


@app.get("/")
def health() -> dict:
    return {"status": "ok", "scope": "educational prototype, not diagnosis"}


@app.get("/health")
def health_check() -> dict:
    return health()


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    mode: str = Form("improved"),
    remote_url: str = Form(""),
) -> dict:
    UPLOAD_DIR.mkdir(exist_ok=True)
    filename = Path(file.filename or "image.png").name
    suffix = Path(filename).suffix or ".png"
    stem = Path(filename).stem or "image"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    target = UPLOAD_DIR / f"uploaded_{safe_stem}{suffix}"
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    if mode not in {"toy", "baseline", "improved", "medgemma", "mock_medgemma", "remote_medgemma"}:
        mode = "improved"
    return run_pipeline(target, mode=mode, remote_url=remote_url or None)
