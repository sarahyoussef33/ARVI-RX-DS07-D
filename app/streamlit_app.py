from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from PIL import Image

from src.models.medgemma_predictor import load_medgemma_resources
from src.pipeline import run_pipeline


@st.cache_resource(show_spinner=False)
def cached_medgemma_resources():
    return load_medgemma_resources()


def load_split_test_cases() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "data" / "splits" / "test.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))

st.set_page_config(page_title="Assistant radiologue virtuel", layout="wide")
st.title("Assistant radiologue virtuel — prototype pédagogique")
st.warning("Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise.")

uploaded = st.file_uploader("Déposer une radiographie thoracique frontale", type=["png", "jpg", "jpeg"])
mode = st.selectbox("Mode", ["toy", "baseline", "improved", "mock_medgemma", "medgemma"])

test_cases = load_split_test_cases()
selected_case = None
if test_cases:
    use_test_image = st.checkbox("Tester une image du split test")
    if use_test_image:
        selected_case = st.selectbox(
            "Image du split test",
            test_cases,
            format_func=lambda row: f"{Path(row['image_path']).name} | {row['label']}",
        )

if uploaded or selected_case:
    if uploaded:
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)
    else:
        tmp_path = PROJECT_ROOT / selected_case["image_path"]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(Image.open(tmp_path), caption="Image uploadée", use_container_width=True)
    with col2:
        if mode == "medgemma":
            with st.spinner("Chargement/analyse MedGemma en cours, cela peut prendre plusieurs minutes."):
                try:
                    resources = cached_medgemma_resources()
                except Exception:
                    resources = None
                pred = run_pipeline(tmp_path, mode=mode, medgemma_resources=resources)
        else:
            pred = run_pipeline(tmp_path, mode=mode)
        st.metric("Classe", pred["predicted_class"])
        st.metric("Confiance", pred["confidence"])
        st.write("**Observations**", pred["visual_evidence"])
        st.write("**Justification**", pred["justification"])
        st.write("**Limites**", pred["limitations"])
        if pred.get("error_detail"):
            st.error(pred["error_detail"])
        st.json(pred)
else:
    st.info("Utiliser les images synthétiques dans data/sample_images pour tester le flux.")
