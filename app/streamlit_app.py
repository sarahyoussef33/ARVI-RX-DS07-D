from __future__ import annotations

import csv
import html
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from PIL import Image

from src.models.medgemma_predictor import load_medgemma_resources
from src.pipeline import run_pipeline


APP_MODES = ["toy", "baseline", "improved", "mock_medgemma", "medgemma", "remote_medgemma"]
CLASS_LABELS = {
    "normal": "Aspect normal",
    "suspected_opacity": "Opacité suspectée",
    "pneumonia_suspected": "Opacité suspectée",
    "uncertain": "Résultat incertain",
}
DEMO_PATIENTS = [
    {"id": "demo-001", "name": "Patient demo 001", "age": "42 ans", "exam": "Radio thoracique"},
    {"id": "demo-002", "name": "Patient demo 002", "age": "67 ans", "exam": "Radio thoracique"},
    {"id": "demo-003", "name": "Patient demo 003", "age": "29 ans", "exam": "Radio thoracique"},
]
RADIOX_TUTORIAL_JSON = {
    "objectif": (
        "RadioX est un prototype pedagogique pour analyser une radiographie "
        "thoracique avec une sortie JSON prudente."
    ),
    "accueil": "Bonjour, que puis-je faire pour vous ?",
    "fonctionnalites": [
        "ouvrir les radios patient fictives",
        "choisir une radiographie de demonstration ou uploader une image",
        "lancer une analyse avec remote_medgemma ou mock_medgemma",
        "afficher un JSON structure avec classe, confiance, observations, limites et warning",
        "expliquer le JSON avec le chatbot local de la page Radio thoracique",
    ],
    "boutons": {
        "dashboard": "Cliquer sur le bouton bleu nomme Ouvrir les radios.",
        "patients": "Choisir un patient fictif avec le bouton Ouvrir.",
        "analyse": "Sur Radio thoracique, cliquer sur le bouton Analyser la radio.",
    },
    "etapes": [
        "Cliquer sur Radios patient.",
        "Choisir un patient fictif.",
        "Ouvrir Analyse radio.",
        "Deposer une radiographie thoracique frontale.",
        "Choisir le mode remote_medgemma ou mock_medgemma.",
        "Lancer l'analyse.",
        "Lire la classe, la confiance, les observations, les limites et l'avertissement.",
    ],
    "rappels": [
        "RadioX ne fournit pas de diagnostic.",
        "Les patients sont fictifs.",
        "La sortie doit etre validee par un professionnel qualifie.",
        "Attention: les resultats sont une validation technique, pas une performance medicale.",
    ],
    "json_fields": {
        "predicted_class": "classe retournee par le modele",
        "confidence": "niveau de confiance entre 0 et 1",
        "visual_evidence": "observations visuelles retournees",
        "justification": "raisonnement court du modele",
        "limitations": "limites de l'analyse",
        "warning": "avertissement medical obligatoire",
    },
}


@st.cache_resource(show_spinner=False)
def cached_medgemma_resources():
    return load_medgemma_resources()


def rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    st.experimental_rerun()


def ensure_session_state() -> None:
    defaults = {
        "authenticated": False,
        "current_page": "login",
        "last_prediction_json": None,
        "chat_history": [],
        "selected_patient": DEMO_PATIENTS[0],
        "dashboard_guide_open": False,
        "dashboard_guide_answer": "",
        "dashboard_guide_question": "",
        "guide_messages": [],
        "guide_is_typing": False,
        "guide_pending_answer": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def navigate(page: str) -> None:
    st.session_state.current_page = page
    rerun()


def load_split_test_cases() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "data" / "splits" / "test.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _safe(value: Any) -> str:
    return html.escape(str(value))


def _confidence(pred: dict[str, Any]) -> float:
    try:
        return float(pred.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def badge_class(predicted_class: str, has_error: bool = False) -> str:
    if has_error:
        return "badge badge-error"
    if predicted_class == "normal":
        return "badge badge-normal"
    if predicted_class in {"suspected_opacity", "pneumonia_suspected"}:
        return "badge badge-opacity"
    return "badge badge-uncertain"


def readable_class(predicted_class: str) -> str:
    return CLASS_LABELS.get(predicted_class, "Resultat incertain")


def analysis_summary(predicted_class: str) -> str:
    if predicted_class == "normal":
        return "Aucune anomalie évidente n'a été détectée par le modèle sur cette radiographie."
    if predicted_class in {"suspected_opacity", "pneumonia_suspected"}:
        return (
            "Le modèle signale une zone d'opacité suspecte sur la radiographie. "
            "Cela peut correspondre à une anomalie à vérifier, sans constituer un diagnostic."
        )
    return "Le modèle n'est pas suffisamment sûr pour proposer une interprétation fiable."


def confidence_level(confidence: float) -> str:
    if confidence >= 0.80:
        return "confiance élevée"
    if confidence >= 0.50:
        return "confiance moyenne"
    return "confiance faible"


def confidence_percent(confidence: float) -> int:
    return round(confidence * 100)


def html_list(items: list[str], empty_text: str) -> str:
    if not items:
        return f'<div class="card-muted">{_safe(empty_text)}</div>'
    rows = "".join(f"<li>{_safe(item)}</li>" for item in items)
    return f'<ul class="result-list">{rows}</ul>'


def section_title(icon: str, title: str) -> str:
    return (
        '<div class="section-title">'
        f'<span class="section-icon">{_safe(icon)}</span>'
        f'<span>{_safe(title)}</span>'
        '</div>'
    )


def helper_box(text: str) -> str:
    return f'<div class="helper-box">{_safe(text)}</div>'


def field_label(icon: str, label: str, help_text: str | None = None) -> str:
    help_html = f'<div class="field-help">{_safe(help_text)}</div>' if help_text else ""
    return (
        '<div class="field-label">'
        f'<span class="field-icon">{_safe(icon)}</span>'
        f'<span>{_safe(label)}</span>'
        '</div>'
        f'{help_html}'
    )


def apply_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --night: #0b1f3a;
            --night-2: #123257;
            --medical: #1f8ed6;
            --cyan: #21c7d9;
            --mint: #55d6b5;
            --ink: #16324f;
            --muted: #637487;
            --line: rgba(33, 199, 217, 0.18);
            --glass: rgba(255, 255, 255, 0.76);
        }
        .stApp {
            background:
                radial-gradient(circle at 12% 8%, rgba(255, 255, 255, 0.90), transparent 18rem),
                radial-gradient(circle at 84% 12%, rgba(33, 199, 217, 0.16), transparent 24rem),
                radial-gradient(circle at 20% 92%, rgba(85, 214, 181, 0.12), transparent 26rem),
                linear-gradient(135deg, #f7fcff 0%, #edf8fb 52%, #f9fffc 100%);
            color: var(--ink);
        }
        header[data-testid="stHeader"] {
            height: 0;
            visibility: hidden;
        }
        #MainMenu, footer {
            visibility: hidden;
        }
        .block-container {
            max-width: 100% !important;
            padding-top: 0.25rem !important;
            padding-bottom: 0.25rem !important;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
        }
        div[data-testid="stVerticalBlock"] {
            gap: 0.42rem;
        }
        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 20% 0%, rgba(33, 199, 217, 0.24), transparent 15rem),
                linear-gradient(180deg, #071b32 0%, #0b1f3a 48%, #123b63 100%);
        }
        [data-testid="stSidebar"] * {
            color: #f7fbff;
        }
        [data-testid="stSidebar"] .stButton > button {
            justify-content: flex-start;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: none;
            color: #f7fbff;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(33, 199, 217, 0.18);
            border-color: rgba(85, 214, 181, 0.55);
        }
        .radiox-shell {
            padding: 0.25rem 0 0.55rem;
        }
        .st-key-login_shell {
            min-height: calc(100vh - 28px);
            display: flex;
            align-items: center;
            justify-content: center;
            max-width: 1280px;
            margin: 0 auto;
            padding: 0.45rem 1rem;
        }
        .st-key-login_shell > div[data-testid="stVerticalBlock"] {
            width: 100%;
        }
        .st-key-login_shell div[data-testid="stHorizontalBlock"] {
            align-items: center;
            gap: 2rem;
        }
        .login-left-panel {
            min-height: min(650px, calc(100vh - 56px));
            max-height: calc(100vh - 44px);
            border-radius: 30px;
            padding: 2.45rem 2.65rem;
            color: #ffffff;
            overflow: hidden;
            position: relative;
            background:
                linear-gradient(115deg, rgba(11, 31, 58, 0.96), rgba(15, 58, 93, 0.94) 48%, rgba(12, 121, 142, 0.88)),
                radial-gradient(circle at 20% 20%, rgba(34, 211, 238, 0.38), transparent 18rem),
                radial-gradient(circle at 72% 76%, rgba(45, 212, 191, 0.30), transparent 17rem);
            box-shadow: 0 26px 64px rgba(11, 31, 58, 0.26);
        }
        .login-left-panel::before {
            content: "";
            position: absolute;
            right: -7rem;
            top: -6rem;
            width: 22rem;
            height: 22rem;
            border-radius: 999px;
            background: rgba(34, 211, 238, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.14);
            filter: blur(2px);
        }
        .login-left-panel::after {
            content: "";
            position: absolute;
            left: -4rem;
            bottom: -4rem;
            width: 18rem;
            height: 18rem;
            border-radius: 999px;
            border: 1px solid rgba(45, 212, 191, 0.22);
            box-shadow: 0 0 70px rgba(45, 212, 191, 0.16);
        }
        .login-kicker {
            position: relative;
            z-index: 1;
            display: inline-flex;
            align-items: center;
            padding: 0.34rem 0.7rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.14);
            border: 1px solid rgba(255, 255, 255, 0.22);
            color: #e8feff;
            font-weight: 760;
            font-size: 0.78rem;
            margin-bottom: 1.05rem;
            letter-spacing: 0.02em;
        }
        .login-hero-title {
            position: relative;
            z-index: 1;
            color: #ffffff;
            font-size: 2.7rem;
            line-height: 1.05;
            font-weight: 850;
            letter-spacing: 0;
            margin: 0 0 0.7rem;
            max-width: 34rem;
        }
        .login-hero-subtitle {
            position: relative;
            z-index: 1;
            color: #d8f7ff;
            font-size: 1.08rem;
            font-weight: 650;
            max-width: 34rem;
            margin-bottom: 0.85rem;
        }
        .login-hero-copy {
            position: relative;
            z-index: 1;
            color: rgba(255, 255, 255, 0.82);
            font-size: 0.94rem;
            line-height: 1.55;
            max-width: 35rem;
            margin-bottom: 1rem;
        }
        .login-badges {
            position: relative;
            z-index: 1;
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1.05rem;
        }
        .login-badge {
            padding: 0.34rem 0.62rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.20);
            color: #f5feff;
            font-weight: 720;
            font-size: 0.78rem;
            letter-spacing: 0.01em;
        }
        .login-card {
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid rgba(255, 255, 255, 0.86);
            box-shadow: 0 20px 48px rgba(9, 37, 68, 0.13);
            padding: 1.55rem 1.55rem;
            min-height: 26.5rem;
        }
        .st-key-login_card {
            border-radius: 28px;
            background: #ffffff;
            border: 1px solid rgba(16, 42, 67, 0.08);
            box-shadow: 0 22px 54px rgba(9, 37, 68, 0.16);
            padding: 1.55rem 1.85rem 1.45rem;
            min-height: auto;
        }
        .st-key-login_card > div[data-testid="stVerticalBlock"] {
            gap: 0.42rem;
        }
        .login-title {
            color: var(--night);
            font-size: 1.85rem;
            font-weight: 830;
            margin: 0 0 0.55rem;
        }
        .login-subtitle {
            color: var(--muted);
            font-size: 0.92rem;
            margin: 0.18rem 0 0.9rem;
        }
        .login-card input,
        .st-key-login_card input {
            background: #ffffff !important;
            border: 1px solid rgba(16, 42, 67, 0.22) !important;
            border-radius: 10px !important;
            color: #102a43 !important;
            min-height: 2.55rem !important;
        }
        .login-card label,
        .st-key-login_card label {
            color: #102a43 !important;
            font-weight: 700 !important;
        }
        .login-card button[kind="secondary"],
        .st-key-login_card button[kind="secondary"] {
            background: #f6feff !important;
            color: #102a43 !important;
            border: 1px solid rgba(34, 211, 238, 0.42) !important;
            box-shadow: none !important;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        .login-card button[kind="secondary"]:hover,
        .st-key-login_card button[kind="secondary"]:hover {
            background: #eafcff !important;
            border-color: rgba(16, 185, 129, 0.58) !important;
            transform: translateY(-1px);
        }
        .radiox-header {
            position: relative;
            overflow: hidden;
            padding: 1.05rem 1.2rem;
            border: 1px solid rgba(255, 255, 255, 0.78);
            border-radius: 20px;
            background:
                radial-gradient(circle at 92% 18%, rgba(33, 199, 217, 0.18), transparent 15rem),
                linear-gradient(135deg, rgba(255, 255, 255, 0.88), rgba(246, 253, 255, 0.66));
            box-shadow: 0 14px 34px rgba(9, 37, 68, 0.10);
            backdrop-filter: blur(14px);
            margin-bottom: 0.7rem;
        }
        .radiox-header::after {
            content: "";
            position: absolute;
            right: -5rem;
            top: -5rem;
            width: 10rem;
            height: 10rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.72);
            filter: blur(18px);
        }
        .radiox-title {
            margin: 0;
            color: var(--night);
            font-size: 1.75rem;
            font-weight: 820;
            letter-spacing: 0;
            position: relative;
            z-index: 1;
        }
        .radiox-subtitle {
            margin-top: 0.25rem;
            color: var(--muted);
            font-size: 0.9rem;
            max-width: 52rem;
            position: relative;
            z-index: 1;
        }
        .glass-card {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.78);
            border-radius: 18px;
            background:
                linear-gradient(145deg, rgba(255, 255, 255, 0.88), rgba(248, 253, 255, 0.70));
            box-shadow: 0 12px 28px rgba(9, 37, 68, 0.08);
            backdrop-filter: blur(12px);
            padding: 0.78rem 0.9rem;
            margin-bottom: 0.62rem;
        }
        .glass-card::before {
            content: "";
            position: absolute;
            inset: 0;
            border-left: 3px solid rgba(33, 199, 217, 0.56);
            pointer-events: none;
        }
        .stTextInput label,
        .stSelectbox label,
        .stFileUploader label,
        .stCheckbox label {
            color: var(--ink) !important;
            font-weight: 720 !important;
        }
        .stTextInput input,
        .stTextInput [data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {
            min-height: 2.9rem !important;
            border-radius: 14px !important;
            background: #ffffff !important;
            color: var(--ink) !important;
            border: 1px solid #bfeaf2 !important;
            box-shadow: 0 8px 18px rgba(9, 37, 68, 0.06) !important;
        }
        .stTextInput input:focus,
        .stTextInput [data-baseweb="input"] > div:focus-within,
        div[data-baseweb="select"] > div:focus-within {
            border-color: rgba(33, 199, 217, 0.72) !important;
            box-shadow: 0 0 0 3px rgba(33, 199, 217, 0.14) !important;
            outline: none !important;
        }
        .stTextInput input::placeholder {
            color: #7f91a5 !important;
        }
        .stTextInput input,
        .stTextInput input:hover,
        .stTextInput input:active,
        .stTextInput input:focus,
        .stTextInput textarea,
        .stTextInput textarea:hover,
        .stTextInput textarea:active,
        .stTextInput textarea:focus {
            background: #ffffff !important;
            color: var(--ink) !important;
            border-color: #bfeaf2 !important;
            caret-color: #2563eb !important;
            outline: none !important;
        }
        div[data-baseweb="select"] * {
            color: var(--ink) !important;
        }
        div[data-baseweb="select"] svg,
        div[data-baseweb="input"] svg,
        div[data-testid="stFileUploader"] svg {
            color: #2563eb !important;
            fill: none !important;
            stroke: #2563eb !important;
        }
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        div[role="listbox"] {
            background: #ffffff !important;
            color: var(--ink) !important;
            border: 1px solid #bfeaf2 !important;
            border-radius: 14px !important;
            box-shadow: 0 16px 34px rgba(9, 37, 68, 0.12) !important;
        }
        div[role="option"] {
            background: #ffffff !important;
            color: var(--ink) !important;
        }
        div[role="option"]:hover,
        div[aria-selected="true"] {
            background: #eafbff !important;
            color: var(--ink) !important;
        }
        div[data-testid="stFileUploaderDropzone"] {
            min-height: 7rem;
            border-radius: 18px !important;
            background:
                linear-gradient(135deg, #ffffff, #f0fcff) !important;
            border: 1px dashed #bfeaf2 !important;
            box-shadow: 0 10px 24px rgba(9, 37, 68, 0.07) !important;
        }
        div[data-testid="stFileUploaderDropzone"] * {
            color: var(--ink) !important;
        }
        div[data-testid="stFileUploaderDropzone"] button,
        .stFileUploader button {
            border-radius: 999px !important;
            background: linear-gradient(135deg, #2563eb 0%, #22d3ee 62%, #2dd4bf 100%) !important;
            color: #ffffff !important;
            border: 1px solid rgba(34, 211, 238, 0.48) !important;
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.16) !important;
        }
        div[data-testid="stFileUploaderDropzone"] button *,
        .stFileUploader button * {
            color: #ffffff !important;
        }
        [data-testid="stFileUploaderFile"] {
            border-radius: 14px !important;
            background: #ffffff !important;
            color: var(--ink) !important;
            border: 1px solid #bfeaf2 !important;
            box-shadow: 0 8px 18px rgba(9, 37, 68, 0.055) !important;
        }
        [data-testid="stFileUploaderFile"] * {
            color: var(--ink) !important;
        }
        [data-testid="stFileUploaderFile"] button,
        [data-testid="stFileUploaderFile"] button:hover {
            background: #eefcff !important;
            border: 1px solid #bfeaf2 !important;
            color: #2563eb !important;
            box-shadow: none !important;
        }
        .stCheckbox [data-testid="stWidgetLabel"] p {
            color: var(--ink) !important;
        }
        .stCheckbox input[type="checkbox"] {
            accent-color: #22d3ee !important;
        }
        .stCheckbox div[role="checkbox"] {
            border-radius: 6px !important;
            background: #ffffff !important;
            border: 1px solid #bfeaf2 !important;
            box-shadow: 0 4px 10px rgba(9, 37, 68, 0.055) !important;
        }
        .stCheckbox div[role="checkbox"][aria-checked="true"] {
            background: linear-gradient(135deg, #2563eb, #22d3ee) !important;
            border-color: #22d3ee !important;
        }
        .stCheckbox div[role="checkbox"] svg {
            color: #ffffff !important;
            stroke: #ffffff !important;
        }
        div[data-testid="stExpander"] {
            border-radius: 16px !important;
            border: 1px solid #bfeaf2 !important;
            background: rgba(255, 255, 255, 0.86) !important;
            overflow: hidden;
            box-shadow: 0 10px 22px rgba(9, 37, 68, 0.055);
        }
        div[data-testid="stExpander"] details,
        div[data-testid="stExpander"] summary {
            background: #ffffff !important;
            color: var(--ink) !important;
        }
        div[data-testid="stExpander"] summary p,
        div[data-testid="stExpander"] p {
            color: var(--ink) !important;
        }
        .stTextInput input:disabled,
        .stButton > button:disabled,
        .stFormSubmitButton > button:disabled {
            background: #edf8fb !important;
            color: #7f91a5 !important;
            border: 1px solid rgba(33, 199, 217, 0.20) !important;
            box-shadow: none !important;
            opacity: 1 !important;
        }
        .stFormSubmitButton > button {
            border-radius: 999px !important;
            background: linear-gradient(135deg, #2563eb 0%, #22d3ee 58%, #2dd4bf 100%) !important;
            color: #ffffff !important;
            border: 1px solid rgba(34, 211, 238, 0.46) !important;
            box-shadow: 0 10px 22px rgba(37, 99, 235, 0.18) !important;
        }
        .stFormSubmitButton > button *,
        .stButton > button *,
        .stDownloadButton > button * {
            color: inherit !important;
        }
        .dashboard-card {
            min-height: 7rem;
            cursor: default;
            box-shadow: 0 14px 34px rgba(9, 37, 68, 0.075);
        }
        .dashboard-card::before {
            display: none;
        }
        .dashboard-header {
            margin: 0.2rem 0 0.9rem;
        }
        .dashboard-title {
            color: var(--night);
            font-size: 2rem;
            font-weight: 840;
            letter-spacing: 0;
            margin: 0;
        }
        .dashboard-subtitle {
            color: var(--muted);
            font-size: 0.96rem;
            margin-top: 0.22rem;
            max-width: 55rem;
        }
        .card-icon {
            width: 2.15rem;
            height: 2.15rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
            margin-bottom: 0.55rem;
            color: var(--night);
            background:
                linear-gradient(135deg, rgba(33, 199, 217, 0.20), rgba(85, 214, 181, 0.20));
            border: 1px solid rgba(33, 199, 217, 0.26);
            font-size: 1rem;
            font-weight: 800;
        }
        .card-title {
            color: var(--night);
            font-weight: 760;
            font-size: 0.98rem;
            margin-bottom: 0.25rem;
        }
        .section-title {
            display: flex;
            align-items: center;
            gap: 0.52rem;
            color: var(--night);
            font-weight: 800;
            font-size: 1rem;
            margin-bottom: 0.42rem;
        }
        .section-icon,
        .field-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            flex: 0 0 auto;
            border-radius: 999px;
            color: #2563eb;
            background: linear-gradient(135deg, rgba(34, 211, 238, 0.16), rgba(45, 212, 191, 0.18));
            border: 1px solid #bfeaf2;
            font-weight: 850;
            line-height: 1;
        }
        .section-icon {
            width: 1.72rem;
            height: 1.72rem;
            font-size: 0.88rem;
        }
        .field-icon {
            width: 1.35rem;
            height: 1.35rem;
            font-size: 0.72rem;
        }
        .helper-box {
            border-radius: 13px;
            background: rgba(247, 252, 253, 0.86);
            border: 1px solid rgba(191, 234, 242, 0.86);
            color: #6c7f93;
            font-size: 0.84rem;
            line-height: 1.38;
            font-style: italic;
            padding: 0.54rem 0.68rem;
            margin: 0.25rem 0 0.72rem;
        }
        .field-label {
            display: flex;
            align-items: center;
            gap: 0.44rem;
            color: var(--ink);
            font-size: 0.88rem;
            font-weight: 760;
            margin: 0.56rem 0 0.28rem;
        }
        .field-help {
            color: #72869a;
            font-size: 0.78rem;
            font-style: italic;
            margin: -0.08rem 0 0.32rem 1.8rem;
        }
        .card-muted {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.35;
        }
        .metric-value {
            color: var(--night);
            font-size: 1.25rem;
            font-weight: 800;
        }
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.28rem 0.58rem;
            border-radius: 999px;
            font-weight: 750;
            font-size: 0.76rem;
            border: 1px solid transparent;
            position: relative;
            z-index: 1;
        }
        .brand-badge {
            color: #063852;
            background: linear-gradient(135deg, rgba(33, 199, 217, 0.22), rgba(85, 214, 181, 0.24));
            border-color: rgba(33, 199, 217, 0.34);
            box-shadow: 0 8px 18px rgba(33, 199, 217, 0.12);
        }
        .badge-normal {
            color: #0f6b4f;
            background: #dff8ef;
            border-color: #b9ecd9;
        }
        .badge-opacity {
            color: #9a5600;
            background: #fff1d7;
            border-color: #ffd997;
        }
        .badge-uncertain {
            color: #345066;
            background: #e8f0f6;
            border-color: #cddce8;
        }
        .badge-error {
            color: #9f2d2d;
            background: #ffe4e4;
            border-color: #ffc8c8;
        }
        .result-summary-card {
            border-radius: 18px;
            background:
                linear-gradient(145deg, rgba(255, 255, 255, 0.94), rgba(240, 252, 255, 0.78));
            border: 1px solid rgba(33, 199, 217, 0.24);
            box-shadow: 0 16px 34px rgba(9, 37, 68, 0.08);
            padding: 1rem 1.05rem;
            margin-bottom: 0.72rem;
        }
        .result-summary-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.65rem;
            margin-bottom: 0.7rem;
        }
        .result-summary-title {
            color: var(--night);
            font-size: 1.04rem;
            font-weight: 820;
        }
        .result-summary-text {
            color: var(--ink);
            font-size: 0.94rem;
            line-height: 1.5;
            margin-bottom: 0.7rem;
        }
        .result-confidence {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.35rem 0.68rem;
            color: #0d4664;
            background: rgba(224, 248, 255, 0.92);
            border: 1px solid rgba(33, 199, 217, 0.28);
            font-size: 0.82rem;
            font-weight: 760;
        }
        .result-section {
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(33, 199, 217, 0.16);
            padding: 0.8rem 0.9rem;
            margin-bottom: 0.62rem;
        }
        .result-section-title {
            color: var(--night);
            font-size: 0.9rem;
            font-weight: 800;
            margin-bottom: 0.38rem;
        }
        .result-section-text,
        .result-list {
            color: var(--ink);
            font-size: 0.86rem;
            line-height: 1.45;
        }
        .result-list {
            margin: 0;
            padding-left: 1.05rem;
        }
        .result-list li {
            margin-bottom: 0.22rem;
        }
        .result-warning {
            border-radius: 16px;
            padding: 0.82rem 0.9rem;
            background: rgba(255, 248, 230, 0.88);
            border: 1px solid rgba(255, 217, 151, 0.95);
            color: #684c0a;
            font-size: 0.86rem;
            font-weight: 650;
            line-height: 1.45;
            margin-bottom: 0.68rem;
        }
        .warning-strip {
            padding: 0.48rem 0.7rem;
            border-radius: 12px;
            background: rgba(255, 248, 230, 0.80);
            border: 1px solid rgba(255, 226, 168, 0.85);
            color: #684c0a;
            font-weight: 620;
            font-size: 0.84rem;
            margin-bottom: 0.55rem;
        }
        .dashboard-cta {
            text-align: center;
            margin: clamp(3.2rem, 11vh, 7.5rem) auto 0.45rem;
            max-width: 30rem;
        }
        .dashboard-cta-logo {
            width: 4.1rem;
            height: 4.1rem;
            margin: 0 auto 0.72rem;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            color: #ffffff;
            font-weight: 860;
            font-size: 1.22rem;
            letter-spacing: 0;
            background: linear-gradient(135deg, #2563eb 0%, #22d3ee 58%, #2dd4bf 100%);
            border: 1px solid rgba(255, 255, 255, 0.72);
            box-shadow: 0 18px 38px rgba(37, 99, 235, 0.22);
        }
        .dashboard-cta-logo::before,
        .dashboard-cta-logo::after {
            content: "";
            position: absolute;
            left: 18%;
            right: 18%;
            height: 2px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.72);
        }
        .dashboard-cta-logo::before {
            top: 31%;
        }
        .dashboard-cta-logo::after {
            bottom: 31%;
        }
        .dashboard-cta-title {
            color: var(--night);
            font-size: 0.95rem;
            font-weight: 760;
            margin-bottom: 0.45rem;
        }
        .dashboard-cta-copy {
            color: var(--muted);
            font-size: 0.82rem;
            margin-bottom: 0.45rem;
        }
        .st-key-open_radios_cta {
            max-width: 300px;
            margin: 0.35rem auto 0;
        }
        .st-key-open_radios_cta button {
            min-height: 3rem !important;
            border-radius: 999px !important;
            font-size: 1.04rem !important;
            font-weight: 820 !important;
            justify-content: center !important;
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
        }
        [data-testid="stSidebar"] div.stButton {
            margin-bottom: 1.5rem;
        }
        .sidebar-active {
            padding: 0.72rem 0.78rem;
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(33, 199, 217, 0.22), rgba(85, 214, 181, 0.14));
            border: 1px solid rgba(85, 214, 181, 0.34);
            font-weight: 750;
            margin-bottom: 1.5rem;
        }
        .sidebar-separator {
            height: 1px;
            margin: 1.55rem 0 1.8rem;
            background: rgba(255, 255, 255, 0.16);
        }
        .sidebar-patient {
            padding: 0.52rem 0.62rem;
            border-radius: 13px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.10);
            margin: 0.4rem 0 1.15rem;
        }
        .st-key-guide_float_button {
            position: fixed;
            right: 24px;
            bottom: 24px;
            width: 62px;
            z-index: 10001;
        }
        .st-key-guide_float_button .stButton > button,
        .st-key-toggle_dashboard_guide_v2 button {
            width: 58px;
            height: 58px;
            min-height: 58px;
            border-radius: 999px;
            padding: 0 !important;
            justify-content: center !important;
            font-size: 1.35rem;
            font-weight: 850;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.58) !important;
            background: linear-gradient(135deg, #2563eb 0%, #22d3ee 58%, #2dd4bf 100%) !important;
            box-shadow: 0 16px 34px rgba(37, 99, 235, 0.30) !important;
        }
        .st-key-guide_float_button .stButton > button:hover,
        .st-key-toggle_dashboard_guide_v2 button:hover {
            transform: translateY(-2px);
            box-shadow: 0 20px 38px rgba(34, 211, 238, 0.32) !important;
        }
        .st-key-guide_float_panel {
            position: fixed;
            right: 24px;
            bottom: 92px;
            width: min(400px, calc(100vw - 48px));
            height: min(540px, calc(100vh - 118px));
            max-height: min(540px, calc(100vh - 118px));
            z-index: 10000;
            padding: 0;
            border-radius: 22px;
            overflow: hidden;
            background: #f7fcfd;
            border: 1px solid #bfeaf2;
            box-shadow: 0 20px 48px rgba(34, 98, 141, 0.18);
        }
        .st-key-guide_float_panel * {
            box-sizing: border-box;
        }
        .st-key-guide_float_panel > div[data-testid="stVerticalBlock"] {
            height: 100%;
            max-height: min(540px, calc(100vh - 118px));
            gap: 0;
        }
        .st-key-guide_float_panel .stTextInput label {
            display: none;
        }
        .st-key-guide_chat_header {
            padding: 0.88rem 1rem 0.78rem;
            background: linear-gradient(135deg, #f2f9ff 0%, #ecfeff 55%, #effdf8 100%);
            border-bottom: 1px solid #bfeaf2;
        }
        .st-key-guide_chat_header > div[data-testid="stVerticalBlock"] {
            gap: 0;
        }
        .guide-title {
            color: #163a63;
            font-weight: 800;
            font-size: 1.02rem;
            margin: 0;
        }
        .guide-copy {
            color: #55738b;
            font-size: 0.78rem;
            line-height: 1.32;
            margin-top: 0.12rem;
        }
        .guide-chat-body {
            height: 348px;
            padding: 0.9rem 0.95rem 0.7rem;
            background:
                radial-gradient(circle at 90% 10%, rgba(34, 211, 238, 0.10), transparent 8rem),
                #f7fcfd;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            gap: 0.66rem;
        }
        .guide-bubble {
            max-width: 90%;
            padding: 0.66rem 0.78rem;
            border-radius: 16px;
            font-size: 0.84rem;
            line-height: 1.42;
            margin-bottom: 0;
        }
        .guide-bubble-assistant {
            background: #ffffff;
            color: #16324f;
            border: 1px solid #cceff5;
            box-shadow: 0 8px 18px rgba(34, 98, 141, 0.055);
        }
        .guide-bubble-user {
            margin-left: auto;
            background: linear-gradient(135deg, #e3f7ff, #e7fffb);
            color: #16324f;
            border: 1px solid #bfeaf2;
        }
        .guide-chat-form {
            padding: 0.95rem 0.75rem 1rem;
            background: #ffffff;
            border-top: 1px solid #d8eef3;
        }
        .st-key-guide_float_panel div[data-testid="stForm"] {
            border: 0;
            padding: 0;
            margin: 0;
            width: 100%;
            max-width: 100%;
            overflow: hidden;
        }
        .st-key-guide_float_panel div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
            gap: 0.35rem;
            align-items: center;
            width: 100%;
            max-width: 100%;
            flex-wrap: nowrap;
        }
        .st-key-guide_float_panel div[data-testid="column"] {
            min-width: 0;
        }
        .st-key-guide_float_panel div[data-testid="stForm"] div[data-testid="column"]:first-child {
            flex: 1 1 auto;
            width: auto !important;
            max-width: none !important;
        }
        .st-key-guide_float_panel div[data-testid="stForm"] div[data-testid="column"]:last-child {
            flex: 0 0 86px;
            width: 86px !important;
            max-width: 86px !important;
        }
        .st-key-guide_float_panel .stTextInput,
        .st-key-guide_float_panel .stFormSubmitButton {
            width: 100%;
        }
        .st-key-guide_float_panel .stTextInput > div,
        .st-key-guide_float_panel .stTextInput > div > div {
            width: 100%;
        }
        .st-key-guide_float_panel .stTextInput > div > div,
        .st-key-guide_float_panel .st-b3 {
            border-radius: 22px !important;
            border: 1px solid #bfeaf2 !important;
            background: #fbfeff !important;
            box-shadow: 0 5px 14px rgba(34, 98, 141, 0.045) !important;
        }
        .st-key-guide_float_panel .stTextInput > div > div:focus-within {
            border-color: #22d3ee !important;
            box-shadow: 0 0 0 2px rgba(34, 211, 238, 0.18) !important;
        }
        .st-key-guide_float_panel input {
            width: 100% !important;
            min-height: 2.45rem !important;
            border-radius: 22px !important;
            border: 0 !important;
            background: #fbfeff !important;
            color: #16324f !important;
            box-shadow: none !important;
            padding-left: 0.9rem !important;
            padding-right: 0.9rem !important;
        }
        .st-key-guide_float_panel input:focus {
            border: 0 !important;
            box-shadow: none !important;
            outline: none !important;
        }
        .st-key-guide_float_panel input::placeholder {
            color: #6f8799 !important;
        }
        .st-key-guide_float_panel .stButton > button {
            min-height: 2.55rem;
            background: #eefcff !important;
            color: #163a63 !important;
            border: 1px solid #bfeaf2 !important;
            box-shadow: none !important;
        }
        .st-key-guide_chat_header .stButton > button,
        .st-key-close_dashboard_guide_v2 button {
            width: 2rem;
            height: 2rem;
            min-height: 2rem;
            padding: 0 !important;
            border-radius: 999px;
            background: linear-gradient(135deg, #e6fbff 0%, #d9fbf5 100%) !important;
            color: #2563eb !important;
            border: 1px solid #8be7f4 !important;
            font-size: 1rem;
            box-shadow: none !important;
            justify-content: center !important;
        }
        .st-key-guide_chat_header .stButton > button:hover,
        .st-key-close_dashboard_guide_v2 button:hover {
            background: linear-gradient(135deg, #dff8ff 0%, #cdf9ee 100%) !important;
            color: #163a63 !important;
            border-color: #22d3ee !important;
        }
        .st-key-guide_float_panel .stFormSubmitButton > button {
            min-height: 2.45rem;
            border-radius: 22px;
            padding-left: 0.55rem !important;
            padding-right: 0.55rem !important;
            background: linear-gradient(135deg, #2563eb 0%, #22d3ee 58%, #2dd4bf 100%) !important;
            color: #ffffff !important;
            border: 1px solid rgba(34, 211, 238, 0.46) !important;
            box-shadow: 0 10px 22px rgba(37, 99, 235, 0.18) !important;
            font-size: 0.9rem;
        }
        .chat-box {
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.74);
            border-radius: 13px;
            padding: 0.58rem;
            margin-bottom: 0.42rem;
        }
        .chat-user {
            border-left: 4px solid var(--medical);
        }
        .chat-assistant {
            border-left: 4px solid var(--mint);
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 999px;
            border: 1px solid rgba(33, 199, 217, 0.34);
            background: linear-gradient(135deg, #145f9a 0%, #1f8ed6 46%, #21c7d9 100%);
            color: white;
            font-weight: 760;
            min-height: 2.35rem;
            box-shadow: 0 8px 20px rgba(31, 142, 214, 0.18);
            transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
        }
        .stButton > button:hover {
            border-color: rgba(85, 214, 181, 0.82);
            color: white;
            transform: translateY(-1px);
            box-shadow: 0 12px 26px rgba(33, 199, 217, 0.22);
        }
        div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"] {
            min-height: 2.55rem;
            font-size: 0.94rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.62);
            border: 1px solid rgba(33, 199, 217, 0.12);
            border-radius: 14px;
            padding: 0.45rem 0.55rem;
        }
        div[data-testid="stMetricValue"] {
            color: var(--night);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f'<div class="radiox-subtitle">{_safe(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="radiox-header">
            <div class="badge brand-badge">+ RadioX</div>
            <h1 class="radiox-title">{_safe(title)}</h1>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    if not st.session_state.authenticated:
        return
    with st.sidebar:
        st.markdown("## RadioX")
        st.caption("Prototype pedagogique")
        page = st.session_state.current_page
        if page == "dashboard":
            st.markdown('<div class="sidebar-active">Dashboard</div>', unsafe_allow_html=True)
        elif st.button("Tableau de bord", use_container_width=True):
            navigate("dashboard")
        if page == "patient_radios":
            st.markdown('<div class="sidebar-active">Mes radios patient</div>', unsafe_allow_html=True)
        elif st.button("Radios patient", use_container_width=True):
            navigate("patient_radios")
        if page == "thoracic_xray":
            st.markdown('<div class="sidebar-active">Radio thoracique</div>', unsafe_allow_html=True)
        elif st.button("Analyse radio", use_container_width=True):
            navigate("thoracic_xray")
        st.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html=True)
        patient = st.session_state.selected_patient or {}
        st.caption("Patient selectionne")
        st.markdown(
            f'<div class="sidebar-patient">{_safe(patient.get("name", "Aucun patient"))}</div>',
            unsafe_allow_html=True,
        )
        if st.button("Quitter la session", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.current_page = "login"
            st.session_state.last_prediction_json = None
            st.session_state.chat_history = []
            rerun()


def card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="glass-card dashboard-card">
            <div class="card-title">{_safe(title)}</div>
            <div class="card-muted">{_safe(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def answer_dashboard_guide(question: str) -> str:
    question = question.strip().lower()
    if not question:
        return (
            "Bonjour, que puis-je faire pour vous ? Je peux vous guider pour ouvrir les radios, "
            "choisir un patient fictif, lancer une analyse ou comprendre le JSON."
        )

    if question in {"bonjour", "salut", "hello", "coucou", "bonsoir"}:
        return (
            "Bonjour, que puis-je faire pour vous ? Vous pouvez me demander comment commencer, "
            "quel bouton cliquer, quels modes sont disponibles ou ce que signifie le JSON."
        )
    if question in {"ok", "d'accord", "dac", "merci", "super"}:
        return "Avec plaisir. Pour continuer, cliquez sur le bouton bleu nomme Ouvrir les radios."
    if any(term in question for term in ["aide", "help", "quoi faire", "que faire", "comment utiliser"]):
        return (
            "Les seules fonctionnalites proposees ici sont: ouvrir des patients fictifs, selectionner ou uploader "
            "une radio thoracique, lancer une analyse technique, afficher le JSON et expliquer ce JSON."
        )
    if any(term in question for term in ["fonction", "fonctionnalite", "possible", "propose"]):
        return "Fonctionnalites: " + "; ".join(RADIOX_TUTORIAL_JSON["fonctionnalites"]) + "."
    if any(term in question for term in ["bouton", "cliquer", "clic", "ouvrir les radios"]):
        buttons = RADIOX_TUTORIAL_JSON["boutons"]
        return f"{buttons['dashboard']} Ensuite, {buttons['patients']} Enfin, {buttons['analyse']}"
    if any(term in question for term in ["attention", "warning", "risque", "clinique", "medical"]):
        return (
            "Attention: RadioX est un prototype pedagogique. Il ne fournit pas de diagnostic medical, "
            "ne remplace pas un professionnel de sante et n'est pas valide cliniquement."
        )
    if any(term in question for term in ["diagnostic", "diagnosti", "maladie", "traitement", "soigner"]):
        return (
            "RadioX ne fournit pas de diagnostic. Il explique seulement une sortie experimentale "
            "qui doit etre validee par un professionnel qualifie."
        )
    if "commencer" in question or "debut" in question or "premier" in question:
        return (
            "Pour commencer, cliquez sur le bouton bleu nomme Ouvrir les radios. "
            "Choisissez ensuite un patient fictif, puis ouvrez la page Radio thoracique."
        )
    if "analyser" in question or "analyse" in question or "radio" in question:
        return (
            "Pour analyser une radio, ouvrez un patient fictif, deposez une radiographie thoracique frontale, "
            "choisissez remote_medgemma ou mock_medgemma, puis lancez l'analyse."
        )
    if "deposer" in question or "upload" in question or "image" in question:
        return "L'image se depose dans la page Radio thoracique, dans la colonne Parametres, via le champ d'upload."
    if "modele" in question or "mode" in question or "medgemma" in question:
        return "Utilisez remote_medgemma pour la demo avec Colab actif. Si Colab n'est pas disponible, utilisez mock_medgemma pour tester l'interface localement."
    if "json" in question:
        fields = RADIOX_TUTORIAL_JSON["json_fields"]
        return (
            "Le JSON structure la sortie du prototype: classe, confiance, observations, justification, limites et warning. "
            f"Par exemple, confidence signifie: {fields['confidence']}."
        )
    if "confiance" in question or "confidence" in question:
        return "La confiance est une valeur entre 0 et 1 retournee par le prototype. Elle ne doit pas etre interpretee comme une probabilite clinique calibree."
    if "suspected" in question or "opacity" in question or "opacite" in question:
        return "Opacite suspectee signifie que le prototype signale une zone dense possible. Ce n'est pas un diagnostic et cela doit etre verifie par un professionnel qualifie."
    if "colab" in question or "ngrok" in question or "marche pas" in question or "erreur" in question:
        return "Si Colab ou ngrok ne marche pas, verifiez l'URL remote_medgemma. Pour continuer la demo sans GPU distant, choisissez mock_medgemma."

    return (
        "Desolee, je n'ai pas compris. Vous pouvez demander: comment commencer, quel bouton cliquer, "
        "quelles fonctionnalites sont proposees, ce que signifie le JSON, ou quoi faire si Colab/ngrok ne marche pas."
    )


def render_floating_dashboard_guide() -> None:
    if not st.session_state.dashboard_guide_open:
        with st.container(key="guide_float_button"):
            if st.button("💬", key="open_dashboard_guide", help="Ouvrir le Guide RadioX"):
                st.session_state.dashboard_guide_open = True
                if not st.session_state.dashboard_guide_answer:
                    st.session_state.dashboard_guide_answer = answer_dashboard_guide("")
                rerun()
        return

    with st.container(key="guide_float_panel"):
        with st.container(key="guide_chat_header"):
            head_col, close_col = st.columns([0.82, 0.18], vertical_alignment="center")
            with head_col:
                st.markdown(
                    """
                    <div class="guide-title">Assistant RadioX</div>
                    <div class="guide-copy">Guide rapide</div>
                    """,
                    unsafe_allow_html=True,
                )
            with close_col:
                if st.button("×", key="close_dashboard_guide", help="Fermer"):
                    st.session_state.dashboard_guide_open = False
                    rerun()

        st.markdown('<div class="guide-chat-body">', unsafe_allow_html=True)
        st.markdown(
            '<div class="guide-bubble guide-bubble-assistant">Bonjour. Je peux vous aider a utiliser RadioX. Posez une question simple.</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.dashboard_guide_question:
            st.markdown(
                f'<div class="guide-bubble guide-bubble-user">{_safe(st.session_state.dashboard_guide_question)}</div>',
                unsafe_allow_html=True,
            )
        answer = st.session_state.dashboard_guide_answer or answer_dashboard_guide("")
        st.markdown(f'<div class="guide-bubble guide-bubble-assistant">{_safe(answer)}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="guide-chat-form">', unsafe_allow_html=True)
        with st.form("dashboard-guide-floating-form", clear_on_submit=True):
            input_col, send_col = st.columns([0.72, 0.28], vertical_alignment="center")
            with input_col:
                question = st.text_input(
                    "Question au guide",
                    placeholder="Posez votre question...",
                    label_visibility="collapsed",
                )
            with send_col:
                submitted = st.form_submit_button("Envoyer", use_container_width=True)
        if submitted:
            st.session_state.dashboard_guide_question = question.strip()
            st.session_state.dashboard_guide_answer = answer_dashboard_guide(question)
            rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def render_floating_dashboard_guide_v2() -> None:
    with st.container(key="guide_float_button"):
        if st.button("?", key="toggle_dashboard_guide_v2", help="Ouvrir ou fermer le Guide RadioX"):
            st.session_state.dashboard_guide_open = not st.session_state.dashboard_guide_open
            if st.session_state.dashboard_guide_open and not st.session_state.guide_messages:
                st.session_state.guide_messages = [
                    {
                            "role": "assistant",
                            "content": "Bonjour, que puis-je faire pour vous ? Je peux vous guider dans la demo RadioX.",
                    }
                ]
            rerun()

    if not st.session_state.dashboard_guide_open:
        return

    with st.container(key="guide_float_panel"):
        with st.container(key="guide_chat_header"):
            head_col, close_col = st.columns([0.82, 0.18], vertical_alignment="center")
            with head_col:
                st.markdown(
                    """
                    <div class="guide-title">Assistant RadioX</div>
                    <div class="guide-copy">Guide rapide</div>
                    """,
                    unsafe_allow_html=True,
                )
            with close_col:
                if st.button("x", key="close_dashboard_guide_v2", help="Fermer"):
                    st.session_state.dashboard_guide_open = False
                    rerun()

        if not st.session_state.guide_messages:
            st.session_state.guide_messages = [
                {
                    "role": "assistant",
                    "content": "Bonjour, que puis-je faire pour vous ? Je peux vous guider dans la demo RadioX.",
                }
            ]
        elif st.session_state.guide_messages[0].get("content", "").startswith("Bonjour. Je peux vous aider"):
            st.session_state.guide_messages[0] = {
                "role": "assistant",
                "content": "Bonjour, que puis-je faire pour vous ? Je peux vous guider dans la demo RadioX.",
            }
        message_html = ['<div class="guide-chat-body">']
        for message in st.session_state.guide_messages[-6:]:
            css_class = "guide-bubble-user" if message.get("role") == "user" else "guide-bubble-assistant"
            message_html.append(
                f'<div class="guide-bubble {css_class}">{_safe(message.get("content", ""))}</div>'
            )
        if st.session_state.guide_is_typing:
            message_html.append('<div class="guide-bubble guide-bubble-assistant">Assistant RadioX ecrit...</div>')
        message_html.append("</div>")
        st.markdown("".join(message_html), unsafe_allow_html=True)

        st.markdown('<div class="guide-chat-form">', unsafe_allow_html=True)
        with st.form("dashboard-guide-floating-form-v2", clear_on_submit=True):
            input_col, send_col = st.columns([0.68, 0.32], vertical_alignment="center")
            with input_col:
                question = st.text_input(
                    "Question au guide",
                    placeholder="Posez votre question...",
                    label_visibility="collapsed",
                )
            with send_col:
                submitted = st.form_submit_button("Envoyer", use_container_width=True)
        if submitted:
            cleaned_question = question.strip()
            if cleaned_question:
                st.session_state.guide_messages.append({"role": "user", "content": cleaned_question})
                st.session_state.dashboard_guide_question = cleaned_question
                st.session_state.guide_pending_answer = answer_dashboard_guide(cleaned_question)
                st.session_state.guide_is_typing = True
                rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.guide_is_typing:
        time.sleep(3)
        st.session_state.guide_messages.append(
            {"role": "assistant", "content": st.session_state.guide_pending_answer or answer_dashboard_guide("")}
        )
        st.session_state.dashboard_guide_answer = st.session_state.guide_pending_answer
        st.session_state.guide_pending_answer = ""
        st.session_state.guide_is_typing = False
        rerun()


def login_page() -> None:
    with st.container(key="login_shell"):
        left, right = st.columns([0.58, 0.42], gap="large", vertical_alignment="center")
        with left:
            st.markdown(
                """
                <div class="login-left-panel">
                    <div class="login-kicker">Portail RadioX</div>
                    <h1 class="login-hero-title">Bienvenue sur RadioX</h1>
                    <div class="login-hero-subtitle">
                        Assistant radiologue virtuel responsable
                    </div>
                    <div class="login-hero-copy">
                        Analyse prudente de radiographies thoraciques avec sortie JSON structuree,
                        tracabilite et avertissement medical.
                    </div>
                    <div class="login-badges">
                        <span class="login-badge">JSON</span>
                        <span class="login-badge">Warning</span>
                        <span class="login-badge">Tracable</span>
                        <span class="login-badge">Pedagogique</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            with st.container(key="login_card"):
                st.markdown(
                    """
                    <h2 class="login-title">Connexion</h2>
                    """,
                    unsafe_allow_html=True,
                )
                username = st.text_input("Identifiant", placeholder="demo")
                password = st.text_input("Mot de passe", placeholder="demo", type="password")
                if st.button("SE CONNECTER", type="primary", use_container_width=True):
                    if username.strip() and password.strip():
                        st.session_state.authenticated = True
                        st.session_state.current_page = "dashboard"
                        rerun()
                    else:
                        st.error("Renseigner un identifiant et un mot de passe non vides.")
                if st.button("CREER UN NOUVEAU COMPTE", use_container_width=True):
                    st.info("La creation de compte n'est pas activee dans ce prototype pedagogique.")
                    if hasattr(st, "toast"):
                        st.toast("Creation de compte non activee dans ce prototype pedagogique.")


def dashboard_page() -> None:
    st.markdown(
        """
        <div class="dashboard-header">
            <h1 class="dashboard-title">Bienvenue sur RadioX</h1>
            <div class="dashboard-subtitle">
                Plateforme demo pour analyser une radiographie thoracique avec un pipeline prudent.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    with cols[0]:
        card("Mes radios patient", "Consulter les dossiers fictifs prevus pour la demonstration.")
    with cols[1]:
        card("Analyses recentes", "Retrouver la derniere sortie JSON conservee dans la session.")
    with cols[2]:
        card("Evaluation technique", "Mini-echantillon equilibre valide pour integration, pas performance clinique.")
    with cols[3]:
        card("Aide medicale prudente", "Le chatbot explique uniquement le JSON sans poser de diagnostic.")

    st.markdown(
        """
        <div class="dashboard-cta">
            <div class="dashboard-cta-logo" aria-hidden="true">RX</div>
            <div class="dashboard-cta-title">Commencer une analyse patient</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, middle, right = st.columns([1.1, 1.35, 1.1])
    with middle:
        if st.button("Ouvrir les radios", key="open_radios_cta", type="primary", use_container_width=True):
            navigate("patient_radios")


def patient_radios_page() -> None:
    header("Mes radios patient", "Donnees patient fictives pour demonstration.")
    st.info("Donnees patient fictives pour demonstration. Ne jamais saisir de donnees personnelles reelles.")
    for patient in DEMO_PATIENTS:
        col_info, col_action = st.columns([4, 1])
        with col_info:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="card-title">{_safe(patient["name"])}</div>
                    <div class="card-muted">Age fictif: {_safe(patient["age"])}<br>Examen disponible: {_safe(patient["exam"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_action:
            st.write("")
            st.write("")
            if st.button("Ouvrir", key=f"open-{patient['id']}", use_container_width=True):
                st.session_state.selected_patient = patient
                navigate("thoracic_xray")


def prediction_items(pred: dict[str, Any]) -> dict[str, Any]:
    predicted = pred.get("predicted_class") or pred.get("class") or "uncertain"
    observations = _as_text_list(pred.get("visual_evidence") or pred.get("observations"))
    limitations = _as_text_list(pred.get("limitations") or pred.get("limits"))
    return {
        "class": str(predicted),
        "confidence": _confidence(pred),
        "observations": observations,
        "justification": pred.get("justification") or "Aucune justification retournee.",
        "limits": limitations,
        "warning": pred.get("warning") or "Prototype pedagogique. Non destine au diagnostic.",
        "latency_ms": pred.get("latency_ms"),
        "pipeline_mode": pred.get("pipeline_mode"),
        "error_detail": pred.get("error_detail"),
    }


def render_result_cards(pred: dict[str, Any]) -> None:
    items = prediction_items(pred)
    predicted_class = items["class"]
    confidence = items["confidence"]
    has_error = bool(items["error_detail"])
    badge = badge_class(predicted_class, has_error)
    readable = readable_class(predicted_class)
    confidence_pct = confidence_percent(confidence)
    confidence_text = confidence_level(confidence)

    if has_error:
        st.error(str(items["error_detail"]))
    if _as_text_list(pred.get("guardrail_errors")):
        st.warning("Sortie corrigee par les garde-fous du prototype.")

    st.markdown(
        f"""
        <div class="result-summary-card">
            <div class="result-summary-head">
                <div class="result-summary-title">Résumé de l'analyse</div>
                <div class="{badge}">{_safe(readable)}</div>
            </div>
            <div class="result-summary-text">{_safe(analysis_summary(predicted_class))}</div>
            <div class="result-confidence">
                Confiance du modèle : {confidence_pct} % &middot; {_safe(confidence_text)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    observations = items["observations"]
    st.markdown(
        f"""
        <div class="result-section">
            <div class="result-section-title">Ce que le modèle observe</div>
            {html_list(observations, "Aucune observation visuelle exploitable n'a été retournée.")}
        </div>
        <div class="result-section">
            <div class="result-section-title">Pourquoi le modèle propose ce résultat</div>
            <div class="result-section-text">{_safe(items["justification"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="result-section">
            <div class="result-section-title">Limites</div>
            {html_list(items["limits"], "Limites non précisées par le modèle.")}
        </div>
        <div class="result-warning">
            <div class="result-section-title">Avertissement</div>
            {_safe(items["warning"])}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Comprendre les classes", expanded=False):
        st.markdown(
            """
            - **Aspect normal** : aucune anomalie évidente détectée par le modèle.
            - **Opacité suspectée** : zone blanche ou dense pouvant indiquer une anomalie à vérifier.
            - **Résultat incertain** : le modèle ne peut pas conclure de manière fiable.
            """
        )
    with st.expander("Détails techniques / JSON brut", expanded=False):
        st.json(pred)


def answer_from_prediction(question: str, prediction: dict[str, Any] | None) -> str:
    if not prediction:
        return "Aucune analyse disponible pour le moment. Veuillez d'abord lancer le pipeline."

    q = question.lower()
    clinical_terms = [
        "diagnostic",
        "diagnostique",
        "traitement",
        "soigner",
        "medicament",
        "grave",
        "urgence",
        "conduite",
        "que faire",
    ]
    if any(term in q for term in clinical_terms):
        return (
            "Je ne peux pas fournir de diagnostic ou de conduite medicale. "
            "Je peux seulement expliquer la sortie JSON du prototype. "
            "Validation par un professionnel qualifie requise."
        )

    items = prediction_items(prediction)
    predicted = items["class"]
    confidence = items["confidence"]
    confidence_pct = confidence_percent(confidence)
    confidence_text = confidence_level(confidence)
    readable = readable_class(predicted)
    observations = items["observations"]
    limits = items["limits"]

    if "classe" in q or "predit" in q or "prediction" in q:
        return f"Le resultat affiche est: {readable}."
    if "confiance" in q or "score" in q or "niveau" in q:
        return f"Confiance du modele : {confidence_pct} % · {confidence_text}."
    if "pourquoi" in q or "justification" in q or "dit ca" in q:
        return f"Justification retournee: {items['justification']}"
    if "observation" in q or "voit" in q or "evidence" in q:
        if observations:
            return "Observations disponibles: " + "; ".join(observations)
        return "Le JSON ne contient pas d'observation exploitable."
    if "limite" in q or "limits" in q:
        if limits:
            return "Limites indiquees: " + "; ".join(limits)
        return "Le JSON ne precise pas de limite detaillee, mais le systeme reste un prototype pedagogique."
    if "suspected" in q or "opacity" in q or "opacite" in q or "opacité" in q:
        return (
            "Opacité suspectée signifie que le modèle signale une zone blanche ou dense possible "
            "sur la radiographie. Ce n'est pas un diagnostic; cela doit être vérifié par un "
            "professionnel qualifié."
        )
    if "latence" in q or "temps" in q:
        latency = items.get("latency_ms")
        if latency is None:
            return "Le JSON ne contient pas de latence."
        return f"La latence retournee est d'environ {latency} ms."
    if "mode" in q or "pipeline" in q:
        return f"Mode pipeline indique dans le JSON: `{items.get('pipeline_mode') or 'non precise'}`."
    if "resume" in q or "résume" in q or "simple" in q or "resumer" in q or "résumer" in q:
        if predicted == "normal":
            return (
                f"Le modele ne detecte pas d'anomalie evidente sur la radio, avec une {confidence_text} "
                f"de {confidence_pct} %. Cela ne constitue pas un diagnostic; une verification medicale "
                "reste necessaire si le contexte clinique l'exige."
            )
        if predicted in {"suspected_opacity", "pneumonia_suspected"}:
            return (
                f"Le modele pense voir une opacite suspecte sur la radio avec une {confidence_text} "
                f"de {confidence_pct} %. Cela ne constitue pas un diagnostic, mais indique qu'une "
                "verification par un professionnel est necessaire."
            )
        return (
            f"Le modele n'est pas assez sur pour conclure de maniere fiable, avec une {confidence_text} "
            f"de {confidence_pct} %. Cela ne constitue pas un diagnostic et doit etre interprete "
            "avec prudence."
        )

    return (
        f"Je peux expliquer le resultat affiche. Synthese: {readable}, confiance du modele: "
        f"{confidence_pct} % · {confidence_text}. Cette sortie est pedagogique et non diagnostique."
    )


def render_chatbot() -> None:
    st.markdown(
        f"""
        <div class="glass-card">
            {section_title("?", "Assistant d'aide RadioX")}
            {helper_box("Réponses locales basées uniquement sur le JSON retourné.")}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not os.environ.get("RADIOX_CHATBOT_API_KEY"):
        st.markdown(helper_box("Mode local actif : aucune API externe appelée."), unsafe_allow_html=True)

    for idx, message in enumerate(st.session_state.chat_history[-8:]):
        role = message.get("role", "assistant")
        css = "chat-user" if role == "user" else "chat-assistant"
        label = "Vous" if role == "user" else "RadioX"
        st.markdown(
            f'<div class="chat-box {css}"><b>{_safe(label)}</b><br>{_safe(message.get("content", ""))}</div>',
            unsafe_allow_html=True,
        )

    with st.form("radiox-chat-form", clear_on_submit=True):
        st.markdown(
            field_label("?", "Question sur le résultat JSON", "Demandez un résumé simple ou une explication d'un champ."),
            unsafe_allow_html=True,
        )
        question = st.text_input(
            "Question sur le resultat JSON",
            placeholder="Resume le resultat simplement.",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Envoyer", use_container_width=True)
    if submitted and question.strip():
        answer = answer_from_prediction(question, st.session_state.last_prediction_json)
        st.session_state.chat_history.append({"role": "user", "content": question.strip()})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        rerun()


def selected_image_from_inputs(uploaded_file: Any, selected_case: dict[str, str] | None) -> Path | None:
    if uploaded_file:
        suffix = Path(uploaded_file.name).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            return Path(tmp.name)
    if selected_case:
        return PROJECT_ROOT / selected_case["image_path"]
    return None


def thoracic_xray_page() -> None:
    patient = st.session_state.selected_patient or DEMO_PATIENTS[0]
    header("Radio thoracique", f"{patient.get('name', 'Patient demo')} - donnees fictives pour demonstration.")

    left, center, right = st.columns([0.9, 1.35, 1.05])

    with left:
        st.markdown(
            f'<div class="glass-card">{section_title("⚙", "Paramètres")}'
            f'{helper_box("Choisissez une image, un mode d’analyse, puis lancez la lecture technique.")}',
            unsafe_allow_html=True,
        )
        st.markdown(
            field_label("↑", "Upload radiographie thoracique frontale", "Formats acceptés : PNG, JPG ou JPEG."),
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Upload radiographie thoracique frontale",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        st.markdown(
            field_label("◇", "Modèle / pipeline", "remote_medgemma pour Colab, mock_medgemma pour tester l’interface."),
            unsafe_allow_html=True,
        )
        mode = st.selectbox(
            "Modele / pipeline",
            APP_MODES,
            index=APP_MODES.index("remote_medgemma"),
            label_visibility="collapsed",
        )
        remote_url = ""
        if mode == "remote_medgemma":
            st.markdown(
                field_label("↗", "URL API Colab/ngrok", "Collez ici l’adresse publique de votre session Colab."),
                unsafe_allow_html=True,
            )
            remote_url = st.text_input(
                "URL API Colab/ngrok",
                value=os.environ.get("REMOTE_MEDGEMMA_URL", ""),
                placeholder="https://xxxx.ngrok-free.dev",
                label_visibility="collapsed",
            )

        test_cases = load_split_test_cases()
        selected_case = None
        if test_cases:
            st.markdown(
                field_label("✓", "Image de démonstration", "Optionnel : utiliser une image RSNA déjà présente dans le projet."),
                unsafe_allow_html=True,
            )
            use_test_image = st.checkbox("Tester une image du split test RSNA", label_visibility="collapsed")
            if use_test_image:
                st.markdown(
                    field_label("▣", "Choisir une image RSNA", "Sélectionnez un exemple de test pour la démonstration."),
                    unsafe_allow_html=True,
                )
                selected_case = st.selectbox(
                    "Image du split test",
                    test_cases,
                    format_func=lambda row: f"{Path(row['image_path']).name} | {row['label']}",
                    label_visibility="collapsed",
                )
        analyze = st.button("Analyser la radio", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if mode == "remote_medgemma":
            cleaned = remote_url.strip()
            with st.expander("Détails techniques de connexion", expanded=False):
                st.caption("Cette section sert uniquement à vérifier l'appel vers l'API Colab/ngrok.")
                st.write("Mode selectionne:", mode)
                st.write("URL remote utilisee:", cleaned or "(vide)")
                st.write("Endpoint final appele:", f"{cleaned.rstrip('/')}/predict" if cleaned else "(vide)")

    image_path: Path | None = None
    if analyze:
        try:
            image_path = selected_image_from_inputs(uploaded, selected_case)
            if image_path is None or not image_path.exists():
                st.error("Image absente ou chemin introuvable.")
                st.stop()
        except Exception as exc:
            st.error(f"Image absente ou illisible: {exc}")
            st.stop()

        try:
            if mode == "medgemma":
                with st.spinner("Chargement/analyse MedGemma en cours, cela peut prendre plusieurs minutes."):
                    try:
                        resources = cached_medgemma_resources()
                    except Exception:
                        resources = None
                    prediction = run_pipeline(image_path, mode=mode, medgemma_resources=resources)
            elif mode == "remote_medgemma":
                if not remote_url.strip():
                    st.error("API indisponible: renseigner l'URL Colab/ngrok avant de lancer remote_medgemma.")
                    st.stop()
                with st.spinner("Analyse remote MedGemma en cours via Colab, cela peut prendre plusieurs minutes."):
                    prediction = run_pipeline(image_path, mode=mode, remote_url=remote_url.strip())
            else:
                prediction = run_pipeline(image_path, mode=mode)
        except Exception as exc:
            st.error(f"Erreur pendant l'analyse: {exc}")
            st.stop()
        st.session_state.last_prediction_json = prediction
        st.session_state.chat_history = []
        st.session_state.last_image_path = str(image_path)

    with center:
        st.markdown(
            f'<div class="glass-card">{section_title("▣", "Image analysée")}'
            f'{helper_box("Chargez une image ou choisissez une image RSNA de test, puis lancez l’analyse.")}',
            unsafe_allow_html=True,
        )
        last_image = st.session_state.get("last_image_path")
        preview_path = image_path or (Path(last_image) if last_image else None)
        if preview_path and preview_path.exists():
            try:
                st.image(Image.open(preview_path), caption="Radiographie analysee", use_container_width=True)
            except Exception as exc:
                st.error(f"Image absente ou illisible: {exc}")
        else:
            st.info("Aucune image n’est affichée pour le moment.")
        st.markdown("</div>", unsafe_allow_html=True)

        prediction = st.session_state.last_prediction_json
        if prediction:
            render_result_cards(prediction)
        else:
            st.info("Aucune analyse disponible pour le moment.")

    with right:
        render_chatbot()


def main() -> None:
    st.set_page_config(page_title="RadioX", page_icon="RX", layout="wide")
    ensure_session_state()
    apply_css()

    if not st.session_state.authenticated:
        st.session_state.current_page = "login"
        login_page()
        return

    render_sidebar()
    page = st.session_state.current_page
    if page == "dashboard":
        dashboard_page()
    elif page == "patient_radios":
        patient_radios_page()
    elif page == "thoracic_xray":
        thoracic_xray_page()
    else:
        dashboard_page()
    render_floating_dashboard_guide_v2()


if __name__ == "__main__":
    main()

