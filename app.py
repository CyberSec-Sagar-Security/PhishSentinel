"""
PhishLens — Streamlit Web Interface.

Full-stack phishing email analysis dashboard powered by:
  - 5 ML models (LightGBM, XGBoost, Random Forest, CatBoost, Logistic Regression)
  - 961-feature vector (header + URL + HTML + text + embedding + TF-IDF + intel + anomaly)
  - 6 threat-intelligence APIs (VirusTotal, GSB, URLScan, URLhaus, AbuseIPDB, IPQS)
  - ChatGPT gpt-4.1-mini forensic analysis layer
  - SHAP + LIME explainability
  - MITRE ATT&CK technique mapping
  - IOC extraction (MISP-compatible JSON export)
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ── Load .env before anything else (API keys) ────────────────────────────────
# The .env file lives in phishlens/ subfolder
_ENV_FILE = Path(__file__).parent / "phishlens" / ".env"
if _ENV_FILE.exists():
    load_dotenv(dotenv_path=_ENV_FILE, override=True)
else:
    # Fallback: try .env in repo root
    load_dotenv(override=True)

# ── Streamlit page config must be first ──────────────────────────────────────
st.set_page_config(
    page_title="PhishLens",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/CyberSec-Sagar-Security/PhishSentinel",
        "Report a bug": "https://github.com/CyberSec-Sagar-Security/PhishSentinel/issues",
        "About": "PhishLens — ML-powered phishing email detection system",
    },
)

# ── Ensure src/ is importable when run from repo root ────────────────────────
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Lazy module imports with user-friendly error ──────────────────────────────
try:
    from src.ingestion.eml_parser import parse_eml_file, parse_eml_string
    from src.features.pipeline import FeaturePipeline
    from src.ioc_extractor import extract_iocs, generate_ioc_explanations, iocs_to_misp_json, iocs_to_syslog
    from src.attack_mapping import map_attack_techniques
    from src.features.openai_analyzer import analyse_email_with_openai
    from src.utils.config import DEFAULT_CONFIG
    from src.utils.logger import get_logger
except ImportError as _err:
    st.error(f"**Import error:** {_err}\n\nEnsure you activated the venv and installed requirements.")
    st.stop()

log = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MODELS_DIR = _ROOT / "models" / "models"
AVAILABLE_MODELS = {
    "LightGBM":          "lightgbm.pkl",
    "XGBoost":           "xgboost.pkl",
    "Random Forest":     "rf.pkl",
    "CatBoost":          "catboost.pkl",
    "Logistic Regression": "lr.pkl",
}
FEATURE_PIPELINE_PATH = MODELS_DIR / "feature_pipeline.pkl"
ANOMALY_DETECTOR_PATH = MODELS_DIR / "anomaly_detector.pkl"
LR_SCALER_PATH        = MODELS_DIR / "lr_scaler.pkl"

# Verdict thresholds
THRESH_PHISHING    = 0.65
THRESH_UNCERTAIN   = 0.40

# Model performance (from reports/metrics.json)
MODEL_METRICS = {
    "LightGBM":            {"f1": 0.9505, "auc": 0.9941, "fnr": 0.0564, "fpr": 0.0311},
    "XGBoost":             {"f1": 0.9482, "auc": 0.9935, "fnr": 0.0648, "fpr": 0.0278},
    "Random Forest":       {"f1": 0.9436, "auc": 0.9926, "fnr": 0.0619, "fpr": 0.0373},
    "CatBoost":            {"f1": 0.9356, "auc": 0.9895, "fnr": 0.0676, "fpr": 0.0452},
    "Logistic Regression": {"f1": 0.9334, "auc": 0.9905, "fnr": 0.0987, "fpr": 0.0222},
}

# Risk level colours
RISK_COLOURS = {
    "CRITICAL": "#d32f2f",
    "HIGH":     "#f57c00",
    "MEDIUM":   "#fbc02d",
    "LOW":      "#388e3c",
    "UNKNOWN":  "#757575",
}
ACTION_ICONS = {
    "DELETE":    "🗑️",
    "QUARANTINE": "🔒",
    "MONITOR":   "👁️",
    "SAFE":      "✅",
}


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global: clean white main area ─────────────────────────────────────────── */
.main .block-container { background: #ffffff; }

/* ── Verdict banner ─────────────────────────────────────────────────────────── */
.verdict-phishing {
    background: #fff5f5;
    border-left: 5px solid #e53e3e;
    border-radius: 8px;
    padding: 16px 24px;
    margin-bottom: 12px;
}
.verdict-legitimate {
    background: #f0fff4;
    border-left: 5px solid #38a169;
    border-radius: 8px;
    padding: 16px 24px;
    margin-bottom: 12px;
}
.verdict-uncertain {
    background: #fffbeb;
    border-left: 5px solid #d69e2e;
    border-radius: 8px;
    padding: 16px 24px;
    margin-bottom: 12px;
}
.verdict-phishing h2   { margin:0; font-size:1.5rem; color:#c53030; }
.verdict-legitimate h2 { margin:0; font-size:1.5rem; color:#276749; }
.verdict-uncertain h2  { margin:0; font-size:1.5rem; color:#975a16; }

/* ── IOC pill ────────────────────────────────────────────────────────────────── */
.ioc-pill {
    display: inline-block;
    background: #edf2f7;
    border: 1px solid #cbd5e0;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.8rem;
    font-family: monospace;
    margin: 2px;
    color: #2b6cb0;
    word-break: break-all;
}

/* ── ATT&CK technique card ──────────────────────────────────────────────────── */
.att-card {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #4299e1;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 8px;
}
.att-card .tid   { color: #2b6cb0; font-weight: bold; font-family: monospace; }
.att-card .tname { color: #1a202c; font-size: 0.92rem; }
.att-card .ttac  { color: #718096; font-size: 0.8rem; }

/* ── Finding card ────────────────────────────────────────────────────────────── */
.finding-card {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 10px 16px;
    margin-bottom: 6px;
}

/* ── Feature bar ─────────────────────────────────────────────────────────────── */
.feat-bar-container { background:#e2e8f0; border-radius:4px; overflow:hidden; height:8px; }
.feat-bar-fill      { height:8px; border-radius:4px; transition:width .3s; }

/* ── Section header ──────────────────────────────────────────────────────────── */
h3 { color: #1a202c !important; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }

/* ── TI tool card ────────────────────────────────────────────────────────────── */
.ti-card {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    text-align: center;
}

/* ── Dark Sidebar (kept dark per design) ────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #30363d !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] span:not([data-testid="stMetricDelta"]),
[data-testid="stSidebar"] div { color: #c9d1d9 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 { color: #e6edf3 !important; }
[data-testid="stSidebar"] .stTextInput input {
    background: #161b22 !important; border-color: #30363d !important; color: #c9d1d9 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #161b22 !important; border-color: #30363d !important; color: #c9d1d9 !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] div[role="slider"] { background: #58a6ff !important; }
[data-testid="stSidebar"] .stSlider > div > div > div { background: #30363d !important; }
[data-testid="stSidebar"] hr { border-color: #30363d !important; }
[data-testid="stSidebar"] .stToggle label { color: #c9d1d9 !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #8b949e !important; }
[data-testid="stSidebar"] .stTextInput label { color: #8b949e !important; }
[data-testid="stSidebar"] .stSlider label { color: #8b949e !important; }
</style>
""", unsafe_allow_html=True)


# ── Model / pipeline loaders (cached) ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_feature_pipeline() -> Optional[FeaturePipeline]:
    if not FEATURE_PIPELINE_PATH.exists():
        return None
    try:
        return FeaturePipeline.load(str(FEATURE_PIPELINE_PATH))
    except Exception as exc:
        log.error(f"Failed to load feature_pipeline: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def load_anomaly_detector():
    if not ANOMALY_DETECTOR_PATH.exists():
        return None
    try:
        return joblib.load(ANOMALY_DETECTOR_PATH)
    except Exception as exc:
        log.error(f"Failed to load anomaly_detector: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def load_lr_scaler():
    if not LR_SCALER_PATH.exists():
        return None
    try:
        return joblib.load(LR_SCALER_PATH)
    except Exception as exc:
        log.error(f"Failed to load lr_scaler: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def load_model(model_filename: str):
    path = MODELS_DIR / model_filename
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        log.error(f"Failed to load {model_filename}: {exc}")
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar() -> Dict:
    """Render sidebar controls and return configuration dict."""

    # ── Badge ─────────────────────────────────────────────────────────────
    st.sidebar.markdown(
        '<div style="background:linear-gradient(135deg,#0969da,#218bff);color:white;'
        'border-radius:10px;padding:14px 20px;text-align:center;margin-bottom:12px;'
        'font-weight:900;font-size:1.15em;letter-spacing:1.5px;'
        'box-shadow:0 2px 8px rgba(9,105,218,.4);">🛡️&nbsp;&nbsp;PHISHLENS V1.0</div>',
        unsafe_allow_html=True,
    )

    # ── ⚙️ Settings ───────────────────────────────────────────────────────
    st.sidebar.markdown("### ⚙️ Settings")

    models_dir = st.sidebar.text_input(
        "Models directory",
        value=str(MODELS_DIR),
        help="Path to the directory containing .pkl model files.",
    )

    selected_model = st.sidebar.selectbox(
        "Active Model",
        list(AVAILABLE_MODELS.keys()),
        index=0,
        help="LightGBM achieves best F1 (0.9505) on the test set.",
    )

    threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.50,
        step=0.01,
        format="%.2f",
        help="Probability above this → PHISHING. Lower = more sensitive.",
    )

    st.sidebar.markdown("---")

    # ── 🔌 Enrichment ─────────────────────────────────────────────────────
    st.sidebar.markdown("### 🔌 Enrichment")

    use_intelligence = st.sidebar.toggle(
        "Threat Intelligence APIs",
        value=False,
        help="Queries VirusTotal, Google Safe Browsing, URLScan, URLhaus, AbuseIPDB, IPQS. Requires API keys in .env.",
    )
    use_chatgpt = st.sidebar.toggle(
        "ChatGPT Analysis (gpt-4.1-mini)",
        value=False,
        help="Sends forensic context to ChatGPT gpt-4.1-mini. Requires OPENAI_API_KEY.",
    )
    use_shap = st.sidebar.toggle(
        "SHAP Explainability",
        value=True,
        help="Compute SHAP feature importance for the verdict.",
    )
    use_network = st.sidebar.toggle(
        "Network Lookups (WHOIS/cert)",
        value=False,
        help="Enables WHOIS and crt.sh lookups for URL features. Adds ~3-10s per email.",
    )

    st.sidebar.markdown("---")

    # ── Footer ────────────────────────────────────────────────────────────
    st.sidebar.markdown(
        "<small style='color:#8b949e;line-height:1.7'>"
        "Datasets: 8 sources · 412,265 emails · Best model: LightGBM F1=0.9505<br>"
        "Models: XGBoost + RF + LightGBM + CatBoost + LR<br>"
        "Explainability: SHAP + LIME<br>"
        "by Sagar | MSc Cybersecurity Portfolio"
        "</small>",
        unsafe_allow_html=True,
    )

    return {
        "model_name": selected_model,
        "models_dir": models_dir,
        "threshold": threshold,
        "use_network": use_network,
        "use_intelligence": use_intelligence,
        "use_chatgpt": use_chatgpt,
        "use_shap": use_shap,
    }


def _show_api_status():
    keys = {
        "OPENAI_API_KEY":    "ChatGPT",
        "VT_API_KEY":        "VirusTotal",
        "GSB_API_KEY":       "Google Safe Browsing",
        "URLSCAN_API_KEY":   "URLScan.io",
        "ABUSEIPDB_API_KEY": "AbuseIPDB",
        "IPQS_API_KEY":      "IPQS",
    }
    for env_var, label in keys.items():
        val = os.getenv(env_var, "")
        icon = "🟢" if val else "🔴"
        st.sidebar.markdown(f"<small>{icon} {label}</small>", unsafe_allow_html=True)


# ── Main content areas ────────────────────────────────────────────────────────
def render_input_tabs() -> Tuple[Optional[str], Optional[str]]:
    """Render email input tabs. Returns (raw_email_text, parsed_subject)."""
    tab_upload, tab_paste, tab_sample = st.tabs(
        ["📁 Upload .eml / .msg", "📋 Paste Raw Email", "🧪 Sample Emails"]
    )

    raw_email: Optional[str] = None
    display_name: Optional[str] = None

    # ── Tab 1: File upload ────────────────────────────────────────────────
    with tab_upload:
        uploaded = st.file_uploader(
            "Upload an .eml or .msg file",
            type=["eml", "msg"],
            help="Drag and drop or click to upload. File is processed locally — never sent to any server.",
        )
        if uploaded:
            raw_bytes = uploaded.read()
            raw_email = raw_bytes.decode("utf-8", errors="replace")
            display_name = uploaded.name
            st.success(f"Loaded: **{uploaded.name}** ({len(raw_bytes):,} bytes)")

    # ── Tab 2: Paste raw ──────────────────────────────────────────────────
    with tab_paste:
        st.markdown("Paste the full raw email including headers:")
        pasted = st.text_area(
            "Raw email",
            height=300,
            placeholder="Received: from ...\nDate: ...\nFrom: ...\nSubject: ...\n\nEmail body here...",
            label_visibility="collapsed",
        )
        if pasted.strip():
            raw_email = pasted
            display_name = "Pasted email"

    # ── Tab 3: Sample emails ──────────────────────────────────────────────
    with tab_sample:
        sample_dir = _ROOT / "samples eamil"
        samples: List[Path] = []
        if sample_dir.exists():
            samples = sorted(
                [p for p in sample_dir.iterdir() if p.suffix.lower() in (".eml", ".msg")],
                key=lambda p: p.name,
            )

        if samples:
            sample_names = [p.name for p in samples]
            chosen = st.selectbox("Choose a sample email", sample_names)
            chosen_path = sample_dir / chosen
            if st.button("Load sample", type="primary"):
                try:
                    raw_bytes = chosen_path.read_bytes()
                    raw_email = raw_bytes.decode("utf-8", errors="replace")
                    display_name = chosen
                    st.success(f"Loaded: **{chosen}**")
                except Exception as exc:
                    st.error(f"Could not read sample: {exc}")
        else:
            st.info("No sample emails found in `samples eamil/` folder.")

    return raw_email, display_name


# ── Analysis engine ────────────────────────────────────────────────────────────
def run_analysis(raw_email: str, config: Dict) -> Dict:
    """Run full PhishLens analysis pipeline. Returns result dict."""
    results: Dict[str, Any] = {}

    with st.spinner("Parsing email structure…"):
        parsed = parse_eml_string(raw_email)
        results["parsed"] = parsed

    with st.spinner("Extracting 961 features…"):
        feature_pipeline = load_feature_pipeline()
        anomaly_detector = load_anomaly_detector()
        lr_scaler        = load_lr_scaler()

        if feature_pipeline is None:
            st.error(
                "**Feature pipeline not found** at `models/models/feature_pipeline.pkl`.\n\n"
                "Run `python train.py` to train models first."
            )
            st.stop()

        # Transform single email to feature vector
        # Set pipeline flags on the instance (models were trained with use_gemini=False)
        feature_pipeline.use_network = config["use_network"]
        feature_pipeline.use_intelligence_apis = config["use_intelligence"]
        feature_pipeline.use_gemini = False  # MUST stay False — models trained without ChatGPT ML feature
        try:
            # transform_single returns (np.ndarray shape [1, n_features], feature_names list)
            X, extracted_feature_names = feature_pipeline.transform_single(raw_email)
            X = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as exc:
            st.error(f"Feature extraction failed: {exc}")
            log.exception("Feature extraction error")
            st.stop()

        results["X"] = X
        results["feature_names"] = extracted_feature_names or [f"f{i}" for i in range(X.shape[1])]

    with st.spinner(f"Running {config['model_name']} classifier…"):
        model_filename = AVAILABLE_MODELS[config["model_name"]]
        model = load_model(model_filename)

        if model is None:
            st.error(
                f"**Model not found:** `models/models/{model_filename}`\n\n"
                "Run `python train.py` to train models first."
            )
            st.stop()

        try:
            X_pred = X
            if config["model_name"] == "Logistic Regression" and lr_scaler is not None:
                X_pred = lr_scaler.transform(X)

            proba = model.predict_proba(X_pred)[0]
            phishing_prob = float(proba[1])
            threshold = config["threshold"]

            if phishing_prob >= threshold:
                verdict = "PHISHING"
            elif phishing_prob >= THRESH_UNCERTAIN:
                verdict = "UNCERTAIN"
            else:
                verdict = "LEGITIMATE"

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            log.exception("Prediction error")
            st.stop()

        results["phishing_prob"] = phishing_prob
        results["verdict"] = verdict
        results["model_name"] = config["model_name"]

    # ── Anomaly score ──────────────────────────────────────────────────────
    if anomaly_detector is not None:
        try:
            anomaly_score = float(anomaly_detector.score_samples(X)[0])
            results["anomaly_score"] = anomaly_score
        except Exception:
            results["anomaly_score"] = None
    else:
        results["anomaly_score"] = None

    # ── IOC extraction + MITRE ATT&CK mapping ─────────────────────────────
    # extract_iocs handles ATT&CK mapping internally when feature_flags is provided.
    with st.spinner("Extracting IOCs…"):
        try:
            # Build feature dict from the extracted feature vector for ATT&CK mapping
            feature_dict = dict(zip(results["feature_names"], X[0].tolist()))
            iocs = extract_iocs(
                parsed,
                risk_score=phishing_prob,
                feature_flags=feature_dict,
            )
            results["iocs"] = iocs
            results["attack_techniques"] = iocs.get("attack_techniques", [])
            # Generate enriched IOC explanations for Forensic Analysis tab
            try:
                results["ioc_explanations"] = generate_ioc_explanations(parsed, feature_dict, iocs)
            except Exception as _exc:
                log.warning(f"generate_ioc_explanations failed: {_exc}")
                results["ioc_explanations"] = {}
        except Exception as exc:
            log.warning(f"IOC extraction failed: {exc}")
            results["iocs"] = {}
            results["ioc_explanations"] = {}
            # Fallback ATT&CK mapping without feature dict
            try:
                results["attack_techniques"] = map_attack_techniques(
                    features={},
                    iocs={},
                    phishing_probability=phishing_prob,
                    verdict=verdict,
                )
            except Exception:
                results["attack_techniques"] = []

    # ── ChatGPT forensic analysis ──────────────────────────────────────────
    if config["use_chatgpt"]:
        with st.spinner("Sending forensic context to ChatGPT gpt-4.1-mini…"):
            try:
                # Build a minimal all_headers dict from parsed
                all_headers = {
                    "From": parsed.get("from_address", ""),
                    "Subject": parsed.get("subject", ""),
                    "Return-Path": parsed.get("return_path", ""),
                    "Reply-To": parsed.get("reply_to", ""),
                    "Message-ID": parsed.get("message_id", ""),
                    "X-Mailer": parsed.get("x_mailer", ""),
                    "SPF": parsed.get("spf", ""),
                    "DKIM": parsed.get("dkim", ""),
                    "DMARC": parsed.get("dmarc", ""),
                    "Received": " | ".join(parsed.get("received_headers", [])[:5]),
                }
                ai_result = analyse_email_with_openai(
                    subject=parsed.get("subject", ""),
                    from_address=parsed.get("from_address", ""),
                    body_text=parsed.get("body_text", ""),
                    urls=parsed.get("urls", []),
                    all_headers=all_headers,
                    iocs=results.get("iocs", {}),
                    intelligence_result=results.get("intel", {}),
                    ml_verdict=verdict,
                    ml_probability=phishing_prob,
                )
                results["ai_result"] = ai_result
            except Exception as exc:
                log.warning(f"ChatGPT analysis error: {exc}")
                results["ai_result"] = None
    else:
        results["ai_result"] = None

    # ── SHAP explainability ────────────────────────────────────────────────
    if config["use_shap"]:
        with st.spinner("Computing SHAP feature importance…"):
            try:
                import shap
                model_type = _detect_model_type(config["model_name"])
                if model_type == "tree":
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_pred)
                    # shap_values can be:
                    #   list of 2 arrays (RF, older SHAP): use class-1 (phishing) → shap_values[1]
                    #   3D array shape (1, n_features, 2) (newer SHAP): index [:, :, 1]
                    #   2D array shape (1, n_features) (LightGBM/XGB binary): use row 0
                    if isinstance(shap_values, list):
                        sv = np.array(shap_values[1]).ravel()
                    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                        sv = shap_values[0, :, 1]
                    else:
                        sv = np.array(shap_values).ravel()
                else:
                    # LinearExplainer: shap_values is 2D array (n_samples, n_features)
                    explainer = shap.LinearExplainer(model, X_pred, feature_perturbation="correlation_dependent")
                    shap_values = explainer.shap_values(X_pred)
                    sv = np.array(shap_values).ravel()

                feature_names_list = results["feature_names"]
                # Align lengths in case feature count differs (should not happen)
                n = min(len(feature_names_list), len(sv))
                shap_df = pd.DataFrame({"feature": feature_names_list[:n], "shap": sv[:n]})
                shap_df["abs_shap"] = shap_df["shap"].abs()
                shap_df = shap_df.sort_values("abs_shap", ascending=False).head(20)
                results["shap_df"] = shap_df
            except Exception as exc:
                log.warning(f"SHAP failed: {exc}")
                results["shap_df"] = None
    else:
        results["shap_df"] = None

    return results


def _detect_model_type(model_name: str) -> str:
    tree_models = {"LightGBM", "XGBoost", "Random Forest", "CatBoost"}
    return "tree" if model_name in tree_models else "linear"


# ── Result renderers ──────────────────────────────────────────────────────────
def render_verdict_banner(results: Dict):
    verdict       = results["verdict"]
    phishing_prob = results["phishing_prob"]
    model_name    = results["model_name"]

    css_class = {
        "PHISHING":   "verdict-phishing",
        "LEGITIMATE": "verdict-legitimate",
        "UNCERTAIN":  "verdict-uncertain",
    }.get(verdict, "verdict-uncertain")

    icon   = {"PHISHING": "🚨", "LEGITIMATE": "✅", "UNCERTAIN": "⚠️"}.get(verdict, "❓")
    colour = {"PHISHING": "#c53030", "LEGITIMATE": "#276749", "UNCERTAIN": "#975a16"}.get(verdict, "#4a5568")

    st.markdown(f"""
    <div class="{css_class}">
        <h2>{icon} {verdict}</h2>
        <p style="color:#4a5568;margin:4px 0 0 0;font-size:0.95rem">
            Phishing probability: <strong style="color:{colour}">{phishing_prob:.1%}</strong>
            &nbsp;·&nbsp; Model: <strong>{model_name}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")


def render_email_summary(parsed: Dict):
    st.markdown("### 📧 Email Summary")
    cols = st.columns([2, 2, 1, 1])
    cols[0].markdown(f"**From:** `{parsed.get('from_address', 'N/A')}`")
    cols[1].markdown(f"**Subject:** {parsed.get('subject', 'N/A')}")
    cols[2].markdown(f"**Attachments:** {parsed.get('attachments_count', 0)}")
    cols[3].markdown(f"**URLs found:** {len(parsed.get('urls', []))}")

    auth_cols = st.columns(3)
    spf_val   = (parsed.get("spf", "") or "").upper()
    dkim_val  = (parsed.get("dkim", "") or "").upper()
    dmarc_val = (parsed.get("dmarc", "") or "").upper()

    def _auth_badge(label, value):
        if "PASS" in value:
            return f"🟢 **{label}:** PASS"
        elif "FAIL" in value or "REJECT" in value:
            return f"🔴 **{label}:** FAIL"
        else:
            return f"⚪ **{label}:** {value or 'N/A'}"

    auth_cols[0].markdown(_auth_badge("SPF", spf_val))
    auth_cols[1].markdown(_auth_badge("DKIM", dkim_val))
    auth_cols[2].markdown(_auth_badge("DMARC", dmarc_val))


def render_iocs(iocs: Dict):
    st.markdown("### 🔍 Extracted IOCs")
    if not iocs:
        st.info("No IOCs extracted.")
        return

    tabs = st.tabs(["URLs", "IPs", "Domains", "Email Addresses", "Attachment Hashes", "Export"])

    with tabs[0]:
        urls = iocs.get("urls", [])
        cleaned_map = iocs.get("url_cleaning_map", {})
        if urls:
            for url in urls[:50]:
                info = cleaned_map.get(url, {})
                was_wrapped = info.get("was_wrapped", False)
                clean = info.get("cleaned", url)
                pill = f'<span class="ioc-pill">{url}</span>'
                if was_wrapped and clean != url:
                    pill += f' <small style="color:#f57c00">→ {clean}</small>'
                st.markdown(pill, unsafe_allow_html=True)
            if len(urls) > 50:
                st.caption(f"... and {len(urls) - 50} more URLs")
        else:
            st.info("No URLs found.")

    with tabs[1]:
        # extract_iocs stores sender IPs under key 'sender_ips'
        ips = iocs.get("sender_ips", [])
        if ips:
            for ip in ips:
                st.markdown(f'<span class="ioc-pill">{ip}</span>', unsafe_allow_html=True)
        else:
            st.info("No IP addresses found.")

    with tabs[2]:
        domains = iocs.get("domains", [])
        if domains:
            for d in domains[:30]:
                st.markdown(f'<span class="ioc-pill">{d}</span>', unsafe_allow_html=True)
        else:
            st.info("No domains extracted.")

    with tabs[3]:
        # extract_iocs stores sender emails under key 'sender_emails'
        emails = iocs.get("sender_emails", [])
        if emails:
            for e in emails:
                st.markdown(f'<span class="ioc-pill">{e}</span>', unsafe_allow_html=True)
        else:
            st.info("No sender email addresses found.")

    with tabs[4]:
        hashes = iocs.get("attachment_hashes", [])
        if hashes:
            for h in hashes:
                fname  = h.get("filename", "unknown")
                md5    = h.get("md5", "")
                sha256 = h.get("sha256", "")
                st.markdown(f"**File:** `{fname}`")
                if sha256:
                    st.markdown(f'SHA256: <span class="ioc-pill">{sha256}</span>', unsafe_allow_html=True)
                if md5:
                    st.markdown(f'MD5: <span class="ioc-pill">{md5}</span>', unsafe_allow_html=True)
        else:
            st.info("No attachment hashes found.")

    with tabs[5]:
        st.markdown("**IOC JSON export:**")
        # Build clean export (no internal _ keys)
        export = {k: v for k, v in iocs.items() if not k.startswith("_")}
        json_str = json.dumps(export, indent=2, default=str)
        st.download_button(
            label="⬇️ Download IOC JSON",
            data=json_str,
            file_name="phishlens_iocs.json",
            mime="application/json",
        )
        st.code(json_str, language="json")


def render_attack_techniques(techniques: List[Dict]):
    st.markdown("### ⚔️ MITRE ATT&CK Mapping")
    if not techniques:
        st.info("No MITRE ATT&CK techniques mapped (email is likely legitimate).")
        return

    for tech in techniques:
        conf_pct = int(tech.get("confidence", 0) * 100)
        conf_bar_colour = "#e53e3e" if conf_pct >= 70 else ("#ed8936" if conf_pct >= 40 else "#ecc94b")
        evidence_str = "; ".join(tech.get("evidence", []))
        mitre_url = tech.get("mitre_url", f"https://attack.mitre.org/techniques/{tech.get('technique_id', '')}/")

        st.markdown(f"""
        <div class="att-card">
            <div>
                <a href="{mitre_url}" target="_blank" class="tid">{tech.get('technique_id', '')}</a>
                <span class="tname"> — {tech.get('technique_name', '')}</span>
            </div>
            <div class="ttac">Tactic: {tech.get('tactic', '')} &nbsp;·&nbsp; Confidence: {conf_pct}%</div>
            <div class="feat-bar-container" style="margin-top:6px">
                <div class="feat-bar-fill" style="width:{conf_pct}%;background:{conf_bar_colour}"></div>
            </div>
            <div style="color:#718096;font-size:0.78rem;margin-top:4px">{evidence_str}</div>
        </div>
        """, unsafe_allow_html=True)


def render_chatgpt_analysis(ai_result: Optional[Dict]):
    st.markdown("### 🤖 ChatGPT Forensic Analysis")
    if ai_result is None:
        st.info("ChatGPT analysis not enabled. Toggle **ChatGPT forensic analysis** in the sidebar.")
        return

    provider = ai_result.get("_ai_provider", "ChatGPT")
    if "unavailable" in provider.lower():
        st.warning(f"ChatGPT unavailable — set `OPENAI_API_KEY` in your environment.\n\nProvider: `{provider}`")
        return

    is_phishing = ai_result.get("gemini_is_phishing", -1)
    confidence  = ai_result.get("gemini_confidence", 0.0)
    risk_level  = ai_result.get("gemini_risk_level", "UNKNOWN")
    action      = ai_result.get("gemini_recommended_action", "MONITOR")
    explanation = ai_result.get("gemini_explanation", "")
    brand       = ai_result.get("gemini_impersonated_brand")
    signals     = ai_result.get("gemini_phishing_signals", [])
    se_techs    = ai_result.get("gemini_social_engineering", [])
    ioc_verdicts = ai_result.get("gemini_ioc_verdicts", {})

    # Risk + action row
    risk_colour = RISK_COLOURS.get(risk_level, RISK_COLOURS["UNKNOWN"])
    action_icon = ACTION_ICONS.get(action, "❓")
    verdict_icon = "🚨" if is_phishing == 1 else ("✅" if is_phishing == 0 else "❓")

    col1, col2, col3 = st.columns(3)
    col1.metric("AI Verdict", f"{verdict_icon} {'PHISHING' if is_phishing == 1 else 'LEGITIMATE' if is_phishing == 0 else 'UNKNOWN'}")
    col2.metric("AI Confidence", f"{confidence:.0%}")
    col3.metric("Risk Level", f"<span style='color:{risk_colour};font-weight:bold'>{risk_level}</span>", help=None)

    st.markdown(f"**Recommended Action:** {action_icon} `{action}`")
    if brand:
        st.markdown(f"**Impersonated Brand:** 🎭 `{brand}`")

    if explanation:
        st.markdown("**Analyst Explanation:**")
        st.markdown(f"> {explanation}")

    if signals:
        st.markdown("**Phishing Signals Detected:**")
        for sig in signals:
            st.markdown(f"- {sig}")

    if se_techs:
        st.markdown("**Social Engineering Techniques:**")
        for t in se_techs:
            st.markdown(f"- {t}")

    if ioc_verdicts:
        st.markdown("**IOC Verdicts:**")
        for ioc, verdict in ioc_verdicts.items():
            colour = "#c53030" if "MALICIOUS" in str(verdict).upper() else \
                     "#c05621" if "SUSPICIOUS" in str(verdict).upper() else \
                     "#276749" if "CLEAN" in str(verdict).upper() else "#4a5568"
            st.markdown(
                f'`{ioc}` → <span style="color:{colour};font-weight:600">{verdict}</span>',
                unsafe_allow_html=True,
            )

    st.caption(f"Powered by: {provider}")


def render_shap(shap_df: Optional[pd.DataFrame]):
    st.markdown("### 📊 SHAP Feature Importance")
    if shap_df is None:
        st.info("SHAP explainability not enabled or failed. Toggle **SHAP explainability** in the sidebar.")
        return

    if shap_df.empty:
        st.info("No SHAP values computed.")
        return

    import plotly.graph_objects as go

    top = shap_df.head(20).copy()
    top = top.sort_values("shap")
    colours = ["#d32f2f" if v > 0 else "#388e3c" for v in top["shap"]]

    fig = go.Figure(go.Bar(
        x=top["shap"],
        y=top["feature"],
        orientation="h",
        marker_color=colours,
        hovertemplate="<b>%{y}</b><br>SHAP value: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Top 20 features by SHAP contribution (red=phishing↑, green=phishing↓)",
        xaxis_title="SHAP value",
        yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f0f6fc",
        height=500,
        margin={"l": 200, "r": 20, "t": 50, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_raw_features(X: np.ndarray, feature_names: List[str]):
    with st.expander("🔬 Raw feature vector (961 features)", expanded=False):
        feat_df = pd.DataFrame({
            "Feature": feature_names[:X.shape[1]],
            "Value": X[0].tolist(),
        })
        feat_df["Group"] = feat_df["Feature"].apply(lambda n:
            "Header"    if n.startswith("hdr_") else
            "URL"       if n.startswith("url_") else
            "HTML"      if n.startswith("html_") else
            "Text"      if n.startswith("txt_") else
            "TF-IDF"    if n.startswith("tfidf_") else
            "Intel"     if n.startswith("intel_") else
            "Anomaly"   if "anomaly" in n else
            "Embedding" if n.startswith("emb_") else "Other"
        )
        groups = feat_df["Group"].value_counts()
        st.markdown("**Feature group breakdown:**")
        cols = st.columns(len(groups))
        for i, (grp, cnt) in enumerate(groups.items()):
            cols[i].metric(grp, cnt)

        st.dataframe(
            feat_df.style.format({"Value": "{:.6f}"}),
            use_container_width=True,
            height=300,
        )


def render_email_body(parsed: Dict):
    with st.expander("📄 Parsed email content", expanded=False):
        body_tab, header_tab = st.tabs(["Body text", "Headers"])
        with body_tab:
            body = parsed.get("body_text", "") or ""
            if body.strip():
                st.text_area("Body text", value=body[:5000], height=250, disabled=True, label_visibility="collapsed")
                if len(body) > 5000:
                    st.caption(f"Showing first 5,000 of {len(body):,} characters.")
            else:
                st.info("No plain-text body found (may be HTML-only).")
        with header_tab:
            header_info = {
                "From":         parsed.get("from_address", ""),
                "Subject":      parsed.get("subject", ""),
                "Return-Path":  parsed.get("return_path", ""),
                "Reply-To":     parsed.get("reply_to", ""),
                "Message-ID":   parsed.get("message_id", ""),
                "X-Mailer":     parsed.get("x_mailer", ""),
                "SPF":          parsed.get("spf", ""),
                "DKIM":         parsed.get("dkim", ""),
                "DMARC":        parsed.get("dmarc", ""),
                "Timezone":     parsed.get("timezone", ""),
                "Attachments":  parsed.get("attachments_count", 0),
            }
            for k, v in header_info.items():
                if v:
                    st.markdown(f"**{k}:** `{v}`")
            received = parsed.get("received_headers", [])
            if received:
                st.markdown(f"**Received hops:** {len(received)}")
                for i, h in enumerate(received[:10], 1):
                    st.markdown(f"<small>**Hop {i}:** {h[:200]}</small>", unsafe_allow_html=True)


def render_full_report_download(results: Dict, display_name: str):
    """Provide a JSON download of the complete analysis report."""
    report = {
        "source": display_name,
        "model": results.get("model_name"),
        "verdict": results.get("verdict"),
        "phishing_probability": results.get("phishing_prob"),
        "anomaly_score": results.get("anomaly_score"),
        "email_metadata": {
            "from": results["parsed"].get("from_address"),
            "subject": results["parsed"].get("subject"),
            "attachments": results["parsed"].get("attachments_count"),
            "url_count": len(results["parsed"].get("urls", [])),
            "spf": results["parsed"].get("spf"),
            "dkim": results["parsed"].get("dkim"),
            "dmarc": results["parsed"].get("dmarc"),
        },
        "iocs": {k: v for k, v in results.get("iocs", {}).items() if not k.startswith("_")},
        "attack_techniques": results.get("attack_techniques", []),
        "ai_analysis": {
            k: v for k, v in (results.get("ai_result") or {}).items()
            if not k.startswith("_")
        },
    }
    json_str = json.dumps(report, indent=2, default=str)
    st.download_button(
        label="⬇️ Download Full Analysis Report (JSON)",
        data=json_str,
        file_name=f"phishlens_report_{int(time.time())}.json",
        mime="application/json",
    )


# ── Main app ───────────────────────────────────────────────────────────────────
def main():
    config = render_sidebar()

    # ── File upload / paste / sample ────────────────────────────────────
    raw_email, display_name = render_input_tabs()

    if not raw_email:
        # Landing page
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            #### 📁 Upload or paste an email
            Drop a `.eml` / `.msg` file, paste raw email text, or pick a sample
            from the sample folder to analyse.
            """)
        with col2:
            st.markdown("""
            #### 🤖 5 ML Models
            LightGBM · XGBoost · Random Forest · CatBoost · Logistic Regression  
            All trained on **412,265 labelled emails** with 961-dimensional features.
            """)
        with col3:
            st.markdown("""
            #### 🔍 Deep Analysis
            IOC extraction · MITRE ATT&CK mapping · SHAP explainability ·
            ChatGPT gpt-4.1-mini forensic layer · 6 threat-intel APIs.
            """)

        st.markdown("---")
        st.markdown("### 📈 Model Performance")
        perf_df = pd.DataFrame(MODEL_METRICS).T.reset_index()
        perf_df.columns = ["Model", "F1", "AUC-ROC", "FNR", "FPR"]
        perf_df = perf_df.sort_values("F1", ascending=False).reset_index(drop=True)
        st.dataframe(
            perf_df.style.format({
                "F1": "{:.4f}", "AUC-ROC": "{:.4f}", "FNR": "{:.2%}", "FPR": "{:.2%}"
            }).background_gradient(cmap="Greens", subset=["F1", "AUC-ROC"])
              .background_gradient(cmap="Reds_r", subset=["FNR", "FPR"]),
            use_container_width=True,
            hide_index=True,
        )
        return

    # ── Analyse button ───────────────────────────────────────────────────
    st.markdown("---")
    t_start = time.perf_counter()

    analyse_btn = st.button("🔍 Analyse Email", type="primary", use_container_width=True)
    if not analyse_btn:
        st.info(f"Email loaded: **{display_name}**  ·  Click **Analyse Email** to run analysis.")
        return

    results = run_analysis(raw_email, config)
    elapsed = time.perf_counter() - t_start

    st.markdown(f"<small style='color:#8b949e'>Analysis completed in {elapsed:.2f}s</small>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Verdict banner ────────────────────────────────────────────────────
    render_verdict_banner(results)

    # Escalation notice for PHISHING verdict
    if results["verdict"] == "PHISHING":
        st.warning(
            "⚠️ **High-confidence phishing detected.** Recommend immediate quarantine and user notification. "
            "Check the Forensic Analysis and IOCs tabs for actionable evidence.",
            icon="🚨",
        )

    # Metric cards row
    prob = results["phishing_prob"]
    a_score = results.get("anomaly_score")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Phishing Probability", f"{prob:.1%}")
    mc2.metric("Model", results["model_name"])
    mc3.metric("Anomaly Score", f"{a_score:.4f}" if a_score is not None else "N/A")
    mc4.metric("URLs Found", len(results["parsed"].get("urls", [])))

    st.markdown("---")

    # ── 4-tab navigation ──────────────────────────────────────────────────
    tab_ov, tab_forensic, tab_ioc, tab_ti = st.tabs([
        "🔍 Overview",
        "🔬 Forensic Analysis",
        "📋 IOCs & Evidence",
        "🌐 Threat Intelligence",
    ])

    # ── TAB 1: Overview ────────────────────────────────────────────────────
    with tab_ov:
        # Probability ring — centred, no black background box
        gauge_colour = "#d32f2f" if prob >= config["threshold"] else \
                       "#fbc02d" if prob >= THRESH_UNCERTAIN else "#388e3c"
        gauge_label  = "🚨 PHISHING" if prob >= config["threshold"] else \
                       "⚠️ UNCERTAIN" if prob >= THRESH_UNCERTAIN else "✅ LEGITIMATE"
        bar_pct = int(prob * 100)
        _g1, _gc, _g2 = st.columns([1, 2, 1])
        with _gc:
            st.markdown(
                f'<div style="text-align:center;padding:18px 0 6px 0">'
                f'<div style="font-size:3rem;font-weight:900;color:{gauge_colour};line-height:1">{prob:.1%}</div>'
                f'<div style="font-size:1rem;color:{gauge_colour};font-weight:700;margin-top:6px;letter-spacing:1px">{gauge_label}</div>'
                f'<div style="margin:12px auto 0 auto;height:10px;border-radius:5px;background:#e2e8f0;max-width:320px;overflow:hidden">'
                f'<div style="width:{bar_pct}%;height:100%;background:{gauge_colour};border-radius:5px;transition:width .4s"></div>'
                f'</div>'
                f'<div style="color:#718096;font-size:0.8rem;margin-top:4px">Phishing Probability · threshold {config["threshold"]:.0%}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        render_shap(results.get("shap_df"))
        render_attack_techniques(results.get("attack_techniques", []))
        render_chatgpt_analysis(results.get("ai_result"))

    # ── TAB 2: Forensic Analysis ────────────────────────────────────────────
    with tab_forensic:
        render_email_summary(results["parsed"])
        render_email_body(results["parsed"])

        ioc_exp = results.get("ioc_explanations", {})

        # Header findings
        header_findings = ioc_exp.get("header_findings", [])
        if header_findings:
            st.markdown("### 🔒 Header Analysis")
            for finding in header_findings:
                sev = finding.get("severity", "LOW")
                icon = finding.get("icon", "ℹ️")
                title = finding.get("title", "")
                detail = finding.get("detail", "")
                colour = RISK_COLOURS.get(sev, RISK_COLOURS["UNKNOWN"])
                st.markdown(
                    f'<div class="finding-card" style="border-left:4px solid {colour}">'
                    f'<strong style="color:{colour}">{icon} [{sev}] {title}</strong><br>'
                    f'<small style="color:#718096">{detail}</small></div>',
                    unsafe_allow_html=True,
                )

        # Sender address analysis
        email_findings = ioc_exp.get("email_findings", [])
        if email_findings:
            st.markdown("### 📧 Sender Address Analysis")
            for ef in email_findings:
                addr = ef.get("address", "")
                addr_type = ef.get("type", "")
                sev = ef.get("severity", "LOW")
                flags = ef.get("flags", [])
                colour = RISK_COLOURS.get(sev, RISK_COLOURS["UNKNOWN"])
                flags_str = " · ".join(flags) if flags else "—"
                st.markdown(
                    f'<div class="finding-card" style="border-left:4px solid {colour}">'
                    f'<strong style="color:{colour}">[{sev}]</strong> '
                    f'<span class="ioc-pill">{addr}</span> '
                    f'<small style="color:#4a5568">({addr_type})</small><br>'
                    f'<small style="color:#718096">{flags_str}</small></div>',
                    unsafe_allow_html=True,
                )

        # HTML body analysis
        html_findings = ioc_exp.get("html_findings", [])
        if html_findings:
            st.markdown("### 🌐 HTML Body Analysis")
            for hf in html_findings:
                sev = hf.get("severity", "LOW")
                icon = hf.get("icon", "ℹ️")
                title = hf.get("title", "")
                detail = hf.get("detail", "")
                colour = RISK_COLOURS.get(sev, RISK_COLOURS["UNKNOWN"])
                st.markdown(
                    f'<div class="finding-card" style="border-left:4px solid {colour}">'
                    f'<strong style="color:{colour}">{icon} [{sev}] {title}</strong><br>'
                    f'<small style="color:#718096">{detail}</small></div>',
                    unsafe_allow_html=True,
                )

        # URL risk analysis
        url_findings = ioc_exp.get("url_findings", [])
        if url_findings:
            st.markdown("### 🔗 URL Risk Analysis")
            for uf in url_findings:
                url = uf.get("url", "")
                domain = uf.get("domain", "")
                sev = uf.get("severity", "LOW")
                risks = uf.get("risks", [])
                colour = RISK_COLOURS.get(sev, RISK_COLOURS["UNKNOWN"])
                risks_str = " · ".join(risks) if risks else "—"
                st.markdown(
                    f'<div class="finding-card" style="border-left:4px solid {colour}">'
                    f'<strong style="color:{colour}">[{sev}]</strong> '
                    f'<span class="ioc-pill">{url[:100]}{"…" if len(url) > 100 else ""}</span><br>'
                    f'<small style="color:#718096">Domain: {domain} · {risks_str}</small></div>',
                    unsafe_allow_html=True,
                )

        if not any([header_findings, email_findings, html_findings, url_findings]):
            st.info("No forensic findings available. Run analysis with enrichment enabled for full detail.")

    # ── TAB 3: IOCs & Evidence ─────────────────────────────────────────────
    with tab_ioc:
        iocs = results.get("iocs", {})

        if not iocs:
            st.info("No IOCs extracted.")
        else:
            # Grids
            col_a, col_b = st.columns(2)
            with col_a:
                sender_emails = iocs.get("sender_emails", [])
                st.markdown(f"**Sender Emails** ({len(sender_emails)})")
                for e in sender_emails:
                    st.markdown(f'<span class="ioc-pill">{e}</span>', unsafe_allow_html=True)

                ips = iocs.get("sender_ips", [])
                st.markdown(f"**Sender IPs** ({len(ips)})")
                for ip in ips:
                    st.markdown(f'<span class="ioc-pill">{ip}</span>', unsafe_allow_html=True)

            with col_b:
                domains = iocs.get("domains", [])
                st.markdown(f"**Domains** ({len(domains)})")
                for d in domains[:20]:
                    st.markdown(f'<span class="ioc-pill">{d}</span>', unsafe_allow_html=True)
                if len(domains) > 20:
                    st.caption(f"… and {len(domains)-20} more")

                urls = iocs.get("urls", [])
                st.markdown(f"**URLs** ({len(urls)})")
                for u in urls[:10]:
                    st.markdown(f'<span class="ioc-pill">{u[:80]}{"…" if len(u)>80 else ""}</span>', unsafe_allow_html=True)
                if len(urls) > 10:
                    st.caption(f"… and {len(urls)-10} more")

            # Attachment hashes
            hashes = iocs.get("attachment_hashes", [])
            if hashes:
                st.markdown(f"**Attachment Hashes** ({len(hashes)})")
                for h in hashes:
                    st.markdown(f"`{h.get('filename','?')}` — SHA256: `{h.get('sha256','')}`")

            st.markdown("---")
            # Export buttons
            exp_col1, exp_col2, exp_col3 = st.columns(3)

            misp_json = iocs_to_misp_json(iocs, campaign_name=display_name or "PhishLens")
            with exp_col1:
                st.download_button(
                    label="⬇️ MISP JSON",
                    data=misp_json,
                    file_name="phishlens_misp.json",
                    mime="application/json",
                    use_container_width=True,
                )

            syslog_cef = iocs_to_syslog(iocs, verdict=results["verdict"])
            with exp_col2:
                st.download_button(
                    label="⬇️ Syslog CEF",
                    data=syslog_cef,
                    file_name="phishlens_syslog.cef",
                    mime="text/plain",
                    use_container_width=True,
                )

            ioc_txt_lines = []
            for key in ("sender_emails", "sender_ips", "domains", "urls"):
                for val in iocs.get(key, []):
                    ioc_txt_lines.append(str(val))
            with exp_col3:
                st.download_button(
                    label="⬇️ IOC .txt",
                    data="\n".join(ioc_txt_lines),
                    file_name="phishlens_iocs.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            render_raw_features(results["X"], results.get("feature_names", []))

    # ── TAB 4: Threat Intelligence ──────────────────────────────────────────
    with tab_ti:
        intel = results.get("intel", {})
        ai_result = results.get("ai_result")

        # Tool-level verdicts
        TI_TOOLS = [
            ("VirusTotal",         "vt_",        "🦠"),
            ("Google Safe Browsing","gsb_",       "🔍"),
            ("URLhaus",            "urlhaus_",    "🌐"),
            ("URLScan.io",         "urlscan_",    "🔭"),
            ("AbuseIPDB",          "abuseipdb_",  "🚫"),
            ("IPQS",               "ipqs_",       "🔒"),
        ]

        if intel:
            st.markdown("### 🌐 Threat Intelligence Tool Verdicts")
            ti_cols = st.columns(3)
            for idx, (tool_name, prefix, icon) in enumerate(TI_TOOLS):
                verdict_key = f"{prefix}verdict"
                score_key   = f"{prefix}score"
                verdict_val = intel.get(verdict_key, intel.get(tool_name.lower().replace(" ", "_"), "N/A"))
                score_val   = intel.get(score_key, "")
                colour = "#d32f2f" if "MALICIOUS" in str(verdict_val).upper() else \
                         "#f57c00" if "SUSPICIOUS" in str(verdict_val).upper() else \
                         "#388e3c" if "CLEAN" in str(verdict_val).upper() else "#757575"
                with ti_cols[idx % 3]:
                    st.markdown(
                        f'<div class="ti-card">'
                        f'<strong style="color:#1a202c">{icon} {tool_name}</strong><br>'
                        f'<span style="color:{colour};font-weight:700;font-size:1rem">{verdict_val}</span>'
                        f'{f"<br><small style=\'color:#718096\'>{score_val}</small>" if score_val else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Per-IOC verification table
            st.markdown("### 🔎 Per-IOC Verification")
            ioc_verdicts_all = intel.get("ioc_verdicts", {})
            if ioc_verdicts_all:
                rows = []
                for ioc_val, ioc_info in ioc_verdicts_all.items():
                    if isinstance(ioc_info, dict):
                        rows.append({
                            "IOC": ioc_val,
                            "Type": ioc_info.get("type", ""),
                            "Verdict": ioc_info.get("verdict", ""),
                            "Score": ioc_info.get("score", ""),
                            "Source": ioc_info.get("source", ""),
                        })
                    else:
                        rows.append({"IOC": ioc_val, "Type": "", "Verdict": str(ioc_info), "Score": "", "Source": ""})
                if rows:
                    ioc_df = pd.DataFrame(rows)
                    st.dataframe(ioc_df, use_container_width=True, hide_index=True)
        else:
            st.info(
                "Threat Intelligence not enabled. Toggle **Threat Intelligence APIs** in the sidebar, "
                "then re-run the analysis."
            )

        # ChatGPT IOC verdicts (if available)
        if ai_result:
            ioc_verdicts_ai = ai_result.get("gemini_ioc_verdicts", {})
            if ioc_verdicts_ai:
                st.markdown("### 🤖 ChatGPT IOC Verdicts")
                for ioc_v, verdict_v in ioc_verdicts_ai.items():
                    colour = "#c53030" if "MALICIOUS" in str(verdict_v).upper() else \
                             "#c05621" if "SUSPICIOUS" in str(verdict_v).upper() else \
                             "#276749" if "CLEAN" in str(verdict_v).upper() else "#4a5568"
                    st.markdown(
                        f'`{ioc_v}` → <span style="color:{colour};font-weight:600">{verdict_v}</span>',
                        unsafe_allow_html=True,
                    )

    st.markdown("---")
    render_full_report_download(results, display_name or "unknown")


if __name__ == "__main__":
    main()
