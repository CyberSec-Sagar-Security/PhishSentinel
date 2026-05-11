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
    from src.ioc_extractor import extract_iocs
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
/* Dashboard header */
.phishlens-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 60%, #1f2937 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
}
.phishlens-header h1 { color: #f0f6fc; margin: 0; font-size: 2rem; }
.phishlens-header p  { color: #8b949e; margin: 6px 0 0 0; font-size: 0.95rem; }

/* Verdict banner */
.verdict-phishing   { background:#3d0000; border:2px solid #d32f2f; border-radius:10px; padding:16px 24px; }
.verdict-legitimate { background:#003d00; border:2px solid #388e3c; border-radius:10px; padding:16px 24px; }
.verdict-uncertain  { background:#3d2e00; border:2px solid #fbc02d; border-radius:10px; padding:16px 24px; }
.verdict-phishing h2, .verdict-legitimate h2, .verdict-uncertain h2 {
    margin: 0; font-size: 1.6rem;
}

/* IOC pill */
.ioc-pill {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.82rem;
    font-family: monospace;
    margin: 2px;
    color: #79c0ff;
    word-break: break-all;
}

/* Attack technique card */
.att-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
}
.att-card .tid   { color: #79c0ff; font-weight: bold; font-family: monospace; }
.att-card .tname { color: #f0f6fc; font-size: 0.9rem; }
.att-card .ttac  { color: #8b949e; font-size: 0.8rem; }

/* Feature bar */
.feat-bar-container { background:#21262d; border-radius:4px; overflow:hidden; height:10px; }
.feat-bar-fill      { height:10px; border-radius:4px; transition:width .3s; }
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
    st.sidebar.markdown("## ⚙️ Analysis Options")

    # Model selection
    st.sidebar.markdown("### 🤖 ML Model")
    selected_model = st.sidebar.selectbox(
        "Choose classifier",
        list(AVAILABLE_MODELS.keys()),
        index=0,
        help="LightGBM achieves best F1 (0.9505) and AUC-ROC (0.9941) on the test set.",
    )
    if selected_model in MODEL_METRICS:
        m = MODEL_METRICS[selected_model]
        st.sidebar.markdown(
            f"<small>F1: **{m['f1']:.4f}** · AUC: **{m['auc']:.4f}** · "
            f"FNR: **{m['fnr']:.2%}** · FPR: **{m['fpr']:.2%}**</small>",
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")

    # Threshold
    st.sidebar.markdown("### 🎚️ Decision Threshold")
    threshold = st.sidebar.slider(
        "Phishing threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.50,
        step=0.01,
        help="Probability above this → PHISHING verdict. Lower = more sensitive (higher FPR). Higher = more conservative (higher FNR).",
    )

    st.sidebar.markdown("---")

    # Feature toggles
    st.sidebar.markdown("### 🔧 Feature Modules")
    use_network     = st.sidebar.toggle("Network lookups (WHOIS/cert)", value=False, help="Enables WHOIS and crt.sh lookups for URL features. Adds ~3-10s per email.")
    use_intelligence = st.sidebar.toggle("Threat Intelligence APIs", value=False, help="Queries VirusTotal, Google Safe Browsing, URLScan, URLhaus, AbuseIPDB, IPQS. Requires API keys in .env.")
    use_chatgpt     = st.sidebar.toggle("ChatGPT forensic analysis", value=False, help="Sends full forensic context to ChatGPT gpt-4.1-mini. Requires OPENAI_API_KEY.")
    use_shap        = st.sidebar.toggle("SHAP explainability", value=True, help="Compute SHAP feature importance for the verdict. May add 2-5s.")

    st.sidebar.markdown("---")

    # API key status
    st.sidebar.markdown("### 🔑 API Key Status")
    _show_api_status()

    st.sidebar.markdown("---")

    # Model performance table
    with st.sidebar.expander("📊 All model benchmarks"):
        df = pd.DataFrame(MODEL_METRICS).T.reset_index()
        df.columns = ["Model", "F1", "AUC-ROC", "FNR", "FPR"]
        df = df.sort_values("F1", ascending=False)
        st.dataframe(
            df.style.format({"F1": "{:.4f}", "AUC-ROC": "{:.4f}", "FNR": "{:.2%}", "FPR": "{:.2%}"}),
            hide_index=True,
            use_container_width=True,
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<small style='color:#8b949e'>PhishLens v1.0 · 103k-email corpus · "
        "MITRE ATT&CK integrated<br>"
        "[GitHub](https://github.com/CyberSec-Sagar-Security/PhishSentinel)</small>",
        unsafe_allow_html=True,
    )

    return {
        "model_name": selected_model,
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
def render_header():
    st.markdown("""
    <div class="phishlens-header">
        <h1>🛡️ PhishLens</h1>
        <p>ML-powered phishing email detection · 961-feature vector · 5 ensemble models · ChatGPT forensic analysis</p>
    </div>
    """, unsafe_allow_html=True)


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
        except Exception as exc:
            log.warning(f"IOC extraction failed: {exc}")
            results["iocs"] = {}
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
    verdict      = results["verdict"]
    phishing_prob = results["phishing_prob"]
    model_name   = results["model_name"]

    css_class = {
        "PHISHING": "verdict-phishing",
        "LEGITIMATE": "verdict-legitimate",
        "UNCERTAIN": "verdict-uncertain",
    }.get(verdict, "verdict-uncertain")

    icon = {"PHISHING": "🚨", "LEGITIMATE": "✅", "UNCERTAIN": "⚠️"}.get(verdict, "❓")
    colour = {"PHISHING": "#ff6b6b", "LEGITIMATE": "#69db7c", "UNCERTAIN": "#ffd43b"}.get(verdict, "#adb5bd")

    st.markdown(f"""
    <div class="{css_class}">
        <h2 style="color:{colour}">{icon} {verdict}</h2>
        <p style="color:#ccc;margin:4px 0 0 0">
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
        conf_bar_colour = "#d32f2f" if conf_pct >= 70 else ("#f57c00" if conf_pct >= 40 else "#fbc02d")
        evidence_str = "; ".join(tech.get("evidence", []))
        mitre_url = tech.get("mitre_url", f"https://attack.mitre.org/techniques/{tech.get('technique_id', '')}/")

        st.markdown(f"""
        <div class="att-card">
            <div>
                <a href="{mitre_url}" target="_blank" class="tid">{tech.get('technique_id', '')}</a>
                <span class="tname"> — {tech.get('technique_name', '')}</span>
            </div>
            <div class="ttac">Tactic: {tech.get('tactic', '')} · Confidence: {conf_pct}%</div>
            <div class="feat-bar-container" style="margin-top:6px">
                <div class="feat-bar-fill" style="width:{conf_pct}%;background:{conf_bar_colour}"></div>
            </div>
            <div style="color:#8b949e;font-size:0.78rem;margin-top:4px">{evidence_str}</div>
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
            colour = "#d32f2f" if "MALICIOUS" in str(verdict).upper() else \
                     "#f57c00" if "SUSPICIOUS" in str(verdict).upper() else \
                     "#388e3c" if "CLEAN" in str(verdict).upper() else "#757575"
            st.markdown(
                f'`{ioc}` → <span style="color:{colour};font-weight:bold">{verdict}</span>',
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
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
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
    render_header()
    config = render_sidebar()

    raw_email, display_name = render_input_tabs()

    if not raw_email:
        # Landing page — show quick-start info
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
            All trained on **103,067 labelled emails** with 961-dimensional features.
            """)
        with col3:
            st.markdown("""
            #### 🔍 Deep Analysis
            IOC extraction · MITRE ATT&CK mapping · SHAP explainability ·
            ChatGPT gpt-4.1-mini forensic layer · 6 threat-intel APIs.
            """)

        st.markdown("---")
        st.markdown("### 📈 Model Performance (103k-email test set)")
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

    # ── Run analysis ──────────────────────────────────────────────────────
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

    # ── Verdict banner ─────────────────────────────────────────────────────
    render_verdict_banner(results)

    # ── Probability gauge ──────────────────────────────────────────────────
    try:
        import plotly.graph_objects as go
        prob = results["phishing_prob"]
        gauge_colour = "#d32f2f" if prob >= config["threshold"] else \
                       "#fbc02d" if prob >= THRESH_UNCERTAIN else "#388e3c"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 28, "color": "#f0f6fc"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b949e", "tickfont": {"color": "#8b949e"}},
                "bar": {"color": gauge_colour},
                "bgcolor": "#21262d",
                "steps": [
                    {"range": [0, THRESH_UNCERTAIN * 100], "color": "#0d2b0d"},
                    {"range": [THRESH_UNCERTAIN * 100, config["threshold"] * 100], "color": "#2b1d00"},
                    {"range": [config["threshold"] * 100, 100], "color": "#2b0000"},
                ],
                "threshold": {
                    "line": {"color": "#f0f6fc", "width": 2},
                    "thickness": 0.75,
                    "value": config["threshold"] * 100,
                },
            },
            title={"text": "Phishing Probability", "font": {"color": "#8b949e", "size": 14}},
            domain={"x": [0, 1], "y": [0, 1]},
        ))
        fig.update_layout(
            paper_bgcolor="#0d1117",
            font_color="#f0f6fc",
            height=220,
            margin={"l": 20, "r": 20, "t": 40, "b": 10},
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.metric("Phishing Probability", f"{results['phishing_prob']:.1%}")

    # ── Anomaly score ──────────────────────────────────────────────────────
    if results.get("anomaly_score") is not None:
        anomaly = results["anomaly_score"]
        anomaly_pct = max(0, min(100, int((-anomaly + 0.5) * 100)))
        st.metric(
            "Anomaly Score",
            f"{anomaly:.4f}",
            help="Isolation Forest anomaly score. More negative = more anomalous. Typical phishing emails score < -0.1.",
        )

    st.markdown("---")

    # ── Detail tabs ────────────────────────────────────────────────────────
    tab_summary, tab_iocs, tab_attack, tab_chatgpt, tab_shap, tab_raw = st.tabs([
        "📧 Email Summary",
        "🔍 IOCs",
        "⚔️ MITRE ATT&CK",
        "🤖 ChatGPT Analysis",
        "📊 SHAP Explainability",
        "🔬 Raw Features",
    ])

    with tab_summary:
        render_email_summary(results["parsed"])
        render_email_body(results["parsed"])

    with tab_iocs:
        render_iocs(results.get("iocs", {}))

    with tab_attack:
        render_attack_techniques(results.get("attack_techniques", []))

    with tab_chatgpt:
        render_chatgpt_analysis(results.get("ai_result"))

    with tab_shap:
        render_shap(results.get("shap_df"))

    with tab_raw:
        if results.get("X") is not None:
            render_raw_features(results["X"], results.get("feature_names", []))

    st.markdown("---")
    render_full_report_download(results, display_name or "unknown")


if __name__ == "__main__":
    main()
