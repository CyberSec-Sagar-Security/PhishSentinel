"""HuggingFace Hub model downloader for PhishSentinel.

When running inside a HuggingFace Space (SPACE_ID env var is present),
models are not stored in the git repo.  This module downloads them from the
dedicated HF Hub model repository at startup so the Streamlit app can load them.

Usage (called from app.py before any joblib.load calls):
    from src.utils.hf_model_loader import ensure_models
    ensure_models(MODELS_DIR)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
# HF Hub model repo that holds the trained artefacts.
# Must be public, OR the HF_TOKEN Space secret must grant read access.
HF_MODEL_REPO = "SagarTony90265/PhishSentinel-models"

# All artefact files that must be present for the app to function.
REQUIRED_MODEL_FILES: list[str] = [
    "anomaly_detector.pkl",
    "catboost.pkl",
    "feature_pipeline.pkl",
    "lightgbm.pkl",
    "lr.pkl",
    "lr_scaler.pkl",
    "rf.pkl",
    "xgboost.pkl",
]


def ensure_models(models_dir: Path) -> bool:
    """Download missing model artefacts from HF Hub.

    Parameters
    ----------
    models_dir:
        Absolute path to the directory where .pkl files are expected
        (``models/models/`` relative to the repo root).

    Returns
    -------
    bool
        ``True`` when all required files are present (either pre-existing
        or freshly downloaded).  ``False`` on any download failure so the
        caller can surface a meaningful error message rather than an obscure
        FileNotFoundError.
    """
    on_hf_spaces = bool(os.getenv("SPACE_ID"))

    missing = [f for f in REQUIRED_MODEL_FILES if not (models_dir / f).exists()]

    if not missing:
        return True  # Nothing to do — all artefacts present locally.

    if not on_hf_spaces:
        # Local dev environment: models should have been trained locally.
        # Don't try to auto-download — let the app show its normal
        # "model not found" guidance instead.
        log.warning(
            "Model files missing locally: %s.  "
            "Run train.py to generate them, or download from HF Hub manually.",
            missing,
        )
        return False

    # ── We're on HF Spaces and models are missing — download from Hub ────────
    log.info(
        "Running on HF Spaces.  Downloading %d model file(s) from %s …",
        len(missing),
        HF_MODEL_REPO,
    )

    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        log.error(
            "huggingface_hub is not installed — cannot download models.  "
            "Add `huggingface_hub` to requirements.txt."
        )
        return False

    models_dir.mkdir(parents=True, exist_ok=True)
    token: str | None = os.getenv("HF_TOKEN") or None  # None → anonymous (public repos)

    failed: list[str] = []
    for filename in missing:
        try:
            dest = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=filename,
                local_dir=str(models_dir),
                token=token,
            )
            log.info("Downloaded %s → %s", filename, dest)
        except Exception as exc:
            log.error("Failed to download %s: %s", filename, exc)
            failed.append(filename)

    if failed:
        log.error("Could not download: %s", failed)
        return False

    log.info("All model artefacts ready in %s", models_dir)
    return True
