"""
upload_models_to_hub.py — One-time script to upload trained model artefacts
to the HuggingFace Hub model repository.

Run this ONCE from the repo root after training is complete:

    python upload_models_to_hub.py

Prerequisites
─────────────
1.  Create a HF model repo at: https://huggingface.co/new
    Repository name : PhishSentinel-models
    Owner           : SagarTony90265
    Visibility      : Public  (or Private — but then set HF_TOKEN in HF Space Secrets)

2.  Set your HF write-access token in the environment (or .env):
        HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

3.  Activate the project venv first:
        .\\phishlens\\.venv\\Scripts\\Activate.ps1

Then run this script.  It uploads every .pkl file from models/models/
using Git LFS-backed storage so even the 400 MB rf.pkl is handled correctly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
HF_MODEL_REPO   = "SagarTony90265/PhishSentinel-models"
MODELS_DIR      = Path(__file__).parent / "models" / "models"
REQUIRED_FILES  = [
    "anomaly_detector.pkl",
    "catboost.pkl",
    "feature_pipeline.pkl",
    "lightgbm.pkl",
    "lr.pkl",
    "lr_scaler.pkl",
    "rf.pkl",
    "xgboost.pkl",
]


def main() -> None:
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError:
        print("[ERROR] huggingface_hub is not installed.  Run: pip install huggingface_hub")
        sys.exit(1)

    token = os.getenv("HF_TOKEN") or None
    if not token:
        print(
            "[ERROR] HF_TOKEN environment variable is not set.\n"
            "Export it before running:\n"
            "  $env:HF_TOKEN = 'hf_xxxx'  (PowerShell)\n"
            "  export HF_TOKEN=hf_xxxx     (bash / Linux)"
        )
        sys.exit(1)

    api = HfApi()

    # Verify repo exists (will raise if not)
    try:
        api.repo_info(repo_id=HF_MODEL_REPO, repo_type="model", token=token)
        print(f"[OK] Repo found: https://huggingface.co/{HF_MODEL_REPO}")
    except Exception as exc:
        print(
            f"[ERROR] Cannot access repo '{HF_MODEL_REPO}': {exc}\n"
            f"Create it at https://huggingface.co/new first."
        )
        sys.exit(1)

    missing = [f for f in REQUIRED_FILES if not (MODELS_DIR / f).exists()]
    if missing:
        print(f"[ERROR] Missing local model files: {missing}")
        print(f"        Expected in: {MODELS_DIR}")
        sys.exit(1)

    print(f"\nUploading {len(REQUIRED_FILES)} files to {HF_MODEL_REPO} …\n")
    for filename in REQUIRED_FILES:
        path = MODELS_DIR / filename
        size_mb = path.stat().st_size / 1_048_576
        print(f"  → {filename:35s} ({size_mb:7.1f} MB) … ", end="", flush=True)
        try:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=filename,
                repo_id=HF_MODEL_REPO,
                repo_type="model",
                token=token,
            )
            print("done")
        except Exception as exc:
            print(f"FAILED — {exc}")
            sys.exit(1)

    print(f"\n[SUCCESS] All models uploaded.")
    print(f"          View at: https://huggingface.co/{HF_MODEL_REPO}\n")


if __name__ == "__main__":
    main()
