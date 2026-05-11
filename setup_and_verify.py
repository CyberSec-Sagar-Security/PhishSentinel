#!/usr/bin/env python3
"""
PhishLens Automated Setup & Verification Script
================================================
Handles the full project bootstrap from a clean Python install:
  1. Creates / activates a virtual environment
  2. Upgrades pip and installs all requirements
  3. Verifies every core import
  4. Creates missing __init__.py files
  5. Copies .env.example → .env if .env does not exist
  6. Runs the test suite
  7. Prints a final health report

Usage:
    python setup_and_verify.py              # full bootstrap
    python setup_and_verify.py --skip-install  # skip pip install (already done)
    python setup_and_verify.py --skip-tests    # skip pytest run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ── Colours (no external deps needed) ────────────────────────────────────────
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

BASE_DIR = Path(__file__).parent.resolve()

# ── Modules to verify ─────────────────────────────────────────────────────────
IMPORT_CHECKS = [
    # (import_statement, friendly_name)
    ("import numpy",                        "numpy"),
    ("import pandas",                       "pandas"),
    ("import sklearn",                      "scikit-learn"),
    ("import xgboost",                      "xgboost"),
    ("import lightgbm",                     "lightgbm"),
    ("import catboost",                     "catboost"),
    ("import imblearn",                     "imbalanced-learn"),
    ("import shap",                         "shap"),
    ("import lime",                         "lime"),
    ("import torch",                        "torch"),
    ("import transformers",                 "transformers"),
    ("import sentence_transformers",        "sentence-transformers"),
    ("import tldextract",                   "tldextract"),
    ("import whois",                        "python-whois"),
    ("import dns.resolver",                 "dnspython"),
    ("import Levenshtein",                  "python-levenshtein"),
    ("import bs4",                          "beautifulsoup4"),
    ("import lxml",                         "lxml"),
    ("import mailparser",                   "mail-parser"),
    ("import google.generativeai",          "google-generativeai"),
    ("import matplotlib",                   "matplotlib"),
    ("import seaborn",                      "seaborn"),
    ("import plotly",                       "plotly"),
    ("import joblib",                       "joblib"),
    ("import optuna",                       "optuna"),
    ("import mlflow",                       "mlflow"),
    ("import streamlit",                    "streamlit"),
    ("import dotenv",                       "python-dotenv"),
    ("import pydantic",                     "pydantic"),
    ("import loguru",                       "loguru"),
    ("import aiohttp",                      "aiohttp"),
    ("import requests",                     "requests"),
    ("import pytest",                       "pytest"),
]

PHISHLENS_IMPORTS = [
    ("from src.utils.config import DEFAULT_CONFIG",           "src.utils.config"),
    ("from src.utils.logger import get_logger",               "src.utils.logger"),
    ("from src.ingestion.eml_parser import parse_eml_string", "src.ingestion.eml_parser"),
    ("from src.features.header_features import extract_header_features",
                                                              "src.features.header_features"),
    ("from src.features.url_features import extract_url_features",
                                                              "src.features.url_features"),
    ("from src.features.html_features import extract_html_features",
                                                              "src.features.html_features"),
    ("from src.features.text_features import extract_text_features",
                                                              "src.features.text_features"),
    ("from src.features.pipeline import FeaturePipeline",    "src.features.pipeline"),
    ("from src.detection.anomaly import ZeroDayDetector",    "src.detection.anomaly"),
    ("from src.models.trainer import PhishLensTrainer",       "src.models.trainer"),
    ("from src.models.evaluator import PhishLensEvaluator",   "src.models.evaluator"),
    ("from src.models.explainer import PhishExplainer",       "src.models.explainer"),
    ("from src.ioc_extractor import extract_iocs",            "src.ioc_extractor"),
    ("from src.attack_mapping import map_attack_techniques",  "src.attack_mapping"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_header(msg: str) -> None:
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")


def _ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET}  {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}✗{RESET}  {msg}")


def _run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> int:
    result = subprocess.run(
        cmd, cwd=str(cwd or BASE_DIR),
        capture_output=False, text=True,
    )
    if check and result.returncode != 0:
        _fail(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")
        sys.exit(result.returncode)
    return result.returncode


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Virtual Environment
# ─────────────────────────────────────────────────────────────────────────────

def ensure_venv() -> Path:
    venv_dir = BASE_DIR / ".venv"
    if not venv_dir.exists():
        _print_header("Step 1: Creating virtual environment")
        _run([sys.executable, "-m", "venv", str(venv_dir)])
        _ok(f"Virtual environment created at {venv_dir}")
    else:
        _print_header("Step 1: Virtual environment already exists")
        _ok(str(venv_dir))
    return venv_dir


def get_venv_python(venv_dir: Path) -> str:
    """Return path to the Python executable inside the venv."""
    if sys.platform == "win32":
        py = venv_dir / "Scripts" / "python.exe"
    else:
        py = venv_dir / "bin" / "python"
    return str(py)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Pip upgrade + requirements install
# ─────────────────────────────────────────────────────────────────────────────

def install_requirements(python_exe: str) -> None:
    _print_header("Step 2: Installing dependencies")

    print(f"  {YELLOW}→{RESET} Upgrading pip …")
    _run([python_exe, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    _ok("pip / setuptools / wheel upgraded")

    req_file = BASE_DIR / "requirements.txt"
    print(f"  {YELLOW}→{RESET} Installing requirements.txt (this may take 5-10 minutes) …")
    _run([
        python_exe, "-m", "pip", "install",
        "-r", str(req_file),
        "--no-warn-script-location",
    ])
    _ok("All requirements installed")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Project structure checks
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_INIT_DIRS = [
    "src",
    "src/utils",
    "src/ingestion",
    "src/features",
    "src/detection",
    "src/models",
    "tests",
]


def ensure_project_structure() -> None:
    _print_header("Step 3: Verifying project structure")

    for d in REQUIRED_INIT_DIRS:
        init_file = BASE_DIR / d / "__init__.py"
        if not init_file.exists():
            init_file.parent.mkdir(parents=True, exist_ok=True)
            init_file.write_text('"""Package."""\n')
            _ok(f"Created {init_file.relative_to(BASE_DIR)}")
        else:
            _ok(f"{init_file.relative_to(BASE_DIR)} exists")

    # .env setup
    env_file = BASE_DIR / ".env"
    env_example = BASE_DIR / ".env.example"
    if not env_file.exists():
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_file)
            _warn(".env copied from .env.example — add your real API keys!")
        else:
            env_file.write_text(
                "OPENAI_API_KEY=your_openai_key_here\n"
                "VIRUSTOTAL_API_KEY=your_virustotal_key_here\n"
                "GOOGLE_SAFE_BROWSING_API_KEY=your_gsb_key_here\n"
                "ABUSEIPDB_API_KEY=your_abuseipdb_key_here\n"
                "URLSCAN_API_KEY=your_urlscan_key_here\n"
            )
            _warn(".env created with placeholder values — fill in your API keys!")
    else:
        _ok(".env file exists")

    # data directory
    (BASE_DIR / "data" / "raw").mkdir(parents=True, exist_ok=True)
    _ok("data/raw directory ready")

    # models directory
    (BASE_DIR / "models").mkdir(exist_ok=True)
    _ok("models/ directory ready")

    # reports directory
    (BASE_DIR / "reports").mkdir(exist_ok=True)
    _ok("reports/ directory ready")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Import verification
# ─────────────────────────────────────────────────────────────────────────────

def _check_import(python_exe: str, stmt: str, label: str) -> bool:
    result = subprocess.run(
        [python_exe, "-c", stmt],
        capture_output=True, text=True,
        cwd=str(BASE_DIR),
    )
    return result.returncode == 0


def verify_imports(python_exe: str) -> tuple[int, int]:
    _print_header("Step 4a: Verifying third-party imports")
    ok = fail = 0
    for stmt, label in IMPORT_CHECKS:
        if _check_import(python_exe, stmt, label):
            _ok(label)
            ok += 1
        else:
            _fail(f"{label}  ← import failed")
            fail += 1
    return ok, fail


def verify_phishlens_imports(python_exe: str) -> tuple[int, int]:
    _print_header("Step 4b: Verifying PhishLens module imports")
    ok = fail = 0
    for stmt, label in PHISHLENS_IMPORTS:
        if _check_import(python_exe, stmt, label):
            _ok(label)
            ok += 1
        else:
            _fail(f"{label}  ← import failed")
            fail += 1
    return ok, fail


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Run tests
# ─────────────────────────────────────────────────────────────────────────────

def run_tests(python_exe: str) -> int:
    _print_header("Step 5: Running test suite")
    result = subprocess.run(
        [
            python_exe, "-m", "pytest",
            "tests/",
            "-v",
            "--tb=short",
            "--timeout=60",
            "--ignore=tests/test_pipeline.py",  # skip heavy integration test here
            "-q",
        ],
        cwd=str(BASE_DIR),
        text=True,
    )
    return result.returncode


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Final report
# ─────────────────────────────────────────────────────────────────────────────

def print_final_report(
    lib_ok: int, lib_fail: int,
    mod_ok: int, mod_fail: int,
    test_rc: int,
    skip_tests: bool,
) -> None:
    _print_header("Final Health Report")

    total_ok = lib_ok + mod_ok
    total_fail = lib_fail + mod_fail

    print(f"  Third-party libraries :  {GREEN}{lib_ok} OK{RESET}  / {RED}{lib_fail} FAIL{RESET}")
    print(f"  PhishLens modules     :  {GREEN}{mod_ok} OK{RESET}  / {RED}{mod_fail} FAIL{RESET}")

    if not skip_tests:
        test_str = f"{GREEN}PASSED{RESET}" if test_rc == 0 else f"{RED}FAILED{RESET}"
        print(f"  Test suite            :  {test_str}")

    if total_fail == 0:
        print(f"\n  {GREEN}{BOLD}✓ PhishLens is ready to use!{RESET}")
        print(f"\n  Quick commands:")
        print(f"  {CYAN}  streamlit run app.py{RESET}                           # Launch web UI")
        print(f"  {CYAN}  python download_datasets.py{RESET}                   # Download + build datasets")
        print(f"  {CYAN}  python train.py --data-dir data/processed --models xgboost --no-network --save models{RESET}  # Quick train")
        print(f"  {CYAN}  python train.py --help{RESET}                         # All train options")
    else:
        print(f"\n  {RED}{BOLD}✗ {total_fail} imports failed. Run with --skip-install to debug, or re-run full setup.{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PhishLens setup & verification")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install step")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest run")
    parser.add_argument(
        "--use-system-python", action="store_true",
        help="Use current Python interpreter instead of creating a venv",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}PhishLens — Automated Setup & Verification{RESET}")
    print(f"Base directory : {BASE_DIR}")
    print(f"Python         : {sys.version}")

    if args.use_system_python:
        python_exe = sys.executable
    else:
        venv_dir = ensure_venv()
        python_exe = get_venv_python(venv_dir)

    ensure_project_structure()

    if not args.skip_install:
        install_requirements(python_exe)
    else:
        _print_header("Step 2: Skipped (--skip-install)")

    lib_ok, lib_fail = verify_imports(python_exe)
    mod_ok, mod_fail = verify_phishlens_imports(python_exe)

    test_rc = 0
    if not args.skip_tests:
        test_rc = run_tests(python_exe)

    print_final_report(lib_ok, lib_fail, mod_ok, mod_fail, test_rc, args.skip_tests)


if __name__ == "__main__":
    main()
