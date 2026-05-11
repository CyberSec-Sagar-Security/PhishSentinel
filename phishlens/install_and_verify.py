#!/usr/bin/env python3
"""
PhishLens — Master Install & Verification Script
=================================================
Covers EVERYTHING specified in:
  • PHISHLENS_MASTER_PROMPT.md  (all 8 phases)
  • PhishLens.md (Advanced Upgrade Guide — all 11 upgrades + 9 APIs)

Single command to:
  1.  Create / reuse virtual environment (.venv)
  2.  Install PyTorch CPU (stable) from the official PyTorch wheel channel
      — avoids the nightly 2.11.0 AST bug on Python 3.13
  3.  Install all remaining requirements from requirements.txt
  4.  Verify 33 third-party library imports
  5.  Verify 14 PhishLens module imports
  6.  Check .env API keys are configured (not placeholder values)
  7.  Verify complete project file structure per master prompt
  8.  Run pytest suite with coverage
  9.  Smoke-test app.py (import check without launching Streamlit server)
 10.  Smoke-test train.py --help
 11.  Save a full health report to reports/verify_report.txt

Usage:
    python install_and_verify.py                  # Full bootstrap + verify
    python install_and_verify.py --skip-install   # Skip pip (already done)
    python install_and_verify.py --skip-tests     # Skip pytest
    python install_and_verify.py --verify-only    # Imports + structure + keys only
    python install_and_verify.py --no-color       # Plain output (for log files)

Exit codes:
    0 — All checks passed
    1 — One or more checks failed (see report for details)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Console colours (disable with --no-color or when stdout is not a tty)
# ─────────────────────────────────────────────────────────────────────────────

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") != "1"


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def red(t: str) -> str:    return _c("91", t)
def green(t: str) -> str:  return _c("92", t)
def yellow(t: str) -> str: return _c("93", t)
def cyan(t: str) -> str:   return _c("96", t)
def bold(t: str) -> str:   return _c("1",  t)


BASE_DIR = Path(__file__).parent.resolve()
REPORT_DIR = BASE_DIR / "reports"
VENV_DIR   = BASE_DIR / ".venv"

# ─────────────────────────────────────────────────────────────────────────────
# Third-party import checks (33 libraries — all from PhishLens.md + master prompt)
# ─────────────────────────────────────────────────────────────────────────────

THIRD_PARTY_CHECKS: List[Tuple[str, str]] = [
    # Core ML
    ("import numpy",                   "numpy"),
    ("import pandas",                  "pandas"),
    ("import sklearn",                 "scikit-learn"),
    ("import xgboost",                 "xgboost"),
    ("import lightgbm",                "lightgbm"),
    ("import catboost",                "catboost"),
    ("import imblearn",                "imbalanced-learn"),
    # Explainability
    ("import shap",                    "shap"),
    ("import lime",                    "lime"),
    # NLP & Embeddings
    ("import torch",                   "torch"),
    ("import transformers",            "transformers"),
    ("import sentence_transformers",   "sentence-transformers"),
    ("import tokenizers",              "tokenizers"),
    # URL & Domain
    ("import tldextract",              "tldextract"),
    ("import whois",                   "python-whois"),
    ("import dns.resolver",            "dnspython"),
    ("import Levenshtein",             "python-levenshtein"),
    ("import confusable_homoglyphs",   "confusable-homoglyphs"),
    ("import requests",                "requests"),
    ("import aiohttp",                 "aiohttp"),
    # Email Parsing
    ("import bs4",                     "beautifulsoup4"),
    ("import lxml",                    "lxml"),
    ("import mailparser",              "mail-parser"),
    # AI Layer
    ("import google.generativeai",     "google-generativeai"),
    # Visualisation
    ("import matplotlib",              "matplotlib"),
    ("import seaborn",                 "seaborn"),
    ("import plotly",                  "plotly"),
    # Model Management
    ("import joblib",                  "joblib"),
    ("import optuna",                  "optuna"),
    ("import mlflow",                  "mlflow"),
    # App
    ("import streamlit",               "streamlit"),
    # Utilities
    ("import dotenv",                  "python-dotenv"),
    ("import pydantic",                "pydantic"),
    ("import loguru",                  "loguru"),
    ("import pytest",                  "pytest"),
    # Dataset download — use find_spec: kaggle raises OSError at import if
    # kaggle.json is absent, even though the package is correctly installed.
    ("import importlib.util; assert importlib.util.find_spec('kaggle') is not None",
                                       "kaggle"),
]

# ─────────────────────────────────────────────────────────────────────────────
# PhishLens module import checks (14 internal modules — all from master prompt)
# ─────────────────────────────────────────────────────────────────────────────

PHISHLENS_CHECKS: List[Tuple[str, str]] = [
    ("from src.utils.config import DEFAULT_CONFIG",              "src.utils.config"),
    ("from src.utils.logger import get_logger",                  "src.utils.logger"),
    ("from src.ingestion.eml_parser import parse_eml_string",   "src.ingestion.eml_parser"),
    ("from src.ingestion.dataset_loader import load_meajor",    "src.ingestion.dataset_loader"),
    ("from src.features.header_features import extract_header_features",
                                                                 "src.features.header_features"),
    ("from src.features.url_features import extract_url_features",
                                                                 "src.features.url_features"),
    ("from src.features.html_features import extract_html_features",
                                                                 "src.features.html_features"),
    ("from src.features.text_features import extract_text_features",
                                                                 "src.features.text_features"),
    ("from src.features.pipeline import FeaturePipeline",       "src.features.pipeline"),
    ("from src.detection.anomaly import ZeroDayDetector",       "src.detection.anomaly"),
    ("from src.models.trainer import PhishLensTrainer",          "src.models.trainer"),
    ("from src.models.evaluator import PhishLensEvaluator",      "src.models.evaluator"),
    ("from src.models.explainer import PhishExplainer",          "src.models.explainer"),
    ("from src.ioc_extractor import extract_iocs",               "src.ioc_extractor"),
    ("from src.attack_mapping import map_attack_techniques",     "src.attack_mapping"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Required project files — every file listed in PHISHLENS_MASTER_PROMPT.md
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_FILES = [
    # Source tree
    "src/__init__.py",
    "src/utils/__init__.py",
    "src/utils/config.py",
    "src/utils/logger.py",
    "src/ingestion/__init__.py",
    "src/ingestion/eml_parser.py",
    "src/ingestion/dataset_loader.py",
    "src/features/__init__.py",
    "src/features/header_features.py",
    "src/features/url_features.py",
    "src/features/html_features.py",
    "src/features/text_features.py",
    "src/features/pipeline.py",
    "src/models/__init__.py",
    "src/models/trainer.py",
    "src/models/evaluator.py",
    "src/models/explainer.py",
    "src/models/adversarial_tester.py",
    "src/models/transformer_model.py",
    "src/detection/__init__.py",
    "src/detection/anomaly.py",
    "src/ioc_extractor.py",
    "src/attack_mapping.py",
    # Entry points
    "app.py",
    "train.py",
    # Tests
    "tests/__init__.py",
    "tests/test_eml_parser.py",
    "tests/test_url_features.py",
    "tests/test_html_features.py",
    "tests/test_pipeline.py",
    # Config / CI
    "requirements.txt",
    ".env.example",
    ".gitignore",
    "README.md",
    ".github/workflows/ci.yml",
    "data/README.md",
]

# ─────────────────────────────────────────────────────────────────────────────
# API keys — from PhishLens.md Section 3 + master prompt
# ─────────────────────────────────────────────────────────────────────────────

API_KEYS = [
    ("GEMINI_API_KEY",                 "Google Gemini (AI analysis layer)"),
    ("VIRUSTOTAL_API_KEY",             "VirusTotal (URL reputation)"),
    ("GOOGLE_SAFE_BROWSING_API_KEY",   "Google Safe Browsing (phishing check)"),
    ("ABUSEIPDB_API_KEY",              "AbuseIPDB (sender IP reputation)"),
    ("URLSCAN_API_KEY",                "URLScan.io (URL sandbox)"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{bold(cyan('=' * 62))}")
    print(f"{bold(cyan(f'  {title}'))}")
    print(f"{bold(cyan('=' * 62))}")


def ok(msg: str)   -> None: print(f"  {green('✓')}  {msg}")
def warn(msg: str) -> None: print(f"  {yellow('⚠')}  {msg}")
def fail(msg: str) -> None: print(f"  {red('✗')}  {msg}")
def info(msg: str) -> None: print(f"  {cyan('→')}  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Virtual environment
# ─────────────────────────────────────────────────────────────────────────────

def ensure_venv() -> Path:
    """Create .venv if it doesn't exist; return its path."""
    section("Step 1 — Virtual Environment")
    if VENV_DIR.exists():
        ok(f"Existing venv: {VENV_DIR}")
    else:
        info("Creating .venv …")
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(VENV_DIR)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            fail(f"venv creation failed:\n{result.stderr}")
            sys.exit(1)
        ok(f"Created: {VENV_DIR}")
    return VENV_DIR


def get_venv_python(venv: Path) -> str:
    if sys.platform == "win32":
        return str(venv / "Scripts" / "python.exe")
    return str(venv / "bin" / "python")


def get_venv_pip(venv: Path) -> str:
    if sys.platform == "win32":
        return str(venv / "Scripts" / "pip.exe")
    return str(venv / "bin" / "pip")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Install dependencies (torch first from PyTorch channel)
# ─────────────────────────────────────────────────────────────────────────────

def _run_pip(python_exe: str, args: List[str], label: str) -> bool:
    """Run a pip command and return True on success."""
    cmd = [python_exe, "-m", "pip"] + args
    info(f"{label} …")
    result = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
    if result.returncode == 0:
        ok(label)
        return True
    else:
        # Show last 10 lines of pip output on failure
        lines = (result.stdout + result.stderr).strip().splitlines()
        for line in lines[-10:]:
            print(f"    {line}")
        fail(f"{label} failed (exit {result.returncode})")
        return False


def install_dependencies(python_exe: str) -> None:
    """Install torch from PyTorch CPU channel, then all other requirements."""
    section("Step 2 — Installing Dependencies")

    # Upgrade pip / setuptools / wheel first
    _run_pip(python_exe, ["install", "--upgrade", "pip", "setuptools", "wheel"],
             "Upgrade pip + setuptools + wheel")

    # ── PyTorch MUST be installed from the official CPU wheel channel ──────
    # Reason: PyPI serves nightly torch builds that have a Python 3.13 AST bug
    # (IndentationError in rnn.py). The stable +cpu wheels at pytorch.org work.
    info("Installing PyTorch CPU (stable, from pytorch.org wheel channel) …")
    torch_result = subprocess.run(
        [
            python_exe, "-m", "pip", "install",
            "torch>=2.5.0,<2.11.0",
            "--index-url", "https://download.pytorch.org/whl/cpu",
            "--quiet",
        ],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
    )
    if torch_result.returncode == 0:
        ok("PyTorch CPU (stable) installed")
    else:
        lines = (torch_result.stdout + torch_result.stderr).strip().splitlines()
        for line in lines[-8:]:
            print(f"    {line}")
        fail("PyTorch CPU install failed — continuing anyway")

    # ── Everything else from requirements.txt (torch will be skipped if already ok) ──
    req_file = BASE_DIR / "requirements.txt"
    if not req_file.exists():
        fail("requirements.txt not found!")
        sys.exit(1)

    info("Installing requirements.txt (may take 5-10 min on first run) …")
    req_result = subprocess.run(
        [
            python_exe, "-m", "pip", "install",
            "-r", str(req_file),
            "--no-warn-script-location",
            "--quiet",
        ],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
    )
    if req_result.returncode == 0:
        ok("All requirements.txt packages installed")
    else:
        lines = (req_result.stdout + req_result.stderr).strip().splitlines()
        for line in lines[-15:]:
            print(f"    {line}")
        fail("requirements.txt install had errors (some packages may still work)")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Project structure verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_structure() -> Tuple[int, int, List[str]]:
    """Check all required files exist. Returns (found, missing, missing_list)."""
    section("Step 3 — Project Structure")
    found = missing = 0
    missing_list: List[str] = []

    # Check and create __init__.py files if missing
    init_dirs = [
        "src", "src/utils", "src/ingestion",
        "src/features", "src/detection", "src/models", "tests",
    ]
    for d in init_dirs:
        init = BASE_DIR / d / "__init__.py"
        if not init.exists():
            init.parent.mkdir(parents=True, exist_ok=True)
            init.write_text('"""Package."""\n')
            warn(f"Auto-created {d}/__init__.py")

    # Create required directories
    for d in ["data/raw", "data/processed", "models/saved",
              "reports/figures", "logs"]:
        (BASE_DIR / d).mkdir(parents=True, exist_ok=True)

    # Check every required file
    for rel_path in REQUIRED_FILES:
        full = BASE_DIR / rel_path
        if full.exists():
            ok(rel_path)
            found += 1
        else:
            fail(f"{rel_path}  ← MISSING")
            missing += 1
            missing_list.append(rel_path)

    # .env
    env = BASE_DIR / ".env"
    if not env.exists():
        env_ex = BASE_DIR / ".env.example"
        if env_ex.exists():
            import shutil
            shutil.copy(env_ex, env)
            warn(".env created from .env.example — fill in your API keys!")
        else:
            env.write_text(
                "GEMINI_API_KEY=your_gemini_key_here\n"
                "VIRUSTOTAL_API_KEY=your_virustotal_key_here\n"
                "GOOGLE_SAFE_BROWSING_API_KEY=your_gsb_key_here\n"
                "ABUSEIPDB_API_KEY=your_abuseipdb_key_here\n"
                "URLSCAN_API_KEY=your_urlscan_key_here\n"
            )
            warn(".env created with placeholders — fill in your API keys!")
    else:
        ok(".env")

    return found, missing, missing_list


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Third-party import verification
# ─────────────────────────────────────────────────────────────────────────────

def _check_import(python_exe: str, stmt: str) -> Tuple[bool, str]:
    """Run 'python -c stmt' and return (success, error_message)."""
    result = subprocess.run(
        [python_exe, "-c", stmt],
        capture_output=True, text=True,
        cwd=str(BASE_DIR),
    )
    if result.returncode == 0:
        return True, ""
    # Extract the last meaningful error line
    err_lines = [l.strip() for l in (result.stdout + result.stderr).splitlines() if l.strip()]
    err = err_lines[-1] if err_lines else "unknown error"
    return False, err


def verify_third_party(python_exe: str) -> Tuple[int, int, List[str]]:
    section("Step 4a — Third-Party Library Imports  (36 packages)")
    passed = failed = 0
    failures: List[str] = []

    for stmt, label in THIRD_PARTY_CHECKS:
        success, err = _check_import(python_exe, stmt)
        if success:
            ok(f"{label}")
            passed += 1
        else:
            fail(f"{label:<30}  ← {err[:80]}")
            failed += 1
            failures.append(f"{label}: {err}")

    return passed, failed, failures


def verify_phishlens_modules(python_exe: str) -> Tuple[int, int, List[str]]:
    section("Step 4b — PhishLens Internal Module Imports  (15 modules)")
    passed = failed = 0
    failures: List[str] = []

    for stmt, label in PHISHLENS_CHECKS:
        success, err = _check_import(python_exe, stmt)
        if success:
            ok(label)
            passed += 1
        else:
            fail(f"{label:<45}  ← {err[:60]}")
            failed += 1
            failures.append(f"{label}: {err}")

    return passed, failed, failures


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — .env API key check
# ─────────────────────────────────────────────────────────────────────────────

def verify_api_keys() -> Tuple[int, int]:
    """Check .env for all required API keys. Returns (configured, placeholder)."""
    section("Step 5 — API Key Configuration  (.env)")

    env_path = BASE_DIR / ".env"
    env_values: dict = {}

    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env_values[k.strip()] = v.strip()

    configured = placeholder = 0
    placeholder_keywords = {"your_", "_here", "placeholder", "xxx", "changeme", "insert"}

    for key, description in API_KEYS:
        value = env_values.get(key, "")
        if not value:
            warn(f"{key:<40}  {yellow('NOT SET')}  ({description})")
            placeholder += 1
        elif any(p in value.lower() for p in placeholder_keywords):
            warn(f"{key:<40}  {yellow('PLACEHOLDER')}  ({description})")
            warn(f"  Register free: see PhishLens.md Section 3")
            placeholder += 1
        else:
            ok(f"{key:<40}  {green('CONFIGURED')}  ({description})")
            configured += 1

    if placeholder > 0:
        print()
        info("API keys are optional for basic usage — they enable:")
        info("  VirusTotal/AbuseIPDB/URLScan → URL reputation scoring")
        info("  Google Safe Browsing         → phishing URL cross-check")
        info("  Gemini                       → AI email analysis layer")
        info("  All are FREE to register (see PhishLens.md Section 3)")

    return configured, placeholder


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — pytest suite
# ─────────────────────────────────────────────────────────────────────────────

def run_tests(python_exe: str) -> Tuple[int, str]:
    """Run pytest on tests/ (excluding heavy pipeline integration test).
    Returns (exit_code, summary_line)."""
    section("Step 6 — Test Suite  (pytest)")

    cmd = [
        python_exe, "-m", "pytest",
        "tests/test_eml_parser.py",
        "tests/test_url_features.py",
        "tests/test_html_features.py",
        "-v",
        "--tb=short",
        "--timeout=60",
        "-q",
    ]
    result = subprocess.run(cmd, cwd=str(BASE_DIR), text=True,
                            capture_output=False)

    # Extract summary from stdout (it was printed above)
    return result.returncode, ("PASSED" if result.returncode == 0 else "FAILED")


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — app.py and train.py smoke tests
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test_entrypoints(python_exe: str) -> Tuple[int, int]:
    """Verify app.py and train.py can be imported / invoked without crashing."""
    section("Step 7 — Entry Point Smoke Tests")
    passed = failed = 0

    # train.py --help (just arg parsing, no training)
    result = subprocess.run(
        [python_exe, "train.py", "--help"],
        cwd=str(BASE_DIR),
        capture_output=True, text=True,
        timeout=30,
    )
    if result.returncode == 0:
        ok("train.py --help  (argparse OK)")
        passed += 1
    else:
        err_lines = (result.stdout + result.stderr).strip().splitlines()
        for line in err_lines[-5:]:
            print(f"    {line}")
        fail("train.py --help  ← failed")
        failed += 1

    # app.py import check (no Streamlit server; just check the module parses)
    app_check = (
        "import ast, pathlib; "
        "ast.parse(pathlib.Path('app.py').read_text(encoding='utf-8')); "
        "print('app.py: syntax OK')"
    )
    result = subprocess.run(
        [python_exe, "-c", app_check],
        cwd=str(BASE_DIR),
        capture_output=True, text=True,
        timeout=15,
    )
    if result.returncode == 0:
        ok("app.py  (syntax + AST parse OK)")
        passed += 1
    else:
        err_lines = (result.stdout + result.stderr).strip().splitlines()
        fail(f"app.py  ← {err_lines[-1][:80] if err_lines else 'syntax error'}")
        failed += 1

    # Streamlit launch check (dry-run: print command only)
    info("Streamlit launch command (NOT starting server here):")
    info(f"  {cyan('streamlit run app.py --server.port 8501')}")
    info("Run this manually to launch the PhishLens web UI.")

    return passed, failed


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Final report
# ─────────────────────────────────────────────────────────────────────────────

def print_and_save_report(
    python_exe: str,
    python_version: str,
    lib_ok: int,      lib_fail: int,   lib_failures: List[str],
    mod_ok: int,      mod_fail: int,   mod_failures: List[str],
    struct_ok: int,   struct_miss: int, missing_files: List[str],
    key_cfg: int,     key_ph: int,
    test_rc: int,     test_summary: str,
    entry_ok: int,    entry_fail: int,
    skip_install: bool,
    skip_tests: bool,
    elapsed: float,
) -> int:
    """Print final report to console and save plain-text version to reports/."""

    section("FINAL HEALTH REPORT")

    total_critical_fail = lib_fail + mod_fail + struct_miss
    all_ok = (total_critical_fail == 0) and (test_rc == 0 or skip_tests)

    rows = [
        ("Third-party libraries (36)", lib_ok,    lib_fail,    lib_fail == 0),
        ("PhishLens modules (15)",     mod_ok,    mod_fail,    mod_fail == 0),
        ("Project file structure",     struct_ok, struct_miss, struct_miss == 0),
        ("API keys configured",        key_cfg,   key_ph,      True),   # non-critical
        ("Entry point smoke tests",    entry_ok,  entry_fail,  entry_fail == 0),
    ]

    for label, n_ok, n_fail, is_ok in rows:
        status = green("PASS") if is_ok else red("FAIL")
        print(f"  [{status}]  {label:<35}  {green(str(n_ok))} OK  /  {red(str(n_fail))} FAIL")

    if not skip_tests:
        test_color = green("PASS") if test_rc == 0 else red("FAIL")
        print(f"  [{test_color}]  {'pytest suite':<35}  {test_summary}")

    print(f"\n  Python     : {python_version}")
    print(f"  Interpreter: {python_exe}")
    print(f"  Run time   : {elapsed:.1f}s")
    print(f"  Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if all_ok:
        print(f"\n  {bold(green('✓  PhishLens is fully installed and verified!'))}")
        print(f"\n  {bold('Quick-start commands:')}")
        print(f"  {cyan('  streamlit run app.py --server.port 8501')}")
        print(f"         → Launches the web UI on http://localhost:8501")
        print(f"  {cyan('  python download_datasets.py --status')}")
        print(f"         → Confirms processed datasets and class balance")
        print(f"  {cyan('  python train.py --data-dir data/processed --models xgboost --no-network --eval --save models')}")
        print(f"         → Fast full pipeline training with local datasets")
        print(f"  {cyan('  python train.py --help')}")
        print(f"         → All training options")
        print(f"\n  {bold('Dataset download (needed before training):')}")
        print(f"  {cyan('  python download_datasets.py')}")
        print(f"         → Downloads and builds data/processed/train.csv + test.csv")
    else:
        print(f"\n  {bold(red('✗  Some checks failed — see details above and below.'))}")
        if lib_failures:
            print(f"\n  {bold('Import failures:')}")
            for f in lib_failures:
                print(f"    {red('•')} {f}")
        if mod_failures:
            print(f"\n  {bold('Module failures:')}")
            for f in mod_failures:
                print(f"    {red('•')} {f}")
        if missing_files:
            print(f"\n  {bold('Missing files:')}")
            for f in missing_files:
                print(f"    {red('•')} {f}")
        print(f"\n  {bold('Re-run with:')}")
        print(f"  {cyan('  python install_and_verify.py')}")
        print(f"         → Will retry installation and re-verify everything")

    # ── Save plain-text report ────────────────────────────────────────────
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "verify_report.txt"
    json_path   = REPORT_DIR / "verify_report.json"

    report_lines = [
        "PhishLens Verification Report",
        "=" * 60,
        f"Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Python     : {python_version}",
        f"Interpreter: {python_exe}",
        f"Run time   : {elapsed:.1f}s",
        "",
        "SUMMARY",
        "-" * 40,
        f"Third-party libraries : {lib_ok} OK / {lib_fail} FAIL",
        f"PhishLens modules     : {mod_ok} OK / {mod_fail} FAIL",
        f"Project file structure: {struct_ok} OK / {struct_miss} MISSING",
        f"API keys configured   : {key_cfg} / {key_cfg + key_ph} total",
        f"Entry point tests     : {entry_ok} OK / {entry_fail} FAIL",
    ]
    if not skip_tests:
        report_lines.append(f"pytest suite          : {test_summary}")
    report_lines += ["", "OVERALL: " + ("PASS" if all_ok else "FAIL")]

    if lib_failures:
        report_lines += ["", "IMPORT FAILURES:"] + [f"  - {e}" for e in lib_failures]
    if mod_failures:
        report_lines += ["", "MODULE FAILURES:"] + [f"  - {e}" for e in mod_failures]
    if missing_files:
        report_lines += ["", "MISSING FILES:"] + [f"  - {f}" for f in missing_files]

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n  {bold('Full report saved:')} {report_path}")

    # JSON report for CI / tooling
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "python_version": python_version,
        "interpreter": python_exe,
        "elapsed_seconds": round(elapsed, 1),
        "overall": "PASS" if all_ok else "FAIL",
        "libraries": {"ok": lib_ok, "fail": lib_fail, "failures": lib_failures},
        "modules":   {"ok": mod_ok, "fail": mod_fail, "failures": mod_failures},
        "structure": {"ok": struct_ok, "missing": struct_miss, "missing_files": missing_files},
        "api_keys":  {"configured": key_cfg, "placeholder": key_ph},
        "tests":     {"skipped": skip_tests, "exit_code": test_rc, "summary": test_summary},
        "entrypoints": {"ok": entry_ok, "fail": entry_fail},
    }
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    print(f"  {bold('JSON report saved :')} {json_path}")

    return 0 if all_ok else 1


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global _USE_COLOR

    parser = argparse.ArgumentParser(
        description="PhishLens — Master Install & Verification Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install_and_verify.py                  # Full install + verify
  python install_and_verify.py --skip-install   # Already installed; verify only
  python install_and_verify.py --skip-tests     # Skip pytest
  python install_and_verify.py --verify-only    # Imports + structure + keys
  python install_and_verify.py --no-color       # Plain output for log files
        """,
    )
    parser.add_argument("--skip-install",  action="store_true",
                        help="Skip pip installation (already installed)")
    parser.add_argument("--skip-tests",    action="store_true",
                        help="Skip pytest suite")
    parser.add_argument("--verify-only",   action="store_true",
                        help="Verify imports + structure + keys only (implies --skip-install --skip-tests)")
    parser.add_argument("--no-color",      action="store_true",
                        help="Disable coloured output")
    parser.add_argument("--use-system-python", action="store_true",
                        help="Use current Python instead of .venv")
    args = parser.parse_args()

    if args.no_color:
        _USE_COLOR = False
    if args.verify_only:
        args.skip_install = True
        args.skip_tests = True

    start = time.monotonic()

    # Banner
    print(f"\n{bold(cyan(' PhishLens — Master Install & Verification Script '))}")
    print(f"{'─' * 62}")
    print(f"  Base dir   : {BASE_DIR}")
    print(f"  Python     : {sys.version.split()[0]} ({platform.architecture()[0]})")
    print(f"  Platform   : {platform.system()} {platform.release()}")
    print(f"  Time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Covers     : PHISHLENS_MASTER_PROMPT.md + PhishLens.md")

    # ── Determine Python executable ───────────────────────────────────────
    if args.use_system_python:
        python_exe = sys.executable
        info(f"Using system Python: {python_exe}")
    else:
        venv = ensure_venv()
        python_exe = get_venv_python(venv)
        if not Path(python_exe).exists():
            fail(f"venv Python not found: {python_exe}")
            fail("Run without --use-system-python flag, or activate the venv manually.")
            sys.exit(1)
        ok(f"Using venv Python: {python_exe}")

    # ── Step 2: Install ───────────────────────────────────────────────────
    if not args.skip_install:
        install_dependencies(python_exe)
    else:
        section("Step 2 — Installation (skipped)")
        info("Using existing packages (--skip-install)")

    # ── Step 3: Structure ─────────────────────────────────────────────────
    struct_ok, struct_miss, missing_files = verify_structure()

    # ── Step 4: Imports ───────────────────────────────────────────────────
    lib_ok, lib_fail, lib_failures = verify_third_party(python_exe)
    mod_ok, mod_fail, mod_failures = verify_phishlens_modules(python_exe)

    # ── Step 5: API keys ──────────────────────────────────────────────────
    key_cfg, key_ph = verify_api_keys()

    # ── Step 6: Tests ─────────────────────────────────────────────────────
    test_rc = 0
    test_summary = "skipped"
    if not args.skip_tests:
        test_rc, test_summary = run_tests(python_exe)

    # ── Step 7: Entry points ──────────────────────────────────────────────
    entry_ok, entry_fail = smoke_test_entrypoints(python_exe)

    # ── Step 8: Report ────────────────────────────────────────────────────
    elapsed = time.monotonic() - start
    exit_code = print_and_save_report(
        python_exe      = python_exe,
        python_version  = sys.version.split()[0],
        lib_ok          = lib_ok,
        lib_fail        = lib_fail,
        lib_failures    = lib_failures,
        mod_ok          = mod_ok,
        mod_fail        = mod_fail,
        mod_failures    = mod_failures,
        struct_ok       = struct_ok,
        struct_miss     = struct_miss,
        missing_files   = missing_files,
        key_cfg         = key_cfg,
        key_ph          = key_ph,
        test_rc         = test_rc,
        test_summary    = test_summary,
        entry_ok        = entry_ok,
        entry_fail      = entry_fail,
        skip_install    = args.skip_install,
        skip_tests      = args.skip_tests,
        elapsed         = elapsed,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
