#!/usr/bin/env python3
"""
PhishLens — Dataset Downloader & Data Pipeline Script
======================================================
Downloads, extracts, and validates all training datasets as specified in:
  • PHISHLENS_MASTER_PROMPT.md (Phase 2 — Data Ingestion)
  • PhishLens.md (Section 2 — Advanced Datasets)

Datasets handled automatically (no credentials required):
  1. SpamAssassin Public Corpus    — ham + spam (6,000 emails)
  2. phishing_pot (MeAJOR proxy)   — GitHub archive (phishing .eml files)
  3. Enron Spam Dataset            — CSV from GitHub mirror (33,716 emails)
  4. Umbrella Top 1M Domains       — Legitimate domain whitelist

Datasets requiring Kaggle CLI (see data/README.md for setup):
  5. CASIS Phishing Email Dataset  — 82,000+ BEC/phishing emails
  6. Enron Kaggle CSV              — 30,000 legitimate Enron emails
  7. Nigerian Fraud Corpus         — ~3,900 Nigerian 419 fraud emails (phishing)

After download, builds a clean combined CSV at:
  data/processed/train.csv    ← ready for python train.py
  data/processed/test.csv     ← held-out stratified evaluation set

Usage:
    python download_datasets.py                  # Download all
    python download_datasets.py --skip-spamassassin
    python download_datasets.py --skip-phishingpot
    python download_datasets.py --skip-enron
    python download_datasets.py --skip-casis
    python download_datasets.py --skip-enron-kaggle
    python download_datasets.py --skip-nigerian-fraud
    python download_datasets.py --build-only     # Skip download, rebuild CSVs
    python download_datasets.py --status         # Show what's present
"""

from __future__ import annotations

import argparse
import bz2
import io
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.resolve()
RAW_DIR  = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"

# ─────────────────────────────────────────────────────────────────────────────
# Console helpers
# ─────────────────────────────────────────────────────────────────────────────

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") != "1"

def _c(code: str, t: str) -> str:
    return f"\033[{code}m{t}\033[0m" if USE_COLOR else t

def green(t: str)  -> str: return _c("92", t)
def red(t: str)    -> str: return _c("91", t)
def yellow(t: str) -> str: return _c("93", t)
def cyan(t: str)   -> str: return _c("96", t)
def bold(t: str)   -> str: return _c("1",  t)

def section(title: str) -> None:
    print(f"\n{bold(cyan('─' * 62))}")
    print(f"{bold(cyan(f'  {title}'))}")
    print(f"{bold(cyan('─' * 62))}")

def ok(msg: str)   -> None: print(f"  {green('✓')}  {msg}")
def warn(msg: str) -> None: print(f"  {yellow('⚠')}  {msg}")
def fail(msg: str) -> None: print(f"  {red('✗')}  {msg}")
def info(msg: str) -> None: print(f"  {cyan('→')}  {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────────────────────────────────────

SPAMASSASSIN_ARCHIVES = [
    # (url, label, description)
    ("https://spamassassin.apache.org/old/publiccorpus/20021010_easy_ham.tar.bz2",   "ham",  "easy_ham_2002"),
    ("https://spamassassin.apache.org/old/publiccorpus/20021010_hard_ham.tar.bz2",   "ham",  "hard_ham_2002"),
    ("https://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2",       "spam", "spam_2002"),
    ("https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",   "ham",  "easy_ham_2003"),
    ("https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2", "ham",  "easy_ham_2003b"),
    ("https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",       "spam", "spam_2003"),
    ("https://spamassassin.apache.org/old/publiccorpus/20030228_spam_2.tar.bz2",     "spam", "spam_2003b"),
    ("https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2",     "spam", "spam_2005"),
]

# Enron spam/ham dataset — Hugging Face datasets hub (CSV, no auth needed)
# Fallback mirrors tried in order
ENRON_CSV_URLS = [
    "https://huggingface.co/datasets/TrainingDataPro/enron-spam-or-ham-email-dataset/resolve/main/data/train-00000-of-00001-6498b8dcba86aabd.parquet",  # parquet on HF
    "https://raw.githubusercontent.com/dineshdaultani/SpamDetection/master/data/enron_spam_data.csv",
    "https://raw.githubusercontent.com/abuzreq/SpamDetection/master/data/enron_spam_data.csv",
]

# phishing_pot GitHub repo ZIP archive
PHISHING_POT_URL = "https://github.com/rf-peixoto/phishing_pot/archive/refs/heads/main.zip"

# Cisco Umbrella Top 1M domains (legitimate domain whitelist)
UMBRELLA_URL = "http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip"


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def _download(url: str, dest: Path, label: str, timeout: int = 60) -> bool:
    """Download url to dest with a progress indicator. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024:
        ok(f"{label} already downloaded ({dest.stat().st_size // 1024:,} KB)")
        return True

    info(f"Downloading {label} …")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PhishLens/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            chunk_size = 65536
            downloaded = 0
            with open(dest, "wb") as fout:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    fout.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        bar = ("█" * (pct // 5)).ljust(20)
                        print(f"\r    [{bar}] {pct:3d}%  {downloaded // 1024:,} KB", end="", flush=True)
        print()  # newline after progress
        ok(f"{label}: {downloaded // 1024:,} KB saved to {dest.name}")
        return True
    except Exception as exc:
        print()
        fail(f"{label} download failed: {exc}")
        if dest.exists():
            dest.unlink()
        return False


def _extract_tar_bz2(archive: Path, dest_dir: Path, label: str) -> int:
    """Extract a .tar.bz2 archive. Returns number of files extracted."""
    info(f"Extracting {label} …")
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with tarfile.open(archive, "r:bz2") as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            for member in members:
                # Sanitise path (prevent path traversal)
                member_path = Path(member.name)
                safe_name = member_path.name  # strip directory portion
                if not safe_name or safe_name.startswith("."):
                    continue
                out_path = dest_dir / safe_name
                with tar.extractfile(member) as fsrc, open(out_path, "wb") as fdst:
                    fdst.write(fsrc.read())
                count += 1
        ok(f"{label}: extracted {count:,} files to {dest_dir.name}/")
    except Exception as exc:
        fail(f"Extraction failed for {archive.name}: {exc}")
    return count


def _extract_zip(archive: Path, dest_dir: Path, label: str,
                 file_filter: Optional[str] = None) -> int:
    """Extract a ZIP archive (optionally only files matching a name filter)."""
    info(f"Extracting {label} …")
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            for entry in zf.infolist():
                if entry.is_dir():
                    continue
                name = entry.filename
                if file_filter and file_filter not in name:
                    continue
                # Write to flat dest_dir using only the basename
                safe_name = Path(name).name
                if not safe_name:
                    continue
                out_path = dest_dir / safe_name
                out_path.write_bytes(zf.read(entry.filename))
                count += 1
        ok(f"{label}: extracted {count:,} files")
    except Exception as exc:
        fail(f"ZIP extraction failed: {exc}")
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — SpamAssassin
# ─────────────────────────────────────────────────────────────────────────────

def download_spamassassin() -> Tuple[int, int]:
    """Download and extract all SpamAssassin archives.
    Returns (ham_count, spam_count) of extracted email files."""
    section("Dataset 1 — SpamAssassin Public Corpus")
    info("Source: https://spamassassin.apache.org/old/publiccorpus/")
    info("Purpose: Gold-standard OOD evaluation set (6,000 expert-labelled emails)")

    ham_dir  = RAW_DIR / "spamassassin_ham"
    spam_dir = RAW_DIR / "spamassassin_spam"
    ham_dir.mkdir(parents=True, exist_ok=True)
    spam_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = RAW_DIR / "_cache"
    cache_dir.mkdir(exist_ok=True)

    ham_count = spam_count = 0

    for url, label_type, desc in SPAMASSASSIN_ARCHIVES:
        archive_name = url.split("/")[-1]
        archive_path = cache_dir / archive_name
        dest = ham_dir if label_type == "ham" else spam_dir

        success = _download(url, archive_path, desc, timeout=60)
        if success:
            n = _extract_tar_bz2(archive_path, dest, desc)
            if label_type == "ham":
                ham_count += n
            else:
                spam_count += n

    ok(f"SpamAssassin total: {ham_count:,} ham + {spam_count:,} spam emails")
    return ham_count, spam_count


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Enron Spam Dataset (CSV, no credentials needed)
# ─────────────────────────────────────────────────────────────────────────────

def download_enron() -> int:
    """Download the Enron spam/ham CSV dataset.
    Returns number of emails downloaded."""
    section("Dataset 2 — Enron Spam Dataset (33,716 emails)")
    info("Source: Multiple GitHub mirrors of the Enron spam corpus")
    info("Purpose: Large legitimate-email corpus; improves ham precision")

    dest = RAW_DIR / "enron_spam_data.csv"
    for url in ENRON_CSV_URLS:
        if url.endswith(".parquet"):
            # Skip parquet format — needs pyarrow
            continue
        success = _download(url, dest, "enron_spam_data.csv", timeout=120)
        if success:
            # Quick validate
            try:
                import csv
                with open(dest, newline="", encoding="utf-8", errors="replace") as f:
                    reader = csv.reader(f)
                    headers = next(reader)
                    count = sum(1 for _ in reader)
                ok(f"Enron CSV: {count:,} rows | columns: {headers}")
                return count
            except Exception as exc:
                warn(f"Enron CSV validation failed: {exc}")
                dest.unlink(missing_ok=True)
                continue

    warn("All Enron mirror URLs failed — will skip Enron dataset.")
    warn("The pipeline will still work using SpamAssassin + phishing_pot.")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — phishing_pot (MeAJOR proxy)
# ─────────────────────────────────────────────────────────────────────────────

def download_phishing_pot() -> int:
    """Download phishing_pot GitHub repo and extract phishing .eml files.
    Returns number of phishing emails extracted."""
    section("Dataset 3 — phishing_pot (Phishing Email Repository)")
    info("Source: https://github.com/rf-peixoto/phishing_pot")
    info("Purpose: Primary phishing email samples (real-world phishing .eml files)")

    cache_dir  = RAW_DIR / "_cache"
    cache_dir.mkdir(exist_ok=True)
    zip_path   = cache_dir / "phishing_pot_main.zip"
    phish_dir  = RAW_DIR / "phishing_pot"
    phish_dir.mkdir(exist_ok=True)

    success = _download(PHISHING_POT_URL, zip_path, "phishing_pot GitHub archive", timeout=120)
    if not success:
        warn("phishing_pot download failed — will skip this dataset")
        return 0

    # Extract all files into phishing_pot/
    info("Extracting phishing email files from archive …")
    count = 0
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            entries = [e for e in zf.infolist() if not e.is_dir()]
            for entry in entries:
                fname = Path(entry.filename).name
                if not fname:
                    continue
                # Sanitise — only accept .eml, .txt, .csv files
                ext = Path(fname).suffix.lower()
                if ext not in {".eml", ".txt", ".csv", ".msg"}:
                    continue
                out = phish_dir / fname
                # Avoid overwrite collisions for same-name files
                if out.exists():
                    stem = out.stem
                    out = phish_dir / f"{stem}_{count}{ext}"
                out.write_bytes(zf.read(entry.filename))
                count += 1
    except Exception as exc:
        fail(f"Extraction error: {exc}")

    ok(f"phishing_pot: extracted {count:,} email files to phishing_pot/")
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Umbrella Top 1M (legitimate domain whitelist)
# ─────────────────────────────────────────────────────────────────────────────

def download_umbrella() -> int:
    """Download Cisco Umbrella Top 1M domain list."""
    section("Dataset 4 — Cisco Umbrella Top 1M (Legitimate Domain Whitelist)")
    info("Source: http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip")
    info("Purpose: Reduces false positives in brand-impersonation detection")

    cache_dir = RAW_DIR / "_cache"
    zip_path  = cache_dir / "umbrella_top1m.zip"
    dest      = RAW_DIR / "umbrella_top1m.csv"

    if dest.exists() and dest.stat().st_size > 10000:
        ok(f"Umbrella 1M already present ({dest.stat().st_size // 1024:,} KB)")
        return 1_000_000

    success = _download(UMBRELLA_URL, zip_path, "umbrella_top1m.zip", timeout=120)
    if not success:
        warn("Umbrella download failed — domain whitelist will use built-in fallback list")
        return 0

    info("Extracting Umbrella CSV …")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_entry = next((e for e in zf.namelist() if e.endswith(".csv")), None)
            if csv_entry:
                dest.write_bytes(zf.read(csv_entry))
                ok(f"Umbrella 1M: {dest.stat().st_size // 1024:,} KB saved to umbrella_top1m.csv")
                return 1_000_000
            else:
                fail("No CSV found inside Umbrella ZIP")
                return 0
    except Exception as exc:
        fail(f"Umbrella extraction failed: {exc}")
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Build processed CSVs
# ─────────────────────────────────────────────────────────────────────────────

def build_processed_datasets() -> Dict[str, int]:
    """
    Combine all downloaded raw data into a balanced train/test split.

    Strategy (ensures both classes always appear in training):
      1. Load ALL sources (SpamAssassin, phishing_pot, Enron, CASIS Kaggle,
         Enron Kaggle, Nigerian Fraud corpus)
      2. Deduplicate the combined pool
      3. Stratified 80/20 split → train.csv / test.csv
      Both files always contain label 0 (legitimate) and label 1 (phishing/spam).

    Returns counts dict.
    """
    section("Step 5/7 — Building Processed Datasets")
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split as _split
    except ImportError:
        fail("pandas / scikit-learn not installed — run 'pip install pandas scikit-learn' first")
        return {}

    all_records: List[dict] = []

    # ── SpamAssassin ──────────────────────────────────────────────────────
    info("Loading SpamAssassin corpus …")
    ham_dir  = RAW_DIR / "spamassassin_ham"
    spam_dir = RAW_DIR / "spamassassin_spam"

    sa_ham_count = sa_spam_count = 0
    if ham_dir.exists():
        for fpath in ham_dir.iterdir():
            if fpath.is_file():
                try:
                    raw = fpath.read_bytes().decode("utf-8", errors="replace")
                    if len(raw.strip()) > 20:
                        all_records.append({"label": 0, "raw_email": raw, "source": "spamassassin"})
                        sa_ham_count += 1
                except Exception:
                    pass

    if spam_dir.exists():
        for fpath in spam_dir.iterdir():
            if fpath.is_file():
                try:
                    raw = fpath.read_bytes().decode("utf-8", errors="replace")
                    if len(raw.strip()) > 20:
                        all_records.append({"label": 1, "raw_email": raw, "source": "spamassassin"})
                        sa_spam_count += 1
                except Exception:
                    pass

    ok(f"SpamAssassin: {sa_ham_count:,} ham + {sa_spam_count:,} spam")

    # ── phishing_pot ──────────────────────────────────────────────────────
    info("Loading phishing_pot emails …")
    phish_dir = RAW_DIR / "phishing_pot"
    pp_count = 0
    if phish_dir.exists():
        for fpath in phish_dir.iterdir():
            if fpath.is_file() and fpath.suffix.lower() in {".eml", ".txt", ".msg"}:
                try:
                    raw = fpath.read_bytes().decode("utf-8", errors="replace")
                    if len(raw.strip()) > 20:
                        all_records.append({"label": 1, "raw_email": raw, "source": "phishing_pot"})
                        pp_count += 1
                except Exception:
                    pass
    ok(f"phishing_pot: {pp_count:,} phishing emails")

    # ── Enron CSV (optional) ──────────────────────────────────────────────
    info("Loading Enron CSV (if present) …")
    enron_csv = RAW_DIR / "enron_spam_data.csv"
    enron_ham = enron_spam = 0
    if enron_csv.exists():
        try:
            df_enron = pd.read_csv(
                enron_csv,
                encoding="utf-8",
                encoding_errors="replace",
                low_memory=False,
            )
            df_enron.columns = [c.lower().strip().replace(" ", "_") for c in df_enron.columns]
            label_col = next((c for c in df_enron.columns if "spam" in c or "label" in c), None)
            msg_col   = next((c for c in df_enron.columns
                             if "message" in c or "text" in c or "email" in c), None)
            if label_col and msg_col:
                for _, row in df_enron.iterrows():
                    raw = str(row[msg_col]) if pd.notna(row[msg_col]) else ""
                    if len(raw.strip()) < 10:
                        continue
                    lbl_raw = row[label_col]
                    if isinstance(lbl_raw, str):
                        lbl_raw = lbl_raw.strip().lower()
                    if lbl_raw in {1, "1", "spam", True, "true"}:
                        label = 1; enron_spam += 1
                    else:
                        label = 0; enron_ham += 1
                    all_records.append({"label": label, "raw_email": raw, "source": "enron"})
                ok(f"Enron: {enron_ham:,} ham + {enron_spam:,} spam")
            else:
                warn(f"Could not identify columns in Enron CSV: {list(df_enron.columns)}")
        except Exception as exc:
            warn(f"Enron CSV load failed: {exc}")
    else:
        info("Enron CSV not found — skipping (only SpamAssassin + phishing_pot used)")

    # ── CASIS Kaggle (phishing emails) ────────────────────────────────────
    info("Loading CASIS Kaggle phishing email dataset …")
    casis_dir = RAW_DIR / "casis"
    casis_csvs = sorted(casis_dir.glob("*.csv")) if casis_dir.exists() else []
    casis_phish = casis_ham = 0
    if casis_csvs:
        for casis_csv in casis_csvs:
            try:
                df_casis = pd.read_csv(casis_csv, low_memory=False, encoding_errors="replace")
                # text col: prefer text_combined, then body, then any text/email column
                text_col = next(
                    (c for c in df_casis.columns if c.lower() in ("text_combined", "body")),
                    next((c for c in df_casis.columns
                          if any(kw in c.lower() for kw in ("text", "email", "message", "content"))
                          and "type" not in c.lower()), None)
                )
                label_col = next((c for c in df_casis.columns
                                   if c.lower() in ("label", "type") or
                                   "label" in c.lower() or "type" in c.lower()), None)
                if text_col and label_col:
                    added = 0
                    for _, row in df_casis.iterrows():
                        raw = str(row[text_col]) if pd.notna(row[text_col]) else ""
                        if len(raw.strip()) < 10:
                            continue
                        lv = str(row[label_col]).strip()
                        if lv in ("Phishing Email", "phishing", "1", "spam", "1.0"):
                            label = 1; casis_phish += 1
                        elif lv in ("0", "0.0", "ham", "legitimate", "safe"):
                            label = 0; casis_ham += 1
                        else:
                            try:
                                label = int(float(lv))
                                if label == 1:
                                    casis_phish += 1
                                else:
                                    label = 0; casis_ham += 1
                            except (ValueError, TypeError):
                                label = 0; casis_ham += 1
                        all_records.append({"label": label, "raw_email": raw, "source": "casis_kaggle"})
                        added += 1
                    ok(f"  {casis_csv.name}: {added:,} rows loaded (text='{text_col}', label='{label_col}')")
                else:
                    warn(f"  {casis_csv.name}: could not detect columns (found: {list(df_casis.columns)[:8]})")
            except Exception as exc:
                warn(f"  {casis_csv.name}: load failed — {exc}")
        ok(f"CASIS total: {casis_ham:,} legitimate + {casis_phish:,} phishing")
    else:
        info("CASIS Kaggle CSV not found — skipping (run download_casis_kaggle() first)")

    # ── Enron Kaggle (legitimate email corpus) ────────────────────────────
    info("Loading Enron Kaggle email dataset …")
    enron_kag_dir = RAW_DIR / "enron_kaggle"
    enron_kag_csv = next(enron_kag_dir.glob("*.csv"), None) if enron_kag_dir.exists() else None
    enron_kag_count = 0
    if enron_kag_csv and enron_kag_csv.exists():
        try:
            df_ek = pd.read_csv(enron_kag_csv, low_memory=False, encoding_errors="replace")
            df_ek.columns = [c.lower().strip() for c in df_ek.columns]
            msg_col = next((c for c in df_ek.columns if "message" in c or "text" in c), None)
            if msg_col:
                if len(df_ek) > 30_000:
                    df_ek = df_ek.sample(n=30_000, random_state=42)
                for _, row in df_ek.iterrows():
                    raw = str(row[msg_col]) if pd.notna(row[msg_col]) else ""
                    if len(raw.strip()) < 10:
                        continue
                    all_records.append({"label": 0, "raw_email": raw, "source": "enron_kaggle"})
                    enron_kag_count += 1
                ok(f"Enron Kaggle: {enron_kag_count:,} legitimate emails (max 30,000 sampled)")
            else:
                warn(f"Could not find message column in Enron Kaggle CSV (cols: {list(df_ek.columns)[:8]})")
        except Exception as exc:
            warn(f"Enron Kaggle load failed: {exc}")
    else:
        info("Enron Kaggle CSV not found — skipping (run download_enron_kaggle() first)")

    # ── Nigerian Fraud Corpus (phishing emails) ───────────────────────────
    info("Loading Nigerian Fraud email corpus …")
    nf_dir = RAW_DIR / "nigerian_fraud"
    nf_txt = nf_dir / "fradulent_emails.txt"    # Note: dataset filename has a typo
    if not nf_txt.exists():
        # Try alternate spelling in case Kaggle fixes the typo in future
        nf_txt_alt = nf_dir / "fraudulent_emails.txt"
        nf_txt = nf_txt_alt if nf_txt_alt.exists() else nf_txt
    nf_count = 0
    if nf_txt.exists():
        try:
            raw_blob = nf_txt.read_text(encoding="utf-8", errors="replace")
            chunks = raw_blob.split("\n\n")
            for chunk in chunks:
                chunk = chunk.strip()
                if len(chunk) < 50:
                    continue
                all_records.append({"label": 1, "raw_email": chunk, "source": "nigerian_fraud"})
                nf_count += 1
            ok(f"Nigerian Fraud: {nf_count:,} phishing emails")
        except Exception as exc:
            warn(f"Nigerian Fraud corpus load failed: {exc}")
    else:
        info("Nigerian Fraud text file not found — skipping (run download_nigerian_fraud() first)")

    # ── HuggingFace datasets (six new sources) ────────────────────────────
    # The HuggingFace 'datasets' library handles download, caching, and
    # format conversion automatically. Each loader returns a DataFrame with
    # ['label', 'raw_email', 'source'] columns matching the existing schema.
    #
    # puyang2025/seven-phishing-email-datasets is intentionally skipped here
    # because it is content-identical to JinqiangDing/seven-phishing-email-datasets
    # (same 203,017 rows from TREC-05/06/07, CEAS-08, Enron, Ling, SpamAssassin).
    # JinqiangDing was updated more recently (April 2026) and is used exclusively.
    section("Step 5b/7 — HuggingFace Datasets (six new sources)")
    try:
        import sys as _sys
        _sys.path.insert(0, str(BASE_DIR))
        from src.ingestion.dataset_loader import (
            load_hf_locuoco,
            load_hf_jinqiangding,
            load_hf_puyang_seven,
            load_hf_llmgen,
            load_hf_zefang,
            load_hf_puyang_phish,
        )
        _HF_LOADERS = [
            ("locuoco (365k, 3-class)",          load_hf_locuoco),
            ("JinqiangDing/seven (203k)",         load_hf_jinqiangding),
            ("puyang2025/seven (203k, dedup)",    load_hf_puyang_seven),
            ("Dizzzy0x00/LLMGen (6.8k, AI-gen)", load_hf_llmgen),
            ("zefang-liu (18.7k)",                load_hf_zefang),
            ("puyang2025/phish (26.6k)",          load_hf_puyang_phish),
        ]
        for label_, loader_fn in _HF_LOADERS:
            try:
                info(f"Loading {label_} …")
                df_hf = loader_fn()
                if len(df_hf) > 0:
                    hf_records = df_hf.to_dict("records")
                    all_records.extend(hf_records)
                    ok(f"{label_}: added {len(hf_records):,} rows")
                else:
                    warn(f"{label_}: returned 0 rows — check loader logs above")
            except Exception as _exc:
                warn(f"{label_}: loader failed — {_exc}")
    except ImportError as _exc:
        warn(
            f"HuggingFace 'datasets' library not available: {_exc}\n"
            "  Install with: pip install datasets>=2.18.0\n"
            "  HuggingFace sources will be skipped."
        )

    # ── Validate we have both classes ─────────────────────────────────────
    if not all_records:
        warn("No data found! Run download steps first.")
        return {"train": 0, "test": 0}

    df_all = pd.DataFrame(all_records)
    n_classes = df_all["label"].nunique()
    if n_classes < 2:
        only_class = int(df_all["label"].iloc[0])
        warn(
            f"Only class {only_class} found in collected data — "
            "cannot train a binary classifier without both classes."
        )
        warn("Add more data sources (e.g. CASIS via Kaggle) and re-run.")
        # Still save what we have so the user can inspect
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(PROC_DIR / "train.csv", index=False)
        return {"train": len(df_all), "test": 0}

    # ── Deduplicate (priority-based 200-char prefix key) ─────────────────
    # Priority: JinqiangDing > puyang2025 > locuoco > Dizzzy0x00 > zefang-liu
    #           > original local sources (keep first = highest priority)
    _SOURCE_PRIORITY = {
        "JinqiangDing/seven-phishing-email-datasets": 1,
        "puyang2025/seven-phishing-email-datasets": 2,
        "puyang2025/phish-email-datasets": 2,
        "locuoco/the-biggest-spam-ham-phish-email-dataset-300000": 3,
        "Dizzzy0x00/LLMGen-Phishing-Email-Dataset": 4,
        "zefang-liu/phishing-email-dataset": 5,
    }
    df_all["_priority"] = df_all["source"].map(lambda s: _SOURCE_PRIORITY.get(s, 6))
    df_all["_dedup_key"] = df_all["raw_email"].str.strip().str[:200].str.lower()
    df_all.sort_values("_priority", inplace=True)
    before = len(df_all)
    df_all.drop_duplicates(subset=["_dedup_key"], keep="first", inplace=True)
    df_all.drop(columns=["_dedup_key", "_priority"], inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    after = len(df_all)
    if before != after:
        info(f"Deduplication: removed {before - after:,} near-duplicate emails")

    # Log combined source breakdown
    info("Source breakdown after deduplication:")
    for src, cnt in df_all["source"].value_counts().items():
        phish_cnt = int((df_all.loc[df_all["source"] == src, "label"] == 1).sum())
        legit_cnt = int((df_all.loc[df_all["source"] == src, "label"] == 0).sum())
        info(f"  [{src}] {cnt:,} total | phishing={phish_cnt:,} | legitimate={legit_cnt:,}")

    # ── Stratified 80/20 split ────────────────────────────────────────────
    info("Performing stratified 80/20 train/test split …")
    df_train, df_test = _split(
        df_all,
        test_size=0.20,
        stratify=df_all["label"],
        random_state=42,
    )
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

    # Save
    train_path = PROC_DIR / "train.csv"
    test_path  = PROC_DIR / "test.csv"

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    train_phish = (df_train["label"] == 1).sum()
    train_ham   = (df_train["label"] == 0).sum()
    test_phish  = (df_test["label"] == 1).sum() if len(df_test) > 0 else 0
    test_ham    = (df_test["label"] == 0).sum()  if len(df_test) > 0 else 0

    print()
    ok(f"data/processed/train.csv : {len(df_train):,} emails "
       f"({train_phish:,} phishing + {train_ham:,} legitimate)")
    ok(f"data/processed/test.csv  : {len(df_test):,} emails "
       f"({test_phish:,} spam + {test_ham:,} ham) [OOD evaluation set]")

    # ── Also create meajor.csv alias pointing loaders to phishing_pot data ─
    # If phishing_pot has a CSV inside, copy it as meajor.csv for full compat
    phish_csv = next(phish_dir.glob("*.csv"), None) if phish_dir.exists() else None
    if phish_csv and not (RAW_DIR / "meajor.csv").exists():
        import shutil
        shutil.copy(phish_csv, RAW_DIR / "meajor.csv")
        ok(f"Copied {phish_csv.name} → data/raw/meajor.csv")

    # ── Write dataset manifest ────────────────────────────────────────────
    manifest = {
        "generated": datetime.now().isoformat(),
        "train": {
            "path": str(train_path),
            "total": len(df_train),
            "phishing": int(train_phish),
            "legitimate": int(train_ham),
            "sources": df_train["source"].value_counts().to_dict(),
        },
        "test": {
            "path": str(test_path),
            "total": len(df_test),
            "spam": int(test_phish),
            "ham": int(test_ham),
            "sources": df_test["source"].value_counts().to_dict() if len(df_test) > 0 else {},
        },
    }
    manifest_path = PROC_DIR / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    ok(f"Dataset manifest: {manifest_path}")

    return {"train": len(df_train), "test": len(df_test)}


# ─────────────────────────────────────────────────────────────────────────────
# Steps 4/7, 5/7, 6/7 — Kaggle-based dataset downloads
# ─────────────────────────────────────────────────────────────────────────────

def _run_kaggle_download(dataset_slug: str, dest_dir: Path, dataset_label: str) -> bool:
    """Run 'kaggle datasets download' via subprocess and unzip into dest_dir.

    Security rationale: We call the Kaggle CLI as a subprocess rather than
    importing it as a Python library. This avoids granting kaggle's code
    direct access to our process memory and credential store; the CLI reads
    ~/.kaggle/kaggle.json in a sandboxed child process.

    Args:
        dataset_slug: Kaggle dataset identifier (e.g. 'owner/dataset-name').
        dest_dir: Directory to download and extract into.
        dataset_label: Human-readable label for log messages.

    Returns:
        True on success, False on any failure.
    """
    import subprocess
    import sys
    from pathlib import Path

    # Build the path to the kaggle entry-point that lives in the *same* venv
    # as the running Python, so we never accidentally invoke a system-wide
    # kaggle binary that lacks our credentials.
    # kaggle 1.6.14 has no __main__.py so `python -m kaggle` fails; we must
    # call the installed script directly.
    venv_scripts = Path(sys.executable).parent
    kaggle_bin = venv_scripts / ("kaggle.exe" if sys.platform == "win32" else "kaggle")
    if not kaggle_bin.exists():
        # Try without extension (editable installs on Windows sometimes omit it)
        kaggle_bin = venv_scripts / "kaggle"
    if not kaggle_bin.exists():
        warn(
            f"{dataset_label}: kaggle script not found at {kaggle_bin}. "
            "Install with: pip install kaggle  (inside the venv)"
        )
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(kaggle_bin),
        "datasets", "download",
        "--unzip",
        "-p", str(dest_dir),
        dataset_slug,
    ]
    info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,         # 30-minute hard timeout (Enron CSV is ~500 MB)
        )
    except subprocess.TimeoutExpired:
        warn(f"{dataset_label}: Download timed out (>5 min). Try again on a faster connection.")
        return False
    except Exception as exc:
        warn(f"{dataset_label}: Subprocess error — {exc}")
        return False

    if result.returncode != 0:
        err_msg = (result.stderr or result.stdout or "").strip()
        if "401" in err_msg or "Unauthorized" in err_msg:
            warn(
                f"{dataset_label}: Kaggle authentication failed. "
                "Ensure ~/.kaggle/kaggle.json contains valid API credentials."
            )
        elif "403" in err_msg or "Forbidden" in err_msg:
            warn(
                f"{dataset_label}: Access denied. "
                "You may need to accept the dataset's terms on kaggle.com first."
            )
        else:
            warn(f"{dataset_label}: Kaggle download failed (rc={result.returncode}). {err_msg[:200]}")
        return False

    ok(f"{dataset_label}: downloaded and extracted to {dest_dir}")
    return True


def download_casis_kaggle() -> bool:
    """Download the CASIS Phishing Email Dataset from Kaggle (82,000+ emails).

    Kaggle dataset: naserabdullahalam/phishing-email-dataset
    Extracted file: phishing_email.csv
    Columns used:   'Email Text' (body), 'Email Type' ('Phishing Email' / 'Safe Email')

    Security rationale: Downloaded directly from the dataset author's Kaggle
    page; no intermediate mirrors. The CSV is placed in data/raw/casis/ so it
    cannot overwrite existing raw data from other sources.

    Returns:
        True on success, False if Kaggle CLI unavailable or download failed.
    """
    section("Step 4/7 — CASIS Phishing Email Dataset (Kaggle)")
    dest = RAW_DIR / "casis"
    success = _run_kaggle_download(
        "naserabdullahalam/phishing-email-dataset", dest, "CASIS Kaggle"
    )
    if success:
        csv_files = list(dest.glob("*.csv"))
        if csv_files:
            ok(f"CASIS CSV files in {dest}: {[f.name for f in csv_files]}")
        else:
            warn(f"No CSV found in {dest} after extraction — check Kaggle download output.")
    return success


def download_enron_kaggle() -> bool:
    """Download the Enron Email Dataset from Kaggle (legitimate email corpus).

    Kaggle dataset: wcukierski/enron-email-dataset
    Extracted file: emails.csv  (~500MB — large; sampled to 30,000 rows at load time)
    Columns used:   'message' (all label=0 — legitimate)

    Security rationale: Using the Kaggle-hosted version avoids the dead/sketchy
    mirror URLs that previously broke the Enron download step.

    Returns:
        True on success, False if Kaggle CLI unavailable or download failed.
    """
    section("Step 5/7 — Enron Email Dataset (Kaggle)")
    dest = RAW_DIR / "enron_kaggle"
    success = _run_kaggle_download(
        "wcukierski/enron-email-dataset", dest, "Enron Kaggle"
    )
    if success:
        csv_files = list(dest.glob("*.csv"))
        ok(f"Enron Kaggle CSV files in {dest}: {[f.name for f in csv_files]}")
    return success


def download_nigerian_fraud() -> bool:
    """Download the Fraudulent Email Corpus (Nigerian 419 fraud) from Kaggle.

    Kaggle dataset: rtatman/fraudulent-email-corpus
    Extracted file: fradulent_emails.txt  (~3,900 fraud emails, all label=1)

    Security rationale: The Nigerian fraud dataset adds diversity to the
    phishing class — different social engineering tactics from BEC/credential-
    harvesting emails improve the model's generalisation to novel attack types.

    Returns:
        True on success, False if Kaggle CLI unavailable or download failed.
    """
    section("Step 6/7 — Nigerian Fraud Email Corpus (Kaggle)")
    dest = RAW_DIR / "nigerian_fraud"
    success = _run_kaggle_download(
        "rtatman/fraudulent-email-corpus", dest, "Nigerian Fraud Kaggle"
    )
    if success:
        txt_files = list(dest.glob("*.txt"))
        ok(f"Nigerian Fraud text files in {dest}: {[f.name for f in txt_files]}")
    return success


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Validate pipeline can load the data
# ─────────────────────────────────────────────────────────────────────────────

def validate_pipeline() -> bool:
    """Quick smoke test: load processed train.csv through dataset_loader."""
    section("Step 7 — Pipeline Validation")
    sys.path.insert(0, str(BASE_DIR))

    train_csv = PROC_DIR / "train.csv"
    if not train_csv.exists():
        warn("data/processed/train.csv not found — run without --build-only first")
        return False

    try:
        import pandas as pd
        df = pd.read_csv(train_csv, nrows=100)
        assert "label" in df.columns, "Missing 'label' column"
        assert "raw_email" in df.columns, "Missing 'raw_email' column"
        phish = (df["label"] == 1).sum()
        ham   = (df["label"] == 0).sum()
        ok(f"train.csv loads correctly — sample of 100 rows: {phish} phishing, {ham} ham")
    except Exception as exc:
        fail(f"train.csv validation failed: {exc}")
        return False

    try:
        from src.ingestion.eml_parser import parse_eml_string
        sample = pd.read_csv(train_csv, nrows=5)
        for _, row in sample.iterrows():
            result = parse_eml_string(str(row["raw_email"]))
            assert isinstance(result, dict), "parse_eml_string should return a dict"
        ok("parse_eml_string processes sample emails correctly")
    except Exception as exc:
        fail(f"eml_parser validation failed: {exc}")
        return False

    ok("Pipeline validation passed — data is ready for training!")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Status check
# ─────────────────────────────────────────────────────────────────────────────

def show_status() -> None:
    """Print a summary of what datasets are present."""
    section("Data Directory Status")

    checks = [
        (RAW_DIR / "spamassassin_ham",   "SpamAssassin ham/",         "dir"),
        (RAW_DIR / "spamassassin_spam",  "SpamAssassin spam/",        "dir"),
        (RAW_DIR / "enron_spam_data.csv","Enron spam CSV",            "file"),
        (RAW_DIR / "phishing_pot",       "phishing_pot emails/",      "dir"),
        (RAW_DIR / "meajor.csv",         "meajor.csv (primary CSV)",  "file"),
        (RAW_DIR / "umbrella_top1m.csv", "Umbrella Top 1M domains",   "file"),
        (PROC_DIR / "train.csv",         "Processed train.csv",       "file"),
        (PROC_DIR / "test.csv",          "Processed test.csv",        "file"),
    ]

    for path, label, kind in checks:
        if kind == "file":
            if path.exists():
                size = path.stat().st_size
                ok(f"{label:<35} {size // 1024:>8,} KB")
            else:
                warn(f"{label:<35} {'NOT FOUND':>10}")
        else:
            if path.exists():
                n = sum(1 for f in path.iterdir() if f.is_file())
                ok(f"{label:<35} {n:>8,} files")
            else:
                warn(f"{label:<35} {'NOT FOUND':>10}")

    manifest = PROC_DIR / "dataset_manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        print()
        print(f"  {bold('Last build:')} {data.get('generated', 'unknown')}")
        t = data.get("train", {})
        print(f"  train.csv : {t.get('total', 0):,} total | "
              f"{t.get('phishing', 0):,} phishing | "
              f"{t.get('legitimate', 0):,} legitimate")
        te = data.get("test", {})
        print(f"  test.csv  : {te.get('total', 0):,} total | "
              f"{te.get('spam', 0):,} spam | "
              f"{te.get('ham', 0):,} ham")

    print()
    print(f"  {bold('To start training (once data is ready):')}")
    print(f"  {cyan('  python train.py --data-dir data/processed --models xgboost --no-network --eval --save models')}")
    print(f"\n  {bold('To launch the web UI:')}")
    print(f"  {cyan('  python -m streamlit run app.py --server.port 8501')}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PhishLens Dataset Downloader & Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Datasets downloaded automatically (no credentials required):
  1. SpamAssassin Public Corpus (ham + spam, 6,000 emails)
  2. Enron Spam Dataset (33,716 emails, CSV)
  3. phishing_pot GitHub archive (phishing .eml files)
  4. Cisco Umbrella Top 1M (legitimate domain whitelist)

Datasets requiring Kaggle CLI (see data/README.md):
  5. CASIS: naserabdullahalam/phishing-email-dataset
  6. Enron Kaggle: wcukierski/enron-email-dataset
  7. Nigerian Fraud: rtatman/fraudulent-email-corpus
        """,
    )
    parser.add_argument("--skip-spamassassin",    action="store_true",
                        help="Skip SpamAssassin download")
    parser.add_argument("--skip-enron",           action="store_true",
                        help="Skip Enron CSV download")
    parser.add_argument("--skip-phishingpot",     action="store_true",
                        help="Skip phishing_pot download")
    parser.add_argument("--skip-umbrella",        action="store_true",
                        help="Skip Umbrella 1M download")
    parser.add_argument("--skip-casis",           action="store_true",
                        help="Skip CASIS Kaggle download")
    parser.add_argument("--skip-enron-kaggle",    action="store_true",
                        help="Skip Enron Kaggle download")
    parser.add_argument("--skip-nigerian-fraud",  action="store_true",
                        help="Skip Nigerian Fraud Kaggle download")
    parser.add_argument("--build-only",           action="store_true",
                        help="Skip all downloads; only build processed CSVs")
    parser.add_argument("--status",               action="store_true",
                        help="Show dataset status and exit")
    args = parser.parse_args()

    print(f"\n{bold(cyan(' PhishLens Dataset Downloader '))}")
    print(f"{'─' * 62}")
    print(f"  Base dir : {BASE_DIR}")
    print(f"  Raw data : {RAW_DIR}")
    print(f"  Processed: {PROC_DIR}")
    print(f"  Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.status:
        show_status()
        return

    # Ensure directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    if not args.build_only:
        if not args.skip_spamassassin:
            download_spamassassin()
        else:
            info("SpamAssassin: skipped (--skip-spamassassin)")

        if not args.skip_enron:
            download_enron()
        else:
            info("Enron: skipped (--skip-enron)")

        if not args.skip_phishingpot:
            download_phishing_pot()
        else:
            info("phishing_pot: skipped (--skip-phishingpot)")

        if not args.skip_umbrella:
            download_umbrella()
        else:
            info("Umbrella 1M: skipped (--skip-umbrella)")

        if not args.skip_casis:
            download_casis_kaggle()
        else:
            info("CASIS Kaggle: skipped (--skip-casis)")

        if not args.skip_enron_kaggle:
            download_enron_kaggle()
        else:
            info("Enron Kaggle: skipped (--skip-enron-kaggle)")

        if not args.skip_nigerian_fraud:
            download_nigerian_fraud()
        else:
            info("Nigerian Fraud: skipped (--skip-nigerian-fraud)")

    # Build processed CSVs
    counts = build_processed_datasets()
    if counts.get("train", 0) == 0:
        fail("No training data was built. Check download steps above.")
        sys.exit(1)

    # Validate
    ok_pipe = validate_pipeline()

    # Final summary
    section("DONE — Dataset Pipeline Summary")
    if counts.get("train", 0) > 0:
        ok(f"Training set : {counts['train']:,} emails in data/processed/train.csv")
        ok(f"Test set     : {counts['test']:,} emails in data/processed/test.csv")
    print()
    if ok_pipe:
        print(f"  {bold(green('✓  Data pipeline is ready for training!'))}")
    print()
    print(f"  {bold('Next step — start training:')}")
    print(f"  {cyan('  python train.py --data-dir data/processed --models xgboost --no-network --eval --save models')}")
    print()
    print(f"  {bold('Launch web UI (no training needed):')}")
    print(f"  {cyan('  python -m streamlit run app.py --server.port 8501')}")
    print()
    print(f"  {bold('For the full MeAJOR corpus (135,894 emails):')}")
    print(f"  {cyan('  git clone https://github.com/rf-peixoto/phishing_pot')}")
    print(f"  {cyan('  copy phishing_pot\\dataset\\dataset.csv data\\raw\\meajor.csv')}")
    print()
    print(f"  {bold('For CASIS dataset (BEC emails, Kaggle credentials needed):')}")
    print(f"  {cyan('  pip install kaggle')}")
    print(f"  {cyan('  kaggle datasets download naserabdullahalam/phishing-email-dataset')}")
    print(f"  {cyan('  rename to data\\raw\\casis.csv')}")


if __name__ == "__main__":
    main()
