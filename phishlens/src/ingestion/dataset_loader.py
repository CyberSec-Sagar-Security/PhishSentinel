"""
PhishLens Dataset Loader Module.

Loads and standardises multiple email corpora into a unified DataFrame format:
  - MeAJOR Phishing Email Corpus (primary training data)
  - SpamAssassin Public Corpus (stress testing / OOD evaluation)
  - CASIS Phishing Dataset (BEC samples)

All loaders normalise to columns: ['label', 'raw_email', 'source']
where label=0 for legitimate email, label=1 for phishing.

Security rationale: Maintaining a 'source' column allows post-hoc analysis
of which corpus is contributing false negatives — critical for identifying
dataset-specific blind spots.
"""

from __future__ import annotations

import mailbox
import os
from pathlib import Path
from typing import List

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# MeAJOR Loader
# ---------------------------------------------------------------------------


def load_meajor(data_dir: str) -> pd.DataFrame:
    """Load the MeAJOR Phishing Email Corpus from CSV.

    Expected file: data_dir/meajor.csv
    Columns expected: any columns containing 'label' and 'email'/'text'/'body'

    Args:
        data_dir: Path to directory containing meajor.csv.

    Returns:
        DataFrame with columns ['label', 'raw_email', 'source'].
        label=0 for legitimate, label=1 for phishing.
    """
    candidates = ["meajor.csv", "dataset.csv", "phishing_dataset.csv"]
    csv_path: Path | None = None

    for name in candidates:
        p = Path(data_dir) / name
        if p.exists():
            csv_path = p
            break

    if csv_path is None:
        log.error(
            f"MeAJOR CSV not found in '{data_dir}'. "
            "Expected: meajor.csv or dataset.csv. "
            "Download from: https://github.com/rf-peixoto/phishing_pot"
        )
        return pd.DataFrame(columns=["label", "raw_email", "source"])

    log.info(f"Loading MeAJOR corpus from '{csv_path}' ...")
    df = pd.read_csv(csv_path, low_memory=False)

    # Normalise column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Find label column
    label_col = _find_column(df, ["label", "class", "category", "spam", "phishing"])
    # Find text column
    text_col = _find_column(df, ["email", "body", "text", "raw_email", "message", "content"])

    if label_col is None or text_col is None:
        log.error(
            f"Could not identify label/text columns in MeAJOR CSV. "
            f"Available columns: {list(df.columns)}"
        )
        return pd.DataFrame(columns=["label", "raw_email", "source"])

    df = df[[label_col, text_col]].copy()
    df.columns = ["label", "raw_email"]
    df["raw_email"] = df["raw_email"].fillna("").astype(str)

    # Normalise labels to binary 0/1
    df["label"] = _normalise_labels(df["label"])
    df["source"] = "meajor"

    # Drop rows where raw_email is empty
    df = df[df["raw_email"].str.strip() != ""]

    phish_count = (df["label"] == 1).sum()
    legit_count = (df["label"] == 0).sum()
    log.info(
        f"MeAJOR loaded: {len(df):,} emails | "
        f"phishing={phish_count:,} | legitimate={legit_count:,}"
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# SpamAssassin Loader
# ---------------------------------------------------------------------------


def load_spamassassin(data_dir: str) -> pd.DataFrame:
    """Load the SpamAssassin Public Corpus from directory structure.

    Expected structure (from tar.bz2 extraction):
      data_dir/spamassassin_ham/   → legitimate emails (raw .eml files)
      data_dir/spamassassin_spam/  → spam/phishing emails

    Also supports mbox format: spamassassin_ham.mbox / spamassassin_spam.mbox

    Args:
        data_dir: Path to directory containing SpamAssassin files.

    Returns:
        DataFrame with columns ['label', 'raw_email', 'source'].
    """
    records: List[dict] = []

    ham_dir = Path(data_dir) / "spamassassin_ham"
    spam_dir = Path(data_dir) / "spamassassin_spam"
    ham_mbox = Path(data_dir) / "spamassassin_ham.mbox"
    spam_mbox = Path(data_dir) / "spamassassin_spam.mbox"

    # Directory mode
    if ham_dir.exists():
        records.extend(_load_eml_directory(ham_dir, label=0))
    if spam_dir.exists():
        records.extend(_load_eml_directory(spam_dir, label=1))

    # Mbox fallback
    if ham_mbox.exists():
        records.extend(_load_mbox(ham_mbox, label=0))
    if spam_mbox.exists():
        records.extend(_load_mbox(spam_mbox, label=1))

    if not records:
        log.warning(
            f"SpamAssassin corpus not found in '{data_dir}'. "
            "Download from: https://spamassassin.apache.org/old/publiccorpus/"
        )
        return pd.DataFrame(columns=["label", "raw_email", "source"])

    df = pd.DataFrame(records)
    df["source"] = "spamassassin"
    phish_count = (df["label"] == 1).sum()
    legit_count = (df["label"] == 0).sum()
    log.info(
        f"SpamAssassin loaded: {len(df):,} emails | "
        f"spam/phishing={phish_count:,} | ham={legit_count:,}"
    )
    return df.reset_index(drop=True)


def _load_eml_directory(directory: Path, label: int) -> List[dict]:
    """Load all .eml files from a directory."""
    records = []
    for fpath in directory.iterdir():
        if fpath.is_file():
            try:
                raw = fpath.read_bytes().decode("utf-8", errors="replace")
                records.append({"label": label, "raw_email": raw})
            except Exception as exc:
                log.debug(f"Skipping '{fpath}': {exc}")
    return records


def _load_mbox(mbox_path: Path, label: int) -> List[dict]:
    """Load emails from an mbox file."""
    records = []
    try:
        mbox = mailbox.mbox(str(mbox_path))
        for msg in mbox:
            try:
                raw = str(msg)
                records.append({"label": label, "raw_email": raw})
            except Exception:
                pass
    except Exception as exc:
        log.warning(f"Failed to read mbox '{mbox_path}': {exc}")
    return records


# ---------------------------------------------------------------------------
# CASIS Loader
# ---------------------------------------------------------------------------


def load_casis(data_dir: str) -> pd.DataFrame:
    """Load the CASIS Phishing Dataset (Kaggle) from CSV.

    Expected file: data_dir/casis.csv

    Args:
        data_dir: Path to directory containing casis.csv.

    Returns:
        DataFrame with columns ['label', 'raw_email', 'source'].
    """
    csv_path = Path(data_dir) / "casis.csv"
    if not csv_path.exists():
        log.warning(
            f"CASIS CSV not found at '{csv_path}'. Skipping."
            " Download from: https://www.kaggle.com/datasets/"
            "naserabdullahalam/phishing-email-dataset"
        )
        return pd.DataFrame(columns=["label", "raw_email", "source"])

    log.info(f"Loading CASIS corpus from '{csv_path}' ...")
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]

    label_col = _find_column(df, ["label", "class", "category", "email type"])
    text_col = _find_column(df, ["email", "body", "text", "message", "email text"])

    if label_col is None or text_col is None:
        log.error(f"Cannot identify columns in CASIS. Columns: {list(df.columns)}")
        return pd.DataFrame(columns=["label", "raw_email", "source"])

    df = df[[label_col, text_col]].copy()
    df.columns = ["label", "raw_email"]
    df["raw_email"] = df["raw_email"].fillna("").astype(str)
    df["label"] = _normalise_labels(df["label"])
    df["source"] = "casis"
    df = df[df["raw_email"].str.strip() != ""]

    log.info(f"CASIS loaded: {len(df):,} emails")
    return df.reset_index(drop=True)


def load_casis_kaggle(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load the CASIS Phishing Email Dataset downloaded via Kaggle CLI.

    Expected file: data/raw/casis/phishing_email.csv
    Columns:       'Email Text' (body), 'Email Type' ('Phishing Email' / 'Safe Email')

    Security rationale: Provides 82,000+ Business Email Compromise (BEC) samples
    from the Kaggle-hosted CASIS corpus, significantly expanding coverage of
    executive-impersonation and wire-transfer fraud patterns.

    Args:
        data_dir: Root raw data directory.

    Returns:
        DataFrame with ['label', 'raw_email', 'source'] columns.
        Returns empty DataFrame with correct schema if file is missing.
    """
    _COLS = ["label", "raw_email", "source"]
    casis_dir = Path(data_dir) / "casis"
    csv_path = casis_dir / "phishing_email.csv"

    # Try any CSV in the directory if the expected name is missing
    if not csv_path.exists():
        candidates = list(casis_dir.glob("*.csv")) if casis_dir.exists() else []
        if candidates:
            csv_path = candidates[0]
            log.info(f"CASIS Kaggle: using {csv_path.name} as fallback")
        else:
            log.warning(
                "CASIS Kaggle CSV not found at %s. "
                "Run: python download_datasets.py --skip-spamassassin ..., "
                "or manually: kaggle datasets download naserabdullahalam/phishing-email-dataset",
                csv_path,
            )
            return pd.DataFrame(columns=_COLS)

    try:
        df_raw = pd.read_csv(csv_path, low_memory=False, encoding_errors="replace")
        text_col  = _find_column(df_raw, ["Email Text", "email_text", "text", "body"])
        label_col = _find_column(df_raw, ["Email Type", "email_type", "type", "label"])
        if text_col is None or label_col is None:
            log.warning(
                "CASIS Kaggle: could not locate text/label columns. "
                "Found: %s", list(df_raw.columns)[:10]
            )
            return pd.DataFrame(columns=_COLS)

        df = pd.DataFrame()
        df["raw_email"] = df_raw[text_col].fillna("").astype(str)
        df["label"]     = _normalise_labels(df_raw[label_col])
        df["source"]    = "casis_kaggle"
        df = df[df["raw_email"].str.strip() != ""]
        df = df[df["label"].isin([0, 1])]
        log.info(f"CASIS Kaggle loaded: {len(df):,} emails")
        return df.reset_index(drop=True)
    except Exception as exc:
        log.error(f"CASIS Kaggle load error: {exc}")
        return pd.DataFrame(columns=_COLS)


def load_enron_kaggle(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load the Enron Email Dataset downloaded via Kaggle CLI (legitimate only).

    Expected file: data/raw/enron_kaggle/emails.csv
    Column used:   'message' (all emails are legitimate, label=0)
    Capped at:     30,000 rows (random_state=42) to avoid class imbalance.

    Security rationale: The Enron corpus is the gold-standard legitimate email
    dataset in NLP research. Adding it improves the false-positive rate by
    training on real business email patterns.

    Args:
        data_dir: Root raw data directory.

    Returns:
        DataFrame with ['label', 'raw_email', 'source'] columns (all label=0).
        Returns empty DataFrame with correct schema if file is missing.
    """
    _COLS = ["label", "raw_email", "source"]
    kag_dir  = Path(data_dir) / "enron_kaggle"
    csv_path = kag_dir / "emails.csv"

    if not csv_path.exists():
        candidates = list(kag_dir.glob("*.csv")) if kag_dir.exists() else []
        if candidates:
            csv_path = candidates[0]
            log.info(f"Enron Kaggle: using {csv_path.name} as fallback")
        else:
            log.warning(
                "Enron Kaggle CSV not found at %s. "
                "Run: kaggle datasets download wcukierski/enron-email-dataset",
                csv_path,
            )
            return pd.DataFrame(columns=_COLS)

    try:
        df_raw = pd.read_csv(csv_path, low_memory=False, encoding_errors="replace")
        df_raw.columns = [c.lower().strip() for c in df_raw.columns]
        msg_col = _find_column(df_raw, ["message", "text", "email", "body", "content"])
        if msg_col is None:
            log.warning("Enron Kaggle: could not locate message column. Found: %s", list(df_raw.columns)[:10])
            return pd.DataFrame(columns=_COLS)

        # Sample to avoid overwhelming the training set with legitimate emails
        if len(df_raw) > 30_000:
            df_raw = df_raw.sample(n=30_000, random_state=42)

        df = pd.DataFrame()
        df["raw_email"] = df_raw[msg_col].fillna("").astype(str)
        df["label"]     = 0
        df["source"]    = "enron_kaggle"
        df = df[df["raw_email"].str.strip() != ""]
        log.info(f"Enron Kaggle loaded: {len(df):,} legitimate emails")
        return df.reset_index(drop=True)
    except Exception as exc:
        log.error(f"Enron Kaggle load error: {exc}")
        return pd.DataFrame(columns=_COLS)


def load_nigerian_fraud(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load the Nigerian 419 Fraudulent Email Corpus (Kaggle).

    Expected file: data/raw/nigerian_fraud/fradulent_emails.txt
                   (note: 'fradulent' is the dataset's own typo)
    Format:        Raw email blobs separated by double newlines, all label=1.

    Security rationale: Nigerian advance-fee fraud ('419 scam') uses distinct
    social engineering patterns (urgency, large monetary promises, secrecy)
    that differ from modern phishing. Including this corpus improves coverage
    of financially-motivated social engineering beyond credential harvesting.

    Args:
        data_dir: Root raw data directory.

    Returns:
        DataFrame with ['label', 'raw_email', 'source'] columns (all label=1).
        Returns empty DataFrame with correct schema if file is missing.
    """
    _COLS = ["label", "raw_email", "source"]
    nf_dir  = Path(data_dir) / "nigerian_fraud"
    txt_path = nf_dir / "fradulent_emails.txt"   # Dataset's own spelling

    # Fallback to alternate spelling in case Kaggle corrects the typo
    if not txt_path.exists():
        alt = nf_dir / "fraudulent_emails.txt"
        if alt.exists():
            txt_path = alt

    if not txt_path.exists():
        log.warning(
            "Nigerian Fraud text file not found at %s. "
            "Run: kaggle datasets download rtatman/fraudulent-email-corpus",
            txt_path,
        )
        return pd.DataFrame(columns=_COLS)

    try:
        raw_blob = txt_path.read_text(encoding="utf-8", errors="replace")
        chunks = [c.strip() for c in raw_blob.split("\n\n")]
        valid  = [c for c in chunks if len(c) >= 50]
        df = pd.DataFrame({
            "raw_email": valid,
            "label":     1,
            "source":    "nigerian_fraud",
        })
        log.info(f"Nigerian Fraud loaded: {len(df):,} phishing emails")
        return df.reset_index(drop=True)
    except Exception as exc:
        log.error(f"Nigerian Fraud load error: {exc}")
        return pd.DataFrame(columns=_COLS)


# ---------------------------------------------------------------------------
# HuggingFace Dataset Loaders
# ---------------------------------------------------------------------------

def _hf_import():
    """Import HuggingFace datasets library, raising a clear error if missing."""
    try:
        from datasets import load_dataset as _load  # type: ignore
        return _load
    except ImportError:
        raise ImportError(
            "HuggingFace 'datasets' library is not installed. "
            "Run: pip install datasets>=2.18.0"
        )


def load_hf_locuoco(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load locuoco/the-biggest-spam-ham-phish-email-dataset-300000 from HuggingFace.

    Three-class dataset: 0=Ham (legitimate), 1=Phish, 2=Spam.
    Mapping applied: 0→0 (legitimate), 1→1 (phishing), 2→1 (spam treated as malicious).
    ~365,448 rows. Text column contains email/message body only (no full RFC-822 headers).

    Args:
        data_dir: Unused — HuggingFace datasets library manages its own cache.

    Returns:
        DataFrame with ['label', 'raw_email', 'source'] columns.
    """
    _COLS = ["label", "raw_email", "source"]
    _SOURCE = "locuoco/the-biggest-spam-ham-phish-email-dataset-300000"
    try:
        _load = _hf_import()
        log.info(f"Downloading {_SOURCE} from HuggingFace (cached after first run) ...")
        ds = _load(_SOURCE, split="train")
        df = ds.to_pandas()
        df.columns = [c.lower().strip() for c in df.columns]

        text_col  = _find_column(df, ["text", "email", "body", "message", "content", "email_text"])
        label_col = _find_column(df, ["label", "class", "category", "type"])

        if text_col is None or label_col is None:
            log.error(
                f"locuoco: cannot find text/label columns. "
                f"Available columns: {list(df.columns)[:10]}"
            )
            return pd.DataFrame(columns=_COLS)

        # Three-class to binary: 0=Ham→0, 1=Phish→1, 2=Spam→1
        def _map_locuoco(val):
            try:
                v = int(val)
            except (TypeError, ValueError):
                v = str(val).strip().lower()
                if v in ("ham", "legitimate", "legit", "0"): return 0
                if v in ("phish", "phishing", "spam", "1", "2"): return 1
                return -1
            if v == 0: return 0
            if v in (1, 2): return 1
            return -1

        result = pd.DataFrame()
        result["raw_email"] = df[text_col].fillna("").astype(str)
        result["label"]     = df[label_col].map(_map_locuoco)
        result["source"]    = _SOURCE
        result = result[result["raw_email"].str.strip() != ""]
        result = result[result["label"].isin([0, 1])]
        result = result.reset_index(drop=True)

        phish = (result["label"] == 1).sum()
        legit = (result["label"] == 0).sum()
        log.info(
            f"locuoco loaded: {len(result):,} emails | "
            f"phishing/spam={phish:,} | legitimate(ham)={legit:,}"
        )
        if legit == 0 or phish == 0:
            log.warning(
                f"locuoco: one class has 0 rows — label mapping may be wrong! "
                f"Verify the 3-class label values in the dataset."
            )
        return result

    except Exception as exc:
        log.error(f"locuoco HuggingFace load error: {exc}")
        return pd.DataFrame(columns=_COLS)


def load_hf_puyang_seven(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load puyang2025/seven-phishing-email-datasets from HuggingFace.

    NOTE: This dataset is content-identical to JinqiangDing/seven-phishing-email-datasets
    (same 203,017 rows, same sources: TREC-05/06/07, CEAS-08, Enron, Ling, SpamAssassin).
    Prefer load_hf_jinqiangding() which was updated more recently (April 2026).
    If both are called, deduplication in combine_datasets() will remove overlaps.

    Columns: text, subject, label, sender, receiver, date, urls, dataset_name.
    Label: 0=legitimate, 1=phishing/spam. Text is email body only.

    Args:
        data_dir: Unused.

    Returns:
        DataFrame with ['label', 'raw_email', 'source'] columns.
    """
    _COLS = ["label", "raw_email", "source"]
    _SOURCE = "puyang2025/seven-phishing-email-datasets"
    try:
        _load = _hf_import()
        log.info(f"Downloading {_SOURCE} from HuggingFace ...")
        ds = _load(_SOURCE, split="train")
        df = ds.to_pandas()
        df.columns = [c.lower().strip() for c in df.columns]

        text_col  = _find_column(df, ["text", "body", "email", "message", "content"])
        label_col = _find_column(df, ["label", "class", "category"])

        if text_col is None or label_col is None:
            log.error(
                f"puyang2025/seven: cannot find text/label columns. "
                f"Available columns: {list(df.columns)[:10]}"
            )
            return pd.DataFrame(columns=_COLS)

        result = pd.DataFrame()
        result["raw_email"] = df[text_col].fillna("").astype(str)
        result["label"]     = _normalise_labels(df[label_col])
        result["source"]    = _SOURCE
        result = result[result["raw_email"].str.strip() != ""]
        result = result[result["label"].isin([0, 1])]
        result = result.reset_index(drop=True)

        phish = (result["label"] == 1).sum()
        legit = (result["label"] == 0).sum()
        log.info(
            f"puyang2025/seven loaded: {len(result):,} emails | "
            f"phishing={phish:,} | legitimate={legit:,}"
        )
        return result

    except Exception as exc:
        log.error(f"puyang2025/seven HuggingFace load error: {exc}")
        return pd.DataFrame(columns=_COLS)


def load_hf_jinqiangding(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load JinqiangDing/seven-phishing-email-datasets from HuggingFace.

    203,017 emails from seven classic corpora: TREC-05/06/07, CEAS-08, Enron,
    Ling-Spam, SpamAssassin. Updated April 2026. Preferred over the identical
    puyang2025/seven-phishing-email-datasets version.

    Columns: text, subject, label, sender, receiver, date, urls, dataset_name.
    Label: 0=legitimate(ham), 1=phishing/spam.
    Text contains email body only — no full RFC-822 headers. Header features
    (SPF, DKIM, relay hops) will default to -1 for these samples.

    Args:
        data_dir: Unused.

    Returns:
        DataFrame with ['label', 'raw_email', 'source'] columns.
    """
    _COLS = ["label", "raw_email", "source"]
    _SOURCE = "JinqiangDing/seven-phishing-email-datasets"
    try:
        _load = _hf_import()
        log.info(f"Downloading {_SOURCE} from HuggingFace ...")
        ds = _load(_SOURCE, split="train")
        df = ds.to_pandas()
        df.columns = [c.lower().strip() for c in df.columns]

        text_col  = _find_column(df, ["text", "body", "email", "message", "content"])
        label_col = _find_column(df, ["label", "class", "category"])

        if text_col is None or label_col is None:
            log.error(
                f"JinqiangDing/seven: cannot find text/label columns. "
                f"Available columns: {list(df.columns)[:10]}"
            )
            return pd.DataFrame(columns=_COLS)

        result = pd.DataFrame()
        result["raw_email"] = df[text_col].fillna("").astype(str)
        result["label"]     = _normalise_labels(df[label_col])
        result["source"]    = _SOURCE
        result = result[result["raw_email"].str.strip() != ""]
        result = result[result["label"].isin([0, 1])]
        result = result.reset_index(drop=True)

        phish = (result["label"] == 1).sum()
        legit = (result["label"] == 0).sum()
        log.info(
            f"JinqiangDing/seven loaded: {len(result):,} emails | "
            f"phishing={phish:,} | legitimate={legit:,}"
        )
        return result

    except Exception as exc:
        log.error(f"JinqiangDing/seven HuggingFace load error: {exc}")
        return pd.DataFrame(columns=_COLS)


def load_hf_llmgen(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load Dizzzy0x00/LLMGen-Phishing-Email-Dataset from HuggingFace.

    6,776 AI-generated emails (both phishing and legitimate) created by
    DeepSeek (Chinese) and OpenAI (English) LLMs, December 2025.
    Adds adversarial robustness against LLM-crafted phishing attacks.

    Columns: content, label. Label: 0=legitimate, 1=phishing.
    Contains body text only — no email headers (AI-generated emails
    typically omit RFC-822 headers). Header features will default to -1.

    NOTE: This dataset contains AI-generated content and should be treated
    as a separate quality category in evaluation. The model trained on this
    data gains the capability to detect phishing crafted by GPT-4 and Claude.

    Args:
        data_dir: Unused.

    Returns:
        DataFrame with ['label', 'raw_email', 'source'] columns.
    """
    _COLS = ["label", "raw_email", "source"]
    _SOURCE = "Dizzzy0x00/LLMGen-Phishing-Email-Dataset"
    try:
        _load = _hf_import()
        log.info(f"Downloading {_SOURCE} from HuggingFace ...")
        ds = _load(_SOURCE, split="train")
        df = ds.to_pandas()
        df.columns = [c.lower().strip() for c in df.columns]

        # Primary column name from dataset card is 'content'
        text_col  = _find_column(df, ["content", "text", "email", "body", "message"])
        label_col = _find_column(df, ["label", "class", "category", "type"])

        if text_col is None:
            log.error(
                f"LLMGen: cannot find text column. "
                f"Available columns: {list(df.columns)[:10]}"
            )
            return pd.DataFrame(columns=_COLS)

        result = pd.DataFrame()
        result["raw_email"] = df[text_col].fillna("").astype(str)

        if label_col is not None:
            result["label"] = _normalise_labels(df[label_col])
        else:
            # No label column — all rows are phishing (AI-generated dataset)
            log.info("LLMGen: no label column found — assigning all rows label=1 (phishing)")
            result["label"] = 1

        result["source"] = _SOURCE
        result = result[result["raw_email"].str.strip() != ""]
        result = result[result["label"].isin([0, 1])]
        result = result.reset_index(drop=True)

        phish = (result["label"] == 1).sum()
        legit = (result["label"] == 0).sum()
        log.info(
            f"LLMGen loaded: {len(result):,} AI-generated emails | "
            f"phishing={phish:,} | legitimate={legit:,} "
            f"[AI-GENERATED — separate quality category in evaluation]"
        )
        return result

    except Exception as exc:
        log.error(f"LLMGen HuggingFace load error: {exc}")
        return pd.DataFrame(columns=_COLS)


def load_hf_zefang(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load zefang-liu/phishing-email-dataset from HuggingFace (LGPL-3.0).

    18,650 emails from the Kaggle 'Phishing Email Detection' dataset by
    'Cyber Cop' (subhajournal/phishingemails). High community validation
    with 2,063 downloads/month. May overlap with existing casis_kaggle data
    — deduplication in combine_datasets() handles this.

    Columns: Unnamed: 0 (index), Email Text, Email Type.
    Email Type values: 'Phishing Email'=1, 'Safe Email'=0.
    Text is email body. Header features will default to -1.

    Args:
        data_dir: Unused.

    Returns:
        DataFrame with ['label', 'raw_email', 'source'] columns.
    """
    _COLS = ["label", "raw_email", "source"]
    _SOURCE = "zefang-liu/phishing-email-dataset"
    try:
        _load = _hf_import()
        log.info(f"Downloading {_SOURCE} from HuggingFace ...")
        ds = _load(_SOURCE, split="train")
        df = ds.to_pandas()
        # Preserve original mixed-case column names for exact matching
        orig_cols = list(df.columns)
        df.columns = [c.lower().strip() for c in df.columns]

        # This dataset uses 'email text' and 'email type' (from Kaggle)
        text_col  = _find_column(df, ["email text", "email_text", "text", "body", "message", "content"])
        label_col = _find_column(df, ["email type", "email_type", "label", "type", "class"])

        if text_col is None or label_col is None:
            log.error(
                f"zefang-liu: cannot find text/label columns. "
                f"Original columns: {orig_cols[:10]}"
            )
            return pd.DataFrame(columns=_COLS)

        result = pd.DataFrame()
        result["raw_email"] = df[text_col].fillna("").astype(str)
        result["label"]     = _normalise_labels(df[label_col])
        result["source"]    = _SOURCE
        result = result[result["raw_email"].str.strip() != ""]
        result = result[result["label"].isin([0, 1])]
        result = result.reset_index(drop=True)

        phish = (result["label"] == 1).sum()
        legit = (result["label"] == 0).sum()
        log.info(
            f"zefang-liu loaded: {len(result):,} emails | "
            f"phishing={phish:,} | legitimate={legit:,}"
        )
        return result

    except Exception as exc:
        log.error(f"zefang-liu HuggingFace load error: {exc}")
        return pd.DataFrame(columns=_COLS)


def load_hf_puyang_phish(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load puyang2025/phish-email-datasets from HuggingFace.

    26,612 rows containing Nazario phishing emails and Nigerian Fraud emails
    (Nazario.parquet, Nazario_5.parquet, Nigerian_Fraud.parquet).
    May overlap with existing nigerian_fraud and casis corpus entries —
    deduplication in combine_datasets() handles this.

    Columns: sender, receiver, date, subject, body, urls, label.
    Label: 1=phishing. Uses 'body' for email text content.
    All-phishing dataset (no legitimate emails).

    Args:
        data_dir: Unused.

    Returns:
        DataFrame with ['label', 'raw_email', 'source'] columns.
    """
    _COLS = ["label", "raw_email", "source"]
    _SOURCE = "puyang2025/phish-email-datasets"
    parts: list = []

    # ── Part A: Nazario.parquet + Nigerian_Fraud.parquet ─────────────────
    # These two files share a uniform schema: sender/receiver/date/subject/body/urls/label.
    # Nazario_5.parquet is skipped (ArrowInvalid: int64 col stores URL strings).
    # Phishing_Email.parquet is handled separately in Part B (different schema).
    _UNIFORM_FILES = ["Nazario.parquet", "Nigerian_Fraud.parquet"]
    try:
        _load = _hf_import()
        log.info(
            f"Downloading {_SOURCE} from HuggingFace "
            f"(Nazario + Nigerian_Fraud via datasets; Phishing_Email via pandas) ..."
        )
        ds = _load(_SOURCE, data_files={"train": _UNIFORM_FILES}, split="train")
        df = ds.to_pandas()
        df.columns = [c.lower().strip() for c in df.columns]
        text_col  = _find_column(df, ["body", "text", "email", "message", "content"])
        label_col = _find_column(df, ["label", "class", "category", "type"])
        part_a = pd.DataFrame()
        part_a["raw_email"] = df[text_col].fillna("").astype(str) if text_col else pd.Series([""] * len(df))
        part_a["label"] = _normalise_labels(df[label_col]) if label_col else pd.Series([1] * len(df))
        part_a["source"] = _SOURCE
        part_a = part_a[part_a["raw_email"].str.strip() != ""]
        part_a = part_a[part_a["label"].isin([0, 1])]
        log.info(f"Part A (Nazario+Nigerian_Fraud): {len(part_a):,} rows loaded")
        parts.append(part_a)
    except Exception as exc_a:
        log.warning(f"puyang2025/phish Part A failed: {exc_a}")

    # ── Part B: Phishing_Email.parquet ────────────────────────────────────
    # Schema: [Unnamed: 0 (int), Email Text (str), Email Type (str)]
    # Email Type values: "Phishing Email" → 1, "Safe Email" → 0
    try:
        from huggingface_hub import hf_hub_download as _hf_dl
        pe_path = _hf_dl(_SOURCE, "Phishing_Email.parquet", repo_type="dataset")
        df_pe = pd.read_parquet(pe_path)
        df_pe.columns = [c.strip() for c in df_pe.columns]
        if "Email Text" in df_pe.columns and "Email Type" in df_pe.columns:
            part_b = pd.DataFrame()
            part_b["raw_email"] = df_pe["Email Text"].fillna("").astype(str)
            part_b["label"] = (
                df_pe["Email Type"]
                .str.strip()
                .map({"Phishing Email": 1, "Safe Email": 0})
                .fillna(1)
                .astype(int)
            )
            part_b["source"] = _SOURCE
            part_b = part_b[part_b["raw_email"].str.strip() != ""]
            part_b = part_b[part_b["label"].isin([0, 1])]
            log.info(f"Part B (Phishing_Email.parquet): {len(part_b):,} rows loaded via pandas")
            parts.append(part_b)
        else:
            log.warning(
                f"Phishing_Email.parquet has unexpected columns: {list(df_pe.columns)[:8]} — skipping Part B"
            )
    except Exception as exc_b:
        log.warning(f"puyang2025/phish Part B (Phishing_Email.parquet) failed: {exc_b}")

    if not parts:
        log.error("puyang2025/phish: all parts failed — returning empty DataFrame")
        return pd.DataFrame(columns=_COLS)

    result = pd.concat(parts, ignore_index=True)
    result = result.reset_index(drop=True)
    phish = (result["label"] == 1).sum()
    legit = (result["label"] == 0).sum()
    log.info(
        f"puyang2025/phish loaded: {len(result):,} emails | "
        f"phishing={phish:,} | legitimate={legit:,}"
    )
    return result


# ---------------------------------------------------------------------------
# Combine + utilities
# ---------------------------------------------------------------------------


def combine_datasets(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Combine multiple email DataFrames, deduplicate, and shuffle.

    Deduplication uses the first 200 characters of raw_email (stripped,
    lowercased) as a fast near-duplicate key — sufficient to catch exact and
    near-exact duplicates across the ~1M-row corpus without full-text hashing.

    When duplicates are found, the row from the highest-priority source is
    kept according to this order (lower number = keep):
      1 – JinqiangDing/seven-phishing-email-datasets
      2 – puyang2025/* datasets
      3 – locuoco/the-biggest-spam-ham-phish-email-dataset-300000
      4 – Dizzzy0x00/LLMGen-Phishing-Email-Dataset
      5 – zefang-liu/phishing-email-dataset
      6 – all original four local sources

    Args:
        *dfs: DataFrames each with ['label', 'raw_email', 'source'] columns.

    Returns:
        Combined, deduplicated, shuffled DataFrame with detailed class
        distribution logged (total, phishing, legitimate, ratio, per-source).
    """
    _SOURCE_PRIORITY: dict = {
        "JinqiangDing/seven-phishing-email-datasets": 1,
        "puyang2025/seven-phishing-email-datasets": 2,
        "puyang2025/phish-email-datasets": 2,
        "locuoco/the-biggest-spam-ham-phish-email-dataset-300000": 3,
        "Dizzzy0x00/LLMGen-Phishing-Email-Dataset": 4,
        "zefang-liu/phishing-email-dataset": 5,
        # existing local sources — lowest priority
        "meajor": 6,
        "spamassassin": 6,
        "casis": 6,
        "casis_kaggle": 6,
        "enron": 6,
        "enron_kaggle": 6,
        "nigerian_fraud": 6,
        "phishing_pot": 6,
        "processed": 6,
    }

    non_empty = [df for df in dfs if len(df) > 0]
    if not non_empty:
        log.error("All provided DataFrames are empty. Cannot combine.")
        return pd.DataFrame(columns=["label", "raw_email", "source"])

    combined = pd.concat(non_empty, ignore_index=True)

    # Assign priority scores and sort so highest-priority rows come first
    combined["_priority"] = combined["source"].map(
        lambda s: _SOURCE_PRIORITY.get(s, 6)
    )
    combined.sort_values("_priority", inplace=True)

    # Deduplicate on first 200 characters of raw email (fast near-duplicate key)
    before = len(combined)
    combined["_dedup_key"] = (
        combined["raw_email"].str.strip().str[:200].str.lower()
    )
    combined = combined.drop_duplicates(subset=["_dedup_key"], keep="first")
    combined = combined.drop(columns=["_dedup_key", "_priority"])
    after = len(combined)

    if before != after:
        log.info(f"Deduplication: removed {before - after:,} near-duplicate emails.")

    # Shuffle
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Log class distribution per source
    total_phish = (combined["label"] == 1).sum()
    total_legit = (combined["label"] == 0).sum()
    ratio = total_phish / max(total_legit, 1)
    log.info(f"Combined dataset: {len(combined):,} emails total")
    for source in sorted(combined["source"].unique()):
        sub = combined[combined["source"] == source]
        phish = (sub["label"] == 1).sum()
        legit = (sub["label"] == 0).sum()
        log.info(
            f"  [{source}] total={len(sub):,} | phishing={phish:,} | legitimate={legit:,}"
        )
    log.info(
        f"  [TOTAL] phishing={total_phish:,} | legitimate={total_legit:,} | "
        f"imbalance_ratio={ratio:.3f}"
    )

    return combined


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    """Find the first matching column name from a list of candidates."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _normalise_labels(series: pd.Series) -> pd.Series:
    """Normalise heterogeneous label values to binary int (0=legit, 1=phishing).

    Handles: 0/1 integers, 'spam'/'ham', 'phishing'/'legitimate',
    'Phishing Email'/'Safe Email', True/False, etc.
    """
    mapping: dict = {
        # Numeric
        0: 0, 1: 1, 2: 1,
        # String forms
        "0": 0, "1": 1,
        "ham": 0, "spam": 1,
        "legitimate": 0, "phishing": 1,
        "safe email": 0, "phishing email": 1,
        "legit": 0, "malicious": 1,
        "false": 0, "true": 1,
        False: 0, True: 1,
    }

    def _map(val):
        if isinstance(val, str):
            val = val.strip().lower()
        return mapping.get(val, -1)  # -1 = unknown; filtered below

    normalised = series.map(_map)
    unknown_count = (normalised == -1).sum()
    if unknown_count > 0:
        log.warning(
            f"{unknown_count:,} rows had unrecognised label values and will be dropped."
        )
    return normalised[normalised != -1].astype(int)
