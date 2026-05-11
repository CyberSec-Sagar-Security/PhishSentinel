"""
PhishLens Master Feature Assembly Pipeline.

Orchestrates all five feature modules into a single sklearn-compatible
FeaturePipeline class with fit/transform/fit_transform interface.
Handles module failures gracefully — a broken URL lookup never crashes
the entire pipeline; it fills with zeros and logs a warning.

Total feature count (approximate):
  - Header features:     12
  - URL features:        ~29 (12 lexical × 2 aggregations + 3 WHOIS + 3 cert + 1 count)
  - HTML features:       11
  - Text/subject feats:   8 (urgency + subject)
  - Semantic embedding: 384
  - TF-IDF:            500
  - Intelligence APIs:   13 (VT + GSB + URLScan + URLhaus + AbuseIPDB)
  - Anomaly score:        1
  TOTAL:              ~961 features (after TF-IDF; ~461 without TF-IDF)

Security rationale: The multi-layer architecture means an attacker must
simultaneously evade: header authentication checks, URL lexical signals,
HTML obfuscation detection, semantic embedding space patterns, AND five
independent threat intelligence feeds. This defence-in-depth approach
dramatically raises the evasion cost.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse

from src.ingestion.eml_parser import parse_eml_string
from src.features.header_features import extract_header_features
from src.features.url_features import extract_url_features, extract_url_features_with_network
from src.features.html_features import extract_html_features
from src.features.text_features import extract_text_features, extract_tfidf_features
from src.features.intelligence import enrich_email_with_intelligence, get_default_intelligence_features
from src.utils.config import DEFAULT_CONFIG, PhishLensConfig
from src.utils.logger import get_logger

log = get_logger(__name__)

_IP_RE = re.compile(r"\b(\d{1,3}\.){3}\d{1,3}\b")

# Default embedding cache directory (relative to CWD = project root).
_EMBEDDING_CACHE_DIR = Path("data") / "processed" / "embedding_cache"


# Hard cap on raw email bytes fed to the email parser.
# email.message_from_bytes traverses the entire input searching for MIME
# boundaries and decoding parts — a 5 MB email with a base64-inlined image
# can take 30-60 s in a single loky worker, freezing the progress bar.
# 500 KB is well above the largest meaningful text/HTML payload we need.
_MAX_RAW_CHARS = 500_000


def _safe_parse(raw) -> dict:
    """Module-level picklable wrapper around parse_eml_string for joblib workers."""
    from src.ingestion.eml_parser import parse_eml_string as _parse
    try:
        raw_str = str(raw) if raw is not None else ""
        if len(raw_str) > _MAX_RAW_CHARS:
            raw_str = raw_str[:_MAX_RAW_CHARS]
        return _parse(raw_str)
    except Exception:
        return {
            "body_text": "", "body_html": "", "urls": [],
            "received_headers": [], "subject": "", "from_address": "",
            "return_path": "", "reply_to": "", "message_id": "",
            "x_mailer": "", "spf": "", "dkim": "", "dmarc": "",
            "timezone": "", "attachments_count": 0, "attachment_hashes": [],
        }


def _extract_email_no_tfidf(
    parsed: dict,
    preembed,
    use_network: bool,
    config,
    use_intelligence_apis: bool,
    use_gemini: bool,
):
    """Module-level worker for parallel per-email feature extraction.

    Extracts all features EXCEPT TF-IDF (which is batch-computed by the
    caller via a single vectorizer.transform() call over all emails).
    Top-level (non-method) so joblib/loky can pickle it without needing
    to serialise the entire FeaturePipeline instance.
    """
    import re as _re
    import numpy as _np
    from src.features.header_features import extract_header_features
    from src.features.url_features import (
        extract_url_features,
        extract_url_features_with_network,
        _default_url_features,
    )
    from src.features.html_features import extract_html_features, _default_html_features
    from src.features.text_features import extract_text_features
    from src.features.intelligence import (
        enrich_email_with_intelligence,
        get_default_intelligence_features,
    )

    _ip_re = _re.compile(r"\b(\d{1,3}\.){3}\d{1,3}\b")
    feature_vector: list = []
    feature_names: list = []

    # ---- Module 1: Header features (12) --------------------------------
    try:
        header_feats = extract_header_features(parsed, use_network=use_network)
    except Exception:
        from src.features.header_features import _default_header_features as _dhf
        header_feats = _dhf()
    for k, v in header_feats.items():
        feature_vector.append(float(v))
        feature_names.append(f"hdr_{k}")

    # ---- Module 2: URL features (~29) ----------------------------------
    urls = parsed.get("urls", [])
    try:
        if use_network:
            url_feats = extract_url_features_with_network(urls, config)
        else:
            url_feats = extract_url_features(urls, config)
    except Exception:
        url_feats = _default_url_features()
    for k, v in url_feats.items():
        feature_vector.append(float(v) if not isinstance(v, str) else 0.0)
        feature_names.append(f"url_{k}")

    # ---- Module 3: HTML features (11) ----------------------------------
    try:
        html_feats = extract_html_features(parsed.get("body_html", ""))
    except Exception:
        html_feats = _default_html_features()
    for k, v in html_feats.items():
        feature_vector.append(float(v))
        feature_names.append(f"html_{k}")

    # ---- Module 4: Text scalar + embeddings (8 + 384) ------------------
    # tfidf_vectorizer=None so TF-IDF is intentionally skipped here.
    try:
        text_vec, text_names = extract_text_features(
            body_text=parsed.get("body_text", ""),
            subject=parsed.get("subject", ""),
            config=config,
            tfidf_vectorizer=None,
            precomputed_embedding=preembed,
        )
        feature_vector.extend(text_vec.tolist())
        feature_names.extend(text_names)
    except Exception:
        feature_vector.extend([0.0] * (8 + 384))
        feature_names.extend([f"txt_pad_{i}" for i in range(8 + 384)])

    # ---- Module 5 (TF-IDF) is inserted by the caller in batch. --------

    # ---- Module 6: Intelligence API features (13) ----------------------
    if use_intelligence_apis:
        try:
            sender_ip = None
            for header in reversed(parsed.get("received_headers", [])):
                for ip in _ip_re.findall(header):
                    if not (ip.startswith("10.") or ip.startswith("192.168.")
                            or ip.startswith("127.") or ip == "0.0.0.0"):
                        sender_ip = ip
                        break
                if sender_ip:
                    break
            intel_feats = enrich_email_with_intelligence(urls, sender_ip)
        except Exception:
            intel_feats = get_default_intelligence_features()
    else:
        intel_feats = get_default_intelligence_features()
    for k, v in intel_feats.items():
        if isinstance(v, (int, float)):
            feature_vector.append(float(v))
            feature_names.append(f"intel_{k}")

    # ---- Module 7: ChatGPT AI (2) ----------------------------------------
    if use_gemini:
        try:
            from src.features.openai_analyzer import get_openai_ml_feature
            gemini_feats = get_openai_ml_feature(
                subject=parsed.get("subject", ""),
                from_address=parsed.get("from_address", ""),
                body_text=parsed.get("body_text", ""),
                urls=urls,
            )
        except Exception:
            gemini_feats = {"gemini_is_phishing": -1, "gemini_confidence": -1.0}
        for k, v in gemini_feats.items():
            feature_vector.append(float(v))
            feature_names.append(f"ai_{k}")

    return _np.array(feature_vector, dtype=_np.float32), feature_names


def _extract_chunk_matrix(args):
    """Process a chunk of pre-parsed emails and return (matrix, feature_names).

    Each worker receives one large chunk (~n/n_jobs emails) and processes them
    sequentially, returning a single numpy matrix.  Returning a numpy array
    (not a list of individual (vec, names) tuples) lets joblib use memory-mapped
    file transfer, reducing IPC traffic from ~2.4 GB to ~8 small transfers and
    eliminating the Windows IPC queue-overflow deadlock caused by 134K tiny tasks.
    """
    import numpy as _np
    parsed_chunk, embed_chunk, use_network, config, use_intelligence_apis, use_gemini = args
    rows = []
    names = None
    n_feats_fallback = 0
    for i, parsed in enumerate(parsed_chunk):
        preembed = embed_chunk[i] if embed_chunk is not None else None
        result = _extract_email_no_tfidf(
            parsed, preembed, use_network, config, use_intelligence_apis, use_gemini
        )
        if result is not None:
            vec, feat_names = result
            if names is None:
                names = feat_names
                n_feats_fallback = len(feat_names)
            rows.append(vec.astype(_np.float32))
        else:
            rows.append(None)

    if not rows or names is None:
        return _np.empty((0, 0), dtype=_np.float32), []

    final_rows = [
        r if r is not None else _np.zeros(n_feats_fallback, dtype=_np.float32)
        for r in rows
    ]
    return _np.vstack(final_rows), names


class FeaturePipeline:
    """Master feature assembly pipeline for PhishLens.

    Sklearn-compatible interface: fit(), transform(), fit_transform().
    Handles all feature modules with graceful per-module fallback on failure.

    Args:
        config: PhishLensConfig instance (default = DEFAULT_CONFIG).
        use_network: If True, enables WHOIS + crt.sh + API lookups.
                     Set False for fast offline training / CI testing.
        use_intelligence_apis: If True, calls VT / GSB / AbuseIPDB / URLScan.
        use_gemini: If True, calls ChatGPT AI for additional analysis feature.
        use_tfidf: If True, includes TF-IDF sparse features.
    """

    def __init__(
        self,
        config: PhishLensConfig = DEFAULT_CONFIG,
        use_network: bool = True,
        use_intelligence_apis: bool = True,
        use_gemini: bool = False,   # Disabled by default in training (rate limits)
        use_tfidf: bool = True,
    ) -> None:
        self.config = config
        self.use_network = use_network
        self.use_intelligence_apis = use_intelligence_apis
        self.use_gemini = use_gemini
        self.use_tfidf = use_tfidf
        self._tfidf_vectorizer = None
        self._is_fitted = False
        self._feature_names: List[str] = []
        # Pre-parsed cache: populated in fit(), consumed by the next transform() call.
        self._pre_parsed_cache: Optional[List[dict]] = None

    def __repr__(self) -> str:
        return (
            f"FeaturePipeline("
            f"use_network={self.use_network}, "
            f"use_intelligence_apis={self.use_intelligence_apis}, "
            f"use_tfidf={self.use_tfidf}, "
            f"fitted={self._is_fitted})"
        )

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """Fit the pipeline (TF-IDF vectorizer) on the training corpus.

        Args:
            df: DataFrame with 'raw_email' and optionally 'body_text' columns.

        Returns:
            self (for method chaining).
        """
        log.info("Fitting FeaturePipeline on training corpus ...")

        # Parse all emails ONCE in parallel; cache for reuse in transform().
        raw_emails = df["raw_email"].tolist()
        pre_parsed = self._parse_all_parallel(raw_emails)
        self._pre_parsed_cache = pre_parsed

        if self.use_tfidf:
            texts = [
                p.get("body_text", "") or p.get("body_html", "") or ""
                for p in pre_parsed
            ]
            _, vectorizer, _ = extract_tfidf_features(
                texts, config=self.config, fit=True
            )
            self._tfidf_vectorizer = vectorizer
            log.info("TF-IDF vectorizer fitted.")

        self._is_fitted = True
        return self

    def _get_cached_embeddings(self, body_texts: List[str]) -> np.ndarray:
        """Load batch embeddings from disk cache or compute and cache them.

        Cache key is derived from dataset size + a sample of text content, so
        any change in data automatically invalidates the stale cache file.

        Security rationale: Cache files are stored locally under
        data/processed/embedding_cache/ and never transmitted. Filenames are
        MD5 hashes that reveal nothing about email content. The cache directory
        is listed in .gitignore to prevent accidental corpus leakage.

        Args:
            body_texts: List of email body strings to encode.

        Returns:
            np.ndarray of shape (n_samples, 384) with float32 embeddings.
        """
        cache_dir = _EMBEDDING_CACHE_DIR
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning(f"Cannot create embedding cache dir: {exc}")

        # Build a deterministic cache key from size + content sample.
        sample = "".join(body_texts[:10])[:2000]
        cache_key = hashlib.md5(
            f"{len(body_texts)}_{sample}".encode("utf-8", errors="replace")
        ).hexdigest()
        cache_file = cache_dir / f"embeddings_{cache_key}.npy"

        if cache_file.exists():
            try:
                embeddings = np.load(str(cache_file))
                if embeddings.shape == (len(body_texts), 384):
                    log.info(
                        f"Embedding cache HIT: {cache_file.name} "
                        f"({embeddings.shape[0]:,} rows)"
                    )
                    return embeddings
                log.warning(
                    f"Cache shape mismatch {embeddings.shape} vs "
                    f"({len(body_texts)}, 384) — recomputing."
                )
            except Exception as exc:
                log.warning(f"Embedding cache load failed: {exc} — recomputing.")

        # --- Compute embeddings in batch -----------------------------------
        from src.features.text_features import get_embedding_model

        model = get_embedding_model()
        if model is None:
            log.warning("Embedding model unavailable — returning zero embeddings.")
            return np.zeros((len(body_texts), 384), dtype=np.float32)

        max_tokens = self.config.embedding_max_tokens
        truncated: List[str] = []
        for text in body_texts:
            words = text.split()
            truncated.append(
                " ".join(words[:max_tokens]) if len(words) > max_tokens else (text or " ")
            )

        log.info(
            f"Computing batch embeddings for {len(truncated):,} emails "
            "(cache MISS) ..."
        )
        try:
            embeddings = model.encode(
                truncated,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=4096,       # larger batches keep GPU fully fed
                normalize_embeddings=False,
            ).astype(np.float32)
        except Exception as exc:
            log.error(f"Batch encoding failed: {exc} — using zero embeddings.")
            return np.zeros((len(body_texts), 384), dtype=np.float32)

        try:
            np.save(str(cache_file), embeddings)
            log.info(f"Embedding cache saved: {cache_file.name}")
        except Exception as exc:
            log.warning(f"Could not save embedding cache: {exc}")

        return embeddings

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Transform a DataFrame of emails into a feature matrix.

        Embeddings for the entire batch are computed (or loaded from cache)
        before the per-email loop, dramatically reducing CPU time for large
        datasets where each model.encode() call would be paid per row.

        Args:
            df: DataFrame with 'raw_email' column.

        Returns:
            Tuple of (X: np.ndarray shape [n_samples, n_features], feature_names).
        """
        n = len(df)
        log.info(f"Transforming {n:,} emails ...")

        # Reuse pre-parsed results cached by fit() (same DataFrame in fit_transform),
        # otherwise parse in parallel now.
        if self._pre_parsed_cache is not None and len(self._pre_parsed_cache) == n:
            log.info("Reusing pre-parsed email cache from fit() — skipping re-parse.")
            pre_parsed = self._pre_parsed_cache
            self._pre_parsed_cache = None  # consume so we don't hold memory
        else:
            raw_emails = df["raw_email"].tolist()
            pre_parsed = self._parse_all_parallel(raw_emails)

        # Extract body texts directly from pre-parsed dicts (no extra parsing).
        body_texts = [
            p.get("body_text", "") or p.get("body_html", "") or ""
            for p in pre_parsed
        ]

        # Pre-compute all embeddings in one batch (uses cache when available).
        precomputed_embeddings = self._get_cached_embeddings(body_texts)

        # ── Batch TF-IDF (1 vectorizer call instead of N calls) ──────────────
        if self.use_tfidf and self._tfidf_vectorizer is not None:
            log.info(f"Batch TF-IDF transform for {n:,} emails ...")
            tfidf_batch = (
                self._tfidf_vectorizer.transform(body_texts).toarray().astype(np.float32)
            )
            tfidf_names = [
                f"tfidf_{fn}"
                for fn in self._tfidf_vectorizer.get_feature_names_out()
            ]
        else:
            tfidf_batch = np.zeros((n, 0), dtype=np.float32)
            tfidf_names = []

        # ── Sequential per-email feature extraction ──────────────────────────
        # Pure-Python code (BS4, regex, dict ops) holds the GIL, making
        # parallel approaches slower due to context-switching overhead.
        # 8 chunks × sequential processing gives clean 100% CPU utilisation.
        n_jobs = min(os.cpu_count() or 4, 8)
        chunk_size = max(1, (n + n_jobs - 1) // n_jobs)
        log.info(
            f"Extracting features for {n:,} emails "
            f"(sequential, {n_jobs} chunks of ~{chunk_size:,}) ..."
        )
        chunk_args = [
            (
                pre_parsed[start : min(start + chunk_size, n)],
                precomputed_embeddings[start : min(start + chunk_size, n)]
                if precomputed_embeddings is not None
                else None,
                self.use_network,
                self.config,
                self.use_intelligence_apis,
                self.use_gemini,
            )
            for start in range(0, n, chunk_size)
        ]
        chunk_results = [_extract_chunk_matrix(args) for args in chunk_args]

        # ── Assemble: stack chunk matrices, insert TF-IDF columns ────────────
        partial_matrices = [mat for mat, _ in chunk_results if mat.shape[0] > 0]
        partial_names = next((names for _, names in chunk_results if names), [])

        if not partial_matrices:
            return np.empty((0, 0)), []

        partial_matrix = np.vstack(partial_matrices)

        # Find TF-IDF insertion point (right before intel_ features).
        tfidf_insert_pos: Optional[int] = None
        if partial_names:
            try:
                tfidf_insert_pos = next(
                    j for j, nm in enumerate(partial_names) if nm.startswith("intel_")
                )
            except StopIteration:
                tfidf_insert_pos = len(partial_names)
            self._feature_names = (
                list(partial_names[:tfidf_insert_pos])
                + tfidf_names
                + list(partial_names[tfidf_insert_pos:])
            )
        else:
            tfidf_insert_pos = partial_matrix.shape[1]
            self._feature_names = tfidf_names

        if tfidf_batch.shape[1] > 0 and tfidf_insert_pos is not None:
            X = np.concatenate([
                partial_matrix[:, :tfidf_insert_pos],
                tfidf_batch,
                partial_matrix[:, tfidf_insert_pos:],
            ], axis=1)
        else:
            X = partial_matrix

        log.info(f"Feature matrix shape: {X.shape}")
        return X, self._feature_names

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Fit then transform — convenience method for training pipeline."""
        self.fit(df)
        return self.transform(df)

    def transform_single(self, raw_email: str) -> Tuple[np.ndarray, List[str]]:
        """Transform a single raw email string into a feature vector.

        Args:
            raw_email: Raw email content (headers + body) as a string.

        Returns:
            Tuple of (feature_vector shape [1, n_features], feature_names).
        """
        # Apply the same 500KB cap used by _safe_parse during batch training.
        # Without this, very large emails (e.g. 590KB with base64 attachments)
        # produce larger body_text → different TF-IDF scores than seen at training.
        if len(raw_email) > _MAX_RAW_CHARS:
            raw_email = raw_email[:_MAX_RAW_CHARS]
        vec, names = self._extract_single_email(raw_email)
        return vec.reshape(1, -1), names

    def save(self, path: str) -> None:
        """Serialise the fitted pipeline to disk.

        Args:
            path: File path for joblib serialisation (.pkl extension recommended).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "tfidf_vectorizer": self._tfidf_vectorizer,
            "feature_names": self._feature_names,
            "config": self.config,
            "flags": {
                "use_network": self.use_network,
                "use_intelligence_apis": self.use_intelligence_apis,
                "use_gemini": self.use_gemini,
                "use_tfidf": self.use_tfidf,
            },
        }, path)
        log.info(f"FeaturePipeline saved to '{path}'")

    @classmethod
    def load(cls, path: str) -> "FeaturePipeline":
        """Load a serialised pipeline from disk.

        Args:
            path: Path to the saved pipeline .pkl file.

        Returns:
            Fitted FeaturePipeline instance.
        """
        data = joblib.load(path)
        flags = data.get("flags", {})
        instance = cls(
            config=data.get("config", DEFAULT_CONFIG),
            use_network=flags.get("use_network", True),
            use_intelligence_apis=flags.get("use_intelligence_apis", True),
            use_gemini=flags.get("use_gemini", False),
            use_tfidf=flags.get("use_tfidf", True),
        )
        instance._tfidf_vectorizer = data.get("tfidf_vectorizer")
        instance._feature_names = data.get("feature_names", [])
        instance._is_fitted = True
        log.info(f"FeaturePipeline loaded from '{path}'")
        return instance

    # -----------------------------------------------------------------------
    # Internal extraction logic
    # -----------------------------------------------------------------------

    def _extract_features_from_parsed(
        self,
        parsed: dict,
        precomputed_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract all features from a pre-parsed email dict (avoids re-parsing)."""
        return self._extract_single_email(
            "", precomputed_embedding=precomputed_embedding, _pre_parsed=parsed
        )

    def _extract_single_email(
        self,
        raw_email: str,
        precomputed_embedding: Optional[np.ndarray] = None,
        _pre_parsed: Optional[dict] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract all features from a single raw email string.

        Args:
            raw_email: Raw email content (headers + body).
            precomputed_embedding: Optional 384-dim embedding pre-computed by
                _get_cached_embeddings(). When None (single-email inference),
                the embedding is computed on-the-fly inside extract_text_features.
            _pre_parsed: Pre-parsed dict to skip re-parsing (used by transform()).
        """
        parsed = _pre_parsed if _pre_parsed is not None else parse_eml_string(raw_email)
        feature_vector: List[float] = []
        feature_names: List[str] = []

        # ---- Module 1: Header features (12) --------------------------------
        try:
            header_feats = extract_header_features(parsed)
        except Exception as exc:
            log.warning(f"Header features failed: {exc}")
            header_feats = {k: 0 for k in [
                "from_reply_to_mismatch", "from_return_path_mismatch",
                "reply_to_freemail", "received_hop_count", "received_geo_anomaly",
                "spf_result", "dkim_result", "dmarc_result",
                "message_id_suspicious", "timezone_mismatch",
                "x_mailer_suspicious", "header_injection_attempt",
            ]}
        for k, v in header_feats.items():
            feature_vector.append(float(v))
            feature_names.append(f"hdr_{k}")

        # ---- Module 2: URL features (~29) ----------------------------------
        urls = parsed.get("urls", [])
        try:
            if self.use_network:
                url_feats = extract_url_features_with_network(urls, self.config)
            else:
                url_feats = extract_url_features(urls, self.config)
        except Exception as exc:
            log.warning(f"URL features failed: {exc}")
            from src.features.url_features import _default_url_features
            url_feats = _default_url_features()
        for k, v in url_feats.items():
            feature_vector.append(float(v) if not isinstance(v, str) else 0.0)
            feature_names.append(f"url_{k}")

        # ---- Module 3: HTML features (11) ----------------------------------
        try:
            html_feats = extract_html_features(parsed.get("body_html", ""))
        except Exception as exc:
            log.warning(f"HTML features failed: {exc}")
            from src.features.html_features import _default_html_features
            html_feats = _default_html_features()
        for k, v in html_feats.items():
            feature_vector.append(float(v))
            feature_names.append(f"html_{k}")

        # ---- Module 4: Text features (8 + 384 embeddings) -----------------
        try:
            text_vec, text_names = extract_text_features(
                body_text=parsed.get("body_text", ""),
                subject=parsed.get("subject", ""),
                config=self.config,
                tfidf_vectorizer=self._tfidf_vectorizer,
                precomputed_embedding=precomputed_embedding,
            )
            feature_vector.extend(text_vec.tolist())
            feature_names.extend(text_names)
        except Exception as exc:
            log.warning(f"Text features failed: {exc}")
            feature_vector.extend([0.0] * (8 + 384))
            feature_names.extend([f"txt_pad_{i}" for i in range(8 + 384)])

        # ---- Module 5: TF-IDF (500 dims) -----------------------------------
        if self.use_tfidf and self._tfidf_vectorizer is not None:
            try:
                body_text = parsed.get("body_text", "") or parsed.get("body_html", "")
                tfidf_vec = self._tfidf_vectorizer.transform([body_text]).toarray()[0]
                feature_vector.extend(tfidf_vec.tolist())
                feature_names.extend(
                    [f"tfidf_{fn}" for fn in self._tfidf_vectorizer.get_feature_names_out()]
                )
            except Exception as exc:
                log.warning(f"TF-IDF transform failed: {exc}")
                n = self.config.tfidf_max_features
                feature_vector.extend([0.0] * n)
                feature_names.extend([f"tfidf_pad_{i}" for i in range(n)])

        # ---- Module 6: Intelligence API features (13) ----------------------
        if self.use_intelligence_apis:
            try:
                sender_ip = self._extract_sender_ip(parsed.get("received_headers", []))
                intel_feats = enrich_email_with_intelligence(urls, sender_ip)
            except Exception as exc:
                log.warning(f"Intelligence API features failed: {exc}")
                intel_feats = get_default_intelligence_features()
        else:
            intel_feats = get_default_intelligence_features()

        for k, v in intel_feats.items():
            if isinstance(v, (int, float)):
                feature_vector.append(float(v))
                feature_names.append(f"intel_{k}")

        # ---- Module 7: ChatGPT AI feature (2) ------------------------------
        if self.use_gemini:
            try:
                from src.features.openai_analyzer import get_openai_ml_feature
                gemini_feats = get_openai_ml_feature(
                    subject=parsed.get("subject", ""),
                    from_address=parsed.get("from_address", ""),
                    body_text=parsed.get("body_text", ""),
                    urls=urls,
                )
            except Exception as exc:
                log.warning(f"ChatGPT features failed: {exc}")
                gemini_feats = {"gemini_is_phishing": -1, "gemini_confidence": -1.0}
            for k, v in gemini_feats.items():
                feature_vector.append(float(v))
                feature_names.append(f"ai_{k}")

        return np.array(feature_vector, dtype=np.float32), feature_names

    def _parse_all_parallel(self, raw_emails: List[str]) -> List[dict]:
        """Parse all raw email strings in parallel using joblib multiprocessing.

        Uses loky backend (process-based) to bypass the GIL for CPU-bound
        Python email parsing. Results are cached in self._pre_parsed_cache.
        A tqdm progress bar tracks per-email completion so long runs are visible.
        """
        from tqdm.auto import tqdm as _tqdm
        n = len(raw_emails)
        n_jobs = min(os.cpu_count() or 4, 8)
        log.info(f"Parsing {n:,} emails in parallel (n_jobs={n_jobs}) ...")
        # batch_size=1: dispatch one email per worker slot so tqdm updates
        # immediately on each completion instead of waiting for a whole
        # auto-batch (which can be 50+ emails, making the bar appear frozen
        # whenever one slow HTML email blocks a batch).
        results = list(_tqdm(
            joblib.Parallel(
                n_jobs=n_jobs, return_as="generator",
                backend="loky", batch_size=1,
            )(joblib.delayed(_safe_parse)(raw) for raw in raw_emails),
            total=n,
            desc="  Parsing emails",
            unit="email",
            ncols=100,
            colour="cyan",
        ))
        return results

    def _extract_body_texts(self, df: pd.DataFrame) -> List[str]:
        """Extract body text from all emails for TF-IDF fitting."""
        raw_emails = df["raw_email"].tolist()
        pre_parsed = self._parse_all_parallel(raw_emails)
        return [
            p.get("body_text", "") or p.get("body_html", "") or ""
            for p in pre_parsed
        ]

    @staticmethod
    def _extract_sender_ip(received_headers: List[str]) -> Optional[str]:
        """Extract the originating sender IP from Received: headers."""
        for header in reversed(received_headers):   # Outermost = closest to origin
            matches = _IP_RE.findall(header)
            for ip in matches:
                # Skip private/loopback addresses
                if not (ip.startswith("10.") or ip.startswith("192.168.")
                        or ip.startswith("127.") or ip == "0.0.0.0"):
                    return ip
        return None
