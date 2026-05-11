"""
PhishLens Text & NLP Feature Module.

Extracts TF-IDF sparse features, urgency/social-engineering scores,
semantic embeddings (sentence-transformers), and subject-line features.

Security rationale: Phishing emails are engineered to create fear and urgency.
NLP signals — particularly semantic embeddings from pre-trained transformers —
capture the latent 'threat context' of an email that bag-of-words methods miss.
The 384-dimensional all-MiniLM-L6-v2 embedding is the single highest-impact
feature group, representing deep semantic meaning that cannot be easily evaded
by paraphrasing or synonym substitution.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.config import DEFAULT_CONFIG, URGENCY_PHRASES
from src.utils.logger import get_logger

log = get_logger(__name__)

# Sentence-transformers lazy loading (80MB model — load once)
_EMBEDDING_MODEL = None
_DEVICE = "cpu"  # Set to 'cuda' at load time if GPU is available


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load and cache the sentence-transformers embedding model.

    Security rationale: The model is loaded once at module level and reused
    across all emails. This prevents the 80MB model from being loaded per email,
    which would make batch processing impractical.

    Device selection: Checks torch.cuda.is_available() at load time. Falls back
    to CPU gracefully on machines without a GPU — no code changes needed.

    Args:
        model_name: Hugging Face model identifier.

    Returns:
        SentenceTransformer model instance.
    """
    global _EMBEDDING_MODEL, _DEVICE
    if _EMBEDDING_MODEL is None:
        try:
            import torch
            _DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
            log.info(f"Embedding device: {_DEVICE.upper()} "
                     f"({'GPU: ' + torch.cuda.get_device_name(0) if 'cuda' in _DEVICE else 'CPU-only build'})")
            from sentence_transformers import SentenceTransformer
            log.info(f"Loading sentence-transformer model: {model_name}")
            _EMBEDDING_MODEL = SentenceTransformer(model_name, device=_DEVICE)
            # Convert to fp16 so CUDA Tensor Cores are engaged on every
            # matrix-multiply — RTX Ada has dedicated fp16 hardware giving
            # ~2x throughput vs fp32 with negligible quality loss at 384-dim.
            if "cuda" in _DEVICE:
                import torch as _t
                _EMBEDDING_MODEL = _EMBEDDING_MODEL.half()
                log.info("Embedding model converted to fp16 (Tensor Core acceleration).")
            log.info("Embedding model loaded successfully.")
        except Exception as exc:
            log.error(f"Failed to load embedding model: {exc}")
            _EMBEDDING_MODEL = None
    return _EMBEDDING_MODEL


def extract_text_features(
    body_text: str,
    subject: str,
    config=DEFAULT_CONFIG,
    tfidf_vectorizer=None,
    fit_tfidf: bool = False,
    precomputed_embedding: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Extract all text-based features from email body and subject.

    Args:
        body_text: Plain text body of the email.
        subject: Email subject line.
        config: PhishLensConfig instance.
        tfidf_vectorizer: Fitted TfidfVectorizer (None during fit phase).
        fit_tfidf: If True, returns raw text for TF-IDF fitting externally.
        precomputed_embedding: Optional pre-computed 384-dim embedding array
            from the batch cache. When provided, model.encode() is skipped,
            saving ~200ms per email in batch mode.

    Returns:
        Tuple of (feature_vector: np.ndarray, feature_names: List[str]).
        feature_vector contains: urgency score, subject features,
        and semantic embedding (384 dims).
    """
    features: List[float] = []
    feature_names: List[str] = []

    # ---- Urgency / Social Engineering Score --------------------------------
    urgency_score, urgency_count = _compute_urgency_score(body_text, config.urgency_phrases)
    features.append(urgency_score)
    features.append(float(urgency_count))
    feature_names.extend(["urgency_score_normalised", "urgency_phrase_count"])

    # ---- Subject line features ---------------------------------------------
    subject_feats, subject_names = _extract_subject_features(subject, config.brand_list)
    features.extend(subject_feats)
    feature_names.extend(subject_names)

    # ---- Semantic Embedding (384 dims) ------------------------------------
    # Security rationale: If a pre-computed batch embedding is supplied (from
    # the pipeline's embedding cache), we use it directly — this skips the
    # 80MB model call and makes batch transforms ~100× faster on CPU.
    if precomputed_embedding is not None and len(precomputed_embedding) == 384:
        embedding = precomputed_embedding.astype(np.float32)
    else:
        model = get_embedding_model(config.embedding_model)
        if model is not None:
            embedding = _compute_embedding(body_text, model, config.embedding_max_tokens)
        else:
            log.warning("Embedding model unavailable — using zeros for embedding features.")
            embedding = np.zeros(384, dtype=np.float32)

    features.extend(embedding.tolist())
    feature_names.extend([f"embed_{i}" for i in range(len(embedding))])

    return np.array(features, dtype=np.float32), feature_names


def extract_tfidf_features(
    texts: List[str],
    vectorizer=None,
    config=DEFAULT_CONFIG,
    fit: bool = False,
):
    """Fit or transform texts using TF-IDF vectorizer.

    Args:
        texts: List of email body texts.
        vectorizer: Fitted TfidfVectorizer or None if fitting from scratch.
        config: PhishLensConfig instance.
        fit: If True, fits the vectorizer on provided texts.

    Returns:
        Tuple of (sparse_matrix, fitted_vectorizer, feature_names).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    if fit or vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=config.tfidf_max_features,
            ngram_range=config.tfidf_ngram_range,
            sublinear_tf=True,          # Log-scaled TF reduces impact of very frequent terms
            strip_accents="unicode",
            decode_error="replace",
            analyzer="word",
            min_df=2,                   # Ignore terms appearing in < 2 docs (noise reduction)
        )
        X = vectorizer.fit_transform(texts)
        log.info(
            f"TF-IDF fitted: {config.tfidf_max_features} features, "
            f"ngram_range={config.tfidf_ngram_range}"
        )
    else:
        X = vectorizer.transform(texts)

    feature_names = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]
    return X, vectorizer, feature_names


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_urgency_score(text: str, urgency_phrases: List[str]) -> Tuple[float, int]:
    """Compute normalised urgency/social-engineering score.

    Security rationale: Urgency creation is the primary psychological
    manipulation technique in phishing. 'Verify now or your account will be
    closed within 24 hours' — these phrases are statistically concentrated
    in phishing and rare in legitimate email. Normalising by word count
    prevents long legitimate emails from triggering false positives.

    Args:
        text: Email body text.
        urgency_phrases: List of phishing urgency phrases from config.

    Returns:
        Tuple of (normalised_score 0.0–1.0, raw_count).
    """
    if not text:
        return 0.0, 0
    text_lower = text.lower()
    count = sum(1 for phrase in urgency_phrases if phrase.lower() in text_lower)
    word_count = max(len(text.split()), 1)
    normalised = min(count / (word_count / 100), 1.0)   # Phrases per 100 words, capped at 1
    return normalised, count


def _extract_subject_features(subject: str, brand_list: List[str]) -> Tuple[List[float], List[str]]:
    """Extract features from the email subject line.

    Security rationale: Subject lines are crafted to provoke urgency and
    impersonate brands. All-caps words, excessive punctuation, spoofed
    RE:/FW: prefixes, and brand keywords are reliable phishing signals.

    Args:
        subject: Email subject string.
        brand_list: List of brand keywords to check.

    Returns:
        Tuple of (feature_values, feature_names).
    """
    features = []
    names = []

    subject = subject or ""

    # subject_length
    features.append(float(len(subject)))
    names.append("subject_length")

    # exclamation_count
    features.append(float(subject.count("!")))
    names.append("subject_exclamation_count")

    # question_mark_count (rarely legitimate in corporate subject lines)
    features.append(float(subject.count("?")))
    names.append("subject_question_count")

    # all_caps_word_ratio — "URGENT ACTION REQUIRED" pattern
    words = subject.split()
    caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / max(len(words), 1)
    features.append(caps_ratio)
    names.append("subject_caps_ratio")

    # spoofed_re_fw: RE: FW: prefix but it is actually a first-contact phish
    spoofed = int(
        bool(re.match(r"^(re:|fw:|fwd:)\s*(re:|fw:|fwd:)?\s*(re:|fw:|fwd:)?", subject, re.IGNORECASE))
    )
    features.append(float(spoofed))
    names.append("subject_spoofed_re_fw")

    # brand_in_subject: brand keyword found in subject line
    subj_lower = subject.lower()
    brand_in_subj = int(any(brand in subj_lower for brand in brand_list))
    features.append(float(brand_in_subj))
    names.append("subject_brand_keyword")

    # urgency_in_subject: urgency phrase in subject
    urgency_in_subj = int(
        any(phrase in subj_lower for phrase in ["urgent", "action required", "verify", "suspended", "alert"])
    )
    features.append(float(urgency_in_subj))
    names.append("subject_urgency_keyword")

    # subject_has_dollar_signs (prize/lottery phishing pattern)
    features.append(float(subject.count("$")))
    names.append("subject_dollar_count")

    return features, names


def _compute_embedding(
    text: str,
    model,
    max_tokens: int = 512,
) -> np.ndarray:
    """Encode email body text into a 384-dimensional semantic embedding.

    Security rationale: Semantic embeddings capture meaning beyond surface
    vocabulary. A phishing email that replaces all risk keywords with synonyms
    still has a recognisable semantic fingerprint: credential requests, urgency,
    impersonation of authority, financial threat. These patterns are encoded in
    the transformer's latent space and cannot be evaded by simple word substitution.

    Args:
        text: Email body text (first max_tokens words used).
        model: Loaded SentenceTransformer instance.
        max_tokens: Maximum token count before truncation.

    Returns:
        384-dimensional float32 numpy array.
    """
    if not text or not text.strip():
        return np.zeros(384, dtype=np.float32)

    # Truncate to max_tokens words (approximate — transformer handles exact token count)
    words = text.split()
    if len(words) > max_tokens:
        text = " ".join(words[:max_tokens])

    try:
        embedding = model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=256,
            device=_DEVICE,
        )
        return embedding.astype(np.float32)
    except Exception as exc:
        log.warning(f"Embedding encode error: {exc}")
        return np.zeros(384, dtype=np.float32)
