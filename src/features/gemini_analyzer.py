"""
PhishLens ChatGPT AI Analysis Module.

Uses the OpenAI ChatGPT API (gpt-4o-mini) to perform AI-powered phishing analysis as an
additional high-confidence layer on top of the ML pipeline.

ChatGPT analyses email content for:
  - Phishing indicators (linguistic + contextual)
  - Impersonated brand detection
  - Social engineering techniques used
  - Confidence score (0.0–1.0)
  - Plain-English explanation for the Streamlit UI

Security rationale: Large language models trained on vast security corpora
can detect sophisticated social engineering that statistical ML models miss —
particularly spear-phishing emails crafted to look legitimate to a specific
target. ChatGPT's analysis provides a second independent signal and generates
human-readable explanations that analysts can act on immediately.

Note: OpenAI API has rate limits. This module is used for:
  1. Real-time Streamlit analysis of single emails
  2. Batch enrichment of high-uncertainty predictions (ML confidence 0.4–0.6)
  NOT for training data feature extraction (too slow / cost-limited for 135k emails).
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

from src.utils.logger import get_logger

log = get_logger(__name__)

_GEMINI_MODEL = "gemini-2.0-flash"   # Fast, cost-efficient model for analysis

# Lazy-loaded Gemini client
_GEMINI_CLIENT = None


def _get_gemini_client():
    """Lazily initialise the Gemini generative AI client (google-genai SDK)."""
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is not None:
        return _GEMINI_CLIENT
    # Reload key each call in case .env was loaded after module import
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
    log.warning("OPENAI_API_KEY not set — ChatGPT analysis disabled.")
        return None
    try:
        from google import genai
        _GEMINI_CLIENT = genai.Client(api_key=api_key)
        log.info(f"ChatGPT client initialised with model: {_GEMINI_MODEL}")
    except ImportError:
        log.error("google-genai not installed. Run: pip install google-genai")
        _GEMINI_CLIENT = None
    except Exception as exc:
        log.error(f"Failed to initialise ChatGPT client: {exc}")
        _GEMINI_CLIENT = None
    return _GEMINI_CLIENT


# ---------------------------------------------------------------------------
# Phishing analysis prompt
# ---------------------------------------------------------------------------

_ANALYSIS_PROMPT_TEMPLATE = """You are a cybersecurity analyst specialising in phishing email detection.
Analyse the following email and return a JSON response ONLY (no markdown, no explanation outside the JSON).

Email subject: {subject}
Email from: {from_address}
Email body (truncated to 1000 chars): {body_truncated}
URLs found in email: {urls_list}

Return this exact JSON structure:
{{
  "is_phishing": true or false,
  "confidence": 0.0 to 1.0 (1.0 = certain phishing),
  "phishing_signals": ["list of detected phishing indicators"],
  "impersonated_brand": "brand name or null",
  "social_engineering_techniques": ["list of techniques used, e.g. urgency, authority, fear"],
  "risk_level": "LOW" or "MEDIUM" or "HIGH" or "CRITICAL",
  "explanation": "2-3 sentence plain English explanation for a security analyst",
  "recommended_action": "DELETE" or "QUARANTINE" or "MONITOR" or "SAFE"
}}

Be precise. Only flag as phishing if there are clear indicators. Legitimate marketing emails are NOT phishing."""


def analyse_email_with_gemini(
    subject: str,
    from_address: str,
    body_text: str,
    urls: list,
    timeout: int = 10,
) -> Dict:
    """Analyse an email with ChatGPT (OpenAI) for phishing indicators.

    Args:
        subject: Email subject line.
        from_address: Sender address.
        body_text: Email body text.
        urls: List of URLs found in the email.
        timeout: Maximum seconds to wait for ChatGPT response.

    Returns:
        Dict with ChatGPT analysis results. Returns safe defaults on failure.
    """
    client = _get_gemini_client()
    if client is None:
        return _default_gemini_features()

    try:
        # Truncate body to 1000 chars to stay within token limits and control cost
        body_truncated = (body_text or "")[:1000].strip()
        urls_list = str(urls[:5]) if urls else "none"

        prompt = _ANALYSIS_PROMPT_TEMPLATE.format(
            subject=subject or "No subject",
            from_address=from_address or "Unknown",
            body_truncated=body_truncated or "Empty body",
            urls_list=urls_list,
        )

        from google.genai import types as _genai_types
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=prompt,
            config=_genai_types.GenerateContentConfig(
                temperature=0.1,        # Low temperature for deterministic analysis
                max_output_tokens=500,
            ),
        )

        raw_text = response.text.strip()

        # Strip markdown code blocks if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:].strip()

        result = json.loads(raw_text)

        return {
            "gemini_is_phishing": int(result.get("is_phishing", False)),
            "gemini_confidence": float(result.get("confidence", 0.0)),
            "gemini_risk_level": result.get("risk_level", "LOW"),
            "gemini_impersonated_brand": result.get("impersonated_brand", None),
            "gemini_phishing_signals": result.get("phishing_signals", []),
            "gemini_social_engineering": result.get("social_engineering_techniques", []),
            "gemini_explanation": result.get("explanation", ""),
            "gemini_recommended_action": result.get("recommended_action", "MONITOR"),
        }

    except json.JSONDecodeError as exc:
        log.warning(f"ChatGPT returned invalid JSON: {exc}")
    except Exception as exc:
        log.warning(f"ChatGPT analysis error: {exc}")

    return _default_gemini_features()


def get_gemini_ml_feature(
    subject: str,
    from_address: str,
    body_text: str,
    urls: list,
) -> Dict:
    """Extract only the numeric ChatGPT features suitable for ML pipeline inclusion.

    This is a thin wrapper around analyse_email_with_gemini that returns
    only the numeric features (is_phishing and confidence) for use in the
    feature pipeline. The full analysis dict is used by the Streamlit UI.

    Args:
        subject: Email subject line.
        from_address: Sender address.
        body_text: Email body text.
        urls: List of URLs.

    Returns:
        Dict with gemini_is_phishing and gemini_confidence as numeric features.
    """
    result = analyse_email_with_gemini(subject, from_address, body_text, urls)
    return {
        "gemini_is_phishing": result.get("gemini_is_phishing", -1),
        "gemini_confidence": result.get("gemini_confidence", -1.0),
    }


def _default_gemini_features() -> Dict:
    """Safe default ChatGPT analysis result when API is unavailable."""
    return {
        "gemini_is_phishing": -1,
        "gemini_confidence": -1.0,
        "gemini_risk_level": "UNKNOWN",
        "gemini_impersonated_brand": None,
        "gemini_phishing_signals": [],
        "gemini_social_engineering": [],
        "gemini_explanation": "ChatGPT AI analysis unavailable.",
        "gemini_recommended_action": "MONITOR",
    }
