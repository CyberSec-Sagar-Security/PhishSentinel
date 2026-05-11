"""
PhishLens OpenAI ChatGPT Analysis Module — gpt-4.1-mini.

Sends the analyst the full picture before asking for a verdict:
  * All raw email headers (not just From/Subject)
  * Complete extracted IOC list (URLs, IPs, domains, email addresses,
    attachment hashes, phone numbers)
  * Threat-intelligence verdicts from every integrated tool
    (VirusTotal, Google Safe Browsing, AbuseIPDB, URLhaus, URLScan, IPQS)
  * PhishLens ML verdict + probability
  * Email body (up to 3000 chars)

Returns the same result schema as the old gemini_analyzer so the Streamlit
UI can display it unchanged.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from src.utils.logger import get_logger

log = get_logger(__name__)

_OPENAI_MODEL = "gpt-4.1-mini"

_OPENAI_CLIENT = None


def _get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        log.warning("OPENAI_API_KEY not set — OpenAI analysis disabled.")
        return None
    try:
        from openai import OpenAI
        _OPENAI_CLIENT = OpenAI(api_key=api_key)
        log.info(f"OpenAI client initialised with model: {_OPENAI_MODEL}")
    except ImportError:
        log.error("openai not installed. Run: pip install openai")
        _OPENAI_CLIENT = None
    except Exception as exc:
        log.error(f"Failed to initialise OpenAI client: {exc}")
        _OPENAI_CLIENT = None
    return _OPENAI_CLIENT


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a senior cybersecurity analyst specialising in phishing email "
    "forensics. You will be given the full email headers, extracted IOCs, "
    "threat-intelligence verdicts from multiple reputation services, and the "
    "output of a machine-learning phishing classifier. Your task is to "
    "synthesise all of this information and return a single structured "
    "JSON verdict. Do NOT return markdown — return raw JSON only."
)


def _build_headers_section(all_headers: Dict) -> str:
    if not all_headers:
        return "  (no headers extracted)"
    lines = []
    for k, v in all_headers.items():
        v_str = str(v).replace("\n", " ").replace("\r", "")[:300]
        lines.append(f"  {k}: {v_str}")
    return "\n".join(lines[:60])


def _build_ioc_section(iocs: Dict) -> str:
    if not iocs:
        return "  None"
    lines = []

    def _add(label, items):
        for i in (items if isinstance(items, list) else [items]):
            if i:
                lines.append(f"  [{label}] {i}")

    _add("URL",         iocs.get("urls", []))
    _add("IP",          iocs.get("ip_addresses", []))
    _add("DOMAIN",      iocs.get("domains", []))
    _add("EMAIL",       iocs.get("email_addresses", []))
    for h in iocs.get("attachment_hashes", []):
        if h.get("md5"):
            lines.append(f"  [HASH_MD5] {h['md5']}")
        if h.get("sha256"):
            lines.append(f"  [HASH_SHA256] {h['sha256']}")
    _add("PHONE",       iocs.get("phone_numbers", []))
    return "\n".join(lines) or "  None"


def _build_ti_section(intel: Dict) -> str:
    if not intel or "_error" in intel:
        return "  Threat Intelligence was not queried."
    lines = []
    _NA = -1

    vt_mal = intel.get("vt_malicious", _NA)
    vt_sus = intel.get("vt_suspicious", _NA)
    if vt_mal != _NA:
        verdict = "MALICIOUS" if vt_mal > 0 else ("SUSPICIOUS" if vt_sus > 0 else "CLEAN")
        lines.append(f"  VirusTotal: {verdict} (malicious={vt_mal}, suspicious={vt_sus}, reputation={intel.get('vt_reputation', 0)})")

    gsb = intel.get("gsb_is_flagged", _NA)
    if gsb != _NA:
        lines.append(f"  Google Safe Browsing: {'FLAGGED' if gsb == 1 else 'CLEAN'} (matches={intel.get('gsb_threat_count', 0)})")

    uh = intel.get("urlhaus_threat", _NA)
    if uh != _NA:
        lines.append(f"  URLhaus: {'MALICIOUS' if uh == 1 else 'CLEAN'}")

    us = intel.get("urlscan_malicious", _NA)
    if us != _NA:
        brand = " [brand impersonation]" if intel.get("urlscan_brand_impersonated") == 1 else ""
        lines.append(f"  URLScan.io: {'MALICIOUS' if us == 1 else 'CLEAN'}{brand}")

    abuse = intel.get("abuse_confidence_score", _NA)
    if abuse != _NA and "_sender_ip" in intel:
        lines.append(f"  AbuseIPDB: IP={intel['_sender_ip']} score={abuse}% ISP={intel.get('abuse_isp', '')} country={intel.get('abuse_country_code', '')}")

    ipqs_em = intel.get("_ipqs_email", {})
    if ipqs_em and not ipqs_em.get("_ipqs_error"):
        em_score = ipqs_em.get("ipqs_email_fraud_score", _NA)
        if em_score != _NA:
            flags = []
            if ipqs_em.get("ipqs_email_disposable") == 1:
                flags.append("disposable")
            if ipqs_em.get("ipqs_email_spam_trap") == 1:
                flags.append("spam-trap")
            if ipqs_em.get("ipqs_email_recent_abuse") == 1:
                flags.append("recent-abuse")
            lines.append(f"  IPQS Email: fraud_score={em_score}/100 flags=[{', '.join(flags)}]")

    for _i in range(5):
        ipqs_u = intel.get(f"_ipqs_url_{_i}", {})
        url_val = intel.get(f"_ipqs_url_{_i}_url", "")
        if ipqs_u and not ipqs_u.get("_ipqs_error") and url_val:
            u_score = ipqs_u.get("ipqs_url_risk_score", _NA)
            if u_score != _NA:
                uflags = []
                if ipqs_u.get("ipqs_url_phishing") == 1:
                    uflags.append("phishing")
                if ipqs_u.get("ipqs_url_malware") == 1:
                    uflags.append("malware")
                if ipqs_u.get("ipqs_url_suspicious") == 1:
                    uflags.append("suspicious")
                lines.append(f"  IPQS URL ({url_val[:80]}): risk={u_score}/100 flags=[{', '.join(uflags)}]")

    ipqs_ip = intel.get("_ipqs_ip", {})
    if ipqs_ip and not ipqs_ip.get("_ipqs_error"):
        ip_score = ipqs_ip.get("ipqs_ip_fraud_score", _NA)
        if ip_score != _NA:
            iflags = []
            if ipqs_ip.get("ipqs_ip_proxy") == 1:
                iflags.append("proxy")
            if ipqs_ip.get("ipqs_ip_vpn") == 1:
                iflags.append("vpn")
            if ipqs_ip.get("ipqs_ip_tor") == 1:
                iflags.append("tor")
            lines.append(f"  IPQS IP: fraud_score={ip_score}/100 flags=[{', '.join(iflags)}]")

    return "\n".join(lines) or "  No data returned from threat intelligence APIs."


def _build_user_prompt(
    subject: str,
    from_address: str,
    body_text: str,
    urls: List[str],
    all_headers: Dict,
    iocs: Dict,
    intelligence_result: Dict,
    ml_verdict: str,
    ml_probability: float,
) -> str:
    headers_section = _build_headers_section(all_headers)
    ioc_section = _build_ioc_section(iocs)
    ti_section = _build_ti_section(intelligence_result)
    body_truncated = (body_text or "")[:3000].strip()

    return f"""=== PHISHLENS ANALYSIS REQUEST ===

--- MACHINE LEARNING VERDICT ---
ML Model Verdict    : {ml_verdict}
Phishing Probability: {ml_probability:.1%}

--- RAW EMAIL HEADERS ---
{headers_section}

--- EMAIL BODY (first 3000 chars) ---
{body_truncated if body_truncated else "(empty body)"}

--- EXTRACTED IOCs ---
{ioc_section}

--- THREAT INTELLIGENCE VERDICTS ---
{ti_section}

=== TASK ===
Using ALL of the above data — headers, IOCs, TI verdicts, and ML score — provide
a comprehensive phishing analysis. If TI tools flag any IOC as malicious or the
ML model flags phishing, weigh that heavily. If all TI tools return CLEAN and the
ML model says LEGITIMATE, lean toward safe. Do not solely rely on the email body;
let the TI verdicts and ML verdict guide your confidence.

Return ONLY this JSON (no markdown, no code fences):
{{
  "is_phishing": true or false,
  "confidence": 0.0 to 1.0,
  "phishing_signals": ["list of specific detected phishing indicators"],
  "impersonated_brand": "brand name or null",
  "social_engineering_techniques": ["..."],
  "risk_level": "LOW" or "MEDIUM" or "HIGH" or "CRITICAL",
  "explanation": "3-5 sentence analyst-grade explanation referencing headers, IOCs, and TI data",
  "recommended_action": "DELETE" or "QUARANTINE" or "MONITOR" or "SAFE",
  "ioc_verdicts": {{
    "url_0": "MALICIOUS / SUSPICIOUS / CLEAN / UNKNOWN",
    "url_1": "MALICIOUS / SUSPICIOUS / CLEAN / UNKNOWN",
    "sender_ip": "MALICIOUS / SUSPICIOUS / CLEAN / UNKNOWN",
    "sender_email": "MALICIOUS / SUSPICIOUS / CLEAN / UNKNOWN"
  }}
}}"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_email_with_openai(
    subject: str,
    from_address: str,
    body_text: str,
    urls: List[str],
    all_headers: Optional[Dict] = None,
    iocs: Optional[Dict] = None,
    intelligence_result: Optional[Dict] = None,
    ml_verdict: str = "UNCERTAIN",
    ml_probability: float = 0.5,
    timeout: int = 30,
) -> Dict:
    """Analyse an email with ChatGPT gpt-4.1-mini using full forensic context."""
    client = _get_openai_client()
    if client is None:
        return _default_openai_features()

    try:
        user_prompt = _build_user_prompt(
            subject=subject or "No subject",
            from_address=from_address or "Unknown",
            body_text=body_text or "",
            urls=urls or [],
            all_headers=all_headers or {},
            iocs=iocs or {},
            intelligence_result=intelligence_result or {},
            ml_verdict=ml_verdict,
            ml_probability=ml_probability,
        )

        response = client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=900,
            timeout=timeout,
            response_format={"type": "json_object"},
        )

        raw_text = response.choices[0].message.content.strip()
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
            "gemini_ioc_verdicts": result.get("ioc_verdicts", {}),
            "_ai_provider": f"ChatGPT {_OPENAI_MODEL}",
        }

    except json.JSONDecodeError as exc:
        log.warning(f"OpenAI returned invalid JSON: {exc}")
    except Exception as exc:
        log.warning(f"OpenAI analysis error: {exc}")

    return _default_openai_features()


def get_openai_ml_feature(
    subject: str,
    from_address: str,
    body_text: str,
    urls: list,
) -> Dict:
    """Extract only the numeric OpenAI features for use in the ML pipeline."""
    result = analyse_email_with_openai(subject, from_address, body_text, urls)
    return {
        "gemini_is_phishing": result.get("gemini_is_phishing", -1),
        "gemini_confidence": result.get("gemini_confidence", -1.0),
    }


def _default_openai_features() -> Dict:
    return {
        "gemini_is_phishing": -1,
        "gemini_confidence": -1.0,
        "gemini_risk_level": "UNKNOWN",
        "gemini_impersonated_brand": None,
        "gemini_phishing_signals": [],
        "gemini_social_engineering": [],
        "gemini_explanation": "OpenAI analysis unavailable.",
        "gemini_recommended_action": "MONITOR",
        "gemini_ioc_verdicts": {},
        "_ai_provider": f"ChatGPT {_OPENAI_MODEL} (unavailable)",
    }
