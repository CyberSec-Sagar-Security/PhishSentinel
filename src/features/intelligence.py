"""
PhishLens Threat Intelligence API Module.

Integrates four external threat intelligence APIs to enrich URL and IP
analysis with community-sourced reputation data:
  1. VirusTotal — 70+ AV engine votes on URL/domain maliciousness
  2. Google Safe Browsing — Chrome-level phishing/malware database
  3. AbuseIPDB — Sender IP reputation from community reports
  4. URLScan.io — Live page scan with visual phishing detection
  5. URLhaus (no key) — abuse.ch malicious URL database

All API calls use strict timeouts, fallback to safe defaults on failure,
and are designed to be called asynchronously for batch processing.

Security rationale: Combining multiple independent threat feeds using
different detection methodologies (ML-based, signature-based, behavioural)
creates a consensus signal that is extremely hard for attackers to evade —
they would need to remain undetected across all five intelligence sources
simultaneously.
"""

from __future__ import annotations

import asyncio
import base64
import os
from typing import Dict, List, Optional

import aiohttp
import requests

from src.utils.config import API_ENDPOINTS, NETWORK_TIMEOUT
from src.utils.logger import get_logger

log = get_logger(__name__)

# API keys loaded lazily at call time (dotenv may be loaded after module import)
def _vt_key()():    return os.getenv("VIRUSTOTAL_API_KEY", "")
def _gsb_key()():   return os.getenv("GOOGLE_SAFE_BROWSING_API_KEY", "")
def _abuse_key(): return os.getenv("ABUSEIPDB_API_KEY", "")
def _urlscan_key()(): return os.getenv("URLSCAN_API_KEY", "")
def _ipqs_key()():  return os.getenv("IPQS_API_KEY", "")

_IPQS_EMAIL_URL = "https://ipqualityscore.com/api/json/email/{key}/{email}"
_IPQS_URL_URL = "https://ipqualityscore.com/api/json/url/{key}/{url}"
_IPQS_IP_URL = "https://ipqualityscore.com/api/json/ip/{key}/{ip}"


# ---------------------------------------------------------------------------
# VirusTotal
# ---------------------------------------------------------------------------


def query_virustotal(url: str, timeout: int = NETWORK_TIMEOUT) -> Dict:
    """Query VirusTotal API v3 for URL reputation.

    Security rationale: VirusTotal aggregates 70+ independent AV/security
    vendor verdicts. Even a single malicious vote on a URL inside an email
    is a significant risk indicator — false positives from VT are rare.

    Args:
        url: URL string to query.
        timeout: Request timeout in seconds.

    Returns:
        Dict with vt_malicious, vt_suspicious, vt_clean, vt_reputation.
        Returns -1 values on API failure.
    """
    if not _vt_key():
        log.debug("VirusTotal API key not configured — skipping VT lookup.")
        return _default_vt_features()

    try:
        url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
        headers = {"x-apikey": _vt_key()}
        resp = requests.get(
            API_ENDPOINTS["virustotal_url"].format(url_id=url_id),
            headers=headers,
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            attrs = data.get("attributes", {})
            stats = attrs.get("last_analysis_stats", {})
            return {
                "vt_malicious": stats.get("malicious", 0),
                "vt_suspicious": stats.get("suspicious", 0),
                "vt_clean": stats.get("undetected", 0),
                "vt_reputation": attrs.get("reputation", 0),
            }
        elif resp.status_code == 404:
            # URL not in VT database — submit for analysis (async, don't wait)
            _submit_url_to_virustotal(url)
            return _default_vt_features()
        else:
            log.debug(f"VirusTotal API returned {resp.status_code} for '{url[:80]}'")
    except requests.Timeout:
        log.debug(f"VirusTotal timeout for '{url[:80]}'")
    except Exception as exc:
        log.debug(f"VirusTotal error for '{url[:80]}': {exc}")

    return _default_vt_features()


def _submit_url_to_virustotal(url: str) -> None:
    """Submit a new URL to VirusTotal for analysis (fire-and-forget)."""
    if not _vt_key():
        return
    try:
        headers = {"x-apikey": _vt_key(), "content-type": "application/x-www-form-urlencoded"}
        requests.post(
            API_ENDPOINTS["virustotal_submit"],
            headers=headers,
            data={"url": url},
            timeout=2,
        )
    except Exception:
        pass    # Best-effort submission; failure is acceptable


def _default_vt_features() -> Dict:
    return {"vt_malicious": -1, "vt_suspicious": -1, "vt_clean": -1, "vt_reputation": 0}


def query_virustotal_domain(domain: str, timeout: int = NETWORK_TIMEOUT) -> Dict:
    """Query VirusTotal API v3 for domain reputation."""
    if not _vt_key() or not domain:
        return _default_vt_features()
    try:
        headers = {"x-apikey": _vt_key()}
        resp = requests.get(
            f"https://www.virustotal.com/api/v3/domains/{domain}",
            headers=headers,
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            attrs = data.get("attributes", {})
            stats = attrs.get("last_analysis_stats", {})
            return {
                "vt_malicious": stats.get("malicious", 0),
                "vt_suspicious": stats.get("suspicious", 0),
                "vt_clean": stats.get("undetected", 0),
                "vt_reputation": attrs.get("reputation", 0),
            }
        log.debug(f"VirusTotal domain API returned {resp.status_code} for '{domain}'")
    except requests.Timeout:
        log.debug(f"VirusTotal domain timeout for '{domain}'")
    except Exception as exc:
        log.debug(f"VirusTotal domain error for '{domain}': {exc}")
    return _default_vt_features()


# ---------------------------------------------------------------------------
# Google Safe Browsing
# ---------------------------------------------------------------------------


def query_google_safe_browsing(urls: List[str], timeout: int = NETWORK_TIMEOUT) -> Dict:
    """Query Google Safe Browsing API v4 for a batch of URLs.

    Security rationale: Google Safe Browsing is the same database that powers
    Chrome's phishing warnings — used by 3+ billion users. When GSB flags a URL,
    it has been confirmed phishing by Google's threat analysis team. This is
    among the most reliable threat signals available at no cost.

    Args:
        urls: List of URL strings to check (up to 500 per call).
        timeout: Request timeout in seconds.

    Returns:
        Dict with gsb_is_flagged (1 if any URL matches), gsb_threat_count.
    """
    if not _gsb_key():
        log.debug("Google Safe Browsing API key not configured — skipping GSB check.")
        return {"gsb_is_flagged": -1, "gsb_threat_count": -1}

    if not urls:
        return {"gsb_is_flagged": 0, "gsb_threat_count": 0}

    payload = {
        "client": {"clientId": "PhishLens", "clientVersion": "2.0"},
        "threatInfo": {
            "threatTypes": [
                "MALWARE",
                "SOCIAL_ENGINEERING",
                "UNWANTED_SOFTWARE",
                "POTENTIALLY_HARMFUL_APPLICATION",
            ],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": u} for u in urls[:500]],
        },
    }

    try:
        resp = requests.post(
            API_ENDPOINTS["google_safe_browsing"],
            params={"key": _gsb_key()},
            json=payload,
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            matches = data.get("matches", [])
            flagged_urls = {m.get("threat", {}).get("url", "") for m in matches}
            return {
                "gsb_is_flagged": int(len(matches) > 0),
                "gsb_threat_count": len(matches),
                "_gsb_flagged_urls": flagged_urls,
            }
    except requests.Timeout:
        log.debug("Google Safe Browsing request timed out.")
    except Exception as exc:
        log.debug(f"Google Safe Browsing error: {exc}")

    return {"gsb_is_flagged": -1, "gsb_threat_count": -1, "_gsb_flagged_urls": set()}


# ---------------------------------------------------------------------------
# AbuseIPDB
# ---------------------------------------------------------------------------


def query_abuseipdb(ip_address: str, timeout: int = NETWORK_TIMEOUT) -> Dict:
    """Query AbuseIPDB for sender IP reputation.

    Security rationale: Phishing infrastructure reuses IP addresses. An IP
    with 50+ community abuse reports is almost certainly malicious, regardless
    of what the email claims its origin is. AbuseIPDB maintains crowdsourced
    reports from security teams globally — it catches infrastructure that
    commercial threat feeds miss.

    Args:
        ip_address: IPv4 or IPv6 address extracted from email Received headers.
        timeout: Request timeout in seconds.

    Returns:
        Dict with abuse_confidence_score, total_reports, is_tor, country_code, isp.
    """
    if not _abuse_key():
        log.debug("AbuseIPDB API key not configured — skipping IP reputation check.")
        return _default_abuseipdb_features()

    if not ip_address or _is_private_ip(ip_address):
        return _default_abuseipdb_features()

    try:
        headers = {
            "Key": _abuse_key(),
            "Accept": "application/json",
        }
        params = {
            "ipAddress": ip_address,
            "maxAgeInDays": "90",
            "verbose": "",
        }
        resp = requests.get(
            API_ENDPOINTS["abuseipdb_check"],
            headers=headers,
            params=params,
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            return {
                "abuse_confidence_score": data.get("abuseConfidenceScore", 0),
                "abuse_total_reports": data.get("totalReports", 0),
                "abuse_is_tor": int(data.get("isTor", False)),
                "abuse_country_code": data.get("countryCode", ""),
                "abuse_isp": data.get("isp", ""),
            }
    except requests.Timeout:
        log.debug(f"AbuseIPDB timeout for IP '{ip_address}'")
    except Exception as exc:
        log.debug(f"AbuseIPDB error for IP '{ip_address}': {exc}")

    return _default_abuseipdb_features()


def _default_abuseipdb_features() -> Dict:
    return {
        "abuse_confidence_score": -1,
        "abuse_total_reports": -1,
        "abuse_is_tor": -1,
        "abuse_country_code": "",
        "abuse_isp": "",
    }


def _is_private_ip(ip: str) -> bool:
    """Return True if the IP is a private/reserved range (not useful for abuse check)."""
    private_prefixes = ("10.", "192.168.", "127.", "172.16.", "172.17.",
                        "172.18.", "172.19.", "172.20.", "172.21.",
                        "172.22.", "172.23.", "172.24.", "172.25.",
                        "172.26.", "172.27.", "172.28.", "172.29.",
                        "172.30.", "172.31.", "0.0.0.0", "::1", "fe80:")
    return any(ip.startswith(p) for p in private_prefixes)


# ---------------------------------------------------------------------------
# URLScan.io
# ---------------------------------------------------------------------------


def query_urlscan(url: str, timeout: int = NETWORK_TIMEOUT) -> Dict:
    """Search URLScan.io for existing scan results for a URL.

    Security rationale: URLScan.io is used daily by SOC analysts to investigate
    suspicious URLs. It captures screenshots of phishing pages, detects brand
    impersonation, and tracks redirect chains — all features that ML models
    cannot capture directly. Integrating URLScan signals you understand
    real-world analyst tooling.

    Args:
        url: URL string to search for.
        timeout: Request timeout in seconds.

    Returns:
        Dict with urlscan_malicious, urlscan_brand_impersonated,
        urlscan_redirect_count.
    """
    if not _urlscan_key():
        log.debug("URLScan.io API key not configured — skipping URLScan lookup.")
        return _default_urlscan_features()

    try:
        import urllib.parse
        query = urllib.parse.quote(f'page.url:"{url}"')
        headers = {"API-Key": _urlscan_key(), "Content-Type": "application/json"}
        resp = requests.get(
            f"{API_ENDPOINTS['urlscan_search']}?q={query}&size=1",
            headers=headers,
            timeout=timeout,
        )
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                result = results[0]
                verdict = result.get("verdicts", {}).get("overall", {})
                return {
                    "urlscan_malicious": int(verdict.get("malicious", False)),
                    "urlscan_brand_impersonated": int(
                        bool(result.get("verdicts", {}).get("urlscan", {}).get("brands", []))
                    ),
                    "urlscan_redirect_count": len(
                        result.get("page", {}).get("redirects", [])
                    ),
                }
    except requests.Timeout:
        log.debug(f"URLScan.io timeout for '{url[:80]}'")
    except Exception as exc:
        log.debug(f"URLScan.io error for '{url[:80]}': {exc}")

    return _default_urlscan_features()


def _default_urlscan_features() -> Dict:
    return {
        "urlscan_malicious": -1,
        "urlscan_brand_impersonated": -1,
        "urlscan_redirect_count": -1,
    }


# ---------------------------------------------------------------------------
# URLhaus (no key required)
# ---------------------------------------------------------------------------


def query_urlhaus(url: str, timeout: int = NETWORK_TIMEOUT) -> Dict:
    """Query abuse.ch URLhaus for malicious URL classification.

    Security rationale: URLhaus tracks malware distribution and phishing URLs
    submitted by the security community. No API key required — fully open.

    Args:
        url: URL string to query.
        timeout: Request timeout in seconds.

    Returns:
        Dict with urlhaus_threat (0=clean, 1=malicious/phishing, -1=unknown).
    """
    try:
        resp = requests.post(
            API_ENDPOINTS["urlhaus_lookup"],
            data={"url": url},
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            query_status = data.get("query_status", "")
            if query_status == "no_results":
                return {"urlhaus_threat": 0}
            elif query_status in ("is_host", "blacklisted"):
                return {"urlhaus_threat": 1}
    except Exception as exc:
        log.debug(f"URLhaus error for '{url[:80]}': {exc}")

    return {"urlhaus_threat": -1}


# ---------------------------------------------------------------------------
# Combined intelligence enrichment
# ---------------------------------------------------------------------------


def enrich_email_with_intelligence(
    urls: List[str],
    sender_ip: Optional[str] = None,
) -> Dict:
    """Run all intelligence API queries for an email and return combined features.

    Args:
        urls: List of URLs from the email.
        sender_ip: Sender IP extracted from Received: headers (optional).

    Returns:
        Merged dict of all intelligence features.
    """
    features: Dict = {}

    # VT, URLhaus, URLScan — scan each URL individually (up to 5) and
    # store per-URL results under _vt_url_N / _uh_url_N / _us_url_N keys
    # (display-only; ML vector still uses primary URL's aggregated features)
    primary_url = urls[0] if urls else None
    for _i, _u in enumerate(urls[:5]):
        vt_i = query_virustotal(_u)
        features[f"_vt_url_{_i}"] = vt_i
        features[f"_vt_url_{_i}_url"] = _u
        uh_i = query_urlhaus(_u)
        features[f"_uh_url_{_i}"] = uh_i
        us_i = query_urlscan(_u)
        features[f"_us_url_{_i}"] = us_i

    # ML features: derived from the primary URL (backward-compatible)
    if primary_url:
        features.update(features.get("_vt_url_0", _default_vt_features()))
        _uh0 = features.get("_uh_url_0", {})
        features["urlhaus_threat"] = _uh0.get("urlhaus_threat", -1)
        _us0 = features.get("_us_url_0", {})
        features["urlscan_malicious"] = _us0.get("urlscan_malicious", -1)
        features["urlscan_brand_impersonated"] = _us0.get("urlscan_brand_impersonated", -1)
        features["urlscan_redirect_count"] = _us0.get("urlscan_redirect_count", -1)

    # GSB on all URLs (batch check — single API call)
    if urls:
        gsb = query_google_safe_browsing(urls)
        features.update(gsb)

    # AbuseIPDB on sender IP
    if sender_ip:
        abuse = query_abuseipdb(sender_ip)
        # Convert numeric abuse features only (drop strings for ML pipeline)
        features["abuse_confidence_score"] = abuse["abuse_confidence_score"]
        features["abuse_total_reports"] = abuse["abuse_total_reports"]
        features["abuse_is_tor"] = abuse["abuse_is_tor"]
    else:
        features.update({
            "abuse_confidence_score": -1,
            "abuse_total_reports": -1,
            "abuse_is_tor": -1,
        })

    return features


def get_default_intelligence_features() -> Dict:
    """Return zero-filled intelligence features for emails without URLs/IPs."""
    return {
        "vt_malicious": -1,
        "vt_suspicious": -1,
        "vt_clean": -1,
        "vt_reputation": 0,
        "gsb_is_flagged": -1,
        "gsb_threat_count": -1,
        "urlscan_malicious": -1,
        "urlscan_brand_impersonated": -1,
        "urlscan_redirect_count": -1,
        "urlhaus_threat": -1,
        "abuse_confidence_score": -1,
        "abuse_total_reports": -1,
        "abuse_is_tor": -1,
    }


# ---------------------------------------------------------------------------
# IPQualityScore (IPQS)
# ---------------------------------------------------------------------------


def query_ipqs_email(email_address: str, timeout: int = NETWORK_TIMEOUT) -> Dict:
    """Check an email address against IPQualityScore Email Verification API.

    Returns fraud score (0–100), disposable flag, spam trap flag,
    and deliverability status.  All display-only — NOT in the ML vector.

    Args:
        email_address: The sender email address to verify.
        timeout: Request timeout in seconds.

    Returns:
        Dict with ipqs_email_fraud_score, ipqs_email_disposable,
        ipqs_email_spam_trap, ipqs_email_valid, ipqs_email_recent_abuse,
        ipqs_email_deliverability, ipqs_email_dns_valid.
    """
    _default = {
        "ipqs_email_fraud_score": -1,
        "ipqs_email_disposable": -1,
        "ipqs_email_spam_trap": -1,
        "ipqs_email_valid": -1,
        "ipqs_email_recent_abuse": -1,
        "ipqs_email_deliverability": "unknown",
        "ipqs_email_dns_valid": -1,
    }
    if not _ipqs_key() or not email_address:
        return _default
    try:
        import urllib.parse
        url = _IPQS_EMAIL_URL.format(
            key=_ipqs_key(),
            email=urllib.parse.quote(email_address, safe=""),
        )
        params = {"timeout": 7, "fast": "true", "abuse_strictness": 1}
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 200:
            d = resp.json()
            if d.get("success", False):
                return {
                    "ipqs_email_fraud_score": d.get("fraud_score", -1),
                    "ipqs_email_disposable": int(d.get("disposable", False)),
                    "ipqs_email_spam_trap": int(d.get("spam_trap_score", 0) > 50),
                    "ipqs_email_valid": int(d.get("valid", False)),
                    "ipqs_email_recent_abuse": int(d.get("recent_abuse", False)),
                    "ipqs_email_deliverability": d.get("deliverability", "unknown"),
                    "ipqs_email_dns_valid": int(d.get("dns_valid", False)),
                }
            else:
                _default["_ipqs_error"] = d.get("message", "API error")
                return _default
    except requests.Timeout:
        log.debug(f"IPQS email timeout for '{email_address}'")
        _default["_ipqs_error"] = "Request timed out"
    except Exception as exc:
        log.debug(f"IPQS email error for '{email_address}': {exc}")
        _default["_ipqs_error"] = str(exc)
    return _default


def query_ipqs_url(url: str, timeout: int = NETWORK_TIMEOUT) -> Dict:
    """Scan a URL against IPQualityScore Malicious URL Scanner API.

    Returns phishing/malware/suspicious flags and an overall risk score.
    All display-only — NOT in the ML vector.

    Args:
        url: URL string to scan.
        timeout: Request timeout in seconds.

    Returns:
        Dict with ipqs_url_phishing, ipqs_url_malware, ipqs_url_suspicious,
        ipqs_url_unsafe, ipqs_url_risk_score, ipqs_url_domain_rank,
        ipqs_url_short_link_redirect, ipqs_url_spamming.
    """
    _default = {
        "ipqs_url_phishing": -1,
        "ipqs_url_malware": -1,
        "ipqs_url_suspicious": -1,
        "ipqs_url_unsafe": -1,
        "ipqs_url_risk_score": -1,
        "ipqs_url_domain_rank": -1,
        "ipqs_url_short_link_redirect": -1,
        "ipqs_url_spamming": -1,
    }
    if not _ipqs_key() or not url:
        return _default
    try:
        import urllib.parse
        encoded_url = urllib.parse.quote(url, safe="")
        req_url = _IPQS_URL_URL.format(key=_ipqs_key(), url=encoded_url)
        params = {"strictness": 1, "allow_public_access_points": "true", "fast": "false"}
        resp = requests.get(req_url, params=params, timeout=timeout)
        if resp.status_code == 200:
            d = resp.json()
            if d.get("success", False):
                return {
                    "ipqs_url_phishing": int(d.get("phishing", False)),
                    "ipqs_url_malware": int(d.get("malware", False)),
                    "ipqs_url_suspicious": int(d.get("suspicious", False)),
                    "ipqs_url_unsafe": int(d.get("unsafe", False)),
                    "ipqs_url_risk_score": d.get("risk_score", -1),
                    "ipqs_url_domain_rank": d.get("domain_rank", -1),
                    "ipqs_url_short_link_redirect": int(d.get("short_link_redirect", False)),
                    "ipqs_url_spamming": int(d.get("spamming", False)),
                }
            else:
                _default["_ipqs_error"] = d.get("message", "API error")
                return _default
    except requests.Timeout:
        log.debug(f"IPQS URL timeout for '{url[:80]}'")
        _default["_ipqs_error"] = "Request timed out"
    except Exception as exc:
        log.debug(f"IPQS URL error for '{url[:80]}': {exc}")
        _default["_ipqs_error"] = str(exc)
    return _default


def query_ipqs_ip(ip_address: str, timeout: int = NETWORK_TIMEOUT) -> Dict:
    """Check a sender IP against IPQualityScore IP Reputation API.

    Returns fraud score, proxy/VPN/Tor detection, and abuse flags.
    All display-only — NOT in the ML vector.

    Args:
        ip_address: IPv4 or IPv6 address to check.
        timeout: Request timeout in seconds.

    Returns:
        Dict with ipqs_ip_fraud_score, ipqs_ip_proxy, ipqs_ip_vpn,
        ipqs_ip_tor, ipqs_ip_recent_abuse, ipqs_ip_bot_status,
        ipqs_ip_country_code, ipqs_ip_isp, ipqs_ip_connection_type.
    """
    _default = {
        "ipqs_ip_fraud_score": -1,
        "ipqs_ip_proxy": -1,
        "ipqs_ip_vpn": -1,
        "ipqs_ip_tor": -1,
        "ipqs_ip_recent_abuse": -1,
        "ipqs_ip_bot_status": -1,
        "ipqs_ip_country_code": "unknown",
        "ipqs_ip_isp": "unknown",
        "ipqs_ip_connection_type": "unknown",
    }
    if not _ipqs_key() or not ip_address:
        return _default
    if _is_private_ip(ip_address):
        return _default
    try:
        req_url = _IPQS_IP_URL.format(key=_ipqs_key(), ip=ip_address)
        params = {"strictness": 1, "allow_public_access_points": "true"}
        resp = requests.get(req_url, params=params, timeout=timeout)
        if resp.status_code == 200:
            d = resp.json()
            if d.get("success", False):
                return {
                    "ipqs_ip_fraud_score": d.get("fraud_score", -1),
                    "ipqs_ip_proxy": int(d.get("proxy", False)),
                    "ipqs_ip_vpn": int(d.get("vpn", False)),
                    "ipqs_ip_tor": int(d.get("tor", False)),
                    "ipqs_ip_recent_abuse": int(d.get("recent_abuse", False)),
                    "ipqs_ip_bot_status": int(d.get("bot_status", False)),
                    "ipqs_ip_country_code": d.get("country_code", "unknown"),
                    "ipqs_ip_isp": d.get("ISP", d.get("isp", "unknown")),
                    "ipqs_ip_connection_type": d.get("connection_type", "unknown"),
                }
            else:
                _default["_ipqs_error"] = d.get("message", "API error")
                return _default
    except requests.Timeout:
        log.debug(f"IPQS IP timeout for '{ip_address}'")
        _default["_ipqs_error"] = "Request timed out"
    except Exception as exc:
        log.debug(f"IPQS IP error for '{ip_address}': {exc}")
        _default["_ipqs_error"] = str(exc)
    return _default
