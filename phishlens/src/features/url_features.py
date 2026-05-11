"""
PhishLens URL Feature Engineering Module.

Extracts lexical, WHOIS, and certificate transparency features from URLs
found in email bodies. All network calls use strict timeouts and fallbacks.
Per-URL features are aggregated (max/mean/count) across all URLs in an email.

Security rationale: URL analysis is the single most reliable phishing signal
category. Phishers cannot easily avoid: newly registered domains, high-entropy
URLs, brand keywords in subdomains, punycode homoglyphs, and Let's Encrypt
certs on <30-day-old domains. Lexical features require zero network calls,
making them zero-day safe — they work even on unknown phishing infrastructure.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import tldextract
import whois

from src.utils.config import (
    DEFAULT_CONFIG,
    BRAND_LIST,
    RISK_TLD_LIST,
    SAFE_TLD_LIST,
    URL_SHORTENER_DOMAINS,
    SUSPICIOUS_URL_KEYWORDS,
    ABUSE_REGISTRARS,
    API_ENDPOINTS,
    NETWORK_TIMEOUT,
    WHOIS_TIMEOUT,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

# Confusable homoglyph library (Unicode spoofing detection)
try:
    from confusable_homoglyphs import confusables
    _CONFUSABLES_AVAILABLE = True
except ImportError:
    _CONFUSABLES_AVAILABLE = False
    log.warning("confusable_homoglyphs not available — homoglyph detection disabled.")

_IP_URL_RE = re.compile(r"https?://(\d{1,3}\.){3}\d{1,3}")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def extract_url_features(urls: List[str], config=DEFAULT_CONFIG) -> Dict:
    """Extract and aggregate URL features from a list of email URLs.

    Args:
        urls: List of URL strings extracted from an email.
        config: PhishLensConfig instance.

    Returns:
        Dict of aggregated features (max/mean/count across all URLs).
        Returns zero-filled defaults if urls is empty.
    """
    if not urls:
        return _default_url_features()

    per_url_results: List[Dict] = []
    for url in urls[:20]:    # Cap at 20 URLs to prevent DoS via URL flooding
        try:
            features = _extract_single_url_features(url, config)
            per_url_results.append(features)
        except Exception as exc:
            log.debug(f"URL feature extraction error for '{url[:80]}': {exc}")
            per_url_results.append(_default_single_url_features())

    # Keep schema stable: always return the full URL feature key set,
    # even when network lookups are disabled.
    aggregated = _aggregate_url_features(per_url_results, total_url_count=len(urls))
    stable = _default_url_features()
    stable.update(aggregated)
    return stable


def extract_url_features_with_network(
    urls: List[str],
    config=DEFAULT_CONFIG,
) -> Dict:
    """Extract URL features including async WHOIS + certificate transparency.

    This function adds network-dependent features on top of the lexical features.
    Uses ThreadPoolExecutor for parallel WHOIS / crt.sh queries.

    Args:
        urls: List of URL strings.
        config: PhishLensConfig instance.

    Returns:
        Dict with both lexical and network-based features.
    """
    base_features = extract_url_features(urls, config)
    if not urls:
        return base_features

    # Sample first 5 unique domains for network lookups (rate limit protection)
    domains = list({_get_registered_domain(u) for u in urls[:10] if _get_registered_domain(u)})
    domains = [d for d in domains if d][:5]

    whois_features = _aggregate_whois_features(domains, config)
    cert_features = _aggregate_cert_features(domains, config)

    base_features.update(whois_features)
    base_features.update(cert_features)
    return base_features


# ---------------------------------------------------------------------------
# Single URL feature extraction
# ---------------------------------------------------------------------------


def _extract_single_url_features(url: str, config) -> Dict:
    """Extract all lexical features for a single URL."""
    features = _default_single_url_features()
    try:
        parsed = urlparse(url)
        ext = tldextract.extract(url)

        domain = ext.domain or ""
        suffix = ext.suffix or ""
        subdomain = ext.subdomain or ""
        registered_domain = ext.top_domain_under_public_suffix or ""
        full_domain = parsed.netloc or ""
        path = parsed.path or ""

        # domain_length: longer domains = higher phishing probability
        features["domain_length"] = len(registered_domain)

        # subdomain_depth: deep subdomain nesting = obfuscation
        features["subdomain_depth"] = len(subdomain.split(".")) if subdomain else 0

        # hyphen_count: hyphens in domain often mimic legitimate brand names
        features["hyphen_count"] = registered_domain.count("-")

        # digit_ratio: high digit proportion across full hostname = random domain generation
        if full_domain:
            features["digit_ratio"] = sum(c.isdigit() for c in full_domain) / len(full_domain)

        # url_entropy: Shannon entropy of the full URL string
        features["url_entropy"] = _shannon_entropy(url)

        # brand_in_subdomain: e.g., paypal.secure-login.xyz
        features["brand_in_subdomain"] = int(
            _has_brand_in_subdomain(subdomain, registered_domain, config.brand_list)
        )

        # tld_risk_score: .xyz/.tk etc. = 1.0, .com/.ie = 0.0, unknown = 0.5
        tld_with_dot = f".{suffix.lower()}" if suffix else ""
        if tld_with_dot in config.risk_tld_list:
            features["tld_risk_score"] = 1.0
        elif tld_with_dot in config.safe_tld_list:
            features["tld_risk_score"] = 0.0
        else:
            features["tld_risk_score"] = 0.5

        # is_ip_address: raw IP in URL = strong phishing signal
        features["is_ip_address"] = int(bool(_IP_URL_RE.match(url)))

        # punycode_detected: xn-- = internationalised domain (homoglyph risk)
        features["punycode_detected"] = int(
            "xn--" in url.lower() or _has_confusable_homoglyph(full_domain)
        )

        # url_shortener: bit.ly, tinyurl, etc.
        features["url_shortener"] = int(
            any(shortener in full_domain.lower() for shortener in config.url_shortener_domains)
        )

        # path_depth: /verify/account/reset = 3 levels = suspicious
        features["path_depth"] = len([p for p in path.split("/") if p])

        # suspicious_keywords_in_url
        url_lower = url.lower()
        kw_count = sum(
            1 for kw in config.suspicious_url_keywords if kw in url_lower
        )
        # Credential spoofing trick: http://paypal.com@attacker.com/login
        # urlparse treats everything before @ as credentials; the host is attacker.com
        if "@" in (parsed.netloc or ""):
            kw_count += 3  # Heavy penalty — this is almost always malicious
        features["suspicious_keywords_in_url"] = kw_count

    except Exception as exc:
        log.debug(f"_extract_single_url_features error: {exc}")

    return features


def _aggregate_url_features(per_url: List[Dict], total_url_count: int) -> Dict:
    """Aggregate per-URL features into email-level max/mean/count statistics."""
    if not per_url:
        return _default_url_features()

    numeric_keys = [
        "domain_length", "subdomain_depth", "hyphen_count",
        "digit_ratio", "url_entropy", "brand_in_subdomain",
        "tld_risk_score", "is_ip_address", "punycode_detected",
        "url_shortener", "path_depth", "suspicious_keywords_in_url",
    ]

    aggregated: Dict = {"url_count": total_url_count}

    for key in numeric_keys:
        vals = [r.get(key, 0) for r in per_url]
        aggregated[f"{key}_max"] = max(vals)
        aggregated[f"{key}_mean"] = sum(vals) / len(vals)

    return aggregated


# ---------------------------------------------------------------------------
# WHOIS features
# ---------------------------------------------------------------------------


def _get_whois_features(domain: str, config) -> Dict:
    """Query WHOIS for domain age and registrar risk.

    Returns:
        Dict with domain_age_days, domain_age_risk, registrar_risk.
        Falls back to -1 values on timeout or WHOIS failure (~30% miss rate).
    """
    features = {
        "domain_age_days": -1,
        "domain_age_risk": 0.5,
        "registrar_risk": 0.0,
    }
    try:
        w = whois.whois(domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date:
            import datetime
            age_days = (datetime.datetime.now() - creation_date).days
            features["domain_age_days"] = age_days
            if age_days < config.domain_age_risk_days:
                features["domain_age_risk"] = 1.0    # Brand new = high risk
            elif age_days < config.domain_age_warn_days:
                features["domain_age_risk"] = 0.5
            else:
                features["domain_age_risk"] = 0.0

        registrar = str(w.registrar or "").lower()
        features["registrar_risk"] = float(
            any(abuse_reg in registrar for abuse_reg in config.abuse_registrars)
        )
    except Exception as exc:
        log.debug(f"WHOIS lookup failed for '{domain}': {exc}")
    return features


def _aggregate_whois_features(domains: List[str], config) -> Dict:
    """Run WHOIS lookups in parallel and aggregate results."""
    all_results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_get_whois_features, d, config): d for d in domains}
        for future in as_completed(futures, timeout=config.whois_timeout + 2):
            try:
                all_results.append(future.result(timeout=config.whois_timeout))
            except Exception:
                all_results.append({"domain_age_days": -1, "domain_age_risk": 0.5, "registrar_risk": 0.0})

    if not all_results:
        return {"domain_age_days": -1, "domain_age_risk": 0.5, "registrar_risk": 0.0}

    # Use worst-case (highest risk) values across all domains
    return {
        "domain_age_days": min(r["domain_age_days"] for r in all_results),
        "domain_age_risk": max(r["domain_age_risk"] for r in all_results),
        "registrar_risk": max(r["registrar_risk"] for r in all_results),
    }


# ---------------------------------------------------------------------------
# Certificate transparency features (crt.sh)
# ---------------------------------------------------------------------------


def _get_cert_features(domain: str, config) -> Dict:
    """Query crt.sh for certificate transparency data.

    Security rationale: Let's Encrypt certs on domains < 30 days old with
    brand keywords in their SAN is one of the strongest phishing signals
    in modern attack infrastructure.
    """
    features = {
        "cert_age_days": -1,
        "cert_lets_encrypt": 0,
        "cert_brand_mismatch": 0,
    }
    try:
        url = API_ENDPOINTS["crtsh"].format(domain=domain)
        resp = requests.get(url, timeout=config.network_timeout)
        if resp.status_code != 200:
            return features
        certs = resp.json()
        if not certs:
            return features

        import datetime
        # Find oldest cert entry
        min_age = None
        le_found = False
        brand_mismatch = False

        for cert in certs[:50]:   # Cap at 50 entries
            try:
                not_before = cert.get("not_before", "")
                if not_before:
                    issued_dt = datetime.datetime.fromisoformat(not_before.replace("T", " ").split(".")[0])
                    age = (datetime.datetime.utcnow() - issued_dt).days
                    if min_age is None or age < min_age:
                        min_age = age

                issuer = cert.get("issuer_name", "").lower()
                if "let's encrypt" in issuer or "lets encrypt" in issuer:
                    le_found = True

                # Brand in SAN but not registered domain = impersonation
                san = cert.get("name_value", "").lower()
                for brand in BRAND_LIST:
                    if brand in san and brand not in domain.lower():
                        brand_mismatch = True
                        break

            except Exception:
                continue

        features["cert_age_days"] = min_age if min_age is not None else -1
        features["cert_lets_encrypt"] = int(le_found)
        features["cert_brand_mismatch"] = int(brand_mismatch)

    except Exception as exc:
        log.debug(f"crt.sh lookup failed for '{domain}': {exc}")

    return features


def _aggregate_cert_features(domains: List[str], config) -> Dict:
    """Run crt.sh lookups asynchronously and aggregate results.

    Security rationale: Using asyncio + aiohttp for crt.sh HTTP calls reduces
    wall-clock time from O(n × timeout) to O(timeout) for n domains by
    dispatching all HTTP requests concurrently. WHOIS stays in ThreadPoolExecutor
    because the whois library uses blocking socket calls that cannot be adapted
    to asyncio without monkey-patching.
    """
    if not domains:
        return {"cert_age_days": -1, "cert_lets_encrypt": 0, "cert_brand_mismatch": 0}

    try:
        # Try to get the running loop (works in Jupyter / async contexts)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                all_results = pool.submit(
                    asyncio.run, _dispatch_certs_async(domains, config)
                ).result()
        else:
            all_results = asyncio.run(_dispatch_certs_async(domains, config))
    except Exception as exc:
        log.debug(f"Async cert dispatch failed, falling back to sync: {exc}")
        all_results = [_get_cert_features(d, config) for d in domains]

    if not all_results:
        return {"cert_age_days": -1, "cert_lets_encrypt": 0, "cert_brand_mismatch": 0}

    return {
        "cert_age_days": min((r["cert_age_days"] for r in all_results if r["cert_age_days"] >= 0), default=-1),
        "cert_lets_encrypt": max(r["cert_lets_encrypt"] for r in all_results),
        "cert_brand_mismatch": max(r["cert_brand_mismatch"] for r in all_results),
    }


async def _dispatch_certs_async(domains: List[str], config) -> List[Dict]:
    """Dispatch all crt.sh HTTP lookups concurrently using aiohttp.

    Security rationale: aiohttp with a 3-second per-request timeout prevents
    a slow/malicious crt.sh response from blocking the main pipeline for the
    duration of all domain lookups combined. Each coroutine silently returns
    -1 defaults on any failure — the pipeline never crashes on network issues.

    Args:
        domains: List of registered domain strings.
        config: PhishLensConfig with network_timeout setting.

    Returns:
        List of cert feature dicts, one per domain.
    """
    try:
        import aiohttp
    except ImportError:
        log.warning("aiohttp not installed — falling back to sync crt.sh lookups")
        return [_get_cert_features(d, config) for d in domains]

    timeout = aiohttp.ClientTimeout(total=3)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [_fetch_cert_async(d, session, config) for d in domains]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    processed: List[Dict] = []
    for r in results:
        if isinstance(r, Exception) or not isinstance(r, dict):
            processed.append({"cert_age_days": -1, "cert_lets_encrypt": 0, "cert_brand_mismatch": 0})
        else:
            processed.append(r)
    return processed


async def _fetch_cert_async(domain: str, session, config) -> Dict:
    """Fetch and parse crt.sh data for a single domain asynchronously.

    Args:
        domain: Registered domain to query.
        session: Shared aiohttp.ClientSession (connection-pooled).
        config: PhishLensConfig instance.

    Returns:
        Dict with cert_age_days, cert_lets_encrypt, cert_brand_mismatch.
        Returns -1/-0/0 defaults on any network or parse error.
    """
    features = {"cert_age_days": -1, "cert_lets_encrypt": 0, "cert_brand_mismatch": 0}
    try:
        url = API_ENDPOINTS["crtsh"].format(domain=domain)
        async with session.get(url) as resp:
            if resp.status != 200:
                return features
            certs = await resp.json(content_type=None)
        if not certs:
            return features

        import datetime
        min_age = None
        le_found = False
        brand_mismatch = False

        for cert in certs[:50]:
            try:
                not_before = cert.get("not_before", "")
                if not_before:
                    issued_dt = datetime.datetime.fromisoformat(
                        not_before.replace("T", " ").split(".")[0]
                    )
                    age = (datetime.datetime.utcnow() - issued_dt).days
                    if min_age is None or age < min_age:
                        min_age = age

                issuer = cert.get("issuer_name", "").lower()
                if "let's encrypt" in issuer or "lets encrypt" in issuer:
                    le_found = True

                san = cert.get("name_value", "").lower()
                for brand in BRAND_LIST:
                    if brand in san and brand not in domain.lower():
                        brand_mismatch = True
                        break
            except Exception:
                continue

        features["cert_age_days"] = min_age if min_age is not None else -1
        features["cert_lets_encrypt"] = int(le_found)
        features["cert_brand_mismatch"] = int(brand_mismatch)

    except Exception as exc:
        log.debug(f"crt.sh async fetch failed for '{domain}': {exc}")
    return features


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _shannon_entropy(text: str) -> float:
    """Compute Shannon entropy of a string.

    Security rationale: High-entropy URLs (random character sequences) indicate
    algorithmically-generated domains (DGA) or obfuscated phishing infrastructure.
    """
    if not text:
        return 0.0
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    n = len(text)
    return -sum((count / n) * math.log2(count / n) for count in freq.values())


def _has_brand_in_subdomain(subdomain: str, registered_domain: str, brand_list: List[str]) -> bool:
    """Detect brand keyword in subdomain but not in registered domain.

    Security rationale: paypal.secure-login.xyz — 'paypal' in subdomain but
    registered domain is 'secure-login.xyz'. This is the canonical brand
    impersonation pattern in phishing URLs.
    """
    if not subdomain:
        return False
    subdomain_lower = subdomain.lower()
    registered_lower = registered_domain.lower()
    for brand in brand_list:
        if brand in subdomain_lower and brand not in registered_lower:
            return True
    return False


def _has_confusable_homoglyph(domain: str) -> bool:
    """Detect Unicode confusable homoglyphs in the domain.

    Security rationale: Cyrillic 'а' vs Latin 'a', Greek 'ο' vs Latin 'o' etc.
    are used to create visually identical but different domain names.
    """
    if not _CONFUSABLES_AVAILABLE:
        return False
    try:
        for char in domain:
            if confusables.is_dangerous(char):
                return True
        return False
    except Exception:
        return False


def _get_registered_domain(url: str) -> Optional[str]:
    """Extract just the registered domain (e.g., google.com) from a URL."""
    try:
        ext = tldextract.extract(url)
        return ext.registered_domain or None
    except Exception:
        return None


def _default_single_url_features() -> Dict:
    """Zero-value defaults for single URL features."""
    return {
        "domain_length": 0,
        "subdomain_depth": 0,
        "hyphen_count": 0,
        "digit_ratio": 0.0,
        "url_entropy": 0.0,
        "brand_in_subdomain": 0,
        "tld_risk_score": 0.0,
        "is_ip_address": 0,
        "punycode_detected": 0,
        "url_shortener": 0,
        "path_depth": 0,
        "suspicious_keywords_in_url": 0,
    }


def _default_url_features() -> Dict:
    """Zero-value defaults for aggregated email-level URL features."""
    base = {"url_count": 0}
    for key in _default_single_url_features():
        base[f"{key}_max"] = 0
        base[f"{key}_mean"] = 0.0
    base.update({
        "domain_age_days": -1,
        "domain_age_risk": 0.0,
        "registrar_risk": 0.0,
        "cert_age_days": -1,
        "cert_lets_encrypt": 0,
        "cert_brand_mismatch": 0,
    })
    return base
