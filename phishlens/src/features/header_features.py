"""
PhishLens Header Forensics Feature Module.

Extracts 12 security-critical features from email headers.
All feature extraction is wrapped in try/except blocks to guarantee
no single malformed header crashes the pipeline.

Security rationale: Email headers contain the digital fingerprints of
message routing. Phishing campaigns consistently show specific header
anomalies: spoofed sender domains, reply-to hijacking, failed SPF/DKIM/DMARC,
and unusual relay chains. These 12 features capture the most reliable signals.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import dns.resolver
import dns.exception
import tldextract

from src.utils.config import DEFAULT_CONFIG, FREEMAIL_DOMAINS, SUSPICIOUS_XMAILER_PATTERNS
from src.utils.logger import get_logger

log = get_logger(__name__)

# Precompile header injection detection pattern
# CRLF injection in headers is an email security attack vector
_HEADER_INJECTION_RE = re.compile(r"[\r\n\x00]")

# IP in Received header
_IP_RE = re.compile(r"\b(\d{1,3}\.){3}\d{1,3}\b")

# Country/timezone extraction from Received header
_TZ_RE = re.compile(r"([+-]\d{4})\s*\(?([A-Z]{2,5})?\)?")

# Known bulk-sender X-Mailer fingerprints (only flag confirmed bulk-mailers)
_BULK_MAILER_RE = re.compile(
    "|".join(re.escape(p) for p in SUSPICIOUS_XMAILER_PATTERNS if p),
    re.IGNORECASE,
)

# Inline Authentication-Results result-to-score mapping
_AUTH_RESULT_SCORES: Dict[str, int] = {
    "pass": 1,
    "softfail": 0,
    "neutral": 0,
    "none": -1,
    "fail": -1,
    "temperror": -1,
    "permerror": -1,
}


def extract_header_features(parsed_email: Dict, use_network: bool = True) -> Dict:
    """Extract header forensics features from a parsed email dict.

    Args:
        parsed_email: Dict returned by eml_parser.parse_eml_bytes().
        use_network: When False, DNS lookups (SPF/DKIM/DMARC) are skipped
            and those features default to -1.  Set False during offline
            training to avoid blocking on DNS timeouts (~300-400 ms/email).

    Returns:
        Dict with header features, all with numeric or boolean values.
        Falls back to safe defaults on any extraction failure.
    """
    features: Dict = _default_header_features()

    try:
        features["from_reply_to_mismatch"] = _check_from_reply_mismatch(
            parsed_email.get("from_address", ""),
            parsed_email.get("reply_to", ""),
        )
    except Exception as exc:
        log.debug(f"from_reply_to_mismatch error: {exc}")

    try:
        features["from_return_path_mismatch"] = _check_from_return_path_mismatch(
            parsed_email.get("from_address", ""),
            parsed_email.get("return_path", ""),
        )
    except Exception as exc:
        log.debug(f"from_return_path_mismatch error: {exc}")

    try:
        # reply_to freemail = phisher redirecting replies to attacker mailbox
        reply_to = parsed_email.get("reply_to", "")
        features["reply_to_freemail"] = int(
            bool(re.search(r"@(" + "|".join(re.escape(d.split("@")[-1]) for d in FREEMAIL_DOMAINS) + r")", reply_to, re.IGNORECASE))
        )
    except Exception as exc:
        log.debug(f"reply_to_freemail error: {exc}")

    try:
        received = parsed_email.get("received_headers", [])
        features["received_hop_count"] = len(received)
        features["received_geo_anomaly"] = int(_check_geo_anomaly(received))
    except Exception as exc:
        log.debug(f"received_headers error: {exc}")

    # Priority 1: parse inline Authentication-Results / Received-SPF headers
    # (available on almost all emails without any network call)
    try:
        _ar = parsed_email.get("auth_results", "")
        _rspf = parsed_email.get("received_spf_header", "")
        if _ar or _rspf:
            _spf, _dkim, _dmarc = _parse_auth_results_header(_ar, _rspf)
            if _spf is not None:
                features["spf_result"] = _spf
            if _dkim is not None:
                features["dkim_result"] = _dkim
            if _dmarc is not None:
                features["dmarc_result"] = _dmarc
            # If DKIM-Signature header present, override with its presence
            if parsed_email.get("dkim_signed", 0):
                features["dkim_result"] = max(features["dkim_result"], 0)  # at least neutral
        # Priority 2: live DNS lookup when no inline results and network enabled
        elif use_network:
            from_domain = _extract_domain(parsed_email.get("from_address", ""))
            if from_domain:
                spf, dkim, dmarc = _check_email_authentication(from_domain)
                features["spf_result"] = spf
                features["dkim_result"] = dkim
                features["dmarc_result"] = dmarc
    except Exception as exc:
        log.debug(f"SPF/DKIM/DMARC check error: {exc}")

    try:
        features["message_id_suspicious"] = int(
            _check_message_id_suspicious(parsed_email.get("message_id", ""))
        )
    except Exception as exc:
        log.debug(f"message_id_suspicious error: {exc}")

    try:
        features["timezone_mismatch"] = int(
            _check_timezone_mismatch(
                parsed_email.get("date", ""),
                parsed_email.get("received_headers", []),
            )
        )
    except Exception as exc:
        log.debug(f"timezone_mismatch error: {exc}")

    try:
        # Only flag confirmed bulk-mailer fingerprints; missing X-Mailer is
        # normal for modern webmail (Gmail, OWA, Apple Mail omit it).
        features["x_mailer_suspicious"] = int(
            _check_xmailer_suspicious(parsed_email.get("x_mailer", ""))
        )
    except Exception as exc:
        log.debug(f"x_mailer_suspicious error: {exc}")

    try:
        # CRLF injection in any header value = potential header injection attack
        header_raw = parsed_email.get("header_raw", "")
        features["header_injection_attempt"] = int(
            bool(_HEADER_INJECTION_RE.search(header_raw[:2048]))
        )
    except Exception as exc:
        log.debug(f"header_injection_attempt error: {exc}")

    return features


# ---------------------------------------------------------------------------
# Feature implementation functions
# ---------------------------------------------------------------------------


def _check_from_reply_mismatch(from_addr: str, reply_to: str) -> int:
    """Detect mismatch between From and Reply-To registered domains.

    Compares registered domains (e.g. paypal.com) not full hostnames so that
    newsletters.paypal.com → paypal.com does NOT trigger as a mismatch.
    Returns 1 if mismatch detected, 0 if same registered domain or Reply-To absent.
    """
    if not reply_to or not from_addr:
        return 0
    from_domain = _extract_domain(from_addr)
    reply_domain = _extract_domain(reply_to)
    if not from_domain or not reply_domain:
        return 0
    try:
        from_reg = tldextract.extract(from_domain).registered_domain or from_domain
        reply_reg = tldextract.extract(reply_domain).registered_domain or reply_domain
        return int(from_reg.lower() != reply_reg.lower())
    except Exception:
        return int(from_domain.lower() != reply_domain.lower())


def _check_from_return_path_mismatch(from_addr: str, return_path: str) -> int:
    """Detect mismatch between From and Return-Path registered domains.

    Compares registered domains so bounce-handling subdomains (e.g.
    bounce.paypal.com) do not falsely flag as a mismatch against paypal.com.
    """
    if not return_path or not from_addr:
        return 0
    from_domain = _extract_domain(from_addr)
    rp_domain = _extract_domain(return_path)
    if not from_domain or not rp_domain:
        return 0
    try:
        from_reg = tldextract.extract(from_domain).registered_domain or from_domain
        rp_reg = tldextract.extract(rp_domain).registered_domain or rp_domain
        return int(from_reg.lower() != rp_reg.lower())
    except Exception:
        return int(from_domain.lower() != rp_domain.lower())


def _check_geo_anomaly(received_headers: List[str]) -> bool:
    """Detect geographic anomalies in the Received header relay chain.

    Security rationale: A legitimate corporate email typically routes through
    a predictable set of servers. A relay chain jumping between 3+ distinct
    continents or containing Tor exit nodes signals relay abuse.
    """
    if len(received_headers) < 2:
        return False
    # Extract IPs from Received headers and check for >2 distinct /8 subnets
    # as a proxy for geographic diversity (simplified heuristic)
    ips = []
    for h in received_headers:
        matches = _IP_RE.findall(h)
        ips.extend(matches)
    if len(ips) < 2:
        return False
    # Check if IPs span more than 2 distinct /16 subnets (class B)
    subnets = {".".join(ip.split(".")[:2]) for ip in ips if not ip.startswith("10.")
               and not ip.startswith("192.168.") and not ip.startswith("127.")}
    return len(subnets) > 2


def _check_email_authentication(domain: str) -> tuple[int, int, int]:
    """Check SPF, DKIM, and DMARC DNS records for the sender domain.

    Security rationale: SPF/DKIM/DMARC are the three email authentication
    standards. All three failing simultaneously is the strongest possible
    phishing signal — it means the sending server is not authorised,
    the message is not cryptographically signed, and the domain owner has
    not published anti-phishing policy.

    Returns:
        Tuple of (spf_result, dkim_result, dmarc_result).
        -1 = fail/not found, 0 = neutral/softfail, 1 = pass
    """
    spf_result = _check_spf(domain)
    dkim_result = _check_dkim(domain)
    dmarc_result = _check_dmarc(domain)
    return spf_result, dkim_result, dmarc_result


def _check_spf(domain: str) -> int:
    """Query SPF TXT record. Returns -1/0/1."""
    try:
        answers = dns.resolver.resolve(domain, "TXT", lifetime=3.0)
        for rdata in answers:
            txt = "".join(s.decode("utf-8", errors="ignore") for s in rdata.strings)
            if "v=spf1" in txt.lower():
                if "~all" in txt or "?all" in txt:
                    return 0    # Softfail / neutral
                if "-all" in txt:
                    return 1    # Strict pass (record exists with hard reject)
                return 1        # Record exists
        return -1               # No SPF record
    except (dns.exception.DNSException, Exception):
        return -1


def _check_dkim(domain: str) -> int:
    """Query common DKIM selector TXT records. Returns -1/0/1."""
    common_selectors = ["default", "google", "mail", "k1", "s1", "s2"]
    for selector in common_selectors:
        try:
            dkim_domain = f"{selector}._domainkey.{domain}"
            dns.resolver.resolve(dkim_domain, "TXT", lifetime=2.0)
            return 1    # DKIM record found for at least one common selector
        except Exception:
            continue
    return -1           # No DKIM record found


def _check_dmarc(domain: str) -> int:
    """Query DMARC TXT record. Returns -1/0/1."""
    try:
        answers = dns.resolver.resolve(f"_dmarc.{domain}", "TXT", lifetime=3.0)
        for rdata in answers:
            txt = "".join(s.decode("utf-8", errors="ignore") for s in rdata.strings)
            if "v=dmarc1" in txt.lower():
                if "p=reject" in txt.lower():
                    return 1    # Strong policy
                if "p=quarantine" in txt.lower():
                    return 0    # Moderate policy
                return -1       # p=none = no enforcement
        return -1
    except Exception:
        return -1


def _check_message_id_suspicious(message_id: str) -> bool:
    """Detect suspicious or missing Message-ID headers.

    Security rationale: Legitimate mail servers always generate a
    Message-ID in the format <random@domain>. A missing, malformed,
    or freemail-domain Message-ID indicates relay abuse or spoofing.
    """
    if not message_id:
        return True     # Missing = suspicious
    # Check format: should be <something@domain>
    if not re.match(r"<[^@]+@[^>]+>", message_id.strip()):
        return True     # Malformed
    # Freemail domain in Message-ID
    mid_domain = message_id.split("@")[-1].rstrip(">").lower()
    for fm in FREEMAIL_DOMAINS:
        if mid_domain.startswith(fm.split("@")[-1]):
            return True
    return False


def _check_timezone_mismatch(date_header: str, received_headers: List[str]) -> bool:
    """Detect timezone mismatch between Date header and relay chain.

    Security rationale: Automated phishing tools often use incorrect
    timezones or copy timestamps, producing obvious mismatches.
    """
    if not date_header or not received_headers:
        return False
    try:
        date_tz_match = _TZ_RE.search(date_header)
        if not date_tz_match:
            return False
        date_tz = date_tz_match.group(1)
        # Check last received header (closest to origin)
        last_received = received_headers[-1] if received_headers else ""
        recv_tz_match = _TZ_RE.search(last_received)
        if not recv_tz_match:
            return False
        recv_tz = recv_tz_match.group(1)
        return date_tz != recv_tz
    except Exception:
        return False


def _check_xmailer_suspicious(x_mailer: str) -> bool:
    """Detect confirmed bulk-sender X-Mailer fingerprints.

    A MISSING X-Mailer is NOT suspicious — modern webmail clients (Gmail,
    Outlook Web Access, Apple Mail, Yahoo Mail) do not send X-Mailer at all.
    Only flag strings that match known mass-mailing software.
    """
    if not x_mailer:
        return False  # Missing = normal for modern webmail
    return bool(_BULK_MAILER_RE.search(x_mailer))


def _extract_domain(address: str) -> Optional[str]:
    """Extract the domain from an email address or bare domain."""
    if not address:
        return None
    # Handle formats: "Name <email@domain.com>", "email@domain.com", "<email@domain>"
    match = re.search(r"[\w.+-]+@([\w.-]+\.[a-zA-Z]{2,})", address)
    if match:
        return match.group(1)
    # Bare domain
    if re.match(r"^[\w.-]+\.[a-zA-Z]{2,}$", address.strip()):
        return address.strip()
    return None


def _default_header_features() -> Dict:
    """Return zero-value defaults for all header features."""
    return {
        "from_reply_to_mismatch": 0,
        "from_return_path_mismatch": 0,
        "reply_to_freemail": 0,
        "received_hop_count": 0,
        "received_geo_anomaly": 0,
        "spf_result": 0,
        "dkim_result": 0,
        "dmarc_result": 0,
        "message_id_suspicious": 0,
        "timezone_mismatch": 0,
        "x_mailer_suspicious": 0,
        "header_injection_attempt": 0,
    }


def _parse_auth_results_header(
    auth_results: str, received_spf: str = ""
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Parse SPF/DKIM/DMARC results from inline Authentication-Results / Received-SPF.

    Returns (spf, dkim, dmarc) each as 1/0/-1, or None if not found in headers.
    This lets us populate auth features without any DNS network call.
    """
    spf: Optional[int] = None
    dkim: Optional[int] = None
    dmarc: Optional[int] = None
    text = (auth_results + " " + received_spf).lower()
    m = re.search(r"\bspf\s*=\s*(\w+)", text)
    if m:
        spf = _AUTH_RESULT_SCORES.get(m.group(1))
    m = re.search(r"\bdkim\s*=\s*(\w+)", text)
    if m:
        dkim = _AUTH_RESULT_SCORES.get(m.group(1))
    m = re.search(r"\bdmarc\s*=\s*(\w+)", text)
    if m:
        dmarc = _AUTH_RESULT_SCORES.get(m.group(1))
    return spf, dkim, dmarc
