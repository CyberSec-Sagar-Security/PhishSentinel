"""
PhishLens Indicator of Compromise (IOC) Extractor.

Extracts structured IOCs from phishing emails in a format compatible with
threat intelligence platforms (MISP, TheHive, OpenCTI, Splunk SIEM).

IOC types extracted:
  - Sender IP addresses (from Received: headers)
  - URLs (full and domain-only)
  - Email addresses (sender, reply-to, return-path)
  - Attachment file hashes (MD5, SHA256 if computed)
  - Malicious domains (cross-referenced against extracted URL features)

Output format:
  - Python dict for in-memory use
  - MISP-compatible JSON for threat intelligence platform ingestion
  - Syslog-compatible string for SIEM forwarding

Security rationale: IOC extraction transforms PhishLens from a passive detector
into an active threat intelligence source. When a phishing campaign is detected,
extracted IOCs can be immediately pushed to:
  - MISP for community sharing
  - Splunk/ELK for automated SIEM rule creation
  - Firewall blocklists for proactive infrastructure defence
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from src.utils.logger import get_logger

log = get_logger(__name__)

# Regex patterns for IOC extraction
_IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_URL_PATTERN = re.compile(r"https?://[^\s<>\"'\)\(]+", re.IGNORECASE)
_HASH_MD5 = re.compile(r"\b[a-fA-F0-9]{32}\b")
_HASH_SHA256 = re.compile(r"\b[a-fA-F0-9]{64}\b")

# Private/reserved IP ranges to exclude from IOCs
_PRIVATE_PREFIXES = ("10.", "192.168.", "127.", "0.", "172.16.", "172.17.",
                     "172.18.", "172.19.", "172.2", "172.3")

# ---------------------------------------------------------------------------
# MITRE ATT&CK technique mapping (static — curated for phishing context)
# ---------------------------------------------------------------------------

_ATTACK_TECHNIQUES = [
    {
        "technique_id":   "T1566.001",
        "technique_name": "Phishing: Spearphishing Attachment",
        "tactic":         "Initial Access",
        "indicator":      "attachment",
    },
    {
        "technique_id":   "T1566.002",
        "technique_name": "Phishing: Spearphishing Link",
        "tactic":         "Initial Access",
        "indicator":      "url",
    },
    {
        "technique_id":   "T1583.001",
        "technique_name": "Acquire Infrastructure: Domains",
        "tactic":         "Resource Development",
        "indicator":      "domain",
    },
    {
        "technique_id":   "T1036.007",
        "technique_name": "Masquerading: Double File Extension",
        "tactic":         "Defense Evasion",
        "indicator":      "attachment",
    },
]


# ---------------------------------------------------------------------------
# IOCReport dataclass
# ---------------------------------------------------------------------------


@dataclass
class IOCReport:
    """Structured Indicator of Compromise report from a PhishLens analysis.

    This dataclass provides a clean, typed container for all IOCs extracted
    from a single email, together with export methods for threat intelligence
    platforms (MISP) and human-readable summaries.

    Security rationale: A dataclass with explicit fields prevents accidental
    omission of IOC categories and makes the schema self-documenting for
    security analysts integrating PhishLens into their SIEM workflows.

    Attributes:
        sender_emails: List of sender/reply-to/return-path email addresses.
        sender_ips:    List of non-private IPs from Received: headers.
        urls:          List of unique URLs found in the email body.
        domains:       List of unique domains extracted from URLs.
        attachment_hashes: List of dicts with 'filename', 'sha256', 'md5'.
        raw_email_hash: SHA-256 fingerprint of the full raw email (dedup key).
        timestamp:     ISO 8601 UTC timestamp of extraction.
        risk_score:    ML phishing probability (0.0–1.0), -1 if unavailable.
        attack_techniques: List of MITRE ATT&CK technique dicts triggered.
    """

    sender_emails:      List[str]  = field(default_factory=list)
    sender_ips:         List[str]  = field(default_factory=list)
    urls:               List[str]  = field(default_factory=list)
    domains:            List[str]  = field(default_factory=list)
    attachment_hashes:  List[Dict] = field(default_factory=list)
    raw_email_hash:     str        = ""
    timestamp:          str        = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    risk_score:         float      = -1.0
    attack_techniques:  List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain Python dict (JSON-serialisable)."""
        return {
            "sender_emails":     self.sender_emails,
            "sender_ips":        self.sender_ips,
            "urls":              self.urls,
            "domains":           self.domains,
            "attachment_hashes": self.attachment_hashes,
            "raw_email_hash":    self.raw_email_hash,
            "timestamp":         self.timestamp,
            "risk_score":        self.risk_score,
            "attack_techniques": self.attack_techniques,
        }

    def to_misp_json(self, campaign_name: str = "PhishLens Detection") -> str:
        """Export this IOCReport as a MISP-compatible JSON event string.

        MISP JSON is the de facto standard for threat intelligence sharing,
        accepted by MISP, OpenCTI, TAXII servers, and most SIEMs.

        Args:
            campaign_name: Human-readable campaign/event name.

        Returns:
            JSON string formatted for MISP API ingestion.
        """
        return iocs_to_misp_json(self.to_dict(), campaign_name=campaign_name)

    def to_text_summary(self) -> str:
        """Generate a concise plain-text summary of all extracted IOCs.

        Designed for display in Streamlit's st.code() widget and for
        plain-text download by SOC analysts.

        Returns:
            Multi-line string summarising all IOC categories.
        """
        lines = [
            "═" * 60,
            "  PhishLens IOC Report",
            f"  Generated : {self.timestamp}",
            f"  Risk Score: {self.risk_score:.1%}" if self.risk_score >= 0 else "  Risk Score: N/A",
            "═" * 60,
        ]

        def _section(title: str, items: List[str]) -> None:
            lines.append(f"\n  {title}  ({len(items)} found)")
            lines.append("  " + "─" * 40)
            if items:
                for item in items[:50]:
                    lines.append(f"    {item}")
                if len(items) > 50:
                    lines.append(f"    ... and {len(items) - 50} more")
            else:
                lines.append("    (none)")

        _section("Sender Email Addresses", self.sender_emails)
        _section("Sender IPs (from Received headers)", self.sender_ips)
        _section("URLs", self.urls)
        _section("Domains", self.domains)

        if self.attachment_hashes:
            lines.append(f"\n  Attachment Hashes  ({len(self.attachment_hashes)} found)")
            lines.append("  " + "─" * 40)
            for h in self.attachment_hashes:
                fname = h.get("filename", "unknown")
                sha   = h.get("sha256", "")
                md5   = h.get("md5", "")
                lines.append(f"    {fname}")
                if sha:
                    lines.append(f"      SHA256: {sha}")
                if md5:
                    lines.append(f"      MD5:    {md5}")

        if self.attack_techniques:
            lines.append(f"\n  MITRE ATT&CK Techniques  ({len(self.attack_techniques)} matched)")
            lines.append("  " + "─" * 40)
            for t in self.attack_techniques:
                lines.append(f"    [{t.get('technique_id', '?')}] {t.get('technique_name', '?')}")
                lines.append(f"      Tactic: {t.get('tactic', '?')}")

        lines.append(f"\n  Raw Email SHA-256: {self.raw_email_hash or '(not computed)'}")
        lines.append("═" * 60)
        return "\n".join(lines)


def extract_iocs(
    parsed_email: Dict,
    risk_score: float = -1.0,
    feature_flags: Optional[Dict] = None,
) -> Dict:
    """Extract all IOCs from a parsed email dict.

    This function extracts structured IOC data and, when risk_score is provided,
    maps triggered MITRE ATT&CK techniques based on detected indicators.

    Args:
        parsed_email: Output of eml_parser.parse_eml_string() or similar.
        risk_score:   Phishing probability from the ML model (0.0–1.0).
                      Pass -1.0 when not available (e.g., during data ingestion).
        feature_flags: Optional dict of the ML feature vector (from pipeline)
                       used to power the full ATT&CK technique mapping.
                       When provided, map_attack_techniques() is called.
                       When absent, falls back to simple indicator-based mapping.

    Returns:
        IOC dict compatible with the existing app.py interface:
          - sender_emails: List of sender/reply-to/return-path addresses
          - sender_ips: List of unique non-private IPs from Received headers
          - urls: List of unique URLs
          - domains: List of unique domains (from URLs)
          - attachment_hashes: List of attachment hash dicts
          - raw_email_hash: SHA256 of the full raw email (for deduplication)
          - timestamp: ISO 8601 extraction timestamp
          - risk_score: ML phishing probability
          - attack_techniques: MITRE ATT&CK techniques triggered
    """
    iocs: Dict[str, Any] = {
        "sender_emails": [],
        "sender_ips": [],
        "urls": [],
        "cleaned_urls": [],
        "url_cleaning_map": {},
        "domains": [],
        "attachment_hashes": [],
        "raw_email_hash": "",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "risk_score": risk_score,
        "attack_techniques": [],
    }

    # --- Sender email addresses -------------------------------------------
    email_addrs = set()
    for field in ("from_address", "reply_to", "return_path"):
        val = parsed_email.get(field, "")
        if val:
            matches = _EMAIL_PATTERN.findall(val)
            email_addrs.update(matches)
    iocs["sender_emails"] = sorted(email_addrs)

    # --- Sender IPs from Received: headers --------------------------------
    received = parsed_email.get("received_headers", [])
    ips: set = set()
    for header in received:
        for ip in _IP_PATTERN.findall(header):
            if not any(ip.startswith(p) for p in _PRIVATE_PREFIXES):
                ips.add(ip)
    iocs["sender_ips"] = sorted(ips)

    # --- URLs — extract then deobfuscate/clean ----------------------------
    urls_raw = parsed_email.get("urls", [])
    # Also extract from body text (catches obfuscated URLs not caught by parser)
    body = parsed_email.get("body_text", "") or parsed_email.get("body_html", "") or ""
    urls_from_body = _URL_PATTERN.findall(body)
    all_urls = list(set(urls_raw + urls_from_body))
    # Sanitise: remove trailing punctuation
    sanitised_urls = [u.rstrip(".,;)'\"") for u in all_urls]
    sanitised_urls = sorted(set(sanitised_urls))

    # Deobfuscate wrapped URLs (SafeLinks, Proofpoint, Google redirect, etc.)
    try:
        from src.features.url_cleaner import build_url_cleaning_map
        url_cleaning_map = build_url_cleaning_map(sanitised_urls, follow_redirects=True)
    except Exception as _exc:
        log.debug(f"URL cleaning failed: {_exc}")
        url_cleaning_map = {u: {"original": u, "cleaned": u, "method": "none", "was_wrapped": False}
                            for u in sanitised_urls}

    # Store original URLs (as extracted) — used for display
    iocs["urls"] = sanitised_urls
    # Store cleaned (real destination) URLs — used for TI analysis
    iocs["cleaned_urls"] = sorted({v["cleaned"] for v in url_cleaning_map.values()})
    # Store the full cleaning map for per-URL display
    iocs["url_cleaning_map"] = url_cleaning_map

    # --- Domains — from CLEANED urls (reveals real attacker infrastructure) --
    domains: set = set()
    for url in iocs["cleaned_urls"]:
        try:
            parsed = urlparse(url)
            if parsed.netloc:
                domains.add(parsed.netloc.lower().lstrip("www."))
        except Exception:
            pass
    iocs["domains"] = sorted(domains)

    # --- Attachment hashes -----------------------------------------------
    iocs["attachment_hashes"] = parsed_email.get("attachment_hashes", [])

    # --- Raw email hash (for deduplication) ------------------------------
    raw_email = parsed_email.get("raw_email", "") or ""
    if raw_email:
        iocs["raw_email_hash"] = hashlib.sha256(raw_email.encode("utf-8", errors="replace")).hexdigest()

    # --- MITRE ATT&CK technique mapping ----------------------------------
    # When feature_flags (the ML feature dict) is provided, use the full
    # attack_mapping module for richer, evidence-backed ATT&CK mapping.
    # Fall back to simple indicator-type matching when not available.
    if feature_flags:
        try:
            from src.attack_mapping import map_attack_techniques as _map_atk
            iocs["attack_techniques"] = _map_atk(feature_flags, iocs)
        except Exception as exc:
            log.debug(f"attack_mapping fallback: {exc}")
            iocs["attack_techniques"] = _simple_attack_mapping(iocs)
    else:
        iocs["attack_techniques"] = _simple_attack_mapping(iocs)

    log.debug(
        f"IOCs extracted: {len(iocs['sender_emails'])} emails, "
        f"{len(iocs['sender_ips'])} IPs, "
        f"{len(iocs['urls'])} URLs "
        f"({sum(1 for v in iocs['url_cleaning_map'].values() if v['was_wrapped'])} wrapped/cleaned), "
        f"{len(iocs['domains'])} domains"
    )
    return iocs


def _simple_attack_mapping(iocs: Dict) -> List[Dict]:
    """Fallback: simple indicator-type-based ATT&CK technique mapping.

    Used when feature_flags is not provided (e.g., during raw data ingestion).
    The full map_attack_techniques() function requires the ML feature vector.
    """
    triggered: List[Dict] = []
    has_urls = len(iocs.get("urls", [])) > 0
    has_attachments = len(iocs.get("attachment_hashes", [])) > 0
    has_domains = len(iocs.get("domains", [])) > 0
    for technique in _ATTACK_TECHNIQUES:
        indicator = technique.get("indicator", "")
        if indicator == "url" and has_urls:
            triggered.append(technique)
        elif indicator == "attachment" and has_attachments:
            triggered.append(technique)
        elif indicator == "domain" and has_domains:
            triggered.append(technique)
    return triggered


def iocs_to_misp_json(iocs: Dict, campaign_name: str = "PhishLens Detection") -> str:
    """Convert an IOC dict to MISP-compatible JSON event format.

    MISP JSON format is the de facto standard for threat intelligence sharing
    and is accepted by MISP, OpenCTI, TAXII servers, and most SIEMs.

    Args:
        iocs: Output of extract_iocs().
        campaign_name: Human-readable campaign/event name.

    Returns:
        JSON string formatted for MISP API ingestion.
    """
    attributes = []

    for email in iocs.get("sender_emails", []):
        attributes.append({
            "type": "email-src",
            "category": "Payload delivery",
            "value": email,
            "comment": "Sender email address extracted by PhishLens",
            "to_ids": True,
        })

    for ip in iocs.get("sender_ips", []):
        attributes.append({
            "type": "ip-src",
            "category": "Network activity",
            "value": ip,
            "comment": "Sender IP from Received headers",
            "to_ids": True,
        })

    for url in iocs.get("urls", [])[:50]:   # Cap at 50 URLs per MISP event
        attributes.append({
            "type": "url",
            "category": "External analysis",
            "value": url,
            "comment": "Phishing URL extracted by PhishLens",
            "to_ids": True,
        })

    for domain in iocs.get("domains", [])[:50]:
        attributes.append({
            "type": "domain",
            "category": "Network activity",
            "value": domain,
            "comment": "Phishing domain extracted by PhishLens",
            "to_ids": True,
        })

    for h in iocs.get("attachment_hashes", []):
        if isinstance(h, dict):
            if "sha256" in h:
                attributes.append({
                    "type": "sha256",
                    "category": "Payload delivery",
                    "value": h["sha256"],
                    "comment": f"Attachment hash: {h.get('filename', 'unknown')}",
                    "to_ids": True,
                })

    misp_event = {
        "Event": {
            "info": campaign_name,
            "date": iocs.get("timestamp", "")[:10],
            "threat_level_id": "2",     # High
            "analysis": "1",            # Initial
            "distribution": "0",        # Your org only
            "Attribute": attributes,
            "Tag": [{"name": "phishing"}, {"name": "PhishLens"}],
        }
    }
    return json.dumps(misp_event, indent=2)


# ============================================================
# IOC Explanation Engine — human-readable per-IOC risk analysis
# ============================================================

_RISKY_TLDS_EX = {
    ".ru", ".xyz", ".tk", ".ml", ".ga", ".cf", ".gq", ".top", ".icu",
    ".vip", ".bid", ".win", ".download", ".racing", ".review", ".stream",
    ".gdn", ".loan", ".click", ".link", ".online", ".site", ".website",
    ".space", ".fun", ".monster", ".buzz", ".club", ".live", ".store",
    ".cc", ".su", ".pw", ".ws", ".in", ".info",
}

_PHISH_KW = [
    "verify", "secure", "update", "confirm", "login", "signin", "account",
    "banking", "password", "credential", "paypal", "amazon", "appleid",
    "microsoft", "office365", "support", "service", "validate", "suspend",
    "alert", "warning", "unusual", "invoice", "statement", "reset",
]

_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co", "rebrand.ly",
    "tr.im", "is.gd", "buff.ly", "tiny.cc", "cutt.ly", "rb.gy",
}

_BRAND_MAP = {
    "paypal": "paypal.com", "amazon": "amazon.com", "microsoft": "microsoft.com",
    "google": "google.com", "apple": "apple.com", "netflix": "netflix.com",
    "facebook": "facebook.com", "meta": "meta.com", "instagram": "instagram.com",
    "twitter": "twitter.com", "linkedin": "linkedin.com", "dropbox": "dropbox.com",
    "chase": "chase.com", "wellsfargo": "wellsfargo.com",
    "bankofamerica": "bankofamerica.com", "citibank": "citibank.com",
    "ebay": "ebay.com", "dhl": "dhl.com", "fedex": "fedex.com",
    "docusign": "docusign.com", "usps": "usps.com", "ups": "ups.com",
}

_FREEMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "protonmail.com", "icloud.com", "yandex.com", "tutanota.com",
    "gmx.com", "live.com", "mail.com", "zoho.com", "qq.com",
}


def _sev_rank(s: str) -> int:
    return {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "INFO": 0}.get(s, 0)


def _bump(cur: str, new: str) -> str:
    return new if _sev_rank(new) > _sev_rank(cur) else cur


def _entropy(s: str) -> float:
    import math
    if not s:
        return 0.0
    f = {}
    for c in s:
        f[c] = f.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in f.values())


def _url_parts(url: str):
    try:
        import tldextract
        from urllib.parse import urlparse
        ext = tldextract.extract(url)
        p = urlparse(url)
        return ext.subdomain, ext.domain, ext.suffix, p.netloc.lower(), p.path, p.port
    except Exception:
        return "", "", "", "", "", None


def _dom_from_addr(addr: str) -> str:
    import re
    m = re.search(r"@([\w.\-]+)", addr)
    return m.group(1).lower() if m else ""


def _analyze_url_risk_for_display(url: str) -> Optional[Dict]:
    """Lightweight per-URL risk analysis — lexical only, no network calls."""
    import re as _re
    try:
        subdomain, domain, suffix, full_domain, path, port = _url_parts(url)
        risks, sev = [], "INFO"
        tld = f".{suffix.lower()}" if suffix else ""

        if tld in _RISKY_TLDS_EX:
            risks.append(f"Non-reputable TLD ({tld}): this TLD is disproportionately used for phishing and malware distribution — often available at no cost, enabling throwaway infrastructure")
            sev = _bump(sev, "HIGH")

        brand_in_sub = next((b for b in _BRAND_MAP if b in subdomain.lower()), None)
        brand_in_dom = next((b for b in _BRAND_MAP if b in domain.lower()), None)

        if brand_in_sub:
            legit = _BRAND_MAP[brand_in_sub]
            if legit not in full_domain:
                risks.append(f"Brand impersonation in subdomain: '{brand_in_sub.capitalize()}' appears in '{subdomain}' but the actual registered domain is '{domain}.{suffix}' — NOT the official {legit}. This is a classic subdomain-spoofing trick designed to make users trust the URL at first glance.")
                sev = _bump(sev, "CRITICAL")
        if brand_in_dom and not brand_in_sub:
            legit = _BRAND_MAP[brand_in_dom]
            if full_domain != legit and not full_domain.endswith(f".{legit}"):
                risks.append(f"Domain impersonation: '{brand_in_dom.capitalize()}' is embedded in the domain name ('{domain}.{suffix}') but this is NOT the official {legit}. Likely a lookalike/typosquat domain registered specifically for this campaign.")
                sev = _bump(sev, "HIGH")

        if _re.match(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url):
            risks.append("Raw IP address used as host: legitimate organizations always use domain names. Raw IPs bypass domain reputation checks and are a near-universal phishing indicator.")
            sev = _bump(sev, "HIGH")

        shortener = next((s for s in _SHORTENERS if s in full_domain), None)
        if shortener:
            risks.append(f"URL shortener ({shortener}): hides the true destination URL, enabling phishers to bypass URL-based email security filters and obscure the malicious domain from the recipient.")
            sev = _bump(sev, "MEDIUM")

        kws = [kw for kw in _PHISH_KW if kw in url.lower()][:4]
        if kws:
            risks.append(f"Credential-harvesting keywords in URL [{', '.join(kws)}]: these terms are embedded to make the URL appear to be a legitimate login or account verification page.")
            sev = _bump(sev, "MEDIUM")

        if domain:
            dr = sum(c.isdigit() for c in domain) / max(len(domain), 1)
            if dr > 0.4:
                risks.append(f"High digit density in domain name ({dr:.0%} digits): suggests an algorithmically-generated or disposable domain registered specifically for this phishing campaign.")
                sev = _bump(sev, "MEDIUM")

        ent = _entropy(url)
        if ent > 5.2:
            risks.append(f"High URL entropy ({ent:.2f}/6.0): the URL contains highly random characters consistent with obfuscated tracking parameters or auto-generated phishing infrastructure designed to evade signature-based detection.")
            sev = _bump(sev, "MEDIUM")

        if "xn--" in url.lower():
            risks.append("Punycode/internationalized domain detected: domain uses Unicode characters that visually mimic Latin letters (e.g., 'pаypal.com' using Cyrillic 'а') — a homoglyph attack to deceive visual inspection.")
            sev = _bump(sev, "HIGH")

        if "@" in full_domain:
            risks.append("Credential spoofing via @ in URL: browsers treat everything before @ as credentials. A URL like paypal.com@attacker.com actually connects to attacker.com, ignoring the 'paypal.com' portion entirely.")
            sev = _bump(sev, "CRITICAL")

        if port and port not in (80, 443):
            risks.append(f"Non-standard port {port}: legitimate email links exclusively use port 80 (HTTP) or 443 (HTTPS). Custom ports indicate rogue web servers or command-and-control infrastructure.")
            sev = _bump(sev, "MEDIUM")

        if subdomain and len(subdomain.split(".")) >= 3:
            risks.append(f"Deeply nested subdomain ({subdomain}.{domain}.{suffix}): multiple subdomain levels are used to make the URL appear to begin with a legitimate domain while the actual registered domain at the end is malicious.")
            sev = _bump(sev, "LOW")

        if url.startswith("http://") and kws:
            risks.append("Unencrypted HTTP used for a page requesting sensitive data: no legitimate service requests credentials or payments over unencrypted HTTP.")
            sev = _bump(sev, "HIGH")

        path_depth = len([p for p in path.split("/") if p])
        if path_depth >= 4:
            risks.append(f"Deep URL path ({path_depth} levels): multi-level paths like /secure/login/verify/account mimic legitimate site navigation while routing through phishing infrastructure.")

        if not risks:
            risks.append("No high-risk lexical indicators detected for this URL.")
            sev = "INFO"

        return {"url": url, "domain": full_domain or f"{domain}.{suffix}", "severity": sev, "risks": risks}
    except Exception as exc:
        log.debug(f"URL risk analysis error for '{url[:60]}': {exc}")
        return None


def _explain_headers_for_display(parsed_email: Dict, feature_dict: Dict) -> List[Dict]:
    """Generate structured header anomaly findings for UI display."""
    findings = []
    f = feature_dict
    from_addr = parsed_email.get("from_address", "")
    from_dom = _dom_from_addr(from_addr)

    _ABSENT = 99.0

    def _fv(key):
        v = f.get(key, _ABSENT)
        return float(v) if v is not None else _ABSENT

    spf, dkim, dmarc = _fv("hdr_spf_result"), _fv("hdr_dkim_result"), _fv("hdr_dmarc_result")

    if spf != _ABSENT:
        if spf < 0:
            findings.append({"severity": "HIGH", "icon": "⛔", "title": "SPF Authentication FAILED",
                "detail": f"The server delivering this email is NOT authorized in the SPF DNS record for '{from_dom}'. The From address has been forged — the sending server has no permission to represent this domain."})
        elif spf == 0:
            findings.append({"severity": "MEDIUM", "icon": "⚠️", "title": "SPF Neutral / No SPF Policy",
                "detail": f"No SPF enforcement found for '{from_dom}'. Legitimate organizations configure SPF to prevent domain impersonation. The absence of SPF policy allows anyone to send email as this domain."})
        else:
            findings.append({"severity": "INFO", "icon": "✅", "title": "SPF Authentication Passed",
                "detail": f"The sending server is authorized to deliver email for '{from_dom}'."})

    if dkim != _ABSENT:
        if dkim < 0:
            findings.append({"severity": "HIGH", "icon": "⛔", "title": "DKIM Signature Invalid / Missing",
                "detail": "This email lacks a valid cryptographic DKIM signature. All legitimate transactional emails (PayPal, banks, Amazon) include DKIM to cryptographically prove the message was not tampered with in transit."})
        elif dkim == 0:
            findings.append({"severity": "MEDIUM", "icon": "⚠️", "title": "DKIM Signature Not Verified",
                "detail": "DKIM signature could not be verified against the DNS public key. Message integrity cannot be confirmed."})
        else:
            findings.append({"severity": "INFO", "icon": "✅", "title": "DKIM Signature Valid",
                "detail": "Cryptographic signature verified — message content was not modified in transit."})

    if dmarc != _ABSENT:
        if dmarc < 0:
            findings.append({"severity": "CRITICAL", "icon": "🚨", "title": "DMARC Policy VIOLATED",
                "detail": f"The domain '{from_dom}' has a DMARC policy that explicitly rejects or quarantines this message. This is the strongest possible authentication signal: the legitimate domain owner has declared messages failing SPF/DKIM alignment should be rejected outright."})
        elif dmarc == 0:
            findings.append({"severity": "LOW", "icon": "⚠️", "title": "DMARC: Monitor-Only Policy (p=none)",
                "detail": f"'{from_dom}' has a DMARC policy of 'none' — monitoring only, no enforcement. No action is taken against failed messages, which allows impersonation."})

    if f.get("hdr_from_reply_to_mismatch"):
        reply_to = parsed_email.get("reply_to", "—")
        findings.append({"severity": "HIGH", "icon": "⛔", "title": "From ≠ Reply-To Domain Mismatch",
            "detail": f"From: {from_addr}\nReply-To: {reply_to}\n\nReplies are directed to a different domain than the claimed sender. This is a classic phishing technique to intercept victim responses while appearing to originate from a trusted organization."})

    if f.get("hdr_from_return_path_mismatch"):
        ret_path = parsed_email.get("return_path", "—")
        findings.append({"severity": "MEDIUM", "icon": "⚠️", "title": "From ≠ Return-Path Domain Mismatch",
            "detail": f"From: {from_addr}\nReturn-Path: {ret_path}\n\nBounce messages go to a different domain, indicating the email passed through infrastructure controlled by a separate party."})

    if f.get("hdr_reply_to_freemail"):
        findings.append({"severity": "MEDIUM", "icon": "⚠️", "title": "Reply-To Uses Free Email Provider",
            "detail": f"Reply-To: {parsed_email.get('reply_to', '—')}\n\nPhishers redirect victim replies to anonymously-created free email accounts (Gmail, Hotmail, Yahoo) to intercept responses without exposing their identity."})

    hop_count = int(f.get("hdr_received_hop_count", 0))
    if hop_count > 6:
        findings.append({"severity": "MEDIUM", "icon": "⚠️", "title": f"Excessive Relay Chain ({hop_count} hops)",
            "detail": f"Email passed through {hop_count} mail servers before delivery. Legitimate emails typically traverse 2–4 hops. Excessive hops suggest obfuscated routing through compromised relay servers to obscure the true origin."})

    if f.get("hdr_received_geo_anomaly"):
        findings.append({"severity": "MEDIUM", "icon": "⚠️", "title": "Geographic Origin Anomaly",
            "detail": "Email originated from a geographic region inconsistent with the claimed sender domain (based on Received header timezone analysis). Legitimate organizations send from predictable, consistent infrastructure."})

    if f.get("hdr_x_mailer_suspicious"):
        findings.append({"severity": "LOW", "icon": "⚠️", "title": "Suspicious X-Mailer / Sending Tool",
            "detail": f"X-Mailer: {parsed_email.get('x_mailer', '—')}\n\nThe email client identifier matches fingerprints of known bulk-sending or phishing tools."})

    return findings


def _explain_email_addresses_for_display(parsed_email: Dict) -> List[Dict]:
    """Analyze sender addresses for impersonation and fraud indicators."""
    import re as _re
    findings = []

    def _parts(addr: str):
        m = _re.search(r"([\w._%+\-]+)@([\w.\-]+)", addr)
        return (m.group(1), m.group(2).lower()) if m else (None, None)

    from_addr = parsed_email.get("from_address", "")
    if from_addr:
        user, domain = _parts(from_addr)
        if user and domain:
            if domain in _FREEMAIL_DOMAINS:
                findings.append({"address": from_addr, "type": "From", "severity": "MEDIUM",
                    "flags": [f"Sender uses free email provider ({domain}). Legitimate businesses and financial institutions do not use free email for transactional or official communications."]})

            dn_m = _re.match(r"(.+?)\s*<", from_addr)
            if dn_m:
                dn = dn_m.group(1).strip().strip('"')
                brand = next((b for b in _BRAND_MAP if b in dn.lower()), None)
                if brand:
                    legit = _BRAND_MAP[brand]
                    if not domain.endswith(legit) and domain != legit:
                        findings.append({"address": from_addr, "type": "From", "severity": "CRITICAL",
                            "flags": [f"Display name fraud: email claims to be '{dn}' ({brand.capitalize()}) but the actual sending domain is '{domain}' — NOT the official {legit}. This is the primary deception technique used in this phishing email."]})
                    else:
                        findings.append({"address": from_addr, "type": "From", "severity": "INFO",
                            "flags": [f"Legitimate {brand.capitalize()} domain confirmed ({domain} matches {legit})."]})

            if domain.count(".") >= 2:
                root = ".".join(domain.split(".")[-2:])
                brand_in_root = next((b for b in _BRAND_MAP if b in root), None)
                brand_in_sub_part = next((b for b in _BRAND_MAP if b in domain.split(".")[0]), None)
                if brand_in_sub_part and not brand_in_root:
                    findings.append({"address": from_addr, "type": "From", "severity": "HIGH",
                        "flags": [f"Subdomain abuse: '{brand_in_sub_part.capitalize()}' appears in the email subdomain but the actual registered domain '{root}' has no affiliation with {brand_in_sub_part.capitalize()}. Attackers craft addresses like 'noreply@paypal.attacker.com' to deceive visual inspection."]})

    ret_path = parsed_email.get("return_path", "")
    if ret_path:
        _, rp_dom = _parts(ret_path)
        _, fr_dom = _parts(from_addr)
        if rp_dom and fr_dom and rp_dom != fr_dom:
            findings.append({"address": ret_path, "type": "Return-Path", "severity": "MEDIUM",
                "flags": [f"Return-Path domain '{rp_dom}' differs from From domain '{fr_dom}'. Delivery failure bounces route to separate infrastructure — evidence of a third-party relay or compromised mail server in the delivery chain."]})

    return findings


def generate_ioc_explanations(
    parsed_email: Dict,
    feature_dict: Dict,
    iocs: Dict,
) -> Dict:
    """Generate human-readable risk explanations for every extracted IOC.

    Performs lightweight (zero network calls) analysis of each IOC category
    using the ML feature vector and parsed email structure. Designed for
    SOC analyst display to explain exactly WHY each indicator is suspicious.

    Args:
        parsed_email: Output of eml_parser.parse_eml_bytes().
        feature_dict: ML feature dict from pipeline.transform_single().
        iocs: IOC dict from extract_iocs().

    Returns:
        Dict with:
          header_findings   — List[Dict]  header anomaly explanations
          url_findings      — List[Dict]  per-URL risk analysis
          email_findings    — List[Dict]  per-address fraud indicators
          html_findings     — List[Dict]  HTML body obfuscation findings
          summary_findings  — List[str]   high-level risk factors
    """
    result: Dict = {
        "header_findings": [],
        "url_findings": [],
        "email_findings": [],
        "html_findings": [],
        "summary_findings": [],
    }

    try:
        result["header_findings"] = _explain_headers_for_display(parsed_email, feature_dict)
    except Exception as exc:
        log.debug(f"Header explanation error: {exc}")

    try:
        for url in iocs.get("urls", [])[:15]:
            if "w3.org" in url or len(url) < 10:
                continue
            f = _analyze_url_risk_for_display(url)
            if f:
                result["url_findings"].append(f)
    except Exception as exc:
        log.debug(f"URL explanation error: {exc}")

    try:
        result["email_findings"] = _explain_email_addresses_for_display(parsed_email)
    except Exception as exc:
        log.debug(f"Email explanation error: {exc}")

    try:
        fd = feature_dict
        hidden = int(fd.get("html_hidden_text_count", 0))
        ext_form = int(fd.get("html_external_form_action", 0))
        b64 = int(fd.get("html_base64_content_count", 0))
        js_n = int(fd.get("html_javascript_count", 0))
        html_links = int(fd.get("html_total_links", 0))
        link_div = float(fd.get("html_link_domain_diversity", 0))

        if ext_form:
            result["html_findings"].append({"severity": "CRITICAL", "icon": "🚨",
                "title": "External Form Action — Direct Credential Harvester",
                "detail": "The email HTML contains a <form> element submitting data to an external server. Any credentials, card numbers, or personal data entered will be transmitted directly to the attacker's infrastructure. This is a direct credential harvesting mechanism."})
        if hidden:
            result["html_findings"].append({"severity": "HIGH", "icon": "⛔",
                "title": f"Hidden Text Elements ({hidden} found) — Filter Evasion",
                "detail": "HTML contains text hidden via CSS (display:none, font-size:0, color:#ffffff, etc.). Used for: (1) keyword stuffing to confuse spam filters, (2) hiding tracking pixels, (3) concealing true URL destinations, (4) injecting attacker-controlled content invisible to the recipient."})
        if b64:
            result["html_findings"].append({"severity": "HIGH", "icon": "⛔",
                "title": f"Base64-Encoded Content ({b64} instances) — Obfuscation",
                "detail": "Email body contains base64-encoded content embedded in HTML. This obfuscation technique encodes malicious scripts or phishing URLs as base64 strings to bypass content-based security filters that scan for known bad patterns."})
        if js_n > 2:
            result["html_findings"].append({"severity": "MEDIUM", "icon": "⚠️",
                "title": f"JavaScript in Email Body ({js_n} script blocks)",
                "detail": f"{js_n} JavaScript blocks found in email HTML. JavaScript in email can: redirect browsers, harvest credentials on page load, fingerprint the victim's system, or perform drive-by downloads in email clients that render HTML with script execution."})
        if link_div > 0.75 and html_links > 5:
            result["html_findings"].append({"severity": "MEDIUM", "icon": "⚠️",
                "title": f"High Link Domain Diversity ({link_div:.0%} across {html_links} links)",
                "detail": f"Email contains {html_links} links spanning {link_div:.0%} distinct domains. Legitimate emails link to 1–3 domains. High diversity is a hallmark of mixed-content phishing: legitimate-looking links (Google, CDNs) are included alongside malicious links to pass filter scoring."})
    except Exception as exc:
        log.debug(f"HTML findings error: {exc}")

    try:
        sf = []
        if any(_sev_rank(f.get("severity", "INFO")) >= 4 for f in result["header_findings"]):
            sf.append("Email authentication critically failed — DMARC/SPF violation proves the sender domain was forged")
        if any(_sev_rank(f.get("severity", "INFO")) >= 3 for f in result["url_findings"]):
            bad = next((f["url"][:55] for f in result["url_findings"] if _sev_rank(f.get("severity","INFO")) >= 3), "")
            sf.append(f"High-risk URL detected: {bad}{'…' if bad else ''}")
        if any(_sev_rank(f.get("severity", "INFO")) >= 4 for f in result["email_findings"]):
            sf.append("Sender email impersonates a known legitimate brand via display-name fraud")
        if any("credential" in f.get("title","").lower() or "form" in f.get("title","").lower() for f in result["html_findings"]):
            sf.append("Credential harvesting form embedded in email HTML — direct data exfiltration mechanism")
        if any("hidden" in f.get("title","").lower() for f in result["html_findings"]):
            sf.append("Hidden text obfuscation detected — content concealed from reader but processed by email clients")
        fd = feature_dict
        urg = float(fd.get("txt_urgency_score_normalised", 0))
        if urg > 0.4:
            sf.append(f"High urgency language (score {urg:.0%}) — social engineering pressure tactics to override the recipient's critical judgment")
        result["summary_findings"] = sf
    except Exception as exc:
        log.debug(f"Summary findings error: {exc}")

    return result


def iocs_to_syslog(iocs: Dict, verdict: str = "PHISHING") -> str:
    """Format IOCs as a syslog-compatible CEF (Common Event Format) string.

    CEF format is accepted by Splunk, ArcSight, IBM QRadar, and most SIEMs.

    Args:
        iocs: Output of extract_iocs().
        verdict: PhishLens verdict ("PHISHING" or "LEGITIMATE").

    Returns:
        CEF-formatted syslog string.
    """
    ts = iocs.get("timestamp", datetime.utcnow().isoformat() + "Z")
    src_ip = iocs["sender_ips"][0] if iocs.get("sender_ips") else "unknown"
    src_email = iocs["sender_emails"][0] if iocs.get("sender_emails") else "unknown"
    url_count = len(iocs.get("urls", []))
    domain_count = len(iocs.get("domains", []))

    cef = (
        f"CEF:0|PhishLens|PhishLens|1.0|phishing_detection|{verdict}|10|"
        f"rt={ts} "
        f"src={src_ip} "
        f"suser={src_email} "
        f"cnt={url_count} "
        f"domainCount={domain_count} "
        f"rawEmailHash={iocs.get('raw_email_hash', '')} "
        f"verdict={verdict}"
    )
    return cef
