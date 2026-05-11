"""
PhishLens URL Deobfuscation & Cleaning Module.

Phishing emails frequently wrap real destination URLs inside corporate email
security gateways, URL shorteners, and custom redirect services to:
  - Bypass URL-based reputation filters (the wrapper looks legitimate)
  - Hide the real malicious domain from human readers
  - Evade sandboxes that only check the first URL hop

Supported wrappers:
  1. Microsoft SafeLinks  — *.safelinks.protection.outlook.com/?url=...
  2. Proofpoint URLDefense v2 — urldefense.proofpoint.com/v2/url?u=...
  3. Proofpoint URLDefense v3 — urldefense.com/v3/__<url>__
  4. Google Redirect       — google.com/url?q= | google.com/url?sa=t&url=
  5. Mimecast URL Protect  — protect-*.mimecast.com/s/...
  6. Barracuda Email Security — links.barracudanetworks.com/...
  7. Symantec MessageLabs  — symanteccloud.com / messagelabs
  8. Recursive percent-decoding — multi-layer %XX encoding
  9. URL shorteners        — bit.ly, t.co, goo.gl, tinyurl.com, ow.ly, etc.
     (optional: follows the redirect with a HEAD request, with strict timeout)

Security rationale: A SafeLinks URL wrapping cacvil.fin.ec/as/nw/dnt will
show "clean" on all TI checks because the outer domain is microsoft.com.
Unwrapping reveals the actual destination that must be scanned — this single
step can convert a false-negative into a true-positive detection.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Dict, List, Optional, Tuple

import requests

from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# URL shortener domains — we follow HEAD redirect for these
# ---------------------------------------------------------------------------
_SHORTENER_DOMAINS = frozenset([
    "bit.ly", "bitly.com",
    "t.co",
    "goo.gl",
    "tinyurl.com",
    "ow.ly",
    "buff.ly",
    "dlvr.it",
    "ift.tt",
    "is.gd",
    "tiny.cc",
    "rebrand.ly",
    "rb.gy",
    "cutt.ly",
    "short.link",
    "lnkd.in",
    "linktr.ee",
    "go.microsoft.com",
    "aka.ms",
])

# SafeLinks pattern — any Outlook/Exchange safelinks host
_SAFELINKS_RE = re.compile(
    r"https?://[a-z0-9\-]+\.safelinks\.protection\.outlook\.com/",
    re.IGNORECASE,
)

# Proofpoint v2 special encoding: replaces `-XX` hex sequences and `_` for `/`
_PP_V2_HEX_RE = re.compile(r"-([0-9A-Fa-f]{2})")

# Mimecast protect URL pattern
_MIMECAST_RE = re.compile(
    r"https?://protect(?:-[a-z0-9]+)?\.mimecast\.com/s/",
    re.IGNORECASE,
)

# Barracuda / BESS pattern
_BARRACUDA_RE = re.compile(
    r"https?://links\.barracudanetworks\.com/",
    re.IGNORECASE,
)

# Cisco Email Security (Ironport) / ESA pattern
_CISCO_ESA_RE = re.compile(
    r"https?://[a-z0-9\-]+\.cisco\.com/c/r/",
    re.IGNORECASE,
)

# Cleaning methods — human-readable labels for the UI
METHOD_SAFELINKS   = "Microsoft SafeLinks"
METHOD_PP_V2       = "Proofpoint URLDefense v2"
METHOD_PP_V3       = "Proofpoint URLDefense v3"
METHOD_GOOGLE      = "Google Redirect"
METHOD_MIMECAST    = "Mimecast URL Protect"
METHOD_BARRACUDA   = "Barracuda Network Links"
METHOD_PERCENT     = "Percent-Encoding"
METHOD_SHORTENER   = "URL Shortener (redirect)"
METHOD_CLEAN       = "none"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_url(url: str, follow_redirects: bool = True, timeout: int = 5) -> Dict:
    """Deobfuscate a single URL and return original + cleaned form.

    Args:
        url:               The raw URL as extracted from the email.
        follow_redirects:  If True, follow HTTP redirects for shortener domains
                           (single HEAD request, no recursive following).
        timeout:           Seconds for the redirect-following request.

    Returns:
        Dict with keys:
          - original  : str  — the input URL unchanged
          - cleaned   : str  — the real destination URL
          - method    : str  — description of what was detected/cleaned
          - was_wrapped: bool — True if the URL was a wrapper
    """
    result = {
        "original":    url,
        "cleaned":     url,
        "method":      METHOD_CLEAN,
        "was_wrapped": False,
    }

    if not url or not isinstance(url, str):
        return result

    current = url.strip()

    # Apply cleaning passes in order; each pass may enable the next
    for _attempt in range(6):   # max 6 rounds of unwrapping (e.g. double-wrapped)
        prev = current
        current, method = _single_clean_pass(current, follow_redirects, timeout)
        if current != prev:
            result["was_wrapped"] = True
            result["method"] = method
        else:
            break   # nothing changed — we're done

    result["cleaned"] = current
    return result


def clean_urls(urls: List[str], follow_redirects: bool = True, timeout: int = 5) -> List[Dict]:
    """Clean a list of URLs, returning a list of clean_url() dicts."""
    return [clean_url(u, follow_redirects=follow_redirects, timeout=timeout) for u in urls]


def get_real_urls(urls: List[str], follow_redirects: bool = True) -> List[str]:
    """Return only the cleaned (real destination) URLs."""
    return [r["cleaned"] for r in clean_urls(urls, follow_redirects=follow_redirects)]


def build_url_cleaning_map(urls: List[str], follow_redirects: bool = True) -> Dict[str, Dict]:
    """Return a dict mapping each original URL → its clean_url() result dict."""
    return {u: clean_url(u, follow_redirects=follow_redirects) for u in urls}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _single_clean_pass(url: str, follow_redirects: bool, timeout: int) -> Tuple[str, str]:
    """Attempt one round of unwrapping. Returns (cleaned_url, method)."""

    # 1 ── Microsoft SafeLinks ───────────────────────────────────────────────
    if _SAFELINKS_RE.match(url):
        cleaned = _extract_safelinks(url)
        if cleaned:
            return cleaned, METHOD_SAFELINKS

    # 2 ── Proofpoint URLDefense v3 ──────────────────────────────────────────
    parsed_u = urllib.parse.urlparse(url)
    host = parsed_u.netloc.lower()
    if ("urldefense.com" in host or "urldefense.proofpoint.com" in host):
        path = parsed_u.path
        if "/v3/__" in path:
            cleaned = _extract_proofpoint_v3(url)
            if cleaned:
                return cleaned, METHOD_PP_V3
        elif "/v2/url" in path:
            cleaned = _extract_proofpoint_v2(url)
            if cleaned:
                return cleaned, METHOD_PP_V2

    # 3 ── Google redirect ───────────────────────────────────────────────────
    if "google.com" in host and parsed_u.path in ("/url", "/url/"):
        qs = urllib.parse.parse_qs(parsed_u.query)
        target = qs.get("q", qs.get("url", [None]))[0]
        if target:
            return urllib.parse.unquote(target), METHOD_GOOGLE

    # 4 ── Mimecast URL Protect ──────────────────────────────────────────────
    if _MIMECAST_RE.match(url):
        cleaned = _extract_mimecast(url)
        if cleaned:
            return cleaned, METHOD_MIMECAST

    # 5 ── Barracuda Networks ────────────────────────────────────────────────
    if _BARRACUDA_RE.match(url):
        cleaned = _extract_barracuda(url)
        if cleaned:
            return cleaned, METHOD_BARRACUDA

    # 6 ── Generic `?url=` / `?u=` / `?dest=` / `?redirect=` ────────────────
    qs = urllib.parse.parse_qs(parsed_u.query)
    for param in ("url", "u", "dest", "destination", "redirect", "goto",
                  "link", "target", "next", "continue", "return", "returnurl"):
        val = qs.get(param, [None])[0]
        if val and (val.startswith("http://") or val.startswith("https://")):
            decoded = urllib.parse.unquote(val)
            if _is_different_domain(url, decoded):
                return decoded, f"Redirect param (?{param}=)"

    # 7 ── Recursive percent-encoding (detect double/triple encoding) ─────────
    if "%" in url:
        decoded = urllib.parse.unquote(url)
        if decoded != url and (decoded.startswith("http://") or decoded.startswith("https://")):
            return decoded, METHOD_PERCENT

    # 8 ── URL shorteners — follow HEAD redirect ──────────────────────────────
    if follow_redirects:
        bare_host = host.lstrip("www.")
        if bare_host in _SHORTENER_DOMAINS or host in _SHORTENER_DOMAINS:
            followed = _follow_redirect(url, timeout)
            if followed and followed != url:
                return followed, METHOD_SHORTENER

    return url, METHOD_CLEAN


# ── SafeLinks extractor ──────────────────────────────────────────────────────

def _extract_safelinks(url: str) -> Optional[str]:
    """Extract and decode the real URL from a Microsoft SafeLinks URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(parsed.query)
        raw = qs.get("url", [None])[0]
        if raw:
            # May be percent-encoded — decode up to 3 times
            decoded = raw
            for _ in range(3):
                prev = decoded
                decoded = urllib.parse.unquote(decoded)
                if decoded == prev:
                    break
            if decoded.startswith("http://") or decoded.startswith("https://"):
                return decoded
    except Exception as exc:
        log.debug(f"SafeLinks extraction error: {exc}")
    return None


# ── Proofpoint extractors ────────────────────────────────────────────────────

def _extract_proofpoint_v2(url: str) -> Optional[str]:
    """Decode Proofpoint URLDefense v2 encoding.

    Proofpoint v2 uses a custom encoding:
      - `-XX` hex sequences replace special chars (e.g., `-3A` → `:`)
      - `_` replaces `/`
    The `u` query parameter holds the encoded real URL.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(parsed.query)
        u_param = qs.get("u", [None])[0]
        if not u_param:
            return None
        # Replace `-XX` with actual hex chars
        decoded = _PP_V2_HEX_RE.sub(lambda m: chr(int(m.group(1), 16)), u_param)
        # Proofpoint also uses `__` to replace `//` in the path after scheme
        decoded = decoded.replace("__", "//", 1)
        decoded = urllib.parse.unquote(decoded)
        if decoded.startswith("http://") or decoded.startswith("https://"):
            return decoded
    except Exception as exc:
        log.debug(f"Proofpoint v2 extraction error: {exc}")
    return None


def _extract_proofpoint_v3(url: str) -> Optional[str]:
    """Decode Proofpoint URLDefense v3 format.

    Format: https://urldefense.com/v3/__<encoded_url>__;<hash>
    The real URL is between `__` markers in the path.
    """
    try:
        # Pattern: /v3/__<url>__
        match = re.search(r"/v3/__(.+?)__(?:;|$)", url)
        if match:
            inner = match.group(1)
            decoded = urllib.parse.unquote(inner)
            if decoded.startswith("http://") or decoded.startswith("https://"):
                return decoded
    except Exception as exc:
        log.debug(f"Proofpoint v3 extraction error: {exc}")
    return None


# ── Mimecast extractor ───────────────────────────────────────────────────────

def _extract_mimecast(url: str) -> Optional[str]:
    """Follow Mimecast protect URL to get the real destination.

    Mimecast protect URLs (/s/<token>) are opaque tokens — the real URL
    is only obtainable by following the redirect.
    """
    return _follow_redirect(url, timeout=5)


# ── Barracuda extractor ──────────────────────────────────────────────────────

def _extract_barracuda(url: str) -> Optional[str]:
    """Extract real URL from Barracuda Email Security links.

    Format: https://links.barracudanetworks.com/<id>?<encoded_url>
    The actual URL is typically in the query string.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(parsed.query)
        for param in ("url", "u", "dest"):
            val = qs.get(param, [None])[0]
            if val and (val.startswith("http://") or val.startswith("https://")):
                return urllib.parse.unquote(val)
        # Fall back to redirect following
        return _follow_redirect(url, timeout=5)
    except Exception as exc:
        log.debug(f"Barracuda extraction error: {exc}")
    return None


# ── Redirect follower ────────────────────────────────────────────────────────

def _follow_redirect(url: str, timeout: int = 5) -> Optional[str]:
    """Follow a single HTTP redirect and return the final URL.

    Uses HEAD to avoid downloading page content. Does NOT recursively
    follow multiple hops — only one hop.

    Returns None on any failure (network error, timeout, non-redirect).
    """
    try:
        resp = requests.head(
            url,
            allow_redirects=False,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (PhishLens URL Inspector)"},
        )
        if resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("Location", "")
            if location and (location.startswith("http://") or location.startswith("https://")):
                return location
    except Exception as exc:
        log.debug(f"Redirect follow failed for '{url[:80]}': {exc}")
    return None


# ── Helper ───────────────────────────────────────────────────────────────────

def _is_different_domain(original: str, candidate: str) -> bool:
    """Return True if candidate URL has a different domain than original."""
    try:
        orig_host = urllib.parse.urlparse(original).netloc.lower()
        cand_host = urllib.parse.urlparse(candidate).netloc.lower()
        return cand_host and cand_host != orig_host
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Convenience: summarise cleaning results for logging / UI
# ---------------------------------------------------------------------------

def summarise_cleaning(cleaning_map: Dict[str, Dict]) -> List[str]:
    """Return human-readable lines describing what was cleaned.

    Args:
        cleaning_map: Output of build_url_cleaning_map().

    Returns:
        List of strings, one per wrapped URL that was cleaned.
    """
    lines = []
    for original, result in cleaning_map.items():
        if result["was_wrapped"]:
            short_orig = original[:80] + ("…" if len(original) > 80 else "")
            short_clean = result["cleaned"][:80] + ("…" if len(result["cleaned"]) > 80 else "")
            lines.append(
                f"[{result['method']}] {short_orig}\n"
                f"  → Real destination: {short_clean}"
            )
    return lines
