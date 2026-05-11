"""
PhishLens HTML Structural Anomaly Feature Module.

Extracts 11 features from the HTML body of emails by parsing with BeautifulSoup.
Phishing emails rely heavily on HTML tricks to hide malicious content, redirect
users, and harvest credentials.

Security rationale: HTML-based obfuscation is a primary evasion technique.
Hidden text (display:none), form POST to attacker-controlled domains, and
href/visible-text mismatches are reliable signals that cannot be faked without
triggering feature flags. These features complement NLP features which only
see the rendered/visible text.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import tldextract

from src.utils.logger import get_logger

# Truncate HTML before BeautifulSoup to prevent exponential parse time on
# monster HTML emails (multi-MB newsletters, base64-inlined images, etc.).
# All structural signals (links, forms, hidden elements) appear early in the
# document so truncating at 200 KB does not materially affect feature quality.
_MAX_HTML_CHARS = 200_000  # 200 KB

log = get_logger(__name__)

# Hidden text CSS patterns — all known obfuscation techniques
_HIDDEN_TEXT_PATTERNS = [
    re.compile(r"display\s*:\s*none", re.IGNORECASE),
    re.compile(r"font-size\s*:\s*0", re.IGNORECASE),
    re.compile(r"color\s*:\s*(white|#fff|#ffffff|rgba\(255,255,255)", re.IGNORECASE),
    re.compile(r"visibility\s*:\s*hidden", re.IGNORECASE),
    re.compile(r"opacity\s*:\s*0(?!\.\d)", re.IGNORECASE),
    # Modern phishing CSS obfuscation techniques:
    re.compile(r"height\s*:\s*0px|height\s*:\s*0;", re.IGNORECASE),
    re.compile(r"max-height\s*:\s*0", re.IGNORECASE),
    re.compile(r"overflow\s*:\s*hidden", re.IGNORECASE),
    re.compile(r"text-indent\s*:\s*-\d{3,}", re.IGNORECASE),
    re.compile(r"clip\s*:\s*rect\s*\(\s*0", re.IGNORECASE),
    re.compile(r"mso-hide\s*:\s*all", re.IGNORECASE),  # Outlook-specific hiding
]

# Base64 data URI pattern
_BASE64_DATA_RE = re.compile(r"data:[^;]+;base64,", re.IGNORECASE)

# Tracking pixel pattern (1x1 images)
_TRACKING_PIXEL_RE = re.compile(r'(width|height)\s*[=:]\s*["\']?1["\']?', re.IGNORECASE)


def extract_html_features(html_body: str) -> Dict:
    """Extract 11 HTML structural anomaly features from an email HTML body.

    Args:
        html_body: Raw HTML string from the email body.

    Returns:
        Dict with 11 numeric HTML features. Returns zero defaults if html_body
        is empty or unparseable.
    """
    if not html_body or not html_body.strip():
        return _default_html_features()

    # Truncate oversized HTML to keep parse time bounded.
    if len(html_body) > _MAX_HTML_CHARS:
        html_body = html_body[:_MAX_HTML_CHARS]

    features = _default_html_features()

    try:
        soup = BeautifulSoup(html_body, "lxml")
    except Exception as exc:
        log.debug(f"BeautifulSoup parse error: {exc}")
        try:
            soup = BeautifulSoup(html_body, "html.parser")
        except Exception:
            return features

    try:
        features["href_text_mismatch_count"] = _count_href_text_mismatches(soup)
    except Exception as exc:
        log.debug(f"href_text_mismatch_count error: {exc}")

    try:
        features["external_form_action"] = int(_has_external_form_action(soup))
    except Exception as exc:
        log.debug(f"external_form_action error: {exc}")

    try:
        features["hidden_text_count"] = _count_hidden_text_elements(soup)
    except Exception as exc:
        log.debug(f"hidden_text_count error: {exc}")

    try:
        features["image_to_text_ratio"] = _compute_image_to_text_ratio(soup)
    except Exception as exc:
        log.debug(f"image_to_text_ratio error: {exc}")

    try:
        features["tracking_pixel_count"] = _count_tracking_pixels(soup)
    except Exception as exc:
        log.debug(f"tracking_pixel_count error: {exc}")

    try:
        features["base64_content_count"] = _count_base64_content(soup)
    except Exception as exc:
        log.debug(f"base64_content_count error: {exc}")

    try:
        features["javascript_count"] = len(soup.find_all("script"))
    except Exception as exc:
        log.debug(f"javascript_count error: {exc}")

    try:
        features["external_css_count"] = _count_external_css(soup)
    except Exception as exc:
        log.debug(f"external_css_count error: {exc}")

    try:
        links = soup.find_all("a", href=True)
        features["total_links"] = len(links)
        domains = [_extract_link_domain(a["href"]) for a in links]
        domains = [d for d in domains if d]
        unique_domains = set(domains)
        features["unique_domains_in_links"] = len(unique_domains)
        if features["total_links"] > 0:
            features["link_domain_diversity"] = len(unique_domains) / features["total_links"]
    except Exception as exc:
        log.debug(f"link domain features error: {exc}")

    return features


# ---------------------------------------------------------------------------
# Feature implementations
# ---------------------------------------------------------------------------


def _count_href_text_mismatches(soup: BeautifulSoup) -> int:
    """Count anchor tags where visible text ≠ href URL.

    Also catches the IP-in-href trick: href points to a raw IP address
    but visible text shows a legitimate brand domain.
    """
    count = 0
    for a in soup.find_all("a", href=True):
        href = str(a["href"]).strip()
        visible_text = a.get_text(strip=True)

        if not href or not visible_text:
            continue

        if not re.match(r"https?://", href):
            continue

        # Case 1: href is a raw IP but visible text looks like a domain
        if re.match(r"https?://(?:\d{1,3}\.){3}\d{1,3}", href):
            if re.search(r"[a-zA-Z]{3,}\.[a-zA-Z]{2,}", visible_text):
                count += 1
                continue

        # Case 2: domain in href ≠ domain in visible text
        if not re.search(r"[a-zA-Z0-9][.-][a-zA-Z]{2,}", visible_text):
            continue
        try:
            href_domain = urlparse(href).netloc.lower().lstrip("www.")
            text_domain = re.search(r"[\w-]+\.[a-zA-Z]{2,}", visible_text)
            if text_domain:
                text_d = text_domain.group(0).lower().lstrip("www.")
                if href_domain and text_d and href_domain != text_d:
                    count += 1
        except Exception:
            pass

    return count


def _has_external_form_action(soup: BeautifulSoup) -> bool:
    """Detect forms that POST to a different domain than the email sender.

    Security rationale: Credential-harvesting forms in phishing emails
    POST login data to attacker-controlled servers. Any form action
    pointing to an external URL is a strong indicator.
    """
    forms = soup.find_all("form")
    for form in forms:
        action = form.get("action", "")
        if action and re.match(r"https?://", action):
            return True     # External form action found
    return False


def _count_hidden_text_elements(soup: BeautifulSoup) -> int:
    """Count HTML elements that visually hide text using CSS tricks.

    Security rationale: Hidden white-on-white text, zero-font-size content,
    and display:none elements are used to stuff keywords that evade spam
    filters while remaining invisible to human readers.
    """
    count = 0
    for element in soup.find_all(style=True):
        style = element.get("style", "")
        for pattern in _HIDDEN_TEXT_PATTERNS:
            if pattern.search(style):
                count += 1
                break
    # Also check elements with the 'hidden' attribute
    count += len(soup.find_all(hidden=True))
    return count


def _compute_image_to_text_ratio(soup: BeautifulSoup) -> float:
    """Compute ratio of img tags to total word count.

    Security rationale: Pure-image phishing emails contain no analysable text
    by design — the phishing content is baked into images to evade text-based
    filters. A high image-to-text ratio is a strong phishing signal.
    """
    img_count = len(soup.find_all("img"))
    word_count = len(soup.get_text().split())
    if word_count == 0:
        return float(img_count)    # All images, no text
    return img_count / word_count


def _count_tracking_pixels(soup: BeautifulSoup) -> int:
    """Count 1×1 tracking pixel images.

    Security rationale: Tracking pixels confirm delivery to a live email
    address. Phishers use them to validate target lists and time follow-up attacks.
    """
    count = 0
    for img in soup.find_all("img"):
        width = img.get("width", "")
        height = img.get("height", "")
        src = img.get("src", "")
        # 1x1 pixel images
        if (str(width) == "1" and str(height) == "1") or "tracking" in src.lower():
            count += 1
    return count


def _count_base64_content(soup: BeautifulSoup) -> int:
    """Count inline base64-encoded content (images, scripts, etc.).

    Security rationale: Base64-encoded content embedded directly in HTML
    bypasses URL-based phishing filters entirely. Legitimate email rarely
    uses inline base64 for anything other than small icons.
    """
    html_str = str(soup)
    return len(_BASE64_DATA_RE.findall(html_str))


def _count_external_css(soup: BeautifulSoup) -> int:
    """Count externally loaded CSS stylesheets.

    Security rationale: External CSS can be used to dynamically alter the
    appearance of email after delivery (e.g., hiding/showing content based
    on when it is opened — a sign of delayed activation phishing).
    """
    count = 0
    for link in soup.find_all("link", rel=True):
        if "stylesheet" in str(link.get("rel", [])).lower():
            href = link.get("href", "")
            if href.startswith("http"):
                count += 1
    return count


def _extract_link_domain(href: str) -> Optional[str]:
    """Extract the registered domain from an href value."""
    try:
        if not href.startswith("http"):
            return None
        ext = tldextract.extract(href)
        return ext.top_domain_under_public_suffix or None
    except Exception:
        return None


def _has_meta_refresh(soup: BeautifulSoup) -> bool:
    """Detect meta refresh redirect tags.

    Meta refresh is used by phishers to redirect victims to a malicious
    page after a short delay, often with a blank/loading placeholder page
    shown first to evade automated scanners.
    """
    for meta in soup.find_all("meta"):
        http_equiv = meta.get("http-equiv", "").lower()
        content = meta.get("content", "").lower()
        if http_equiv == "refresh" and "url=" in content:
            return True
    return False


def _default_html_features() -> Dict:
    """Return zero-value defaults for all HTML features."""
    return {
        "href_text_mismatch_count": 0,
        "external_form_action": 0,
        "hidden_text_count": 0,
        "image_to_text_ratio": 0.0,
        "tracking_pixel_count": 0,
        "base64_content_count": 0,
        "javascript_count": 0,
        "external_css_count": 0,
        "total_links": 0,
        "unique_domains_in_links": 0,
        "link_domain_diversity": 0.0,
    }
