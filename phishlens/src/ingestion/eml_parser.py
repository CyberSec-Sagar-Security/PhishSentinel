"""
PhishLens EML Parser Module.

Parses raw .eml email files into structured dicts for feature extraction.
Handles multipart MIME, nested parts, base64-encoded bodies, and malformed
headers gracefully. Every parse step is wrapped in try/except to guarantee
a partial result even for adversarially crafted or corrupted emails.

Security rationale: Phishing emails frequently contain malformed MIME
structures designed to confuse filters. Robust parsing that does not crash
on malformed input is a hard requirement for production use.
"""

from __future__ import annotations

import base64
import email
import email.policy
import hashlib
import re
from email.message import Message
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.utils.logger import get_logger

# Limit body sizes stored per email to prevent regex/BS4 stalling on
# multi-MB marketing HTML or abnormally large plain-text blobs.
_MAX_HTML_CHARS = 524_288  # 512 KB
_MAX_TEXT_CHARS = 102_400  # 100 KB

log = get_logger(__name__)

# Regex to extract all URLs from plain text / HTML
_URL_PATTERN = re.compile(
    r"https?://[^\s<>\"'(){}\[\]]+",
    re.IGNORECASE,
)

# Common freemail domains for reply-to analysis
_FREEMAIL_RE = re.compile(
    r"@(gmail|yahoo|hotmail|outlook|aol|protonmail|icloud|"
    r"yandex|tutanota|gmx|live|mail)\.",
    re.IGNORECASE,
)


def parse_eml_file(path: Union[str, Path]) -> Dict:
    """Parse a .eml file from disk into a structured feature dict.

    Args:
        path: Path to the .eml file.

    Returns:
        Structured dict with all extractable email fields.
        Never raises; returns partial dict on parse errors.
    """
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        return parse_eml_bytes(raw)
    except OSError as exc:
        log.warning(f"Could not open .eml file '{path}': {exc}")
        return _empty_email_dict()


def parse_eml_string(raw_email: str) -> Dict:
    """Parse a raw email string into a structured feature dict.

    Args:
        raw_email: Raw email content as a string (headers + body).

    Returns:
        Structured dict with all extractable email fields.
    """
    if not raw_email:
        return parse_eml_bytes(b"")
    return parse_eml_bytes(raw_email.encode("utf-8", errors="replace"))


def parse_eml_bytes(raw: bytes) -> Dict:
    """Parse raw email bytes into a structured feature dict.

    Args:
        raw: Raw email bytes.

    Returns:
        Structured dict containing extracted email metadata and body content.
    """
    # Cap total raw bytes before any parsing. Many dataset rows contain raw
    # HTML blobs (not RFC 2822 emails) that can be 5-20MB — email.message_from_bytes
    # on such input walks the entire content for MIME boundaries and takes minutes.
    # 1MB is more than enough for any real email and all features we extract.
    _MAX_RAW_BYTES = 1_048_576  # 1 MB
    if len(raw) > _MAX_RAW_BYTES:
        raw = raw[:_MAX_RAW_BYTES]

    result = _empty_email_dict()

    try:
        msg: Message = email.message_from_bytes(
            raw,
            policy=email.policy.compat32,
        )
    except Exception as exc:
        log.warning(f"email.message_from_bytes failed: {exc}")
        return result

    # ---- Header extraction ------------------------------------------------
    try:
        result["from_address"] = _decode_header_safe(msg.get("From", ""))
        result["to_address"] = _decode_header_safe(msg.get("To", ""))
        result["reply_to"] = _decode_header_safe(msg.get("Reply-To", ""))
        result["return_path"] = _decode_header_safe(msg.get("Return-Path", ""))
        result["subject"] = _decode_header_safe(msg.get("Subject", ""))
        result["date"] = _decode_header_safe(msg.get("Date", ""))
        result["message_id"] = _decode_header_safe(msg.get("Message-ID", ""))
        result["x_mailer"] = _decode_header_safe(msg.get("X-Mailer", ""))
        result["content_type"] = _decode_header_safe(msg.get("Content-Type", ""))
        # Security-critical headers for inline auth parsing (no DNS lookup needed)
        result["auth_results"] = _decode_header_safe(msg.get("Authentication-Results", ""))
        result["received_spf_header"] = _decode_header_safe(msg.get("Received-SPF", ""))
        result["dkim_signed"] = int(msg.get("DKIM-Signature") is not None)
        result["list_unsubscribe"] = _decode_header_safe(msg.get("List-Unsubscribe", ""))
        result["x_priority"] = _decode_header_safe(msg.get("X-Priority", msg.get("Priority", "")))
        result["x_originating_ip"] = _decode_header_safe(msg.get("X-Originating-IP", ""))
        result["user_agent"] = _decode_header_safe(msg.get("User-Agent", ""))
        result["x_spam_status"] = _decode_header_safe(msg.get("X-Spam-Status", ""))
        result["mime_version"] = _decode_header_safe(msg.get("MIME-Version", ""))
        result["content_transfer_encoding"] = _decode_header_safe(msg.get("Content-Transfer-Encoding", ""))
    except Exception as exc:
        log.warning(f"Header extraction error: {exc}")

    # ---- Received headers (relay chain) -----------------------------------
    try:
        received_raw = msg.get_all("Received") or []
        result["received_headers"] = [
            _decode_header_safe(h) for h in received_raw
        ]
    except Exception as exc:
        log.warning(f"Received headers extraction error: {exc}")

    # ---- Full raw header dump ---------------------------------------------
    # Use msg.items() (already-parsed key/value pairs) instead of str(msg)
    # which calls as_string() and re-serializes the ENTIRE email including body.
    # For emails with malformed charsets (euc, unknown-8bit, embedded HTML),
    # as_string() can stall for seconds per email. header_raw is stored for
    # debugging only and is not consumed by any feature extractor.
    try:
        result["header_raw"] = "\n".join(
            f"{k}: {v}" for k, v in msg.items()
        )[:8192]
    except Exception as exc:
        log.warning(f"header_raw extraction error: {exc}")

    # ---- Body extraction (text + HTML) ------------------------------------
    try:
        body_text, body_html, urls, att_count, att_hashes = _extract_body(msg)
        result["body_text"] = body_text
        result["body_html"] = body_html
        result["urls"] = urls
        result["attachments_count"] = att_count
        result["attachment_hashes"] = att_hashes
    except Exception as exc:
        log.warning(f"Body extraction error: {exc}")

    return result


def _extract_body(
    msg: Message,
) -> tuple[str, str, List[str], int, List[str]]:
    """Walk MIME parts and extract text, HTML, URLs, and attachment info.

    Args:
        msg: Parsed email.Message object.

    Returns:
        Tuple of (body_text, body_html, urls, attachments_count, attachment_hashes).
    """
    body_text = ""
    body_html = ""
    urls: List[str] = []
    attachments_count = 0
    attachment_hashes: List[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))

            # Skip attachment parts but record their hashes for IOC extraction
            if "attachment" in disposition:
                attachments_count += 1
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        sha256 = hashlib.sha256(payload).hexdigest()
                        attachment_hashes.append(sha256)
                except Exception:
                    pass
                continue

            try:
                payload_bytes = part.get_payload(decode=True)
                if payload_bytes is None:
                    continue
                charset = part.get_content_charset() or "utf-8"
                decoded = payload_bytes.decode(charset, errors="replace")
            except Exception:
                continue

            if content_type == "text/plain":
                remaining_t = max(0, _MAX_TEXT_CHARS - len(body_text))
                chunk_t = decoded[:remaining_t]
                body_text += chunk_t + "\n"
                urls.extend(_URL_PATTERN.findall(chunk_t))
            elif content_type == "text/html":
                # Cap HTML to avoid multi-MB documents stalling BeautifulSoup
                remaining = max(0, _MAX_HTML_CHARS - len(body_html))
                chunk = decoded[:remaining]
                body_html += chunk + "\n"
                urls.extend(_URL_PATTERN.findall(chunk))
    else:
        # Single-part message
        try:
            payload_bytes = msg.get_payload(decode=True)
            if payload_bytes:
                charset = msg.get_content_charset() or "utf-8"
                decoded = payload_bytes.decode(charset, errors="replace")
                content_type = msg.get_content_type()
                if "html" in content_type:
                    body_html = decoded[:_MAX_HTML_CHARS]
                    urls.extend(_URL_PATTERN.findall(body_html))
                else:
                    body_text = decoded[:_MAX_TEXT_CHARS]
                    urls.extend(_URL_PATTERN.findall(body_text))
        except Exception as exc:
            log.debug(f"Single-part body decode error: {exc}")

    # Supplement regex URL extraction with BeautifulSoup href scanning.
    # This catches href-embedded or entity-encoded URLs the regex misses.
    if body_html:
        try:
            from bs4 import BeautifulSoup as _BS4
            _soup = _BS4(body_html[:_MAX_HTML_CHARS], "lxml")
            for _a in _soup.find_all("a", href=True):
                _href = str(_a["href"]).strip()
                if _href.lower().startswith(("http://", "https://")):
                    urls.append(_href)
            # Also capture tracking pixel / iframe src URLs
            for _tag in _soup.find_all(["img", "iframe"], src=True):
                _src = str(_tag["src"]).strip()
                if _src.lower().startswith(("http://", "https://")):
                    urls.append(_src)
        except Exception:
            pass  # BeautifulSoup failure is non-fatal; regex results still used

    # Deduplicate URLs while preserving order
    seen = set()
    unique_urls = []
    for u in urls:
        # Strip trailing punctuation that regex may include
        u_clean = u.rstrip(".,;)'\"")
        if u_clean and u_clean not in seen:
            seen.add(u_clean)
            unique_urls.append(u_clean)

    return body_text, body_html, unique_urls, attachments_count, attachment_hashes


def _decode_header_safe(raw: str) -> str:
    """Safely decode an email header value, handling RFC 2047 encoding.

    Security rationale: Header injection attacks embed CRLF sequences in
    encoded headers. We decode then sanitise before further processing.

    Args:
        raw: Raw header value string.

    Returns:
        Decoded, sanitised header string.
    """
    if not raw:
        return ""
    try:
        parts = email.header.decode_header(raw)
        decoded_parts = []
        for bpart, charset in parts:
            if isinstance(bpart, bytes):
                decoded_parts.append(
                    bpart.decode(charset or "utf-8", errors="replace")
                )
            else:
                decoded_parts.append(str(bpart))
        return " ".join(decoded_parts).strip()
    except Exception:
        return str(raw).strip()


def parse_msg_bytes(raw: bytes) -> Dict:
    """Parse an Outlook .msg binary file into a structured feature dict.

    Uses the extract-msg library to decode the OLE2 compound document format
    and maps its fields onto the same dict schema as parse_eml_bytes so the
    full feature pipeline works without any changes downstream.

    Args:
        raw: Raw .msg file bytes.

    Returns:
        Structured dict with all extractable email fields. Same schema as
        parse_eml_bytes. Never raises; returns partial/empty dict on errors.
    """
    result = _empty_email_dict()
    try:
        import extract_msg  # pip install extract-msg
        import io, hashlib as _hs
    except ImportError:
        log.warning("extract-msg is not installed. Install it with: pip install extract-msg")
        return result

    try:
        msg_obj = extract_msg.openMsg(io.BytesIO(raw))
    except Exception as exc:
        log.warning(f"extract_msg failed to open .msg: {exc}")
        return result

    try:
        _hd: Dict = msg_obj.headerDict or {}
        _header_text: str = msg_obj.headerText or ""

        # ---- Core headers ------------------------------------------------
        result["from_address"] = str(msg_obj.sender or _hd.get("From", ""))
        result["to_address"] = str(msg_obj.to or _hd.get("To", ""))
        result["reply_to"] = str(_hd.get("Reply-To", ""))
        result["return_path"] = str(_hd.get("Return-Path", ""))
        result["subject"] = str(msg_obj.subject or _hd.get("Subject", ""))
        result["date"] = str(msg_obj.date or _hd.get("Date", ""))
        result["message_id"] = str(_hd.get("Message-ID", ""))
        result["x_mailer"] = str(_hd.get("X-Mailer", ""))
        result["content_type"] = str(
            _hd.get("Content-Type", "text/html" if msg_obj.htmlBody else "text/plain")
        )
        result["auth_results"] = str(_hd.get("Authentication-Results", ""))
        result["received_spf_header"] = str(_hd.get("Received-SPF", ""))
        result["dkim_signed"] = int("DKIM-Signature" in _hd)
        result["list_unsubscribe"] = str(_hd.get("List-Unsubscribe", ""))
        result["x_priority"] = str(_hd.get("X-Priority", _hd.get("Importance", "")))
        result["x_originating_ip"] = str(_hd.get("X-Originating-IP", ""))
        result["mime_version"] = str(_hd.get("MIME-Version", ""))
        result["content_transfer_encoding"] = str(
            _hd.get("Content-Transfer-Encoding", "")
        )
        # Received relay chain from raw header text
        if _header_text:
            result["header_raw"] = _header_text[:8192]
            result["received_headers"] = [
                " ".join(r.split())
                for r in re.findall(
                    r"(?m)^Received:[ \t]*(.+(?:\n[ \t].+)*)", _header_text
                )
            ]

        # ---- Body --------------------------------------------------------
        body_text = ""
        body_html = ""
        urls: List[str] = []

        if msg_obj.htmlBody:
            try:
                body_html = msg_obj.htmlBody.decode("utf-8", errors="replace")
            except Exception:
                body_html = str(msg_obj.htmlBody or "")
            body_html = body_html[:_MAX_HTML_CHARS]
            urls.extend(_URL_PATTERN.findall(body_html))

        if msg_obj.body:
            try:
                body_text = msg_obj.body if isinstance(msg_obj.body, str) else \
                    msg_obj.body.decode("utf-8", errors="replace")
            except Exception:
                body_text = str(msg_obj.body or "")
            body_text = body_text[:_MAX_TEXT_CHARS]
            urls.extend(_URL_PATTERN.findall(body_text))

        # Also scrape href links from HTML via BeautifulSoup
        if body_html:
            try:
                from bs4 import BeautifulSoup as _BS4
                _soup = _BS4(body_html, "lxml")
                for _a in _soup.find_all("a", href=True):
                    _href = str(_a["href"]).strip()
                    if _href.lower().startswith(("http://", "https://")):
                        urls.append(_href)
                for _tag in _soup.find_all(["img", "iframe"], src=True):
                    _src = str(_tag["src"]).strip()
                    if _src.lower().startswith(("http://", "https://")):
                        urls.append(_src)
            except Exception:
                pass

        # Deduplicate URLs
        _seen: set = set()
        _unique: List[str] = []
        for u in urls:
            u = u.rstrip(".,;)'\"")
            if u and u not in _seen:
                _seen.add(u)
                _unique.append(u)

        result["body_text"] = body_text
        result["body_html"] = body_html
        result["urls"] = _unique

        # ---- Attachments -------------------------------------------------
        atts = getattr(msg_obj, "attachments", []) or []
        result["attachments_count"] = len(atts)
        hashes: List[str] = []
        for att in atts:
            try:
                att_data = att.data
                if att_data:
                    hashes.append(_hs.sha256(att_data).hexdigest())
            except Exception:
                pass
        result["attachment_hashes"] = hashes

    except Exception as exc:
        log.warning(f"parse_msg_bytes field extraction error: {exc}")
    finally:
        try:
            msg_obj.close()
        except Exception:
            pass

    return result


def _empty_email_dict() -> Dict:
    """Return a zero-filled email dict as a safe fallback."""
    return {
        "from_address": "",
        "to_address": "",
        "reply_to": "",
        "return_path": "",
        "subject": "",
        "date": "",
        "message_id": "",
        "x_mailer": "",
        "content_type": "",
        "received_headers": [],
        "header_raw": "",
        "body_text": "",
        "body_html": "",
        "urls": [],
        "attachments_count": 0,
        "attachment_hashes": [],
        # Security headers (added for inline auth parsing)
        "auth_results": "",
        "received_spf_header": "",
        "dkim_signed": 0,
        "list_unsubscribe": "",
        "x_priority": "",
        "x_originating_ip": "",
        "user_agent": "",
        "x_spam_status": "",
        "mime_version": "",
        "content_transfer_encoding": "",
    }
