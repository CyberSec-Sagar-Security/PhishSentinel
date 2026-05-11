"""
Tests for eml_parser.py — covers MIME parsing, header extraction, URL discovery,
injection sanitisation, and encoding edge cases.
"""

from __future__ import annotations

import pytest

from src.ingestion.eml_parser import parse_eml_string


# ---------------------------------------------------------------------------
# Sample emails
# ---------------------------------------------------------------------------

SIMPLE_EMAIL = """From: attacker@phish.example.com
To: victim@corp.com
Subject: Urgent: Verify Your Account
Date: Mon, 01 Jan 2024 12:00:00 +0000
Message-ID: <test123@phish.example.com>
Content-Type: text/plain; charset=utf-8

Click here to verify: http://malicious-site.xyz/login?ref=paypal
"""

MULTIPART_EMAIL = """From: sender@example.com
To: user@example.com
Subject: Test multipart
Date: Mon, 01 Jan 2024 12:00:00 +0000
Message-ID: <multipart@example.com>
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="boundary123"

--boundary123
Content-Type: text/plain; charset=utf-8

Plain text part with http://example.com/link

--boundary123
Content-Type: text/html; charset=utf-8

<html><body><a href="http://phishing.example.com/steal">Click here</a></body></html>

--boundary123--
"""

REPLY_TO_MISMATCH_EMAIL = """From: legitimate@bank.com
Reply-To: attacker@freeemail.net
Return-Path: bounces@completely-different.com
Subject: Your account needs attention
Date: Mon, 01 Jan 2024 12:00:00 +0000
Message-ID: <replyto@test.com>

Please verify your account.
"""

INJECTION_EMAIL = """From: attacker@test.com\r\nBcc: victim2@test.com
To: victim@test.com
Subject: Injection test\r\nX-Injected: malicious
Date: Mon, 01 Jan 2024 12:00:00 +0000
Message-ID: <injection@test.com>

Body.
"""

EMPTY_EMAIL = ""

ENCODED_SUBJECT_EMAIL = """From: test@example.com
To: user@example.com
Subject: =?UTF-8?B?VXJnZW50OiBWZXJpZnkgWW91ciBBY2NvdW50?=
Date: Mon, 01 Jan 2024 12:00:00 +0000
Message-ID: <encoded@test.com>

Encoded subject email.
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseSimpleEmail:
    def test_from_address_extracted(self):
        parsed = parse_eml_string(SIMPLE_EMAIL)
        assert "attacker@phish.example.com" in parsed.get("from_address", "")

    def test_subject_extracted(self):
        parsed = parse_eml_string(SIMPLE_EMAIL)
        assert "Verify" in parsed.get("subject", "")

    def test_urls_extracted(self):
        parsed = parse_eml_string(SIMPLE_EMAIL)
        urls = parsed.get("urls", [])
        assert len(urls) > 0
        assert any("malicious-site.xyz" in u for u in urls)

    def test_to_address_extracted(self):
        parsed = parse_eml_string(SIMPLE_EMAIL)
        assert "victim@corp.com" in parsed.get("to_address", "")

    def test_message_id_extracted(self):
        parsed = parse_eml_string(SIMPLE_EMAIL)
        assert "test123" in parsed.get("message_id", "")

    def test_body_text_not_empty(self):
        parsed = parse_eml_string(SIMPLE_EMAIL)
        body = parsed.get("body_text", "")
        assert len(body) > 0


class TestParseMultipartEmail:
    def test_html_body_extracted(self):
        parsed = parse_eml_string(MULTIPART_EMAIL)
        html = parsed.get("body_html", "")
        assert "<a href=" in html

    def test_urls_from_html_extracted(self):
        parsed = parse_eml_string(MULTIPART_EMAIL)
        urls = parsed.get("urls", [])
        assert any("phishing.example.com" in u for u in urls)

    def test_both_body_parts_present(self):
        parsed = parse_eml_string(MULTIPART_EMAIL)
        assert parsed.get("body_text") or parsed.get("body_html")


class TestReplyToMismatch:
    def test_reply_to_extracted(self):
        parsed = parse_eml_string(REPLY_TO_MISMATCH_EMAIL)
        assert "attacker@freeemail.net" in parsed.get("reply_to", "")

    def test_return_path_extracted(self):
        parsed = parse_eml_string(REPLY_TO_MISMATCH_EMAIL)
        assert "completely-different.com" in parsed.get("return_path", "")


class TestHeaderInjectionSanitisation:
    def test_injection_not_in_from(self):
        """Injected headers must NOT appear as separate parsed headers."""
        parsed = parse_eml_string(INJECTION_EMAIL)
        # The injected '\r\nBcc:' should NOT create a valid BCC field
        from_raw = parsed.get("from_address", "")
        assert "\r\n" not in from_raw
        assert "Bcc" not in from_raw

    def test_subject_injection_sanitised(self):
        parsed = parse_eml_string(INJECTION_EMAIL)
        subject = parsed.get("subject", "")
        assert "\r\n" not in subject


class TestEmptyEmail:
    def test_empty_string_does_not_crash(self):
        parsed = parse_eml_string(EMPTY_EMAIL)
        assert isinstance(parsed, dict)
        assert parsed.get("urls") == []

    def test_none_input_handled(self):
        parsed = parse_eml_string(None)
        assert isinstance(parsed, dict)


class TestEncodedHeaders:
    def test_encoded_subject_decoded(self):
        parsed = parse_eml_string(ENCODED_SUBJECT_EMAIL)
        subject = parsed.get("subject", "")
        # Should be decoded to ASCII, not the raw =?UTF-8?B?...?=
        assert "Urgent" in subject or "Verify" in subject
        assert "=?UTF-8?B?" not in subject
