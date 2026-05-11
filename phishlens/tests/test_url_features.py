"""
Tests for url_features.py — covers lexical feature extraction,
entropy calculation, brand detection, TLD risk scoring, and
edge cases (empty URL list, IP-based URLs, Punycode domains).
"""

from __future__ import annotations

import pytest

from src.features.url_features import extract_url_features
from src.utils.config import DEFAULT_CONFIG


class TestLexicalFeatures:
    def test_empty_url_list_returns_zeros(self):
        feats = extract_url_features([], DEFAULT_CONFIG)
        assert feats["url_count"] == 0

    def test_domain_length_extracted(self):
        feats = extract_url_features(["http://verylongdomainname.example.com/page"], DEFAULT_CONFIG)
        assert feats["domain_length_max"] > 0

    def test_hyphen_count_detected(self):
        feats = extract_url_features(["http://paypal-secure-login.com/"], DEFAULT_CONFIG)
        assert feats["hyphen_count_max"] >= 2

    def test_digit_ratio_calculated(self):
        feats = extract_url_features(["http://1234567890.example.com/"], DEFAULT_CONFIG)
        assert feats["digit_ratio_max"] > 0

    def test_url_entropy_calculated(self):
        feats = extract_url_features(
            ["http://aXbYcZdW1234-random.phishing.example/verify?token=abc123def456"],
            DEFAULT_CONFIG,
        )
        assert feats["url_entropy_max"] > 2.0

    def test_subdomain_depth_measured(self):
        feats = extract_url_features(
            ["http://one.two.three.deep.example.com/"], DEFAULT_CONFIG
        )
        assert feats["subdomain_depth_max"] >= 3

    def test_path_depth_measured(self):
        feats = extract_url_features(
            ["http://example.com/a/b/c/d/e"], DEFAULT_CONFIG
        )
        assert feats["path_depth_max"] >= 5


class TestBrandDetection:
    def test_brand_in_subdomain_detected(self):
        feats = extract_url_features(
            ["http://paypal.malicious.xyz/login"], DEFAULT_CONFIG
        )
        assert feats["brand_in_subdomain_max"] == 1

    def test_no_brand_in_legitimate_url(self):
        feats = extract_url_features(
            ["https://www.paypal.com/login"], DEFAULT_CONFIG
        )
        # Brand is in the APEX domain (paypal.com), not a subdomain — should not flag
        assert feats["brand_in_subdomain_max"] == 0


class TestIPAndPunycode:
    def test_ip_address_url_detected(self):
        feats = extract_url_features(
            ["http://192.168.1.1/login"], DEFAULT_CONFIG
        )
        assert feats["is_ip_address_max"] == 1

    def test_punycode_domain_detected(self):
        feats = extract_url_features(
            ["http://xn--pypl-poa.com/login"], DEFAULT_CONFIG
        )
        assert feats["punycode_detected_max"] == 1


class TestURLShorteners:
    def test_url_shortener_detected(self):
        feats = extract_url_features(["http://bit.ly/abc123"], DEFAULT_CONFIG)
        assert feats["url_shortener_max"] == 1

    def test_regular_url_not_flagged_as_shortener(self):
        feats = extract_url_features(["https://www.google.com/"], DEFAULT_CONFIG)
        assert feats["url_shortener_max"] == 0


class TestTLDRisk:
    def test_risky_tld_scores_higher(self):
        feats_risk = extract_url_features(["http://phishing.xyz/"], DEFAULT_CONFIG)
        feats_safe = extract_url_features(["http://legit.com/"], DEFAULT_CONFIG)
        # .xyz should have higher risk score than .com
        assert feats_risk["tld_risk_score_max"] >= feats_safe["tld_risk_score_max"]


class TestAggregation:
    def test_max_and_mean_computed(self):
        feats = extract_url_features([
            "http://short.co/",
            "http://paypal.evil.xyz/verify?token=abc123def456abc",
        ], DEFAULT_CONFIG)
        assert "domain_length_max" in feats
        assert "domain_length_mean" in feats
        assert feats["domain_length_max"] >= feats["domain_length_mean"]

    def test_url_count_correct(self):
        urls = ["http://a.com/", "http://b.com/", "http://c.com/"]
        feats = extract_url_features(urls, DEFAULT_CONFIG)
        assert feats["url_count"] == 3
