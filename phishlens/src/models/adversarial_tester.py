"""
PhishLens Adversarial Robustness Tester.

Tests model robustness against adversarial perturbations — carefully crafted
modifications to phishing emails designed to evade detection. This mirrors
real-world adversarial attacks observed in the wild.

Attack types simulated:
  1. Whitespace insertion: Adding zero-width spaces, non-breaking spaces
     between characters to break tokenisation.
  2. Homoglyph substitution: Replacing Latin characters with visually
     identical Unicode characters (e.g., 'а' Cyrillic for 'a' Latin).
  3. URL obfuscation: Adding random subdomains, query parameters, redirectors.
  4. Header spoofing: Altering Received headers to look geographically benign.
  5. Brand name mutation: Adding or removing characters from brand names.
  6. Urgency dilution: Adding neutral phrases alongside urgency phrases.

Security rationale: Adversarial testing is mandated by security ML best practice
(e.g., MITRE ATLAS). An ML model that achieves 99% accuracy on test data but
drops to 70% after simple perturbations is not production-ready. PhishLens targets
>90% robustness score — F1 after adversarial perturbation / F1 on clean data.
"""

from __future__ import annotations

import random
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score

from src.utils.logger import get_logger

log = get_logger(__name__)

# Homoglyph map: Latin → visually similar Unicode characters
_HOMOGLYPHS: Dict[str, str] = {
    "a": "а",   # Cyrillic а
    "e": "е",   # Cyrillic е
    "o": "о",   # Cyrillic о
    "p": "р",   # Cyrillic р
    "c": "с",   # Cyrillic с
    "x": "х",   # Cyrillic х
    "A": "А",   # Cyrillic А
    "B": "В",   # Cyrillic В
    "E": "Е",   # Cyrillic Е
    "H": "Н",   # Cyrillic Н
    "M": "М",   # Cyrillic М
    "O": "О",   # Cyrillic О
    "T": "Т",   # Cyrillic Т
    "X": "Х",   # Cyrillic Х
}

# Zero-width characters for whitespace insertion attacks
_ZWS_CHARS = ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"]

# URL shorteners / redirectors to insert
_REDIRECTORS = [
    "https://bit.ly/xyz?url=",
    "https://tinyurl.com/redirect?to=",
    "https://t.co/go?href=",
]


class AdversarialTester:
    """Tests PhishLens model robustness against adversarial email perturbations.

    Args:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def run_full_test(
        self,
        model: Any,
        pipeline,
        phishing_emails: List[str],
        y_true_phishing: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict:
        """Run all adversarial attack types and compute robustness score.

        The robustness score is:
            robustness = F1(adversarial) / F1(clean)

        Target: robustness >= 0.90

        Args:
            model: Fitted PhishLens classifier.
            pipeline: Fitted FeaturePipeline instance.
            phishing_emails: List of raw phishing email strings to perturb.
            y_true_phishing: Labels (all should be 1 for phishing emails).
            threshold: Classification threshold.

        Returns:
            Dict with per-attack type results and overall robustness score.
        """
        import pandas as pd

        log.info(f"Running adversarial tests on {len(phishing_emails):,} phishing emails ...")

        # Baseline F1 on clean phishing emails
        df_clean = pd.DataFrame({"raw_email": phishing_emails})
        X_clean, _ = pipeline.transform(df_clean)
        X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=0.0, neginf=0.0)
        proba_clean = model.predict_proba(X_clean)[:, 1]
        y_pred_clean = (proba_clean >= threshold).astype(int)
        f1_clean = float(f1_score(y_true_phishing, y_pred_clean, zero_division=0))
        log.info(f"Clean F1: {f1_clean:.4f}")

        attack_results: Dict[str, Dict] = {}

        attacks = {
            "whitespace_insertion": self.whitespace_insertion,
            "homoglyph_substitution": self.homoglyph_substitution,
            "url_obfuscation": self.url_obfuscation,
            "urgency_dilution": self.urgency_dilution,
            "brand_mutation": self.brand_mutation,
        }

        for attack_name, attack_fn in attacks.items():
            log.info(f"Attack: {attack_name}")
            perturbed = [attack_fn(email) for email in phishing_emails]
            df_perturbed = pd.DataFrame({"raw_email": perturbed})
            X_adv, _ = pipeline.transform(df_perturbed)
            X_adv = np.nan_to_num(X_adv, nan=0.0, posinf=0.0, neginf=0.0)
            proba_adv = model.predict_proba(X_adv)[:, 1]
            y_pred_adv = (proba_adv >= threshold).astype(int)
            f1_adv = float(f1_score(y_true_phishing, y_pred_adv, zero_division=0))
            evasion_rate = float(1.0 - (y_pred_adv.sum() / max(y_true_phishing.sum(), 1)))

            attack_results[attack_name] = {
                "f1_clean": f1_clean,
                "f1_adversarial": f1_adv,
                "robustness": f1_adv / f1_clean if f1_clean > 0 else 0.0,
                "evasion_rate": evasion_rate,
            }
            log.info(
                f"  {attack_name}: F1={f1_adv:.4f}, "
                f"Robustness={attack_results[attack_name]['robustness']:.3f}, "
                f"Evasion={evasion_rate:.3f}"
            )

        overall_robustness = float(
            np.mean([v["robustness"] for v in attack_results.values()])
        )
        log.info(f"\nOverall robustness score: {overall_robustness:.4f} "
                 f"(target: >= 0.90)")

        return {
            "overall_robustness": overall_robustness,
            "f1_clean": f1_clean,
            "attack_results": attack_results,
            "passed": overall_robustness >= 0.90,
        }

    def whitespace_insertion(self, email: str) -> str:
        """Insert zero-width characters into email body to disrupt tokenisation."""
        # Only perturb body lines (after header block)
        parts = email.split("\n\n", 1)
        if len(parts) < 2:
            return email
        headers, body = parts[0], parts[1]
        perturbed_body = ""
        for char in body:
            perturbed_body += char
            if char.isalpha() and self.rng.random() < 0.1:
                perturbed_body += self.rng.choice(_ZWS_CHARS)
        return headers + "\n\n" + perturbed_body

    def homoglyph_substitution(self, email: str) -> str:
        """Replace Latin chars with Cyrillic homoglyphs in 20% of occurrences."""
        result = []
        for char in email:
            if char in _HOMOGLYPHS and self.rng.random() < 0.2:
                result.append(_HOMOGLYPHS[char])
            else:
                result.append(char)
        return "".join(result)

    def url_obfuscation(self, email: str) -> str:
        """Wrap URLs in a redirect chain to break URL feature extraction."""
        url_pattern = re.compile(r"https?://[^\s<>\"']+")
        def replace_url(m: re.Match) -> str:
            original = m.group(0)
            redirector = self.rng.choice(_REDIRECTORS)
            return redirector + original
        return url_pattern.sub(replace_url, email)

    def urgency_dilution(self, email: str) -> str:
        """Add neutral filler sentences around urgency phrases to lower urgency score."""
        filler_phrases = [
            " Please take your time to review this message at your convenience.",
            " There is no rush to respond immediately.",
            " This is a routine notification for your records.",
        ]
        urgency_words = ["immediately", "urgent", "action required", "verify now", "expires"]
        result = email
        for word in urgency_words:
            if word.lower() in result.lower():
                filler = self.rng.choice(filler_phrases)
                result = re.sub(
                    re.escape(word),
                    word + filler,
                    result,
                    count=1,
                    flags=re.IGNORECASE,
                )
        return result

    def brand_mutation(self, email: str) -> str:
        """Add characters around brand names to confuse brand detection."""
        brands = ["PayPal", "Apple", "Microsoft", "Amazon", "Google", "Netflix",
                  "Bank of Ireland", "AIB", "An Post", "Revenue", "DHL", "FedEx"]
        result = email
        for brand in brands:
            if brand in result:
                # Insert a zero-width space in the middle of the brand name
                mid = len(brand) // 2
                mutated = brand[:mid] + "\u200b" + brand[mid:]
                result = result.replace(brand, mutated, 1)
        return result
