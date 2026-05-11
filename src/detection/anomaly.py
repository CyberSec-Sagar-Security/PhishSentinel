"""
PhishLens Zero-Day Anomaly Detection Module.

Implements an Isolation Forest trained exclusively on legitimate (label=0)
email features. Emails that are anomalous relative to this "normal" baseline
are flagged even if they don't match known phishing patterns — this is the
zero-day detection layer.

Security rationale: Supervised ML (XGBoost, RF) can only detect phishing
patterns seen in training data. Novel attack campaigns — new brands being
targeted, new social engineering lures, new URL structures — will not match
training distribution and may evade the supervised classifier. Isolation Forest
provides a distribution-free anomaly layer that flags structurally unusual emails
regardless of whether they match known phishing signatures.

Reference:
    Liu, F.T., Ting, K.M., & Zhou, Z.H. (2008). Isolation Forest.
    ICDM 2008. https://doi.org/10.1109/ICDM.2008.17
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from src.utils.config import DEFAULT_CONFIG
from src.utils.logger import get_logger

log = get_logger(__name__)


class ZeroDayDetector:
    """Isolation Forest based zero-day phishing anomaly detector.

    Trained only on legitimate email features. Anomaly scores are inverted
    and normalised to [0, 1] where 1.0 = most anomalous.

    Args:
        config: PhishLensConfig instance.
    """

    def __init__(self, config=DEFAULT_CONFIG) -> None:
        self.config = config
        self._model: Optional[IsolationForest] = None
        self._scaler: Optional[MinMaxScaler] = None
        self._is_fitted: bool = False
        self._feature_names: List[str] = []

    def __repr__(self) -> str:
        return (
            f"ZeroDayDetector("
            f"fitted={self._is_fitted}, "
            f"contamination={self.config.anomaly_contamination})"
        )

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "ZeroDayDetector":
        """Train Isolation Forest on legitimate email features only.

        Security rationale: Training only on label=0 (legitimate) emails ensures
        the model learns what 'normal' looks like. Any email outside this distribution
        — including novel phishing campaigns — receives a high anomaly score.

        Args:
            X: Feature matrix (all emails — method filters to label=0 internally).
            y: Binary labels (0=legitimate, 1=phishing).
            feature_names: Optional list of feature names for explanation.

        Returns:
            self (for method chaining).
        """
        # Filter to legitimate emails only
        legit_mask = y == 0
        X_legit = X[legit_mask]

        log.info(
            f"Training ZeroDayDetector on {X_legit.shape[0]:,} legitimate emails "
            f"({legit_mask.sum():,} of {len(y):,} total) ..."
        )

        # Replace NaN/inf with 0 (network feature failures produce -1/NaN)
        X_legit = np.nan_to_num(X_legit, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features to [0, 1] before Isolation Forest
        self._scaler = MinMaxScaler()
        X_scaled = self._scaler.fit_transform(X_legit)

        self._model = IsolationForest(
            contamination=self.config.anomaly_contamination,
            n_estimators=200,          # More trees = more stable anomaly scores
            max_samples="auto",
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        self._model.fit(X_scaled)
        self._is_fitted = True
        if feature_names:
            self._feature_names = feature_names

        log.info("ZeroDayDetector fitted successfully.")
        return self

    def predict_proba_anomaly(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly probability scores for a feature matrix.

        Scores are normalised to [0, 1] where:
          - 1.0 = highly anomalous (likely novel phishing or zero-day)
          - 0.0 = behaves like typical legitimate email

        Security rationale: The raw Isolation Forest score is negative (more
        anomalous = more negative). We invert and normalise to a probability-like
        score that is interpretable in the Streamlit UI.

        Args:
            X: Feature matrix shape [n_samples, n_features].

        Returns:
            Anomaly scores shape [n_samples], float in [0, 1].
        """
        if not self._is_fitted or self._model is None:
            log.warning("ZeroDayDetector not fitted — returning neutral scores.")
            return np.full(len(X), 0.5)

        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self._scaler.transform(X_clean)

        # Raw score: more negative = more anomalous
        raw_scores = self._model.score_samples(X_scaled)

        # Invert and normalise to [0, 1]
        # Lower raw_score (more anomalous) → higher anomaly probability
        min_score, max_score = raw_scores.min(), raw_scores.max()
        if max_score == min_score:
            return np.full(len(X), 0.5)

        anomaly_scores = 1.0 - (raw_scores - min_score) / (max_score - min_score)
        return anomaly_scores.astype(np.float32)

    def predict_single_anomaly(self, x: np.ndarray) -> float:
        """Return anomaly score for a single email feature vector.

        Args:
            x: Single email feature vector, shape [n_features].

        Returns:
            Float anomaly score in [0, 1].
        """
        return float(self.predict_proba_anomaly(x.reshape(1, -1))[0])

    def explain_anomaly(self, x: np.ndarray, top_n: int = 10) -> List[Dict]:
        """Identify which features contribute most to the anomaly score.

        Uses a perturbation approach: zero out each feature and measure
        how much the anomaly score changes. Features with the largest impact
        are the primary anomaly drivers.

        Security rationale: When an email is flagged as anomalous, analysts
        need to know WHY — which structural properties deviate from normal.
        This explanation bridges the gap between the black-box anomaly score
        and actionable SOC investigation.

        Args:
            x: Single email feature vector, shape [n_features].
            top_n: Number of top contributing features to return.

        Returns:
            List of dicts: [{'feature': name, 'value': val, 'impact': delta_score}, ...]
        """
        if not self._is_fitted:
            return []

        base_score = self.predict_single_anomaly(x)
        contributions: List[Tuple[str, float, float]] = []

        for i in range(len(x)):
            x_perturbed = x.copy()
            x_perturbed[i] = 0.0    # Zero out this feature
            perturbed_score = self.predict_single_anomaly(x_perturbed)
            impact = base_score - perturbed_score   # Positive = this feature increased anomaly

            feature_name = self._feature_names[i] if i < len(self._feature_names) else f"feat_{i}"
            contributions.append((feature_name, float(x[i]), float(impact)))

        # Sort by absolute impact (most influential first)
        contributions.sort(key=lambda t: abs(t[2]), reverse=True)

        return [
            {"feature": name, "value": value, "impact": impact}
            for name, value, impact in contributions[:top_n]
        ]

    def save(self, path: str) -> None:
        """Serialise the fitted detector to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self._model,
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "config": self.config,
        }, path)
        log.info(f"ZeroDayDetector saved to '{path}'")

    @classmethod
    def load(cls, path: str) -> "ZeroDayDetector":
        """Load a serialised ZeroDayDetector from disk."""
        data = joblib.load(path)
        instance = cls(config=data.get("config", DEFAULT_CONFIG))
        instance._model = data["model"]
        instance._scaler = data["scaler"]
        instance._feature_names = data.get("feature_names", [])
        instance._is_fitted = True
        log.info(f"ZeroDayDetector loaded from '{path}'")
        return instance
