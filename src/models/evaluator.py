"""
PhishLens Model Evaluator.

Produces comprehensive evaluation metrics, visualisations, and stress tests
for trained PhishLens classifiers. All outputs are saved to
`reports/figures/` and logged to MLflow.

Key security metrics:
  - False Negative Rate (FNR): Fraction of phishing emails classified as
    legitimate — the most critical security failure mode. Target FNR < 5%.
  - False Positive Rate (FPR): Fraction of legitimate emails flagged as phishing.
    High FPR causes alert fatigue and user trust erosion.
  - Matthews Correlation Coefficient (MCC): Balanced metric robust to class imbalance.
  - AUC-ROC: Discrimination ability across all confidence thresholds.
  - Confusion matrix: Visualised and saved as PNG.

Security rationale: A phishing detector with 99% accuracy but 20% FNR is
dangerous — it misses 1 in 5 phishing emails. Evaluator explicitly surfaces FNR
and FPR as primary dashboard metrics, not just accuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    _MLFLOW_AVAILABLE = False
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.utils.config import DEFAULT_CONFIG
from src.utils.logger import get_logger

log = get_logger(__name__)

FIGURES_DIR = Path("reports/figures")


class PhishLensEvaluator:
    """Evaluation engine for PhishLens classifiers.

    Args:
        threshold: Classification threshold (default 0.5).
                   Raise to reduce FPR (at cost of higher FNR).
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.results: Dict[str, Dict] = {}

    def evaluate(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model",
        scaler: Optional[Any] = None,
        log_to_mlflow: bool = True,
    ) -> Dict:
        """Evaluate a classifier and compute all security-relevant metrics.

        Args:
            model: Fitted classifier with predict_proba() method.
            X_test: Test feature matrix.
            y_test: True labels.
            model_name: Name for logging and file naming.
            scaler: Optional StandardScaler (for LR models).
            log_to_mlflow: Whether to log metrics to MLflow.

        Returns:
            Dict of evaluation metrics.
        """
        X_eval = scaler.transform(X_test) if scaler else X_test
        X_eval = np.nan_to_num(X_eval, nan=0.0, posinf=0.0, neginf=0.0)

        proba = model.predict_proba(X_eval)[:, 1]   # P(phishing)
        y_pred = (proba >= self.threshold).astype(int)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "model": model_name,
            "threshold": self.threshold,
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_test, proba)),
            "mcc": float(matthews_corrcoef(y_test, y_pred)),
            "fnr": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
            "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "n_test": len(y_test),
        }

        log.info(
            f"\n[{model_name.upper()}] "
            f"F1={metrics['f1']:.4f} | "
            f"AUC={metrics['auc_roc']:.4f} | "
            f"FNR={metrics['fnr']:.4f} | "
            f"FPR={metrics['fpr']:.4f} | "
            f"MCC={metrics['mcc']:.4f}"
        )

        if log_to_mlflow and _MLFLOW_AVAILABLE:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"test_{k}", v)

        self.results[model_name] = metrics
        return metrics

    def plot_confusion_matrix(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model",
        scaler: Optional[Any] = None,
    ) -> str:
        """Generate and save a confusion matrix PNG.

        Returns:
            File path to the saved PNG.
        """
        X_eval = scaler.transform(X_test) if scaler else X_test
        X_eval = np.nan_to_num(X_eval, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = (model.predict_proba(X_eval)[:, 1] >= self.threshold).astype(int)

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=["Legitimate", "Phishing"],
            cmap="Blues",
            ax=ax,
        )
        ax.set_title(f"PhishLens — {model_name.upper()} Confusion Matrix")
        plt.tight_layout()
        out_path = str(FIGURES_DIR / f"cm_{model_name}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log.info(f"Confusion matrix saved to '{out_path}'")
        return out_path

    def compare_models(self) -> pd.DataFrame:
        """Produce a comparison DataFrame of all evaluated models.

        Returns:
            DataFrame sorted by F1 (descending).
        """
        if not self.results:
            return pd.DataFrame()
        df = pd.DataFrame(self.results.values())
        df = df.sort_values("f1", ascending=False).reset_index(drop=True)
        return df

    def stress_test(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model",
        noise_levels: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.2),
        scaler: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Stress test robustness against Gaussian feature noise.

        Security rationale: Real phishing emails contain natural variation.
        We simulate this by adding Gaussian noise to feature vectors and
        measuring F1 degradation. A robust model should degrade gracefully.

        Args:
            model: Fitted classifier.
            X_test: Test feature matrix.
            y_test: True labels.
            model_name: For logging.
            noise_levels: Sigma values for Gaussian noise.
            scaler: Optional StandardScaler.

        Returns:
            DataFrame with noise_level and corresponding F1 score.
        """
        records: List[Dict] = []
        rng = np.random.default_rng(seed=42)

        for sigma in noise_levels:
            if sigma == 0.0:
                X_noisy = X_test.copy()
            else:
                noise = rng.normal(0, sigma, size=X_test.shape).astype(np.float32)
                X_noisy = X_test + noise

            X_eval = scaler.transform(X_noisy) if scaler else X_noisy
            X_eval = np.nan_to_num(X_eval, nan=0.0, posinf=0.0, neginf=0.0)
            proba = model.predict_proba(X_eval)[:, 1]
            y_pred = (proba >= self.threshold).astype(int)
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            records.append({"model": model_name, "noise_sigma": sigma, "f1": f1})
            log.info(f"Stress test [{model_name}] noise={sigma:.2f}: F1={f1:.4f}")

        return pd.DataFrame(records)

    def find_failure_modes(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        top_n: int = 20,
        scaler: Optional[Any] = None,
    ) -> Dict:
        """Analyse false negatives (missed phishing) and false positives.

        Returns the most common feature patterns in misclassified emails
        to help identify weaknesses and adversarial attack surfaces.

        Args:
            model: Fitted classifier.
            X_test: Test feature matrix.
            y_test: True labels.
            feature_names: List of feature names.
            top_n: Number of top features to show per failure mode.
            scaler: Optional StandardScaler.

        Returns:
            Dict with 'false_negatives' and 'false_positives' feature summaries.
        """
        X_eval = scaler.transform(X_test) if scaler else X_test
        X_eval = np.nan_to_num(X_eval, nan=0.0, posinf=0.0, neginf=0.0)
        proba = model.predict_proba(X_eval)[:, 1]
        y_pred = (proba >= self.threshold).astype(int)

        fn_mask = (y_test == 1) & (y_pred == 0)    # Phishing missed
        fp_mask = (y_test == 0) & (y_pred == 1)    # Legitimate flagged

        def top_features(X_subset: np.ndarray) -> List[Dict]:
            if len(X_subset) == 0:
                return []
            means = X_subset.mean(axis=0)
            top_idx = np.argsort(means)[::-1][:top_n]
            return [
                {"feature": feature_names[i] if i < len(feature_names) else f"feat_{i}",
                 "mean_value": float(means[i])}
                for i in top_idx
            ]

        return {
            "false_negative_count": int(fn_mask.sum()),
            "false_positive_count": int(fp_mask.sum()),
            "false_negatives_top_features": top_features(X_test[fn_mask]),
            "false_positives_top_features": top_features(X_test[fp_mask]),
        }
