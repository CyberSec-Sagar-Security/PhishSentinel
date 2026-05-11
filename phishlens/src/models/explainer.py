"""
PhishLens SHAP + LIME Explainability Module.

Provides dual-method explainability for all PhishLens classifiers:
  - SHAP (SHapley Additive exPlanations) via TreeExplainer (tree models) or
    LinearExplainer (Logistic Regression)
  - LIME (Local Interpretable Model-agnostic Explanations) as independent check

An "agreement score" quantifies how well SHAP and LIME agree on the top
contributing features — high agreement increases analyst trust in the explanation.

Security rationale: ML model explanations are a critical part of any security
tool deployed in a real SOC environment. Analysts need to understand WHY an
email was flagged to:
  1. Verify the flag is correct (not a false positive)
  2. Document the phishing indicators for incident reports
  3. Feed intelligence back into detection rules
  4. Avoid over-trusting black-box predictions

Reference:
  - Lundberg, S.M. & Lee, S.I. (2017). A unified approach to interpreting model predictions.
    NeurIPS 2017. https://arxiv.org/abs/1705.07874
  - Ribeiro, M.T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?"
    ICML 2016. https://arxiv.org/abs/1602.04938
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression

from src.utils.logger import get_logger

log = get_logger(__name__)


class PhishExplainer:
    """Dual SHAP + LIME explainer for PhishLens classifiers.

    Args:
        model: Fitted classifier (must have predict_proba).
        feature_names: List of feature names corresponding to X columns.
        X_train: Training data (required for LIME background and TreeExplainer).
        model_type: One of 'tree', 'linear', 'generic'. Auto-detected if None.
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        X_train: np.ndarray,
        model_type: Optional[str] = None,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.X_train = np.nan_to_num(X_train.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self.model_type = model_type or self._detect_model_type(model)
        self._shap_explainer = None
        self._lime_explainer = None
        self._init_explainers()

    def _detect_model_type(self, model: Any) -> str:
        """Detect model type for choosing the best SHAP explainer."""
        model_class = type(model).__name__.lower()
        if any(t in model_class for t in ["xgb", "lgbm", "catboost", "randomforest", "gradientboosting"]):
            return "tree"
        if "logistic" in model_class or "linear" in model_class:
            return "linear"
        return "generic"

    def _init_explainers(self) -> None:
        """Initialise SHAP and LIME explainers."""
        log.info(f"Initialising {self.model_type} SHAP explainer ...")
        try:
            if self.model_type == "tree":
                self._shap_explainer = shap.TreeExplainer(self.model)
            elif self.model_type == "linear":
                self._shap_explainer = shap.LinearExplainer(
                    self.model, self.X_train, feature_names=self.feature_names
                )
            else:
                # Fallback: KernelExplainer with 100-sample background (slow but universal)
                background = shap.sample(self.X_train, 100)
                self._shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba, background
                )
            log.info("SHAP explainer ready.")
        except Exception as exc:
            log.warning(f"SHAP initialisation failed: {exc}")

        try:
            self._lime_explainer = LimeTabularExplainer(
                training_data=self.X_train,
                feature_names=self.feature_names,
                class_names=["Legitimate", "Phishing"],
                mode="classification",
                discretize_continuous=True,
                random_state=42,
            )
            log.info("LIME explainer ready.")
        except Exception as exc:
            log.warning(f"LIME initialisation failed: {exc}")

    def explain_single(
        self,
        x: np.ndarray,
        top_n: int = 15,
    ) -> Dict:
        """Generate SHAP + LIME explanations for a single email.

        Args:
            x: Single email feature vector shape [n_features].
            top_n: Number of top features to return per method.

        Returns:
            Dict with:
              - shap_features: List of {feature, shap_value, value} dicts
              - lime_features: List of {feature, weight, value} dicts
              - agreement_score: Float 0–1 measuring SHAP/LIME agreement
              - phishing_risk_features: Top features pushing toward phishing
        """
        x_clean = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        result: Dict = {
            "shap_features": [],
            "lime_features": [],
            "agreement_score": 0.0,
            "phishing_risk_features": [],
        }

        # --- SHAP -----------------------------------------------------------
        shap_top: List[Tuple[str, float]] = []
        if self._shap_explainer is not None:
            try:
                shap_vals = self._shap_explainer.shap_values(x_clean.reshape(1, -1))
                # For binary classifiers, shap_values is either a 2D array or list
                if isinstance(shap_vals, list):
                    vals = shap_vals[1][0]      # Class 1 (phishing) shap values
                else:
                    vals = shap_vals[0] if shap_vals.ndim == 2 else shap_vals

                # Sort by absolute value
                sorted_idx = np.argsort(np.abs(vals))[::-1][:top_n]
                shap_top = []
                for col_idx in sorted_idx:
                    fname = self.feature_names[col_idx] if col_idx < len(self.feature_names) else f"feat_{col_idx}"
                    shap_top.append((fname, float(vals[col_idx])))
                    result["shap_features"].append({
                        "feature": fname,
                        "shap_value": float(vals[col_idx]),
                        "value": float(x_clean[col_idx]),  # correct column index
                    })

                # Phishing risk features: positive SHAP (push toward phishing)
                result["phishing_risk_features"] = [
                    {"feature": f, "shap_value": v}
                    for f, v in sorted(shap_top, key=lambda t: t[1], reverse=True)
                    if v > 0
                ][:top_n]

            except Exception as exc:
                log.warning(f"SHAP explanation failed: {exc}")

        # --- LIME -----------------------------------------------------------
        lime_top: List[Tuple[str, float]] = []
        if self._lime_explainer is not None:
            try:
                lime_exp = self._lime_explainer.explain_instance(
                    x_clean,
                    self.model.predict_proba,
                    num_features=top_n,
                    labels=(1,),    # Explain phishing class
                )
                lime_top = lime_exp.as_list(label=1)
                # Build a name->column_index lookup for correct value retrieval
                _name_to_col: dict = {}
                if self.feature_names:
                    _name_to_col = {n: i for i, n in enumerate(self.feature_names)}

                lime_features_out = []
                for feat, weight in lime_top:
                    # Extract base feature name from LIME condition string
                    # e.g. "url_domain_length > 0.50" -> "url_domain_length"
                    base_name = feat
                    for op in (" <= ", " > ", " < ", " >= ", " = "):
                        if op in feat:
                            base_name = feat.split(op)[0].strip()
                            break
                    col_idx = _name_to_col.get(base_name)
                    feat_val = float(x_clean[col_idx]) if col_idx is not None else 0.0
                    lime_features_out.append({
                        "feature": feat,
                        "weight": float(weight),
                        "value": feat_val,
                    })
                result["lime_features"] = lime_features_out
            except Exception as exc:
                log.warning(f"LIME explanation failed: {exc}")

        # --- Agreement score ------------------------------------------------
        result["agreement_score"] = self._compute_agreement(shap_top, lime_top, top_n=5)

        return result

    def _compute_agreement(
        self,
        shap_top: List[Tuple[str, float]],
        lime_top: List[Tuple[str, float]],
        top_n: int = 5,
    ) -> float:
        """Compute Jaccard similarity between top-N SHAP and LIME features.

        Agreement score = |SHAP_top ∩ LIME_top| / |SHAP_top ∪ LIME_top|

        Args:
            shap_top: SHAP top features (name, value) tuples.
            lime_top: LIME top features (condition, weight) tuples.
            top_n: Top-N features to compare.

        Returns:
            Float in [0, 1]. 1.0 = perfect agreement.
        """
        if not shap_top or not lime_top:
            return 0.0

        # Extract base feature names from LIME (conditions include comparison operators)
        def _extract_name(lime_feat: str) -> str:
            # LIME produces e.g. "url_domain_length > 0.50" — extract base name
            for op in [" <= ", " > ", " < ", " >= ", " = "]:
                if op in lime_feat:
                    return lime_feat.split(op)[0].strip()
            return lime_feat.strip()

        shap_names = set(name for name, _ in shap_top[:top_n])
        lime_names = set(_extract_name(feat) for feat, _ in lime_top[:top_n])

        intersection = len(shap_names & lime_names)
        union = len(shap_names | lime_names)
        return float(intersection / union) if union > 0 else 0.0

    def batch_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values for a batch of emails.

        Used for population-level feature importance analysis and
        generating SHAP summary plots.

        Args:
            X: Feature matrix shape [n_samples, n_features].

        Returns:
            SHAP values array shape [n_samples, n_features].
        """
        if self._shap_explainer is None:
            return np.zeros_like(X)

        X_clean = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        try:
            vals = self._shap_explainer.shap_values(X_clean)
            if isinstance(vals, list):
                return vals[1]      # Return phishing class values
            return vals
        except Exception as exc:
            log.warning(f"Batch SHAP failed: {exc}")
            return np.zeros_like(X_clean)
