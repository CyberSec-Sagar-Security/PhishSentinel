"""Standalone evaluation script for all 5 PhishLens models.

Evaluates on BOTH:
  - Inner test split  (20% of train.csv = 82,453 emails)  — same split used during training
  - Held-out test.csv (103,067 emails)                    — never seen by any model
"""
import sys, json, joblib, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.model_selection import train_test_split
from src.utils.config import DEFAULT_CONFIG
from src.features.pipeline import FeaturePipeline
from src.models.evaluator import PhishLensEvaluator
from src.utils.logger import configure_logger, get_logger

configure_logger(level="INFO")
log = get_logger(__name__)

# ── Determine which test set to use ────────────────────────────────────────
# Prefer the truly held-out test.csv (built by download_datasets.py).
# Fall back to an inner 20% split of train.csv if test.csv is absent.
_heldout = Path("data/processed/test.csv")
if _heldout.exists():
    print("[1/3] Loading HELD-OUT test set (data/processed/test.csv) ...")
    df_test = pd.read_csv(_heldout, low_memory=False)
    _test_label = "held-out test.csv"
else:
    print("[1/3] test.csv not found — using inner 20% split of train.csv ...")
    df = pd.read_csv("data/processed/train.csv", low_memory=False)
    _, df_test = train_test_split(
        df, test_size=0.2, stratify=df["label"],
        random_state=DEFAULT_CONFIG.random_state,
    )
    _test_label = "inner 20% split of train.csv"
phish = int(df_test["label"].sum())
legit = int((df_test["label"] == 0).sum())
print(f"  Test set ({_test_label}): {len(df_test):,} rows | Phishing: {phish:,} | Legit: {legit:,}")

print("[2/3] Transforming test features (cache HIT expected) ...")
pipeline = FeaturePipeline.load("models/feature_pipeline.pkl")
X_test, feat_names = pipeline.transform(df_test)
y_test = df_test["label"].values.astype(int)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
print(f"  Feature matrix: {X_test.shape}")

print("[3/3] Evaluating all 5 models ...")
ev = PhishLensEvaluator(threshold=DEFAULT_CONFIG.prediction_threshold)
for name in ["lightgbm", "xgboost", "catboost", "rf", "lr"]:
    m = joblib.load(f"models/{name}.pkl")
    sc_path = Path(f"models/{name}_scaler.pkl")
    sc = joblib.load(sc_path) if sc_path.exists() else None
    ev.evaluate(m, X_test, y_test, model_name=name, scaler=sc)
    ev.plot_confusion_matrix(m, X_test, y_test, model_name=name, scaler=sc)

cmp = ev.compare_models()
cols = ["model", "f1", "auc_roc", "fnr", "fpr", "mcc", "precision", "recall"]
print()
print(cmp[cols].to_string(index=False))

Path("reports/metrics.json").write_text(json.dumps(cmp.to_dict(orient="records"), indent=2))
print("\nmetrics.json saved to reports/metrics.json")
