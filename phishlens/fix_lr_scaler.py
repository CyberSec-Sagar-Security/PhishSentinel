"""Regenerate lr_scaler.pkl using the correct 80-20 training split."""
import sys, joblib, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils.config import DEFAULT_CONFIG
from src.features.pipeline import FeaturePipeline
from src.utils.logger import configure_logger, get_logger

configure_logger(level="INFO")
log = get_logger(__name__)

log.info("[1/4] Loading train.csv …")
df = pd.read_csv("data/processed/train.csv", low_memory=False)
log.info(f"  Total rows: {len(df):,}")

log.info("[2/4] 80/20 split (same random_state as training) …")
df_train, df_test = train_test_split(
    df, test_size=0.2, stratify=df["label"],
    random_state=DEFAULT_CONFIG.random_state,
)
log.info(f"  Train: {len(df_train):,}  Test: {len(df_test):,}")

log.info("[3/4] Applying feature pipeline to training set …")
pipeline = FeaturePipeline.load("models/feature_pipeline.pkl")
X_train, _ = pipeline.transform(df_train)
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
log.info(f"  Feature matrix: {X_train.shape}")

log.info("[4/4] Fitting StandardScaler and saving …")
scaler = StandardScaler()
scaler.fit(X_train)
out_path = Path("models/lr_scaler.pkl")
joblib.dump(scaler, out_path)
log.info(f"  lr_scaler.pkl saved → {out_path.stat().st_mtime}")
log.info(f"  mean_ shape: {scaler.mean_.shape}, var_ shape: {scaler.var_.shape}")
log.info("Done. Re-run eval_all.py to get updated LR metrics.")
