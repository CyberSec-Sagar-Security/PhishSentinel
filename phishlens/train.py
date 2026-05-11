"""
PhishLens Training CLI.

End-to-end training pipeline for PhishLens classifiers.
Orchestrates data loading → feature extraction → model training →
evaluation → adversarial testing → report generation.

Usage examples:
    # Full training pipeline with all models + tuning:
    python train.py --data-dir data/processed --models all --tune --eval

    # Fast training (XGBoost only, no Optuna):
    python train.py --data-dir data/processed --models xgboost --eval

    # Offline training (no network calls, no intelligence APIs):
    python train.py --data-dir data/processed --no-network --eval

    # Save pipeline and models to specific directory:
    python train.py --data-dir data/processed --save models/v2

Security note: This script should only be run in a controlled environment
with access to the training datasets. Do NOT run in production environments
with live API keys enabled for full training (rate limits apply).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Load .env BEFORE any src imports so module-level os.getenv() calls in
# intelligence.py, app.py, etc. pick up the real API key values.
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd

# Add src to path so imports work from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.dataset_loader import combine_datasets, load_meajor, load_spamassassin, load_casis
from src.features.pipeline import FeaturePipeline
from src.detection.anomaly import ZeroDayDetector
from src.models.trainer import PhishLensTrainer, AVAILABLE_MODELS
from src.models.evaluator import PhishLensEvaluator
from src.models.explainer import PhishExplainer
from src.utils.config import DEFAULT_CONFIG
from src.utils.logger import configure_logger, get_logger

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_data_dir = "data/processed"
    if not (Path(__file__).parent / default_data_dir / "train.csv").exists():
        default_data_dir = "data/raw"

    parser = argparse.ArgumentParser(
        prog="python train.py",
        description="PhishLens — ML Phishing Email Detector Training CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=default_data_dir,
        help="Directory containing training email datasets.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        choices=list(AVAILABLE_MODELS) + ["all"],
        help="Which model(s) to train.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Run Optuna hyperparameter tuning (50 trials per model).",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=True,
        help="Run evaluation on held-out test set after training.",
    )
    parser.add_argument(
        "--no-eval",
        dest="eval",
        action="store_false",
        help="Skip evaluation (faster for debugging).",
    )
    parser.add_argument(
        "--save",
        type=str,
        nargs="?",
        const="models/",
        default="models/",
        help="Directory to save trained models and pipeline. If provided without a value, defaults to models/.",
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        default=False,
        help="Disable WHOIS/crt.sh/API calls during feature extraction (faster).",
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        default=False,
        help="Disable SMOTE oversampling.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data held out for testing (0.0–0.5).",
    )
    parser.add_argument(
        "--adversarial",
        action="store_true",
        default=False,
        help="Run adversarial robustness tests after training.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    return parser.parse_args()


def main() -> int:
    """Main training pipeline entry point. Returns exit code (0=success)."""
    args = parse_args()
    configure_logger(level=args.log_level)
    start_time = time.time()

    log.info("=" * 70)
    log.info("PhishLens Training Pipeline Starting")
    log.info("=" * 70)
    log.info(f"Data directory : {args.data_dir}")
    log.info(f"Models         : {args.models}")
    log.info(f"Optuna tuning  : {args.tune}")
    log.info(f"Network calls  : {not args.no_network}")
    log.info(f"Save directory : {args.save}")

    # -----------------------------------------------------------------------
    # Step 1: Load training data
    # -----------------------------------------------------------------------
    log.info("\n[1/6] Loading training datasets ...")
    data_path = Path(args.data_dir)

    try:
        # Fast path: use pre-built processed CSV from download_datasets.py
        train_csv = data_path / "train.csv"
        if train_csv.exists():
            log.info(f"Loading pre-built dataset from '{train_csv}' ...")
            df_loaded = pd.read_csv(train_csv, low_memory=False)
            if "source" not in df_loaded.columns:
                df_loaded["source"] = "processed"
            df = combine_datasets(df_loaded)
        else:
            # Raw data path — load from individual source directories
            log.info(f"Loading raw dataset sources from '{data_path}' ...")
            df = combine_datasets(
                load_meajor(str(data_path)),
                load_spamassassin(str(data_path)),
                load_casis(str(data_path)),
            )
    except Exception as exc:
        log.error(f"Failed to load training data: {exc}")
        log.error(
            f"Ensure datasets are in '{data_path}'. "
            "Run: python download_datasets.py  (see data/README.md)"
        )
        return 1

    log.info(f"Loaded {len(df):,} emails | "
             f"Phishing: {df['label'].sum():,} | "
             f"Legitimate: {(df['label'] == 0).sum():,}")

    # -----------------------------------------------------------------------
    # Step 2: Train/test split
    # -----------------------------------------------------------------------
    log.info(f"\n[2/6] Splitting data ({1 - args.test_size:.0%} train / {args.test_size:.0%} test) ...")
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(
        df,
        test_size=args.test_size,
        stratify=df["label"],
        random_state=DEFAULT_CONFIG.random_state,
    )
    log.info(f"Train: {len(df_train):,} | Test: {len(df_test):,}")

    # -----------------------------------------------------------------------
    # Step 3: Feature extraction
    # -----------------------------------------------------------------------
    log.info("\n[3/6] Extracting features ...")
    pipeline = FeaturePipeline(
        config=DEFAULT_CONFIG,
        use_network=not args.no_network,
        use_intelligence_apis=False,    # Disabled in training (rate limits)
        use_gemini=False,               # Disabled in training (rate limits)
        use_tfidf=True,
    )

    X_train, feature_names = pipeline.fit_transform(df_train)
    y_train = df_train["label"].values.astype(int)

    X_test, _ = pipeline.transform(df_test)
    y_test = df_test["label"].values.astype(int)

    log.info(f"Feature matrix: train={X_train.shape}, test={X_test.shape}")

    # Save pipeline
    pipeline_path = Path(args.save) / "feature_pipeline.pkl"
    pipeline.save(str(pipeline_path))

    # -----------------------------------------------------------------------
    # Step 4: Anomaly detector (Zero-day layer)
    # -----------------------------------------------------------------------
    log.info("\n[4/6] Training Isolation Forest anomaly detector ...")
    detector = ZeroDayDetector(config=DEFAULT_CONFIG)
    detector.fit(X_train, y_train, feature_names=feature_names)
    detector.save(str(Path(args.save) / "anomaly_detector.pkl"))

    # -----------------------------------------------------------------------
    # Step 5: Model training
    # -----------------------------------------------------------------------
    model_names = list(AVAILABLE_MODELS) if args.models == "all" else [args.models]
    log.info(f"\n[5/6] Training models: {model_names}")

    trainer = PhishLensTrainer(
        config=DEFAULT_CONFIG,
        model_names=tuple(model_names),
        tune=args.tune,
        use_smote=not args.no_smote,
    )
    trainer.train(X_train, y_train, feature_names=feature_names,
                  save_checkpoint_dir=args.save)
    trainer.save_all(args.save)

    # -----------------------------------------------------------------------
    # Step 6: Evaluation
    # -----------------------------------------------------------------------
    if args.eval:
        log.info("\n[6/6] Evaluating models on test set ...")
        evaluator = PhishLensEvaluator(threshold=DEFAULT_CONFIG.prediction_threshold)

        for name, model in trainer.trained_models.items():
            scaler = trainer.scalers.get(name)
            evaluator.evaluate(model, X_test, y_test, model_name=name, scaler=scaler)
            evaluator.plot_confusion_matrix(model, X_test, y_test, model_name=name, scaler=scaler)

        # Print comparison table
        comparison = evaluator.compare_models()
        log.info("\nModel Comparison:")
        log.info("\n" + comparison.to_string(index=False))

        # Save metrics JSON for the Performance Dashboard
        import json
        metrics_path = Path("reports/metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_records = comparison.to_dict(orient="records")
        with open(metrics_path, "w", encoding="utf-8") as _f:
            json.dump(metrics_records, _f, indent=2)
        log.info(f"Metrics saved to '{metrics_path}'")

        # Stress test best model (highest F1)
        if not comparison.empty:
            best_model_name = comparison.iloc[0]["model"]
            best_model = trainer.trained_models[best_model_name]
            best_scaler = trainer.scalers.get(best_model_name)
            log.info(f"\nStress testing best model: {best_model_name}")
            stress_df = evaluator.stress_test(
                best_model, X_test, y_test, model_name=best_model_name, scaler=best_scaler
            )
            log.info("\n" + stress_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # Adversarial testing (optional)
    # -----------------------------------------------------------------------
    if args.adversarial and trainer.trained_models:
        from src.models.adversarial_tester import AdversarialTester
        log.info("\n[+] Running adversarial robustness tests ...")
        best_model_name = max(trainer.cv_scores, key=trainer.cv_scores.get)
        best_model = trainer.trained_models[best_model_name]
        phish_mask = y_test == 1
        phish_emails = df_test["raw_email"].values[phish_mask].tolist()[:200]  # Cap at 200
        y_phish = y_test[phish_mask][:200]

        tester = AdversarialTester()
        adv_results = tester.run_full_test(best_model, pipeline, phish_emails, y_phish)
        log.info(
            f"\nAdversarial robustness: {adv_results['overall_robustness']:.3f} "
            f"({'PASSED' if adv_results['passed'] else 'FAILED'} — target >= 0.90)"
        )

    elapsed = time.time() - start_time
    log.info(f"\n{'=' * 70}")
    log.info(f"Training complete in {elapsed/60:.1f} minutes.")
    log.info(f"Models saved to '{args.save}'")
    log.info(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
