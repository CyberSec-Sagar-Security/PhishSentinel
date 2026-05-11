"""
PhishLens Model Trainer.

Trains an ensemble of classifiers:
  - Logistic Regression (strong interpretable baseline)
  - Random Forest (captures non-linear feature interactions)
  - XGBoost (gradient boosting, best single model for tabular phishing features)
  - LightGBM (faster alternative to XGBoost for large corpora)
  - CatBoost (handles categorical-like encoded features well)

All models undergo Optuna hyperparameter optimisation (50 trials) with 5-fold
stratified cross-validation and are logged to MLflow.

Security rationale: An ensemble reduces the risk that a targeted adversarial
attack against one model architecture defeats the overall system. Each model
learns slightly different feature interactions, so an attacker must simultaneously
craft an email that evades all five — significantly harder in practice.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.utils.config import DEFAULT_CONFIG, PhishLensConfig
from src.utils.logger import get_logger

log = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)   # Suppress Optuna verbosity

# Models available for training
AVAILABLE_MODELS = ("lr", "rf", "xgboost", "lightgbm", "catboost")


class PhishLensTrainer:
    """Trains and optimises PhishLens classifiers.

    Args:
        config: PhishLensConfig instance with hyperparameter search spaces.
        model_names: Which models to train (default: all).
        tune: If True, run Optuna hyperparameter search (50 trials).
        n_folds: Number of CV folds (default: 5).
        use_smote: If True, oversample minority class with SMOTE.
    """

    def __init__(
        self,
        config: PhishLensConfig = DEFAULT_CONFIG,
        model_names: Tuple[str, ...] = AVAILABLE_MODELS,
        tune: bool = True,
        n_folds: int = 5,
        use_smote: bool = True,
    ) -> None:
        self.config = config
        self.model_names = model_names
        self.tune = tune
        self.n_folds = n_folds
        self.use_smote = use_smote
        self.trained_models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.cv_scores: Dict[str, float] = {}

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        experiment_name: str = "PhishLens",
        save_checkpoint_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train all configured classifiers.

        Args:
            X: Feature matrix shape [n_samples, n_features].
            y: Binary labels (0=legitimate, 1=phishing).
            feature_names: Feature names for MLflow logging.
            experiment_name: MLflow experiment name.

        Returns:
            Dict mapping model name to fitted classifier.
        """
        import torch as _torch
        if _torch.cuda.is_available():
            log.info(f"GPU: {_torch.cuda.get_device_name(0)} | VRAM: "
                     f"{_torch.cuda.get_device_properties(0).total_memory // 1024**2:,} MB")
            log.info("GPU models: XGBoost(cuda:0)  LightGBM(gpu)  CatBoost(GPU)")
        else:
            log.info("CUDA not available — all models will use CPU")
        log.info(f"Training PhishLens models: {self.model_names}")
        log.info(f"Dataset: {X.shape[0]:,} samples, {X.shape[1]:,} features, "
                 f"{y.sum():,} phishing, {(y==0).sum():,} legitimate")

        # Clean NaN/inf from feature matrix
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Balance classes with SMOTE before training
        if self.use_smote:
            log.info("Applying SMOTE oversampling ...")
            smote = SMOTE(random_state=self.config.random_state, k_neighbors=5)
            X, y = smote.fit_resample(X, y)
            log.info(f"After SMOTE: {X.shape[0]:,} samples")

        mlflow.set_experiment(experiment_name)

        from tqdm.auto import tqdm as _tqdm
        n_models = len(self.model_names)
        pbar = _tqdm(
            enumerate(self.model_names, 1), total=n_models,
            desc="  Training models", unit="model", ncols=100, colour="green",
        )
        for idx, model_name in pbar:
            pbar.set_description(f"  [{idx}/{n_models}] {model_name.upper()}")
            log.info(f"\n{'='*60}")
            log.info(f"Training: {model_name.upper()}")
            self._train_one(model_name, X, y, feature_names, experiment_name, save_checkpoint_dir)
            pbar.set_postfix(cv_f1=f"{self.cv_scores.get(model_name, 0):.4f}")

        return self.trained_models

    def _train_one(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]],
        experiment_name: str,
        save_checkpoint_dir: Optional[str] = None,
    ) -> None:
        """Train a single model with optional Optuna tuning."""
        start = time.time()

        with mlflow.start_run(run_name=f"PhishLens_{name}"):
            mlflow.log_param("model", name)
            mlflow.log_param("n_samples", X.shape[0])
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("tune", self.tune)

            if self.tune:
                log.info(f"Running Optuna search ({self.config.optuna_trials} trials) for {name} ...")
                best_params = self._optuna_tune(name, X, y)
                mlflow.log_params(best_params)
            else:
                best_params = {}

            model = self._build_model(name, best_params)

            # Scale features for Logistic Regression
            if name == "lr":
                scaler = StandardScaler()
                X_fit = scaler.fit_transform(X)
                self.scalers[name] = scaler
            else:
                X_fit = X

            # Stratified K-Fold cross-validation — manual loop so tqdm can show
            # per-fold progress. 3 folds when not tuning (faster; tune already
            # did 3-fold CV per Optuna trial). GPU folds run sequentially (no
            # parallel CUDA context conflicts).
            n_folds_cv = 3 if not self.tune else self.n_folds
            cv = StratifiedKFold(
                n_splits=n_folds_cv, shuffle=True,
                random_state=self.config.random_state,
            )
            log.info(f"Running {n_folds_cv}-fold CV for {name} ...")
            from tqdm.auto import tqdm as _tqdm
            _cv_scores: List[float] = []
            _fold_bar = _tqdm(
                enumerate(cv.split(X_fit, y), 1), total=n_folds_cv,
                desc=f"    {name} CV", unit="fold", ncols=100,
                leave=True, colour="yellow",
            )
            for _fold_num, (_tr, _val) in _fold_bar:
                _fm = clone(model)
                _fm.fit(X_fit[_tr], y[_tr])
                _fold_f1 = f1_score(y[_val], _fm.predict(X_fit[_val]))
                _cv_scores.append(_fold_f1)
                _fold_bar.set_postfix(
                    fold=f"{_fold_num}/{n_folds_cv}",
                    f1=f"{_fold_f1:.4f}",
                    mean=f"{np.mean(_cv_scores):.4f}",
                )
            cv_f1 = np.array(_cv_scores)
            self.cv_scores[name] = float(cv_f1.mean())
            log.info(f"{name} CV F1: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
            mlflow.log_metric("cv_f1_mean", cv_f1.mean())
            mlflow.log_metric("cv_f1_std", cv_f1.std())

            # Fit on full dataset — enable verbose progress for tree-based models
            log.info(f"Fitting {name} on full training set ({X_fit.shape[0]:,} samples) ...")
            if name == "lightgbm":
                import lightgbm as _lgb
                model.fit(X_fit, y, callbacks=[_lgb.log_evaluation(period=50)])
            elif name == "xgboost":
                model.set_params(verbosity=1)
                model.fit(X_fit, y)
                model.set_params(verbosity=0)
            elif name == "catboost":
                # Flush any CUDA memory held by the embedding model or previous
                # GPU models before CatBoost allocates its own GPU context.
                try:
                    import torch as _t
                    if _t.cuda.is_available():
                        _t.cuda.empty_cache()
                        _free = _t.cuda.get_device_properties(0).total_memory - _t.cuda.memory_reserved(0)
                        log.info(f"CUDA cache cleared. Free VRAM: {_free // 1024**2} MB")
                except Exception:
                    pass
                # Note: CatBoost forbids set_params() after fitting, so verbose
                # is set only before training via _build_model (verbose=0 default).
                model.set_params(verbose=100)
                model.fit(X_fit, y)
                # Do NOT call model.set_params(verbose=0) here — CatBoost raises
                # CatBoostError: You can't change params of fitted model.
            else:
                model.fit(X_fit, y)
            self.trained_models[name] = model

            # Checkpoint: save immediately so a crash later doesn't lose this model.
            if save_checkpoint_dir is not None:
                _ckpt = Path(save_checkpoint_dir) / f"{name}.pkl"
                Path(save_checkpoint_dir).mkdir(parents=True, exist_ok=True)
                joblib.dump(model, _ckpt)
                log.info(f"  Checkpoint saved → {_ckpt.name}")

            elapsed = time.time() - start
            log.info(f"{name} trained in {elapsed:.1f}s")
            mlflow.log_metric("training_time_s", elapsed)

    def _optuna_tune(self, name: str, X: np.ndarray, y: np.ndarray) -> Dict:
        """Run Optuna hyperparameter search for the specified model."""
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.random_state)
        # GPU models must run CV folds sequentially (n_jobs=1) — spawning multiple
        # loky worker processes each initialising a CUDA context on the same device
        # causes context conflicts and silent hangs. CPU models keep n_jobs=-1.
        _gpu_models = {"xgboost", "lightgbm", "catboost"}
        _tune_cv_jobs = 1 if name in _gpu_models else -1

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial, name)
            model = self._build_model(name, params)
            if name == "lr":
                scaler = StandardScaler()
                X_t = scaler.fit_transform(X)
            else:
                X_t = X
            scores = cross_val_score(model, X_t, y, cv=cv, scoring="f1", n_jobs=_tune_cv_jobs)
            return float(scores.mean())

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=self.config.optuna_trials,
            show_progress_bar=False,
        )
        log.info(f"Optuna best F1 for {name}: {study.best_value:.4f}")
        return study.best_params

    def _suggest_params(self, trial: optuna.Trial, name: str) -> Dict:
        """Define Optuna hyperparameter search space per model."""
        spaces = self.config.optuna_search_spaces
        if name == "lr":
            return {
                "C": trial.suggest_float("C", *spaces["lr"]["C"], log=True),
                "max_iter": trial.suggest_int("max_iter", *spaces["lr"]["max_iter"]),
            }
        elif name == "rf":
            return {
                "n_estimators": trial.suggest_int("n_estimators", *spaces["rf"]["n_estimators"]),
                "max_depth": trial.suggest_int("max_depth", *spaces["rf"]["max_depth"]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", *spaces["rf"]["min_samples_leaf"]),
            }
        elif name == "xgboost":
            sp = spaces["xgboost"]
            return {
                "n_estimators": trial.suggest_int("n_estimators", *sp["n_estimators"]),
                "max_depth": trial.suggest_int("max_depth", *sp["max_depth"]),
                "learning_rate": trial.suggest_float("learning_rate", *sp["learning_rate"], log=True),
                "subsample": trial.suggest_float("subsample", *sp["subsample"]),
                "colsample_bytree": trial.suggest_float("colsample_bytree", *sp["colsample_bytree"]),
            }
        elif name == "lightgbm":
            sp = spaces["lightgbm"]
            return {
                "n_estimators": trial.suggest_int("n_estimators", *sp["n_estimators"]),
                "max_depth": trial.suggest_int("max_depth", *sp["max_depth"]),
                "learning_rate": trial.suggest_float("learning_rate", *sp["learning_rate"], log=True),
                "num_leaves": trial.suggest_int("num_leaves", *sp["num_leaves"]),
            }
        elif name == "catboost":
            sp = spaces["catboost"]
            return {
                "iterations": trial.suggest_int("iterations", *sp["iterations"]),
                "depth": trial.suggest_int("depth", *sp["depth"]),
                "learning_rate": trial.suggest_float("learning_rate", *sp["learning_rate"], log=True),
            }
        return {}

    def _build_model(self, name: str, params: Dict) -> Any:
        """Instantiate a classifier with the given hyperparameters."""
        import torch
        _cuda = torch.cuda.is_available()
        rs = self.config.random_state
        if name == "lr":
            return LogisticRegression(
                C=params.get("C", 1.0),
                max_iter=params.get("max_iter", 1000),
                solver="lbfgs",
                class_weight="balanced",
                random_state=rs,
            )
        elif name == "rf":
            return RandomForestClassifier(
                n_estimators=params.get("n_estimators", 300),
                max_depth=params.get("max_depth", 20),
                min_samples_leaf=params.get("min_samples_leaf", 2),
                class_weight="balanced",
                random_state=rs,
                n_jobs=-1,
            )
        elif name == "xgboost":
            # tree_method="hist" + device="cuda:0": GPU-accelerated histogram algorithm.
            # n_jobs=1 when on GPU — XGBoost GPU handles all parallelism via CUDA;
            # setting n_jobs>1 would launch CPU threads competing with the GPU kernel.
            xgb_gpu = {"tree_method": "hist", "device": "cuda:0"} if _cuda else {}
            return XGBClassifier(
                n_estimators=params.get("n_estimators", 300),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.05),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                eval_metric="logloss",
                scale_pos_weight=1,
                random_state=rs,
                n_jobs=1 if _cuda else -1,
                **xgb_gpu,
            )
        elif name == "lightgbm":
            # device="gpu": uses OpenCL GPU acceleration.
            # n_jobs=1 when on GPU — same reason as XGBoost: GPU handles
            # parallelism internally; CPU thread pool would conflict with it.
            return LGBMClassifier(
                n_estimators=params.get("n_estimators", 300),
                max_depth=params.get("max_depth", -1),
                learning_rate=params.get("learning_rate", 0.05),
                num_leaves=params.get("num_leaves", 31),
                class_weight="balanced",
                random_state=rs,
                n_jobs=1 if _cuda else -1,
                verbose=-1,
                device="gpu" if _cuda else "cpu",
            )
        elif name == "catboost":
            import tempfile, os
            _cb_train_dir = os.path.join(tempfile.gettempdir(), "catboost_info")
            return CatBoostClassifier(
                iterations=params.get("iterations", 300),
                depth=params.get("depth", 6),
                learning_rate=params.get("learning_rate", 0.05),
                auto_class_weights="Balanced",
                random_seed=rs,
                verbose=0,
                task_type="GPU" if _cuda else "CPU",
                devices="0" if _cuda else None,
                train_dir=_cb_train_dir,
                gpu_ram_part=0.7 if _cuda else None,
            )
        raise ValueError(f"Unknown model name: {name}")

    def save_all(self, output_dir: str) -> None:
        """Save all trained models to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, model in self.trained_models.items():
            joblib.dump(model, out / f"{name}.pkl")
            log.info(f"Saved {name} to {out / f'{name}.pkl'}")
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, out / f"{name}_scaler.pkl")
        log.info(f"All models saved to '{output_dir}'")
