"""Model definitions, factory functions, saving/loading utilities."""

import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from src.utils import MODELS_DIR, DATA_PROCESSED


# ===========================================================================
# DATA LOADING UTILITIES
# ===========================================================================

def load_feature_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    """
    Load the saved feature-engineered parquet files and feature column list.

    Returns train, val, test DataFrames and the feature column list.
    Requires Step 4 (feature engineering) to have been run first.
    """
    train = pd.read_parquet(DATA_PROCESSED / "train_features.parquet")
    val = pd.read_parquet(DATA_PROCESSED / "val_features.parquet")
    test = pd.read_parquet(DATA_PROCESSED / "test_features.parquet")
    with open(DATA_PROCESSED / "feature_columns_v4.json") as f:
        feature_cols = json.load(f)
    return train, val, test, feature_cols


# ===========================================================================
# MODEL FACTORY FUNCTIONS
# ===========================================================================

def get_baseline_models(seed: int = 42) -> Dict[str, Any]:
    """
    Return a dictionary of baseline models.

    These are simple models used to establish performance floors.
    """
    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "median_baseline": DummyRegressor(strategy="median"),
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=seed),
        "lasso": Lasso(alpha=0.01, random_state=seed, max_iter=5000),
        "decision_tree": DecisionTreeRegressor(
            max_depth=15,
            min_samples_leaf=50,
            random_state=seed,
        ),
    }


def get_advanced_models(seed: int = 42) -> Dict[str, Any]:
    """
    Return a dictionary of advanced models with reasonable default hyperparameters.

    These will be further tuned with Optuna in Phase 7.
    """
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    return {
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=seed,
        ),
        "xgboost": XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=seed,
            verbosity=0,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=500,
            max_depth=-1,
            num_leaves=63,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        ),
        "catboost": CatBoostRegressor(
            iterations=500,
            depth=8,
            learning_rate=0.1,
            l2_leaf_reg=3.0,
            random_seed=seed,
            verbose=0,
        ),
    }


def get_classification_models(seed: int = 42) -> Dict[str, Any]:
    """Return classification models for SLA breach prediction."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        ),
        "random_forest_clf": RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "xgboost_clf": XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.1,
            scale_pos_weight=1,  # Will be set dynamically
            tree_method="hist",
            random_state=seed,
            verbosity=0,
            use_label_encoder=False,
            eval_metric="logloss",
        ),
        "lightgbm_clf": LGBMClassifier(
            n_estimators=500,
            num_leaves=63,
            learning_rate=0.1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        ),
        "catboost_clf": CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.1,
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=0,
        ),
    }



# ===========================================================================
# MODEL PERSISTENCE
# ===========================================================================

def save_model(model, name: str, models_dir: Optional[Path] = None):
    """Save a model to disk."""
    if models_dir is None:
        models_dir = MODELS_DIR
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    path = models_dir / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"  Model saved: {path}")
    return path


def load_model(name: str, models_dir: Optional[Path] = None):
    """Load a model from disk."""
    if models_dir is None:
        models_dir = MODELS_DIR
    path = Path(models_dir) / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def save_feature_columns(feature_cols: list, models_dir: Optional[Path] = None):
    """Save feature column list as JSON."""
    if models_dir is None:
        models_dir = MODELS_DIR
    path = Path(models_dir) / "feature_columns.json"
    with open(path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  Feature columns saved: {path}")


def load_feature_columns(models_dir: Optional[Path] = None) -> list:
    """Load feature column list from JSON."""
    if models_dir is None:
        models_dir = MODELS_DIR
    path = Path(models_dir) / "feature_columns.json"
    with open(path, "r") as f:
        return json.load(f)
