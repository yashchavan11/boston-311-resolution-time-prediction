
"""
Step 6: Intermediate Models (Gradient Boosting)
================================================
Trains gradient boosting models with default hyperparameters:
  - Random Forest (500 trees)
  - XGBoost (CUDA GPU)
  - CatBoost (GPU)

These models represent the "intermediate" step between simple baselines
and the final tuned/ensemble models.

Requires: Step 4 (features), Step 5 (baseline results for comparison)
Output: models/*.joblib, results/intermediate_results.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from src.utils import load_config, set_seed, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, ensure_dirs, get_device
from src.evaluation import regression_metrics, create_comparison_table, compute_baseline_improvement
from src.models import save_model, load_feature_data

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

config = load_config()
SEED = config["random_seed"]
set_seed(SEED)
ensure_dirs()
XGB_DEVICE = get_device("xgb")
CB_DEVICE = get_device("catboost")

print("=" * 70)
print("  STEP 6: INTERMEDIATE MODELS (GRADIENT BOOSTING)")
print("=" * 70)

# -- Load features ---------------------------------------------------------
print("\nLoading feature-engineered data...")
train, val, test, feature_cols = load_feature_data()

X_train = train[feature_cols].fillna(0); y_train = train["resolution_days_log"]
X_val = val[feature_cols].fillna(0);     y_val = val["resolution_days_log"]
X_test = test[feature_cols].fillna(0);   y_test = test["resolution_days_log"]
print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# -- Load baseline for comparison ------------------------------------------
import joblib
mean_bl_model = joblib.load(MODELS_DIR / "mean_baseline.joblib")
mean_bl = regression_metrics(y_test, mean_bl_model.predict(X_test), log_transformed=True)

# -- Device info -----------------------------------------------------------
print(f"  XGBoost device: {XGB_DEVICE}, CatBoost device: {CB_DEVICE}")

# -- Train models ----------------------------------------------------------
models = {
    "random_forest": RandomForestRegressor(
        n_estimators=500, max_depth=20, min_samples_leaf=10,
        n_jobs=-1, random_state=SEED),
    "xgboost": XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        tree_method="hist", device=XGB_DEVICE,
        random_state=SEED, verbosity=0),
    "catboost": CatBoostRegressor(
        iterations=500, depth=8, learning_rate=0.1,
        l2_leaf_reg=3.0, task_type=CB_DEVICE,
        random_seed=SEED, verbose=0),
}

all_results = {}
print("\nTraining and evaluating on test set...")
for name, model in models.items():
    t0 = time.time()
    if name == "xgboost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif name == "catboost":
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    elapsed = time.time() - t0

    m = regression_metrics(y_test, y_pred, log_transformed=True)
    m["train_time_s"] = round(elapsed, 2)
    m["improvement_%"] = round(compute_baseline_improvement(m, mean_bl), 2)
    all_results[name] = m
    save_model(model, name)

    print(f"  {name:25s} | MAE={m['mae_days']:.3f}d | MedAE={m['median_ae_days']:.3f}d | "
          f"R2={m['r2_days']:.4f} | Improv={m['improvement_%']:.1f}% | {elapsed:.1f}s")

# -- Save results ----------------------------------------------------------
table = create_comparison_table(all_results)
table.to_csv(RESULTS_DIR / "intermediate_results.csv", index=False)
print(f"\nResults saved to results/intermediate_results.csv")
print(f"\nBest intermediate: {table.iloc[0]['model']} (MAE={table.iloc[0]['mae_days']:.3f}d)")
print("Step 6 complete.")
