
"""
Step 5: Baseline Models
=======================
Trains simple baseline models to establish performance floors:
  - Mean baseline (predict global mean)
  - Median baseline (predict global median)
  - Linear Regression
  - Ridge Regression (alpha=1.0)
  - Lasso Regression (alpha=0.01)
  - Decision Tree (depth=15)

All models are trained on log1p(resolution_days) and evaluated on the
test set (2023-2024) after back-transformation to original scale.

Requires: Step 4 (data/processed/train_features.parquet, etc.)
Output: models/*.joblib, results/baseline_results.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from src.utils import load_config, set_seed, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, ensure_dirs
from src.evaluation import regression_metrics, create_comparison_table, compute_baseline_improvement
from src.models import save_model, load_feature_data

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor

config = load_config()
SEED = config["random_seed"]
set_seed(SEED)
ensure_dirs()

print("=" * 70)
print("  STEP 5: BASELINE MODELS")
print("=" * 70)

# -- Load features ---------------------------------------------------------
print("\nLoading feature-engineered data...")
train, val, test, feature_cols = load_feature_data()

X_train = train[feature_cols].fillna(0); y_train = train["resolution_days_log"]
X_val = val[feature_cols].fillna(0);     y_val = val["resolution_days_log"]
X_test = test[feature_cols].fillna(0);   y_test = test["resolution_days_log"]
print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# -- Train baselines -------------------------------------------------------
baselines = {
    "mean_baseline": DummyRegressor(strategy="mean"),
    "median_baseline": DummyRegressor(strategy="median"),
    "linear_regression": LinearRegression(),
    "ridge": Ridge(alpha=1.0, random_state=SEED),
    "lasso": Lasso(alpha=0.01, random_state=SEED, max_iter=5000),
    "decision_tree": DecisionTreeRegressor(max_depth=15, min_samples_leaf=50, random_state=SEED),
}

all_results = {}
print("\nTraining and evaluating on test set...")
for name, model in baselines.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    elapsed = time.time() - t0

    m = regression_metrics(y_test, y_pred, log_transformed=True)
    m["train_time_s"] = round(elapsed, 2)
    all_results[name] = m
    save_model(model, name)

    print(f"  {name:25s} | MAE={m['mae_days']:.3f}d | MedAE={m['median_ae_days']:.3f}d | "
          f"R2={m['r2_days']:.4f} | RMSE={m['rmse_days']:.2f}d | {elapsed:.1f}s")

# Add improvement percentages
mean_bl = all_results["mean_baseline"]
for name in all_results:
    all_results[name]["improvement_%"] = round(compute_baseline_improvement(all_results[name], mean_bl), 2)

# -- Save results ----------------------------------------------------------
table = create_comparison_table(all_results)
table.to_csv(RESULTS_DIR / "baseline_results.csv", index=False)
print(f"\nResults saved to results/baseline_results.csv")
print(f"\nBest baseline: {table.iloc[0]['model']} (MAE={table.iloc[0]['mae_days']:.3f}d)")
print("Step 5 complete.")
