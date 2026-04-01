
"""
Step 2: Data Preprocessing and Cleaning
========================================
Loads raw Boston 311 data, applies cleaning and filtering, computes the
target variable, and performs a temporal train/validation/test split.

Cleaning steps:
  1. Standardize column names (lowercase -> canonical)
  2. Parse date columns (OPEN_DT, CLOSED_DT, TARGET_DT)
  3. Compute target: resolution_days = (CLOSED_DT - OPEN_DT) / 24 hours
  4. Filter: closed cases only, remove negative times, 90-day cap,
     exclude administrative closures
  5. Add log-transformed target: log1p(resolution_days)
  6. Temporal split by year: Train 2015-2021, Val 2022, Test 2023-2024
  7. Fill missing values (categoricals -> "Unknown", lat/lon -> median)

Output:
  - data/processed/train_clean.parquet
  - data/processed/val_clean.parquet
  - data/processed/test_clean.parquet
  - results/dataset_summary.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json, gc
import numpy as np, pandas as pd
from src.utils import load_config, set_seed, DATA_PROCESSED, RESULTS_DIR, ensure_dirs
from src.data_loader import load_all_years
from src.preprocessing import (
    parse_dates, compute_target, filter_valid_cases,
    add_log_target, temporal_split, fill_missing_values,
)

config = load_config()
SEED = config["random_seed"]
set_seed(SEED)
ensure_dirs()

print("=" * 70)
print("  STEP 2: DATA PREPROCESSING AND CLEANING")
print("=" * 70)

# -- Load raw data ---------------------------------------------------------
print("\n[2.1] Loading raw data...")
df = load_all_years(list(range(2015, 2025)))

# -- Parse dates -----------------------------------------------------------
print("\n[2.2] Parsing dates and computing target...")
df = parse_dates(df)
df = compute_target(df)

# -- Filter valid cases ----------------------------------------------------
print("\n[2.3] Filtering valid cases...")
MAX_RES_DAYS = config["data"]["max_resolution_days"]
df_clean = filter_valid_cases(df, max_resolution_days=MAX_RES_DAYS, min_resolution_days=0)
df_clean = add_log_target(df_clean)

# -- Temporal split --------------------------------------------------------
print("\n[2.4] Temporal train/val/test split...")
train, val, test = temporal_split(
    df_clean,
    train_years=config["data"]["train_years"],
    val_years=config["data"]["val_years"],
    test_years=config["data"]["test_years"],
)

# -- Fill missing values --------------------------------------------------
print("\n[2.5] Filling missing values...")
train, fill_stats = fill_missing_values(train)          # compute stats from train
val, _ = fill_missing_values(val, fill_stats=fill_stats) # reuse train stats
test, _ = fill_missing_values(test, fill_stats=fill_stats)

# -- Save processed splits ------------------------------------------------
print("\n[2.6] Saving processed data...")
train.to_parquet(DATA_PROCESSED / "train_clean.parquet", index=False)
val.to_parquet(DATA_PROCESSED / "val_clean.parquet", index=False)
test.to_parquet(DATA_PROCESSED / "test_clean.parquet", index=False)

# -- Save dataset summary -------------------------------------------------
all_data = pd.concat([train, val, test])
summary = {
    "train_years": config["data"]["train_years"],
    "val_years": config["data"]["val_years"],
    "test_years": config["data"]["test_years"],
    "max_resolution_days": MAX_RES_DAYS,
    "train_count": int(len(train)),
    "val_count": int(len(val)),
    "test_count": int(len(test)),
    "total_count": int(len(all_data)),
    "mean_resolution_days": float(all_data["resolution_days"].mean()),
    "median_resolution_days": float(all_data["resolution_days"].median()),
    "std_resolution_days": float(all_data["resolution_days"].std()),
    "skewness": float(all_data["resolution_days"].skew()),
    "kurtosis": float(all_data["resolution_days"].kurtosis()),
}
for t in [1, 3, 7, 14, 30, 60, 90]:
    summary[f"pct_within_{t}d"] = float((all_data["resolution_days"] <= t).mean() * 100)

with open(RESULTS_DIR / "dataset_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Saved: data/processed/train_clean.parquet ({len(train):,} rows)")
print(f"  Saved: data/processed/val_clean.parquet ({len(val):,} rows)")
print(f"  Saved: data/processed/test_clean.parquet ({len(test):,} rows)")
print(f"  Saved: results/dataset_summary.json")
print("\nStep 2 complete.")
