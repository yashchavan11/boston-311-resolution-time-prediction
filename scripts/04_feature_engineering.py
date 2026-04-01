
"""
Step 4: Feature Engineering
===========================
Builds 99 features from the preprocessed data across 10 categories:
  1. Temporal (27): hour, day_of_week, year, cyclical encodings, weekend, holiday, season, time-of-day
  2. Categorical encoded (26): Target + frequency encoding for 13 columns
  3. Geographic (5): Lat/Lon, zipcode, distance from city center, has_coordinates
  4. Historical aggregates (15): Mean/median/std resolution by TYPE, REASON, etc.
  5. Workload proxies (2): Previous day/week request counts (lagged)
  6. District backlog (2): Open cases per neighborhood/department (1-day lag)
  7. Department velocity (6): Rolling 7/14/30-day avg resolution
  8. Interaction features (8): type x dept, velocity x backlog, etc.
  9. SLA baseline (6): P25/P75/IQR resolution time per TYPE and REASON
  10. Velocity deviation (2): Current vs historical speed

All features respect the creation-time boundary (no future information).

Requires: Step 2 (data/processed/train_clean.parquet, etc.)
Output: data/processed/train_features.parquet, val_features.parquet,
        test_features.parquet, feature_columns_v4.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json, gc, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from src.utils import load_config, set_seed, DATA_PROCESSED, ensure_dirs
from src.features import (
    run_feature_engineering,
    add_interaction_features,
    add_sla_baseline_features,
    add_velocity_deviation,
    finalize_feature_columns_v4,
)

config = load_config()
SEED = config["random_seed"]
set_seed(SEED)
ensure_dirs()

print("=" * 70)
print("  STEP 4: FEATURE ENGINEERING (99 FEATURES)")
print("=" * 70)

# -- Load preprocessed data ------------------------------------------------
print("\nLoading preprocessed data...")
train = pd.read_parquet(DATA_PROCESSED / "train_clean.parquet")
val = pd.read_parquet(DATA_PROCESSED / "val_clean.parquet")
test = pd.read_parquet(DATA_PROCESSED / "test_clean.parquet")
print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

# -- Core feature engineering (temporal, categorical, geographic,
#    historical, workload, backlog, velocity) ------------------------------
print("\n[4.1] Core feature engineering pipeline...")
train, val, test, artifacts = run_feature_engineering(train, val, test, config)

# -- Interaction features ---------------------------------------------------
print("\n[4.2] Adding interaction features...")
train, val, test = add_interaction_features(train, val, test)
print("  Added 8 interaction features")

# -- SLA baseline features --------------------------------------------------
print("\n[4.3] Adding SLA baseline features...")
train, val, test = add_sla_baseline_features(train, val, test)
print("  Added SLA baseline for TYPE and REASON: p25, p75, IQR")

# -- Velocity deviation -----------------------------------------------------
print("\n[4.4] Adding velocity deviation features...")
train, val, test = add_velocity_deviation(train, val, test)

# -- Finalize feature column list -----------------------------------------
print("\n[4.5] Finalizing feature columns...")
feature_cols = finalize_feature_columns_v4(train, val, test, artifacts["feature_columns"])
print(f"  Total features: {len(feature_cols)}")

# -- Save feature data ----------------------------------------------------
print("\n[4.6] Saving feature-engineered data...")
# Include targets + raw categoricals for downstream analysis (error analysis, ARIMA)
meta_cols = ["resolution_days", "resolution_days_log", "OPEN_DT", "CLOSED_DT",
             "TYPE", "REASON", "Department", "neighborhood", "Source"]
keep_cols = feature_cols + [c for c in meta_cols if c in train.columns]
keep_cols = list(dict.fromkeys(keep_cols)) 
train[keep_cols].to_parquet(DATA_PROCESSED / "train_features.parquet", index=False)
val[keep_cols].to_parquet(DATA_PROCESSED / "val_features.parquet", index=False)
test[keep_cols].to_parquet(DATA_PROCESSED / "test_features.parquet", index=False)

with open(DATA_PROCESSED / "feature_columns_v4.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

print(f"\n  Saved: data/processed/train_features.parquet")
print(f"  Saved: data/processed/val_features.parquet")
print(f"  Saved: data/processed/test_features.parquet")
print(f"  Saved: data/processed/feature_columns_v4.json ({len(feature_cols)} features)")
print("\nStep 4 complete.")
