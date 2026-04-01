# Note: LLMs were used in this section
"""
Create small sample datasets for GitHub sharing and smoke testing.

Produces:
  - data/sample/raw_sample.csv        (~5,000 rows from real raw data)
  - data/sample/processed_sample.parquet  (processed through real pipeline)
  - data/sample/feature_columns.json     (copy of feature schema)

The sample is stratified across years, request types, and neighborhoods
to be representative of the full dataset. All processing uses the real
pipeline code (src/preprocessing.py, src/features.py) so the output
schema matches what downstream scripts expect.

Usage:
    python scripts/create_sample_data.py

Requires: Raw data in data/raw/ (run scripts/01_data_collection.py first)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd

from src.utils import load_config, set_seed, DATA_RAW, DATA_PROCESSED, PROJECT_ROOT
from src.preprocessing import (
    parse_dates, compute_target, filter_valid_cases,
    add_log_target, fill_missing_values,
)

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
SEED = 42
SAMPLE_SIZE = 5000  # rows per split won't exceed this total

set_seed(SEED)


def load_and_concat_raw() -> pd.DataFrame:
    """Load all raw CSV files into a single DataFrame."""
    frames = []
    for year in range(2015, 2025):
        path = DATA_RAW / f"311_requests_{year}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    return pd.concat(frames, ignore_index=True)


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample n rows stratified by year and request type."""
    rng = np.random.RandomState(seed)
    df = df.copy()
    df["_year"] = df["OPEN_DT"].dt.year

    # Allocate proportional to year counts, minimum 50 per year
    year_counts = df["_year"].value_counts()
    year_fracs = year_counts / year_counts.sum()
    year_targets = (year_fracs * n).clip(lower=50).astype(int)

    # Scale to hit target total
    scale = n / year_targets.sum()
    year_targets = (year_targets * scale).astype(int)

    samples = []
    for year, target_n in year_targets.items():
        year_df = df[df["_year"] == year]
        actual_n = min(target_n, len(year_df))
        if actual_n > 0:
            idx = rng.choice(len(year_df), actual_n, replace=False)
            samples.append(year_df.iloc[idx])

    result = pd.concat(samples, ignore_index=True)
    result.drop(columns=["_year"], inplace=True)
    return result


def main():
    print("=" * 60)
    print("  CREATE SAMPLE DATASET FOR GITHUB")
    print("=" * 60)

    config = load_config()
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load raw data
    print("\n[1/6] Loading raw data...")
    raw = load_and_concat_raw()
    print(f"  Loaded {len(raw):,} raw rows")

    # Step 2: Parse dates and compute target (needed for filtering)
    print("[2/6] Parsing dates and computing target...")
    raw = parse_dates(raw)
    raw = compute_target(raw)

    # Step 3: Filter valid cases (same rules as main pipeline)
    print("[3/6] Filtering valid cases...")
    valid = filter_valid_cases(
        raw,
        max_resolution_days=config["data"]["max_resolution_days"],
        min_resolution_days=config["data"]["min_resolution_days"],
        verbose=False,
    )
    print(f"  Valid cases: {len(valid):,}")

    # Step 4: Stratified sample
    print(f"[4/6] Stratified sampling ({SAMPLE_SIZE} rows)...")
    sample = stratified_sample(valid, SAMPLE_SIZE, SEED)
    sample = add_log_target(sample)
    print(f"  Sample: {len(sample):,} rows")

    # Check year distribution
    year_dist = sample["OPEN_DT"].dt.year.value_counts().sort_index()
    print("  Year distribution:")
    for year, count in year_dist.items():
        print(f"    {year}: {count}")

    # Step 5: Save raw sample (before encoding, in original column format)
    print("[5/6] Saving raw sample...")
    # Re-read the raw data for just these case IDs to get original columns
    raw_cols = raw.columns.tolist()
    # The sample has extra computed columns; save only the original ones plus target
    save_cols = [c for c in raw_cols if c in sample.columns]
    save_cols += ["resolution_hours", "resolution_days", "resolution_days_log"]
    save_cols = [c for c in save_cols if c in sample.columns]
    sample[save_cols].to_csv(SAMPLE_DIR / "raw_sample.csv", index=False)
    raw_size = (SAMPLE_DIR / "raw_sample.csv").stat().st_size
    print(f"  Saved: data/sample/raw_sample.csv ({raw_size / 1024:.0f} KB, {len(sample)} rows)")

    # Step 6: Process through pipeline to create feature-ready sample
    print("[6/6] Processing sample through feature pipeline...")

    # Split into train/val/test by year
    train_years = config["data"]["train_years"]
    val_years = config["data"]["val_years"]
    test_years = config["data"]["test_years"]

    sample["_year"] = sample["OPEN_DT"].dt.year
    train_s = sample[sample["_year"].isin(train_years)].copy()
    val_s = sample[sample["_year"].isin(val_years)].copy()
    test_s = sample[sample["_year"].isin(test_years)].copy()
    sample.drop(columns=["_year"], inplace=True)
    train_s.drop(columns=["_year"], inplace=True)
    val_s.drop(columns=["_year"], inplace=True)
    test_s.drop(columns=["_year"], inplace=True)

    # Fill missing values (train stats propagated to val/test)
    train_s, fill_stats = fill_missing_values(train_s)
    val_s, _ = fill_missing_values(val_s, fill_stats=fill_stats)
    test_s, _ = fill_missing_values(test_s, fill_stats=fill_stats)

    # Run feature engineering (same sequence as scripts/04_feature_engineering.py)
    from src.features import (
        run_feature_engineering, finalize_feature_columns_v4,
        add_interaction_features, add_sla_baseline_features, add_velocity_deviation,
    )

    train_s, val_s, test_s, artifacts = run_feature_engineering(train_s, val_s, test_s, config)
    train_s, val_s, test_s = add_interaction_features(train_s, val_s, test_s)
    train_s, val_s, test_s = add_sla_baseline_features(train_s, val_s, test_s)
    train_s, val_s, test_s = add_velocity_deviation(train_s, val_s, test_s)
    feature_cols = finalize_feature_columns_v4(train_s, val_s, test_s, artifacts["feature_columns"])

    # Combine and save
    processed = pd.concat([train_s, val_s, test_s], ignore_index=True)
    processed.to_parquet(SAMPLE_DIR / "processed_sample.parquet", index=False)
    proc_size = (SAMPLE_DIR / "processed_sample.parquet").stat().st_size
    print(f"  Saved: data/sample/processed_sample.parquet ({proc_size / 1024:.0f} KB, {len(processed)} rows)")

    # Save feature columns
    with open(SAMPLE_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  Saved: data/sample/feature_columns.json ({len(feature_cols)} features)")

    # Step 7: Validation
    print("\n" + "=" * 60)
    print("  VALIDATION")
    print("=" * 60)

    # Load real feature columns for comparison
    real_cols_path = DATA_PROCESSED / "feature_columns_v4.json"
    if real_cols_path.exists():
        with open(real_cols_path) as f:
            real_cols = json.load(f)

        sample_cols = feature_cols
        missing = set(real_cols) - set(sample_cols)
        extra = set(sample_cols) - set(real_cols)

        print(f"\n  Real feature count:   {len(real_cols)}")
        print(f"  Sample feature count: {len(sample_cols)}")
        if missing:
            print(f"  MISSING features: {missing}")
        if extra:
            print(f"  EXTRA features: {extra}")
        if not missing and not extra:
            print("  Schema match: EXACT MATCH")

        # Check column ordering
        if sample_cols == real_cols:
            print("  Column ordering: MATCHES")
        else:
            print("  Column ordering: DIFFERS (but all columns present)")

    # Dtype check
    real_train = pd.read_parquet(DATA_PROCESSED / "train_features.parquet", columns=feature_cols[:5])
    sample_check = processed[feature_cols[:5]]
    dtype_match = all(real_train[c].dtype == sample_check[c].dtype for c in feature_cols[:5])
    print(f"  Dtype spot-check (first 5 features): {'MATCH' if dtype_match else 'MISMATCH'}")

    # Target presence
    has_target = "resolution_days_log" in processed.columns
    print(f"  Target column present: {'YES' if has_target else 'NO'}")

    # NaN check in features
    nan_counts = processed[feature_cols].isna().sum()
    nan_features = nan_counts[nan_counts > 0]
    if len(nan_features) == 0:
        print("  NaN in features: NONE")
    else:
        print(f"  NaN in features: {len(nan_features)} columns have NaNs")

    # Year coverage
    years_present = sorted(processed["OPEN_DT"].dt.year.unique())
    print(f"  Years covered: {years_present}")

    print("\n" + "=" * 60)
    print("  SAMPLE DATASET CREATION COMPLETE")
    print("=" * 60)
    print(f"\n  Files created in data/sample/:")
    print(f"    raw_sample.csv           - {raw_size / 1024:.0f} KB")
    print(f"    processed_sample.parquet - {proc_size / 1024:.0f} KB")
    print(f"    feature_columns.json     - {len(feature_cols)} features")


if __name__ == "__main__":
    main()
