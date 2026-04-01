"""Data cleaning, filtering, target computation, and temporal splitting."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional


# -- Column name mapping ---------------------------------------------------
# The data.boston.gov CSVs use lowercase column names.
# Standardized to canonical names for consistency across the project.
COLUMN_MAP = {
    "case_enquiry_id": "CASE_ENQUIRY_ID",
    "open_dt": "OPEN_DT",
    "sla_target_dt": "TARGET_DT",
    "closed_dt": "CLOSED_DT",
    "on_time": "OnTime_Status",
    "case_status": "CASE_STATUS",
    "closure_reason": "CLOSURE_REASON",
    "case_title": "CASE_TITLE",
    "subject": "SUBJECT",
    "reason": "REASON",
    "type": "TYPE",
    "queue": "QUEUE",
    "department": "Department",
    "submitted_photo": "SubmittedPhoto",
    "closed_photo": "ClosedPhoto",
    "location": "Location",
    "fire_district": "fire_district",
    "pwd_district": "pwd_district",
    "city_council_district": "city_council_district",
    "police_district": "police_district",
    "neighborhood": "neighborhood",
    "neighborhood_services_district": "neighborhood_services_district",
    "ward": "ward",
    "precinct": "precinct",
    "location_street_name": "LOCATION_STREET_NAME",
    "location_zipcode": "LOCATION_ZIPCODE",
    "latitude": "LATITUDE",
    "longitude": "LONGITUDE",
    "geom_4326": "Geocoded_Location",
    "source": "Source",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns from lowercase data.boston.gov format to standard names."""
    rename_map = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse all date columns to datetime."""
    # Standardize column names 
    df = standardize_columns(df)

    date_cols = ["OPEN_DT", "TARGET_DT", "CLOSED_DT"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed", dayfirst=False, errors="coerce")
    return df


def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """Compute resolution time in hours and days."""
    df["resolution_hours"] = (df["CLOSED_DT"] - df["OPEN_DT"]).dt.total_seconds() / 3600
    df["resolution_days"] = df["resolution_hours"] / 24
    return df


def filter_valid_cases(
    df: pd.DataFrame,
    max_resolution_days: float = 365,
    min_resolution_days: float = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Filter dataset to cases valid for modeling.

    Filtering steps:
    1. Keep only Closed cases
    2. Remove rows missing OPEN_DT or CLOSED_DT
    3. Remove negative resolution times
    4. Cap resolution time at max_resolution_days
    5. Remove administratively closed cases
    """
    log = {}
    log["initial"] = len(df)

    # Step 1: Keep only Closed cases
    if "CASE_STATUS" in df.columns:
        df = df[df["CASE_STATUS"] == "Closed"].copy()
    log["after_closed_only"] = len(df)

    # Step 2: Remove rows with missing dates
    df = df.dropna(subset=["OPEN_DT", "CLOSED_DT"])
    log["after_drop_missing_dates"] = len(df)

    # Step 3: Remove negative resolution times
    df = df[df["resolution_days"] >= min_resolution_days]
    log["after_remove_negative"] = len(df)

    # Step 4: Cap at max_resolution_days
    df = df[df["resolution_days"] <= max_resolution_days]
    log["after_cap_max"] = len(df)

    # Step 5: Remove administrative closures
    if "CLOSURE_REASON" in df.columns:
        admin_mask = df["CLOSURE_REASON"].str.contains(
            "Administratively Closed|Case Noted", case=False, na=False
        )
        df = df[~admin_mask]
    log["after_remove_admin"] = len(df)

    if verbose:
        print("=" * 60)
        print("DATA FILTERING REPORT")
        print("=" * 60)
        prev = None
        for step, count in log.items():
            dropped = f" (dropped {prev - count:,})" if prev is not None else ""
            print(f"  {step:35s}: {count:>10,}{dropped}")
            prev = count
        total_dropped = log["initial"] - log["after_remove_admin"]
        pct = total_dropped / log["initial"] * 100
        print(f"\n  Total removed: {total_dropped:,} ({pct:.1f}%)")
        print(f"  Final dataset: {log['after_remove_admin']:,} rows")
        print("=" * 60)

    return df


def add_log_target(df: pd.DataFrame, col: str = "resolution_days") -> pd.DataFrame:
    """Add log-transformed target variable."""
    df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
    return df


def temporal_split(
    df: pd.DataFrame,
    train_years: List[int],
    val_years: List[int],
    test_years: List[int],
    date_col: str = "OPEN_DT",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by year for temporal train/validation/test sets."""
    df["_split_year"] = df[date_col].dt.year

    train = df[df["_split_year"].isin(train_years)].copy()
    val = df[df["_split_year"].isin(val_years)].copy()
    test = df[df["_split_year"].isin(test_years)].copy()

    # Clean up temp column
    for split_df in [train, val, test]:
        split_df.drop(columns=["_split_year"], inplace=True, errors="ignore")
    df.drop(columns=["_split_year"], inplace=True, errors="ignore")

    if verbose:
        total = len(train) + len(val) + len(test)
        print("\nTEMPORAL SPLIT:")
        print(f"  Train ({train_years}): {len(train):>10,} rows ({len(train)/total*100:.1f}%)")
        print(f"  Val   ({val_years}):          {len(val):>10,} rows ({len(val)/total*100:.1f}%)")
        print(f"  Test  ({test_years}):      {len(test):>10,} rows ({len(test)/total*100:.1f}%)")
        print(f"  Total:                      {total:>10,}")

        for name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
            stats = split_df["resolution_days"].describe()
            print(f"\n  {name} target stats: mean={stats['mean']:.2f}, "
                  f"median={stats['50%']:.2f}, std={stats['std']:.2f}")

    return train, val, test


def fill_missing_values(
    df: pd.DataFrame,
    fill_stats: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Handle missing values with appropriate strategies.

    Parameters
    ----------
    df : pd.DataFrame
        Data to fill.
    fill_stats : dict, optional
        Pre-computed fill statistics (from training set). If None, statistics
        are computed from ``df`` itself (appropriate only for the training split).

    Returns
    -------
    df : pd.DataFrame
        DataFrame with missing values filled.
    fill_stats : dict
        The statistics used for filling (pass to val/test to prevent leakage).
    """
    if fill_stats is None:
        fill_stats = {}
        if "LATITUDE" in df.columns:
            fill_stats["median_lat"] = float(df["LATITUDE"].median())
        if "LONGITUDE" in df.columns:
            fill_stats["median_lon"] = float(df["LONGITUDE"].median())

    # Categorical columns: fill with 'Unknown'
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna("Unknown")

    # Latitude/Longitude: fill with median (from train)
    if "LATITUDE" in df.columns and df["LATITUDE"].isna().any():
        df["LATITUDE"] = df["LATITUDE"].fillna(fill_stats.get("median_lat", 0))

    if "LONGITUDE" in df.columns and df["LONGITUDE"].isna().any():
        df["LONGITUDE"] = df["LONGITUDE"].fillna(fill_stats.get("median_lon", 0))

    # LOCATION_ZIPCODE: fill with 0
    if "LOCATION_ZIPCODE" in df.columns and df["LOCATION_ZIPCODE"].isna().any():
        df["LOCATION_ZIPCODE"] = df["LOCATION_ZIPCODE"].fillna(0)

    return df, fill_stats


def run_full_preprocessing(
    df: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full preprocessing pipeline."""
    print("Step 1/5: Parsing dates...")
    df = parse_dates(df)

    print("Step 2/5: Computing target variable...")
    df = compute_target(df)

    print("Step 3/5: Filtering valid cases...")
    df = filter_valid_cases(
        df,
        max_resolution_days=config["data"]["max_resolution_days"],
        min_resolution_days=config["data"]["min_resolution_days"],
    )

    print("Step 4/5: Adding log target...")
    df = add_log_target(df)

    print("Step 5/5: Temporal split...")
    train, val, test = temporal_split(
        df,
        train_years=config["data"]["train_years"],
        val_years=config["data"]["val_years"],
        test_years=config["data"]["test_years"],
    )

    # Fill missing values (train stats used for val/test to prevent leakage)
    train, fill_stats = fill_missing_values(train)
    val, _ = fill_missing_values(val, fill_stats=fill_stats)
    test, _ = fill_missing_values(test, fill_stats=fill_stats)

    return train, val, test
