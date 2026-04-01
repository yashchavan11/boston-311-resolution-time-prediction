"""Feature engineering: temporal, categorical, geographic, historical, workload."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pandas.tseries.holiday import USFederalHolidayCalendar


# ===========================================================================
# A. TEMPORAL FEATURES
# ===========================================================================

def add_temporal_features(df: pd.DataFrame, date_col: str = "OPEN_DT") -> pd.DataFrame:
    """
    Extract temporal features from the submission timestamp.

    Features created:
    - hour, day_of_week, day_of_month, month, quarter, week_of_year, year
    - Cyclical encoding: sin/cos for hour, day_of_week, month
    - Binary: is_weekend, is_holiday, time_of_day (morning/afternoon/evening/night)
    """
    dt = df[date_col]

    # Raw temporal components
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek  # 0=Monday
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["year"] = dt.dt.year

    # Cyclical encoding (prevents discontinuity at boundaries)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Binary flags
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)   # Monday effect
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)   # Friday pre-weekend

    # Holiday detection
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=dt.min(), end=dt.max())
    df["is_holiday"] = dt.dt.normalize().isin(holidays).astype(int)

    # Season flags (Boston has strong seasonality)
    df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)
    df["is_spring"] = df["month"].isin([3, 4, 5]).astype(int)
    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
    df["is_fall"]   = df["month"].isin([9, 10, 11]).astype(int)

    # Time of day buckets
    df["is_morning"] = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(int)
    df["is_afternoon"] = ((df["hour"] >= 12) & (df["hour"] < 18)).astype(int)
    df["is_evening"] = ((df["hour"] >= 18) & (df["hour"] < 22)).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(int)
    df["is_business_hours"] = (
        (df["hour"] >= 8) & (df["hour"] < 17) & (df["is_weekend"] == 0)
    ).astype(int)

    # Days until end of week (proxy for workweek runway)
    df["days_to_weekend"] = (4 - df["day_of_week"]).clip(lower=0)

    return df


# ===========================================================================
# B. CATEGORICAL FEATURES (Target Encoding)
# ===========================================================================

class TargetEncoder:
    """
    Target encoder with smoothing to prevent overfitting.

    For each category value, the encoded value is a weighted average of:
    - The category's mean target value
    - The global mean target value

    The weight depends on the number of samples in the category.
    """

    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing
        self.encodings: Dict[str, Dict] = {}
        self.global_mean: float = 0.0

    def fit(self, df: pd.DataFrame, cols: List[str], target_col: str) -> "TargetEncoder":
        """Fit encoder on training data."""
        self.global_mean = df[target_col].mean()
        self.encodings = {}

        for col in cols:
            stats = df.groupby(col)[target_col].agg(["mean", "count"])
            # Smoothed encoding: blend category mean with global mean
            smoothing_factor = stats["count"] / (stats["count"] + self.smoothing)
            stats["encoded"] = smoothing_factor * stats["mean"] + (1 - smoothing_factor) * self.global_mean
            self.encodings[col] = stats["encoded"].to_dict()

        return self

    def transform(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Apply encoding to data."""
        for col in cols:
            if col in self.encodings:
                encoded_col = f"{col}_target_enc"
                df[encoded_col] = df[col].map(self.encodings[col]).fillna(self.global_mean)
        return df

    def fit_transform(self, df: pd.DataFrame, cols: List[str], target_col: str) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, cols, target_col)
        return self.transform(df, cols)


class FrequencyEncoder:
    """Encode categories by their frequency in the training data."""

    def __init__(self):
        self.encodings: Dict[str, Dict] = {}

    def fit(self, df: pd.DataFrame, cols: List[str]) -> "FrequencyEncoder":
        """Fit frequency encoder on training data."""
        self.encodings = {}
        for col in cols:
            freq = df[col].value_counts(normalize=True).to_dict()
            self.encodings[col] = freq
        return self

    def transform(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Apply frequency encoding."""
        for col in cols:
            if col in self.encodings:
                df[f"{col}_freq_enc"] = df[col].map(self.encodings[col]).fillna(0)
        return df


def add_categorical_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cat_cols: List[str],
    target_col: str = "resolution_days_log",
    smoothing: float = 10.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TargetEncoder, FrequencyEncoder]:
    """
    Apply target encoding and frequency encoding to categorical columns.

    Fit on train only to prevent leakage.
    """
    # Target encoding
    target_enc = TargetEncoder(smoothing=smoothing)
    train = target_enc.fit_transform(train, cat_cols, target_col)
    val = target_enc.transform(val, cat_cols)
    test = target_enc.transform(test, cat_cols)

    # Frequency encoding
    freq_enc = FrequencyEncoder()
    freq_enc.fit(train, cat_cols)
    train = freq_enc.transform(train, cat_cols)
    val = freq_enc.transform(val, cat_cols)
    test = freq_enc.transform(test, cat_cols)

    return train, val, test, target_enc, freq_enc


# ===========================================================================
# C. GEOGRAPHIC FEATURES
# ===========================================================================

def add_geographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived geographic features."""
    # Boston city center approximate coordinates
    BOSTON_CENTER_LAT = 42.3601
    BOSTON_CENTER_LON = -71.0589

    if "LATITUDE" in df.columns and "LONGITUDE" in df.columns:
        # Distance from city center (Euclidean approximation)
        df["dist_from_center"] = np.sqrt(
            (df["LATITUDE"] - BOSTON_CENTER_LAT) ** 2
            + (df["LONGITUDE"] - BOSTON_CENTER_LON) ** 2
        )

        # Flag for missing coordinates 
        df["has_coordinates"] = (
            (df["LATITUDE"] != 0) & (df["LONGITUDE"] != 0)
            & df["LATITUDE"].notna() & df["LONGITUDE"].notna()
        ).astype(int)

    return df


# ===========================================================================
# D. HISTORICAL / ROLLING FEATURES
# ===========================================================================


def add_rolling_features_simple(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "resolution_days",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Add historical mean features per category - simplified version.

    Computes global historical means from training data and applies to val/test.
    This is leakage-free and simpler than true rolling windows.
    """
    group_cols = ["TYPE", "REASON", "Department", "neighborhood", "Source"]

    for group_col in group_cols:
        if group_col not in train.columns:
            continue

        feat_name = f"hist_mean_{group_col}"
        feat_name_median = f"hist_median_{group_col}"
        feat_name_std = f"hist_std_{group_col}"

        # Compute stats from training data only
        group_stats = train.groupby(group_col)[target_col].agg(
            ["mean", "median", "std", "count"]
        )
        global_mean = train[target_col].mean()
        global_median = train[target_col].median()
        global_std = train[target_col].std()

        # Apply smoothed stats
        smoothing = 10
        smoothed_mean = (
            group_stats["count"] / (group_stats["count"] + smoothing) * group_stats["mean"]
            + smoothing / (group_stats["count"] + smoothing) * global_mean
        )
        stats_map_mean = smoothed_mean.to_dict()
        stats_map_median = group_stats["median"].to_dict()
        stats_map_std = group_stats["std"].to_dict()

        for df in [train, val, test]:
            df[feat_name] = df[group_col].map(stats_map_mean).fillna(global_mean)
            df[feat_name_median] = df[group_col].map(stats_map_median).fillna(global_median)
            df[feat_name_std] = df[group_col].map(stats_map_std).fillna(global_std)

    return train, val, test


# ===========================================================================
# D2. DISTRICT BACKLOG & DEPARTMENT VELOCITY FEATURES
# ===========================================================================

def add_backlog_features_combined(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    date_col: str = "OPEN_DT",
    close_col: str = "CLOSED_DT",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Leakage-free backlog features computed across all splits chronologically.

    Uses a 1-day lag on close dates: on day D, only closes from day D-1 or
    earlier are subtracted. This ensures no same-day or future close
    information is used, which is the information available at the time a
    new request arrives.
    """
    import datetime

    needs = [date_col, "neighborhood"]
    if not all(c in train.columns for c in needs):
        return train, val, test

    combined = pd.concat([
        train.assign(_split="train"),
        val.assign(_split="val"),
        test.assign(_split="test"),
    ], ignore_index=True)
    combined = combined.sort_values(date_col).copy()
    combined["_open_date"] = combined[date_col].dt.date

    if close_col in combined.columns:
        combined["_close_date"] = combined[close_col].dt.date
    else:
        combined["_close_date"] = pd.NaT

    all_dates = sorted(combined["_open_date"].unique())

    def _compute_backlog(group_col):
        opens = combined.groupby([group_col, "_open_date"]).size().rename("_daily_opens")
        closes = combined.dropna(subset=["_close_date"]).groupby(
            [group_col, "_close_date"]
        ).size().rename("_daily_closes")

        groups = combined[group_col].unique()
        records = []
        for grp in groups:
            open_s = opens.get(grp, pd.Series(dtype=int))
            close_s = closes.get(grp, pd.Series(dtype=int)) if len(closes) else pd.Series(dtype=int)
            running = 0
            prev_day_closes = 0
            prev_d = None
            for d in all_dates:
                running += open_s.get(d, 0)
                if prev_d is not None:
                    running -= prev_day_closes
                running = max(running, 0)
                records.append((grp, d, running))
                prev_day_closes = close_s.get(d, 0)
                prev_d = d
        return records

    nbhd_records = _compute_backlog("neighborhood")
    backlog_df = pd.DataFrame(nbhd_records, columns=["neighborhood", "_open_date", "open_cases_in_district"])
    combined = combined.merge(backlog_df, on=["neighborhood", "_open_date"], how="left")
    combined["open_cases_in_district"] = combined["open_cases_in_district"].fillna(0)

    if "Department" in combined.columns:
        dept_records = _compute_backlog("Department")
        dept_bl = pd.DataFrame(dept_records, columns=["Department", "_open_date", "open_cases_in_dept"])
        combined = combined.merge(dept_bl, on=["Department", "_open_date"], how="left")
        combined["open_cases_in_dept"] = combined["open_cases_in_dept"].fillna(0)

    combined.drop(columns=["_open_date", "_close_date"], errors="ignore", inplace=True)

    train_out = combined[combined["_split"] == "train"].drop(columns=["_split"])
    val_out = combined[combined["_split"] == "val"].drop(columns=["_split"])
    test_out = combined[combined["_split"] == "test"].drop(columns=["_split"])

    train_out.index = train.index
    val_out.index = val.index
    test_out.index = test.index

    return train_out, val_out, test_out


def add_department_velocity(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "resolution_days",
    date_col: str = "OPEN_DT",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Add department velocity features -- rolling average resolution time
    for each TYPE over recent windows (7, 14, 30 days).

    Computed from training data only (leakage-free for val/test).
    For val/test, uses the last available rolling value from train.
    """
    train = train.sort_values(date_col).copy()

    # For TYPE: compute per-day mean resolution, then rolling
    for group_col in ["TYPE", "Department"]:
        if group_col not in train.columns:
            continue
        daily_avg = train.groupby([group_col, train[date_col].dt.date])[target_col].mean()
        daily_avg = daily_avg.reset_index()
        daily_avg.columns = [group_col, "_date", "_daily_avg"]
        daily_avg = daily_avg.sort_values("_date")

        for window in [7, 14, 30]:
            feat = f"velocity_{group_col}_{window}d"
            # Rolling mean per group
            roll = (
                daily_avg.groupby(group_col)["_daily_avg"]
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            daily_avg[feat] = roll.values
            # Build lookup: last available value per group
            last_vals = daily_avg.groupby(group_col)[feat].last().to_dict()
            global_fallback = train[target_col].mean()

            # Map to train
            train["_date_key"] = train[date_col].dt.date
            tmp = daily_avg[[group_col, "_date", feat]].copy()
            tmp.columns = [group_col, "_date_key", feat]
            train = train.merge(tmp, on=[group_col, "_date_key"], how="left")
            train[feat] = train[feat].fillna(train[group_col].map(last_vals)).fillna(global_fallback)
            train.drop(columns=["_date_key"], inplace=True)

            # Map to val/test using last known values from train
            for split_df in [val, test]:
                split_df[feat] = split_df[group_col].map(last_vals).fillna(global_fallback)

    return train, val, test


# ===========================================================================
# E. WORKLOAD PROXY FEATURES
# ===========================================================================

def add_workload_features_combined(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    date_col: str = "OPEN_DT",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Leakage-free workload features using lagged volume counts.

    Instead of same-day/same-week counts (which require future knowledge
    within the day/week), this uses:
    - requests_prev_day: total requests on the PREVIOUS calendar day
    - requests_prev_week: total requests in the PREVIOUS ISO week

    Computed on the concatenated dataset so val/test get correct
    lagged values from training period.
    """
    combined = pd.concat([
        train.assign(_split="train"),
        val.assign(_split="val"),
        test.assign(_split="test"),
    ], ignore_index=True)

    combined["_date"] = combined[date_col].dt.date
    iso = combined[date_col].dt.isocalendar()
    combined["_yearweek"] = iso.year.astype(str) + "-" + iso.week.astype(str).str.zfill(2)

    daily_counts = combined.groupby("_date").size().rename("_daily_count")
    daily_df = daily_counts.reset_index()
    daily_df.columns = ["_date", "_daily_count"]
    daily_df = daily_df.sort_values("_date")
    daily_df["requests_prev_day"] = daily_df["_daily_count"].shift(1).fillna(0)
    prev_day_map = dict(zip(daily_df["_date"], daily_df["requests_prev_day"]))

    weekly_counts = combined.groupby("_yearweek").size().rename("_weekly_count")
    weekly_df = weekly_counts.reset_index()
    weekly_df.columns = ["_yearweek", "_weekly_count"]
    weekly_df = weekly_df.sort_values("_yearweek")
    weekly_df["requests_prev_week"] = weekly_df["_weekly_count"].shift(1).fillna(0)
    prev_week_map = dict(zip(weekly_df["_yearweek"], weekly_df["requests_prev_week"]))

    combined["requests_prev_day"] = combined["_date"].map(prev_day_map).fillna(0)
    combined["requests_prev_week"] = combined["_yearweek"].map(prev_week_map).fillna(0)

    combined.drop(columns=["_date", "_yearweek"], inplace=True)

    train_out = combined[combined["_split"] == "train"].drop(columns=["_split"])
    val_out = combined[combined["_split"] == "val"].drop(columns=["_split"])
    test_out = combined[combined["_split"] == "test"].drop(columns=["_split"])

    train_out.index = train.index
    val_out.index = val.index
    test_out.index = test.index

    return train_out, val_out, test_out


# ===========================================================================
# F. FEATURE SELECTION
# ===========================================================================

def compute_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    n_top: int = 30,
) -> pd.DataFrame:
    """Compute mutual information between features and target."""
    from sklearn.feature_selection import mutual_info_regression

    mi = mutual_info_regression(X.fillna(0), y, random_state=42, n_neighbors=5)
    mi_df = pd.DataFrame({"feature": X.columns, "mi_score": mi})
    mi_df = mi_df.sort_values("mi_score", ascending=False).reset_index(drop=True)

    print(f"\nTop {n_top} features by Mutual Information:")
    print(mi_df.head(n_top).to_string(index=False))

    return mi_df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get the list of feature columns to use for modeling.

    Excludes:
    - Target columns (resolution_days, resolution_hours, resolution_days_log)
    - ID columns (CASE_ENQUIRY_ID)
    - Date columns (OPEN_DT, CLOSED_DT, TARGET_DT)
    - Raw categorical columns (already encoded)
    - Metadata columns (source_year, CASE_STATUS, etc.)
    """
    exclude_patterns = [
        "CASE_ENQUIRY_ID", "OPEN_DT", "CLOSED_DT", "TARGET_DT",
        "resolution_days", "resolution_hours", "resolution_days_log",
        "CASE_STATUS", "CLOSURE_REASON", "CASE_TITLE", "OnTime_Status",
        "source_year", "Location", "Geocoded_Location",
        "LOCATION_STREET_NAME", "Property_ID",
        "SubmittedPhoto", "ClosedPhoto", "sla_breach",
    ]

    feature_cols = []
    for col in df.columns:
        if col in exclude_patterns:
            continue
        # Skip raw categorical columns (the encoded versions are used instead)
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            continue
        feature_cols.append(col)

    return feature_cols


# ===========================================================================
# MASTER FEATURE ENGINEERING PIPELINE
# ===========================================================================

def run_feature_engineering(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Run the full feature engineering pipeline.

    Returns
    -------
    train, val, test : pd.DataFrames with engineered features
    artifacts : dict with fitted encoders and feature lists
    """
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    cat_cols = config["features"]["categorical_cols"]
    # Filter to cols that exist in data
    cat_cols = [c for c in cat_cols if c in train.columns]
    target_col_log = config["data"]["target_col_log"]
    smoothing = config["features"]["target_encoding_smoothing"]

    # A. Temporal features
    print("\n[1/7] Temporal features...")
    for df in [train, val, test]:
        add_temporal_features(df)
    print(f"  Added temporal features. Columns now: {len(train.columns)}")

    # B. Categorical encoding
    print("\n[2/7] Categorical encoding...")
    train, val, test, target_enc, freq_enc = add_categorical_features(
        train, val, test, cat_cols, target_col_log, smoothing
    )
    print(f"  Encoded {len(cat_cols)} categorical columns. Columns now: {len(train.columns)}")

    # C. Geographic features
    print("\n[3/7] Geographic features...")
    for df in [train, val, test]:
        add_geographic_features(df)
    print(f"  Added geographic features. Columns now: {len(train.columns)}")

    # D. Historical features
    print("\n[4/7] Historical / rolling features...")
    train, val, test = add_rolling_features_simple(train, val, test)
    print(f"  Added historical features. Columns now: {len(train.columns)}")

    # E. Workload features (leakage-free: uses lagged counts)
    print("\n[5/7] Workload proxy features (lagged, leakage-free)...")
    train, val, test = add_workload_features_combined(train, val, test)
    print(f"  Added workload features. Columns now: {len(train.columns)}")

    # F. District backlog features (leakage-free: 1-day lag on closes)
    print("\n[6/7] District backlog features (leakage-free, combined)...")
    train, val, test = add_backlog_features_combined(train, val, test)
    print(f"  Added backlog features. Columns now: {len(train.columns)}")

    # G. Department velocity features
    print("\n[7/7] Department velocity (rolling avg resolution time)...")
    train, val, test = add_department_velocity(train, val, test)
    print(f"  Added velocity features. Columns now: {len(train.columns)}")

    # Get feature columns
    feature_cols = get_feature_columns(train)
    print(f"\n  Final feature count: {len(feature_cols)}")

    artifacts = {
        "target_encoder": target_enc,
        "frequency_encoder": freq_enc,
        "feature_columns": feature_cols,
        "categorical_columns": cat_cols,
    }

    return train, val, test, artifacts


# ===========================================================================
# EXTENDED FEATURES (INTERACTION, SLA, VELOCITY DEVIATION)
# ===========================================================================

def add_interaction_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add interaction features (8 features)."""
    for split_df in [train, val, test]:
        if "TYPE_target_enc" in split_df.columns and "Department_target_enc" in split_df.columns:
            split_df["type_dept_interaction"] = split_df["TYPE_target_enc"] * split_df["Department_target_enc"]
        if "TYPE_target_enc" in split_df.columns and "neighborhood_target_enc" in split_df.columns:
            split_df["type_nbhd_interaction"] = split_df["TYPE_target_enc"] * split_df["neighborhood_target_enc"]
        if "hist_mean_TYPE" in split_df.columns:
            split_df["hist_mean_x_weekend"] = split_df["hist_mean_TYPE"] * split_df.get("is_weekend", 0)
            split_df["log_hist_mean_TYPE"] = np.log1p(split_df["hist_mean_TYPE"].clip(lower=0))
        if "hist_mean_REASON" in split_df.columns:
            split_df["log_hist_mean_REASON"] = np.log1p(split_df["hist_mean_REASON"].clip(lower=0))
        if "velocity_TYPE_7d" in split_df.columns and "open_cases_in_district" in split_df.columns:
            split_df["velocity_x_backlog"] = split_df["velocity_TYPE_7d"] * np.log1p(split_df["open_cases_in_district"])
        if "velocity_TYPE_30d" in split_df.columns and "is_monday" in split_df.columns:
            split_df["velocity_x_monday"] = split_df["velocity_TYPE_30d"] * split_df["is_monday"]
        if "open_cases_in_district" in split_df.columns and "open_cases_in_dept" in split_df.columns:
            split_df["backlog_ratio"] = split_df["open_cases_in_district"] / (split_df["open_cases_in_dept"] + 1)
    return train, val, test


def add_sla_baseline_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "resolution_days",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add SLA baseline features (p25, p75, IQR per TYPE and REASON)."""
    for group_col in ["TYPE", "REASON"]:
        if group_col not in train.columns:
            continue
        sla_stats = train.groupby(group_col)[target_col].agg(["mean", "median", "std", "count"])
        sla_stats["p25"] = train.groupby(group_col)[target_col].quantile(0.25)
        sla_stats["p75"] = train.groupby(group_col)[target_col].quantile(0.75)
        sla_stats["iqr"] = sla_stats["p75"] - sla_stats["p25"]
        p25_map = sla_stats["p25"].to_dict()
        p75_map = sla_stats["p75"].to_dict()
        iqr_map = sla_stats["iqr"].to_dict()
        global_p25 = train[target_col].quantile(0.25)
        global_p75 = train[target_col].quantile(0.75)
        global_iqr = global_p75 - global_p25
        for split_df in [train, val, test]:
            split_df[f"sla_p25_{group_col}"] = split_df[group_col].map(p25_map).fillna(global_p25)
            split_df[f"sla_p75_{group_col}"] = split_df[group_col].map(p75_map).fillna(global_p75)
            split_df[f"sla_iqr_{group_col}"] = split_df[group_col].map(iqr_map).fillna(global_iqr)
    return train, val, test


def add_velocity_deviation(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add velocity deviation features (current vs historical speed)."""
    for split_df in [train, val, test]:
        if "velocity_TYPE_7d" in split_df.columns and "hist_mean_TYPE" in split_df.columns:
            split_df["velocity_deviation"] = split_df["velocity_TYPE_7d"] - split_df["hist_mean_TYPE"]
            split_df["velocity_ratio"] = split_df["velocity_TYPE_7d"] / (split_df["hist_mean_TYPE"] + 0.01)
    return train, val, test


def finalize_feature_columns_v4(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    base_feature_cols: List[str],
) -> List[str]:
    """Finalize the feature column list including interaction/SLA/velocity features."""
    feature_cols = list(base_feature_cols)
    extra_feats = [
        "type_dept_interaction", "type_nbhd_interaction", "hist_mean_x_weekend",
        "log_hist_mean_TYPE", "log_hist_mean_REASON",
        "velocity_x_backlog", "velocity_x_monday", "backlog_ratio",
        "sla_p25_TYPE", "sla_p75_TYPE", "sla_iqr_TYPE",
        "sla_p25_REASON", "sla_p75_REASON", "sla_iqr_REASON",
        "velocity_deviation", "velocity_ratio",
    ]
    for e in extra_feats:
        if e in train.columns and e not in feature_cols:
            feature_cols.append(e)
    # Add any remaining numeric features
    exclude = {
        "CASE_ENQUIRY_ID", "OPEN_DT", "CLOSED_DT", "TARGET_DT",
        "resolution_days", "resolution_hours", "resolution_days_log",
        "CASE_STATUS", "CLOSURE_REASON", "CASE_TITLE", "OnTime_Status",
        "source_year", "Location", "Geocoded_Location",
        "LOCATION_STREET_NAME", "Property_ID", "SubmittedPhoto", "ClosedPhoto", "sla_breach",
    }
    for c in train.columns:
        if c not in feature_cols and c in val.columns and c in test.columns:
            if not (train[c].dtype == "object" or pd.api.types.is_string_dtype(train[c])) and c not in exclude:
                feature_cols.append(c)
    feature_cols = list(dict.fromkeys(feature_cols))
    feature_cols = [c for c in feature_cols if c in train.columns and c in val.columns and c in test.columns]
    return feature_cols


def build_monotonic_constraints(feature_cols: List[str]) -> List[int]:
    """Build monotonic constraint vector for gradient boosting models."""
    mono = [0] * len(feature_cols)
    for i, f in enumerate(feature_cols):
        if "open_cases" in f or "backlog" in f:
            mono[i] = 1
        if f.startswith("hist_mean_") or f.startswith("hist_median_"):
            mono[i] = 1
        if f.startswith("velocity_") and "monday" not in f and "deviation" not in f and "ratio" not in f:
            mono[i] = 1
        if f.startswith("sla_p25_") or f.startswith("sla_p75_"):
            mono[i] = 1
    return mono
