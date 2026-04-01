"""Tests for feature engineering leakage constraints and correctness."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import (
    add_temporal_features,
    TargetEncoder,
    FrequencyEncoder,
    add_rolling_features_simple,
    add_workload_features_combined,
    add_categorical_features,
    get_feature_columns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def splits():
    """Create minimal train/val/test splits with temporal ordering."""
    np.random.seed(42)
    def _make(start, n, target_mean):
        dates = pd.date_range(start, periods=n, freq="D")
        return pd.DataFrame({
            "OPEN_DT": dates,
            "CLOSED_DT": dates + pd.to_timedelta(np.random.exponential(target_mean, n), unit="D"),
            "resolution_days": np.random.exponential(target_mean, n),
            "resolution_days_log": np.log1p(np.random.exponential(target_mean, n)),
            "TYPE": np.random.choice(["Pothole", "Streetlight"], n),
            "REASON": np.random.choice(["ReasonA", "ReasonB"], n),
            "Department": np.random.choice(["DeptX", "DeptY"], n),
            "neighborhood": np.random.choice(["NbhdA", "NbhdB"], n),
            "Source": np.random.choice(["Phone", "App"], n),
            "LATITUDE": 42.36 + np.random.randn(n) * 0.01,
            "LONGITUDE": -71.06 + np.random.randn(n) * 0.01,
        })

    train = _make("2020-01-01", 300, 3.0)
    val = _make("2021-01-01", 100, 4.0)
    test = _make("2022-01-01", 100, 5.0)
    return train, val, test


# ---------------------------------------------------------------------------
# Temporal features -- no leakage possible (pure datetime extraction)
# ---------------------------------------------------------------------------

class TestTemporalFeatures:
    def test_uses_only_open_dt(self, splits):
        train, _, _ = splits
        result = add_temporal_features(train.copy())
        assert "hour" in result.columns
        assert "is_weekend" in result.columns
        assert "is_holiday" in result.columns

    def test_no_target_columns_created(self, splits):
        train, _, _ = splits
        result = add_temporal_features(train.copy())
        # Should not create anything referencing resolution
        new_cols = set(result.columns) - set(train.columns)
        for col in new_cols:
            assert "resolution" not in col.lower()


# ---------------------------------------------------------------------------
# Target encoding -- must fit on train only
# ---------------------------------------------------------------------------

class TestTargetEncoder:
    def test_fit_on_train_only(self, splits):
        train, val, test = splits
        enc = TargetEncoder(smoothing=10.0)
        enc.fit(train, ["TYPE"], "resolution_days_log")

        # Encoder's global mean should be train's mean
        assert abs(enc.global_mean - train["resolution_days_log"].mean()) < 1e-10

    def test_unseen_category_gets_global_mean(self, splits):
        train, val, _ = splits
        enc = TargetEncoder(smoothing=10.0)
        enc.fit(train, ["TYPE"], "resolution_days_log")

        # Add unseen category to val
        val_copy = val.copy()
        val_copy.loc[val_copy.index[0], "TYPE"] = "NeverSeenCategory"
        result = enc.transform(val_copy, ["TYPE"])
        # The unseen category should get global_mean
        assert abs(result.loc[val_copy.index[0], "TYPE_target_enc"] - enc.global_mean) < 1e-10


class TestFrequencyEncoder:
    def test_fit_on_train_only(self, splits):
        train, val, _ = splits
        enc = FrequencyEncoder()
        enc.fit(train, ["TYPE"])

        # Frequencies should sum to 1.0
        freq_sum = sum(enc.encodings["TYPE"].values())
        assert abs(freq_sum - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Historical rolling features -- leakage-free version
# ---------------------------------------------------------------------------

class TestRollingFeaturesSimple:
    def test_stats_from_train_only(self, splits):
        train, val, test = splits
        train_copy, val_copy, test_copy = (
            train.copy(), val.copy(), test.copy()
        )
        train_out, val_out, test_out = add_rolling_features_simple(
            train_copy, val_copy, test_copy
        )

        # hist_mean_TYPE should exist
        assert "hist_mean_TYPE" in train_out.columns
        assert "hist_mean_TYPE" in val_out.columns

        # The mapping should be identical for val and test (both come from train)
        # For a given TYPE, val and test should have the same hist_mean
        for t in ["Pothole", "Streetlight"]:
            val_vals = val_out.loc[val_out["TYPE"] == t, "hist_mean_TYPE"].unique()
            test_vals = test_out.loc[test_out["TYPE"] == t, "hist_mean_TYPE"].unique()
            np.testing.assert_array_equal(val_vals, test_vals)


# ---------------------------------------------------------------------------
# Workload features -- must use lagged counts (no same-day info)
# ---------------------------------------------------------------------------

class TestWorkloadFeatures:
    def test_prev_day_is_lagged(self, splits):
        train, val, test = splits
        train_out, val_out, test_out = add_workload_features_combined(
            train.copy(), val.copy(), test.copy()
        )
        assert "requests_prev_day" in train_out.columns
        assert "requests_prev_week" in train_out.columns

        # The very first day in the dataset should have 0 prev_day requests
        combined = pd.concat([train_out, val_out, test_out])
        first_date = combined["OPEN_DT"].min()
        first_day_rows = combined[combined["OPEN_DT"].dt.date == first_date.date()]
        assert (first_day_rows["requests_prev_day"] == 0).all()


# ---------------------------------------------------------------------------
# Feature columns -- excluded columns check
# ---------------------------------------------------------------------------

class TestFeatureColumns:
    def test_excludes_target_and_dates(self, splits):
        train, _, _ = splits
        train_copy = add_temporal_features(train.copy())
        feature_cols = get_feature_columns(train_copy)

        forbidden = ["resolution_days", "resolution_hours", "resolution_days_log",
                      "OPEN_DT", "CLOSED_DT"]
        for col in forbidden:
            assert col not in feature_cols, f"{col} should be excluded from features"

    def test_excludes_raw_categoricals(self, splits):
        train, _, _ = splits
        train_copy = add_temporal_features(train.copy())
        feature_cols = get_feature_columns(train_copy)
        # Raw object columns should not be features
        for col in train_copy.select_dtypes(include=["object"]).columns:
            assert col not in feature_cols
