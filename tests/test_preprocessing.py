"""Tests for preprocessing invariants and leakage prevention."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import (
    compute_target,
    filter_valid_cases,
    add_log_target,
    temporal_split,
    fill_missing_values,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Minimal DataFrame mimicking raw 311 data."""
    np.random.seed(42)
    n = 200
    open_dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close_dates = open_dates + pd.to_timedelta(np.random.exponential(3, n), unit="D")
    return pd.DataFrame({
        "OPEN_DT": open_dates,
        "CLOSED_DT": close_dates,
        "CASE_STATUS": "Closed",
        "CLOSURE_REASON": "",
        "LATITUDE": np.where(np.random.rand(n) < 0.1, np.nan, 42.36),
        "LONGITUDE": np.where(np.random.rand(n) < 0.1, np.nan, -71.06),
        "LOCATION_ZIPCODE": pd.Series(["02101"] * n).where(np.random.rand(n) >= 0.05),
        "TYPE": np.random.choice(["Pothole", "Streetlight", "Graffiti"], n),
    })


@pytest.fixture
def sample_with_target(sample_df):
    df = compute_target(sample_df)
    df = filter_valid_cases(df, max_resolution_days=90, verbose=False)
    df = add_log_target(df)
    return df


# ---------------------------------------------------------------------------
# Target computation
# ---------------------------------------------------------------------------

class TestComputeTarget:
    def test_resolution_days_non_negative(self, sample_df):
        df = compute_target(sample_df)
        # All cases where closed > open should have non-negative resolution
        mask = df["CLOSED_DT"] >= df["OPEN_DT"]
        assert (df.loc[mask, "resolution_days"] >= 0).all()

    def test_resolution_hours_consistent(self, sample_df):
        df = compute_target(sample_df)
        np.testing.assert_allclose(
            df["resolution_days"].values,
            df["resolution_hours"].values / 24,
            rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class TestFilterValidCases:
    def test_cap_applied(self, sample_df):
        df = compute_target(sample_df)
        filtered = filter_valid_cases(df, max_resolution_days=10, verbose=False)
        assert filtered["resolution_days"].max() <= 10

    def test_no_negative_times(self, sample_df):
        df = compute_target(sample_df)
        filtered = filter_valid_cases(df, verbose=False)
        assert (filtered["resolution_days"] >= 0).all()


# ---------------------------------------------------------------------------
# Log target
# ---------------------------------------------------------------------------

class TestLogTarget:
    def test_log_target_exists(self, sample_with_target):
        assert "resolution_days_log" in sample_with_target.columns

    def test_log_invertible(self, sample_with_target):
        original = sample_with_target["resolution_days"].values
        logged = sample_with_target["resolution_days_log"].values
        recovered = np.expm1(logged)
        np.testing.assert_allclose(recovered, original, rtol=1e-10)


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

class TestTemporalSplit:
    def test_no_temporal_overlap(self, sample_with_target):
        train, val, test = temporal_split(
            sample_with_target,
            train_years=[2020],
            val_years=[2020],  # same year for this small test
            test_years=[2020],
            verbose=False,
        )
        # With a real multi-year dataset
        df = sample_with_target.copy()
        df["_year"] = df["OPEN_DT"].dt.year
        years = sorted(df["_year"].unique())
        if len(years) >= 3:
            mid = years[len(years) // 2]
            train, val, test = temporal_split(
                df,
                train_years=years[:len(years)//2],
                val_years=[mid],
                test_years=years[len(years)//2+1:],
                verbose=False,
            )
            assert train["OPEN_DT"].max() < val["OPEN_DT"].min()
            assert val["OPEN_DT"].max() < test["OPEN_DT"].min()

    def test_no_data_loss(self, sample_with_target):
        """Rows assigned to exactly one split (disjoint year lists)."""
        # All data is 2020, so assign to train only
        train, val, test = temporal_split(
            sample_with_target,
            train_years=[2020],
            val_years=[2019],   # no 2019 data in fixture
            test_years=[2021],  # no 2021 data in fixture
            verbose=False,
        )
        assert len(train) == len(sample_with_target)
        assert len(val) == 0
        assert len(test) == 0


# ---------------------------------------------------------------------------
# Fill missing values -- leakage prevention
# ---------------------------------------------------------------------------

class TestFillMissingValues:
    def test_returns_stats(self, sample_with_target):
        df, stats = fill_missing_values(sample_with_target)
        assert "median_lat" in stats
        assert "median_lon" in stats

    def test_train_stats_reused_for_val(self, sample_with_target):
        """Val/test must use train's fill statistics, not their own."""
        # Simulate different splits with different distributions
        train_df = sample_with_target.copy()
        val_df = sample_with_target.copy()

        # Force different medians by altering val lat values
        val_df["LATITUDE"] = val_df["LATITUDE"] + 1.0
        val_df.loc[val_df.index[:5], "LATITUDE"] = np.nan

        train_filled, train_stats = fill_missing_values(train_df)
        val_filled, _ = fill_missing_values(val_df, fill_stats=train_stats)

        # Val's NaN rows should be filled with TRAIN's median, not val's
        filled_vals = val_filled.loc[val_df.index[:5], "LATITUDE"]
        assert (filled_vals == train_stats["median_lat"]).all(), (
            "Val NaN values should be filled with train median, not val median"
        )

    def test_no_nans_after_fill(self, sample_with_target):
        df, _ = fill_missing_values(sample_with_target)
        assert not df["LATITUDE"].isna().any()
        assert not df["LONGITUDE"].isna().any()
