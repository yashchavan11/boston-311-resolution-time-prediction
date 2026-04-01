"""Tests for evaluation metric correctness."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import (
    regression_metrics,
    compute_baseline_improvement,
)


class TestRegressionMetrics:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = regression_metrics(y, y, log_transformed=False)
        assert m["mae_days"] == 0.0
        assert m["rmse_days"] == 0.0
        assert m["r2_days"] == 1.0

    def test_log_backtransform(self):
        """Verify log-space predictions are correctly back-transformed."""
        y_orig = np.array([1.0, 5.0, 10.0, 20.0])
        y_log = np.log1p(y_orig)
        m = regression_metrics(y_log, y_log, log_transformed=True)
        # Perfect prediction in log space and perfect in day space
        assert m["mae_days"] < 1e-10
        assert m["r2_days"] > 0.9999

    def test_mae_is_positive(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        m = regression_metrics(y_true, y_pred, log_transformed=False)
        assert m["mae_days"] > 0

    def test_clip_prevents_negative_predictions(self):
        """Predictions below 0 should be clipped to 0 in day space."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([-1.0, 2.0, 3.0])
        m = regression_metrics(y_true, y_pred, log_transformed=False)
        # MAE should use clipped prediction: |1-0| + |2-2| + |3-3| = 1.0/3
        expected_mae = (1.0 + 0.0 + 0.0) / 3
        assert abs(m["mae_days"] - expected_mae) < 1e-10

    def test_wape_definition(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 33.0])
        m = regression_metrics(y_true, y_pred, log_transformed=False)
        expected_wape = (2 + 2 + 3) / (10 + 20 + 30) * 100
        assert abs(m["wape"] - expected_wape) < 1e-6


class TestBaselineImprovement:
    def test_improvement_calculation(self):
        baseline = {"mae_days": 4.0}
        model = {"mae_days": 3.0}
        imp = compute_baseline_improvement(model, baseline)
        assert abs(imp - 25.0) < 1e-10  # (4-3)/4 * 100 = 25%

    def test_zero_baseline(self):
        baseline = {"mae_days": 0.0}
        model = {"mae_days": 3.0}
        imp = compute_baseline_improvement(model, baseline)
        assert imp == 0.0

    def test_same_as_baseline(self):
        baseline = {"mae_days": 4.0}
        model = {"mae_days": 4.0}
        imp = compute_baseline_improvement(model, baseline)
        assert imp == 0.0
