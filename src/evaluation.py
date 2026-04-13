"""Evaluation metrics and comparison tables."""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    log_transformed: bool = True,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics.

    If log_transformed=True, back-transforms predictions and targets
    from log space to original scale (days) before computing metrics.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.
    log_transformed : bool
        Whether the values are in log(1+x) space.
    prefix : str
        Optional prefix for metric keys.

    Returns
    -------
    dict
        Dictionary of metric names to values.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    metrics = {}

    # Metrics in log space (if applicable)
    if log_transformed:
        metrics[f"{prefix}rmse_log"] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics[f"{prefix}mae_log"] = mean_absolute_error(y_true, y_pred)
        metrics[f"{prefix}r2_log"] = r2_score(y_true, y_pred)

        # Back-transform to original scale
        y_true_orig = np.expm1(y_true)
        y_pred_orig = np.expm1(np.clip(y_pred, None, 20))  # Clip to avoid overflow
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred

    # Metrics in original scale (days)
    y_pred_orig = np.clip(y_pred_orig, 0, None)  # Predictions can't be negative days

    metrics[f"{prefix}rmse_days"] = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    metrics[f"{prefix}mae_days"] = mean_absolute_error(y_true_orig, y_pred_orig)
    metrics[f"{prefix}r2_days"] = r2_score(y_true_orig, y_pred_orig)
    metrics[f"{prefix}median_ae_days"] = median_absolute_error(y_true_orig, y_pred_orig)

    # MAPE (avoid division by zero)
    nonzero_mask = y_true_orig > 0.01
    if nonzero_mask.sum() > 0:
        mape = np.mean(
            np.abs(y_true_orig[nonzero_mask] - y_pred_orig[nonzero_mask])
            / y_true_orig[nonzero_mask]
        ) * 100
        metrics[f"{prefix}mape"] = mape
    else:
        metrics[f"{prefix}mape"] = np.nan

    # WAPE -- Weighted Absolute Percentage Error 
    sum_actual = np.sum(np.abs(y_true_orig))
    if sum_actual > 0:
        metrics[f"{prefix}wape"] = (
            np.sum(np.abs(y_true_orig - y_pred_orig)) / sum_actual * 100
        )
    else:
        metrics[f"{prefix}wape"] = np.nan

    # MASE -- Mean Absolute Scaled Error
    # Denominator is the test-set MAD (mean absolute deviation from the test
    # mean).  This is NOT the MAE of the training-set mean baseline; it is the
    # error of the best constant predictor on the evaluation set itself.
    # Values < 1 indicate the model beats a constant-mean predictor.
    naive_mae = np.mean(np.abs(y_true_orig - np.mean(y_true_orig)))
    if naive_mae > 0:
        metrics[f"{prefix}mase"] = mean_absolute_error(y_true_orig, y_pred_orig) / naive_mae
    else:
        metrics[f"{prefix}mase"] = np.nan

    return metrics


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute classification metrics for SLA breach prediction.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0=on-time, 1=late).
    y_pred : array-like
        Predicted binary labels.
    y_prob : array-like, optional
        Predicted probabilities for the positive class.
    prefix : str
        Optional prefix for metric keys.

    Returns
    -------
    dict
        Dictionary of metric names to values.
    """
    metrics = {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}precision_pos": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        f"{prefix}recall_pos": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        f"{prefix}f1_pos": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics[f"{prefix}auc_roc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics[f"{prefix}auc_roc"] = np.nan

    return metrics


def create_comparison_table(
    results: Dict[str, Dict[str, float]],
    sort_by: str = "mae_days",
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Create a sorted comparison table from model results.

    Parameters
    ----------
    results : dict
        {model_name: {metric_name: value, ...}, ...}
    sort_by : str
        Metric to sort by.
    ascending : bool
        Sort ascending (lower is better for MAE/RMSE) or descending (higher is better for R2).

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by the specified metric.
    """
    rows = []
    for model_name, metrics in results.items():
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

    # Round numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)

    return df



def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Pretty-print metrics."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:30s}: {value:12.4f}")
        else:
            print(f"  {key:30s}: {value}")
    print(f"{'='*50}\n")


def compute_baseline_improvement(
    model_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    metric_name: str = "mae_days",
) -> float:
    """
    Compute percentage improvement over a baseline.

    Returns
    -------
    float
        Percentage improvement (positive = better than baseline).
    """
    baseline_val = baseline_metrics.get(metric_name, 0)
    model_val = model_metrics.get(metric_name, 0)

    if baseline_val == 0:
        return 0.0

    return (baseline_val - model_val) / baseline_val * 100
