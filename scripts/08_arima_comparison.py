
"""
Step 8: ARIMA/SARIMA Literature Comparison
==========================================

SWIFT uses Gaussian CRFs to forecast weekly aggregate response times using a
2-week lookback. It compared against Linear Regression, ARIMA, and SARIMA.

This script implements:
  1. Weekly and daily ARIMA/SARIMA models on aggregate resolution times
  2. Per-request predictions by assigning each test request the forecast for its period
  3. Direct comparison with the per-request ML models

Requires: Step 2 (clean parquets with OPEN_DT and resolution_days)
Output: results/arima_comparison.csv, figures/arima_comparison.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings, json, time, gc
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.utils import load_config, set_seed, DATA_PROCESSED, FIGURES_DIR, RESULTS_DIR, setup_plotting

config = load_config()
SEED = config["random_seed"]
set_seed(SEED)
setup_plotting()

def pf(msg): print(msg, flush=True)

pf("=" * 74)
pf("  ARIMA / SARIMA BASELINE -- SWIFT Comparison")
pf("=" * 74)

# ===========================================================================
# STEP 1: LOAD PREPROCESSED DATA (from Step 2)
# ===========================================================================
pf("\n[1/5] Loading preprocessed data...")
train = pd.read_parquet(DATA_PROCESSED / "train_clean.parquet")
val = pd.read_parquet(DATA_PROCESSED / "val_clean.parquet")
test = pd.read_parquet(DATA_PROCESSED / "test_clean.parquet")

pf(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

# ===========================================================================
# STEP 2: AGGREGATE TIME SERIES
# ===========================================================================
pf("\n[2/5] Building weekly time series...")

# Add date column
for split in [train, val, test]:
    split["open_date"] = split["OPEN_DT"].dt.date

# Weekly aggregation for train+val (training period for ARIMA)
train_val = pd.concat([train, val])
weekly_train = train_val.groupby(pd.Grouper(key="OPEN_DT", freq="W"))["resolution_days"].agg(
    ["mean", "median", "count"]
).dropna()
weekly_train.columns = ["mean_res", "median_res", "count"]
weekly_train = weekly_train[weekly_train["count"] >= 50]  # Remove sparse weeks

# Daily aggregation
daily_train = train_val.groupby(pd.Grouper(key="OPEN_DT", freq="D"))["resolution_days"].agg(
    ["mean", "median", "count"]
).dropna()
daily_train.columns = ["mean_res", "median_res", "count"]
daily_train = daily_train[daily_train["count"] >= 10]

# Weekly test series for evaluation
weekly_test = test.groupby(pd.Grouper(key="OPEN_DT", freq="W"))["resolution_days"].agg(
    ["mean", "median", "count"]
).dropna()
weekly_test.columns = ["mean_res", "median_res", "count"]

pf(f"  Weekly training series: {len(weekly_train)} weeks")
pf(f"  Daily training series: {len(daily_train)} days")
pf(f"  Weekly test series: {len(weekly_test)} weeks")

# ===========================================================================
# STEP 3: FIT ARIMA / SARIMA MODELS
# ===========================================================================
pf("\n[3/5] Fitting ARIMA/SARIMA models...")

results = {}

# --- ARIMA on weekly median resolution ---
pf("\n  [a] ARIMA(2,1,2) on weekly median resolution...")
t0 = time.time()
try:
    arima_weekly = ARIMA(weekly_train["median_res"], order=(2, 1, 2))
    arima_weekly_fit = arima_weekly.fit()
    arima_weekly_time = time.time() - t0
    # Forecast for test period
    n_test_weeks = len(weekly_test)
    arima_weekly_forecast = arima_weekly_fit.forecast(steps=n_test_weeks)
    pf(f"    Fit time: {arima_weekly_time:.1f}s")
    pf(f"    Forecast range: {arima_weekly_forecast.min():.3f} to {arima_weekly_forecast.max():.3f} days")
except Exception as e:
    pf(f"    ARIMA weekly failed: {e}")
    arima_weekly_forecast = None

# --- ARIMA on weekly mean resolution ---
pf("\n  [b] ARIMA(2,1,2) on weekly mean resolution...")
t0 = time.time()
try:
    arima_mean = ARIMA(weekly_train["mean_res"], order=(2, 1, 2))
    arima_mean_fit = arima_mean.fit()
    arima_mean_time = time.time() - t0
    arima_mean_forecast = arima_mean_fit.forecast(steps=n_test_weeks)
    pf(f"    Fit time: {arima_mean_time:.1f}s")
    pf(f"    Forecast range: {arima_mean_forecast.min():.3f} to {arima_mean_forecast.max():.3f} days")
except Exception as e:
    pf(f"    ARIMA mean failed: {e}")
    arima_mean_forecast = None

# --- SARIMA on weekly median resolution ---
pf("\n  [c] SARIMA(1,1,1)(1,1,1,52) on weekly median resolution...")
t0 = time.time()
try:
    sarima_weekly = SARIMAX(weekly_train["median_res"],
                            order=(1, 1, 1),
                            seasonal_order=(1, 1, 1, 52),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
    sarima_weekly_fit = sarima_weekly.fit(disp=False, maxiter=200)
    sarima_weekly_time = time.time() - t0
    sarima_weekly_forecast = sarima_weekly_fit.forecast(steps=n_test_weeks)
    pf(f"    Fit time: {sarima_weekly_time:.1f}s")
    pf(f"    Forecast range: {sarima_weekly_forecast.min():.3f} to {sarima_weekly_forecast.max():.3f} days")
except Exception as e:
    pf(f"    SARIMA weekly failed: {e}")
    sarima_weekly_forecast = None

# --- Daily ARIMA ---
pf("\n  [d] ARIMA(3,1,3) on daily median resolution...")
t0 = time.time()
try:
    arima_daily = ARIMA(daily_train["median_res"], order=(3, 1, 3))
    arima_daily_fit = arima_daily.fit()
    arima_daily_time = time.time() - t0
    n_test_days = (test["OPEN_DT"].max() - test["OPEN_DT"].min()).days + 1
    arima_daily_forecast = arima_daily_fit.forecast(steps=n_test_days)
    pf(f"    Fit time: {arima_daily_time:.1f}s")
    pf(f"    Forecast range: {arima_daily_forecast.min():.3f} to {arima_daily_forecast.max():.3f} days")
except Exception as e:
    pf(f"    ARIMA daily failed: {e}")
    arima_daily_forecast = None

# ===========================================================================
# STEP 4: PER-REQUEST PREDICTION & EVALUATION
# ===========================================================================
pf("\n[4/5] Computing per-request predictions and metrics...")

test_actual = test["resolution_days"].values

# --- Naive baselines (similar to SWIFT's linear regression baseline) ---
# Training-set mean as prediction for every test request
train_mean = train["resolution_days"].mean()
train_median = train["resolution_days"].median()
pf(f"\n  Training set mean: {train_mean:.3f} days, median: {train_median:.3f} days")

# Mean baseline
mean_pred = np.full(len(test), train_mean)
mae_mean = np.mean(np.abs(test_actual - mean_pred))
rmse_mean = np.sqrt(np.mean((test_actual - mean_pred)**2))
medae_mean = np.median(np.abs(test_actual - mean_pred))

# Median baseline
median_pred = np.full(len(test), train_median)
mae_median = np.mean(np.abs(test_actual - median_pred))
rmse_median = np.sqrt(np.mean((test_actual - median_pred)**2))
medae_median = np.median(np.abs(test_actual - median_pred))

results["mean_baseline"] = {"mae": mae_mean, "rmse": rmse_mean, "medae": medae_mean, "method": "Constant (mean)"}
results["median_baseline"] = {"mae": mae_median, "rmse": rmse_median, "medae": medae_median, "method": "Constant (median)"}

pf(f"\n  Mean baseline:   MAE={mae_mean:.3f}d  RMSE={rmse_mean:.3f}d  MedAE={medae_mean:.3f}d")
pf(f"  Median baseline: MAE={mae_median:.3f}d  RMSE={rmse_median:.3f}d  MedAE={medae_median:.3f}d")

# --- ARIMA weekly median: assign weekly forecast to each request ---
if arima_weekly_forecast is not None:
    test_week = test["OPEN_DT"].dt.isocalendar()
    test_year_week = test["OPEN_DT"].dt.to_period("W")

    # Map each test request to the weekly forecast
    forecast_idx = arima_weekly_forecast.index
    test_week_periods = test["OPEN_DT"].dt.to_period("W")

    # Create a mapping from week period to forecast value
    week_forecast_map = {}
    for i, (idx, val) in enumerate(zip(forecast_idx, arima_weekly_forecast.values)):
        week_forecast_map[i] = val

    # Assign by week number relative to first test week
    first_test_week = test["OPEN_DT"].min().to_period("W")
    test_week_num = (test_week_periods - first_test_week).apply(lambda x: x.n)
    arima_w_pred = test_week_num.map(lambda x: week_forecast_map.get(x, arima_weekly_forecast.iloc[-1])).values

    # Clip negative predictions
    arima_w_pred = np.clip(arima_w_pred, 0, 90)
    mae_arima_w = np.mean(np.abs(test_actual - arima_w_pred))
    rmse_arima_w = np.sqrt(np.mean((test_actual - arima_w_pred)**2))
    medae_arima_w = np.median(np.abs(test_actual - arima_w_pred))
    results["arima_weekly_median"] = {"mae": mae_arima_w, "rmse": rmse_arima_w, "medae": medae_arima_w,
                                      "method": "ARIMA(2,1,2) weekly median"}
    pf(f"\n  ARIMA weekly (median): MAE={mae_arima_w:.3f}d  RMSE={rmse_arima_w:.3f}d  MedAE={medae_arima_w:.3f}d")

# --- ARIMA weekly mean ---
if arima_mean_forecast is not None:
    arima_m_pred = test_week_num.map(lambda x: week_forecast_map.get(x,
        arima_mean_forecast.iloc[-1] if x >= len(arima_mean_forecast) else arima_mean_forecast.iloc[min(x, len(arima_mean_forecast)-1)]
    )).values
    # Actually rebuild map for mean
    mean_forecast_map = {}
    for i, val in enumerate(arima_mean_forecast.values):
        mean_forecast_map[i] = val
    arima_m_pred = test_week_num.map(lambda x: mean_forecast_map.get(x, arima_mean_forecast.iloc[-1])).values
    arima_m_pred = np.clip(arima_m_pred, 0, 90)
    mae_arima_m = np.mean(np.abs(test_actual - arima_m_pred))
    rmse_arima_m = np.sqrt(np.mean((test_actual - arima_m_pred)**2))
    medae_arima_m = np.median(np.abs(test_actual - arima_m_pred))
    results["arima_weekly_mean"] = {"mae": mae_arima_m, "rmse": rmse_arima_m, "medae": medae_arima_m,
                                    "method": "ARIMA(2,1,2) weekly mean"}
    pf(f"  ARIMA weekly (mean):   MAE={mae_arima_m:.3f}d  RMSE={rmse_arima_m:.3f}d  MedAE={medae_arima_m:.3f}d")

# --- SARIMA weekly ---
if sarima_weekly_forecast is not None:
    sarima_forecast_map = {}
    for i, val in enumerate(sarima_weekly_forecast.values):
        sarima_forecast_map[i] = val
    sarima_pred = test_week_num.map(lambda x: sarima_forecast_map.get(x, sarima_weekly_forecast.iloc[-1])).values
    sarima_pred = np.clip(sarima_pred, 0, 90)
    mae_sarima = np.mean(np.abs(test_actual - sarima_pred))
    rmse_sarima = np.sqrt(np.mean((test_actual - sarima_pred)**2))
    medae_sarima = np.median(np.abs(test_actual - sarima_pred))
    results["sarima_weekly"] = {"mae": mae_sarima, "rmse": rmse_sarima, "medae": medae_sarima,
                                 "method": "SARIMA(1,1,1)(1,1,1,52) weekly"}
    pf(f"  SARIMA weekly:         MAE={mae_sarima:.3f}d  RMSE={rmse_sarima:.3f}d  MedAE={medae_sarima:.3f}d")

# --- Daily ARIMA ---
if arima_daily_forecast is not None:
    test_day_offset = (test["OPEN_DT"].dt.normalize() - test["OPEN_DT"].min().normalize()).dt.days
    daily_forecast_map = {}
    for i, val in enumerate(arima_daily_forecast.values):
        daily_forecast_map[i] = val
    arima_d_pred = test_day_offset.map(lambda x: daily_forecast_map.get(x, arima_daily_forecast.iloc[-1])).values
    arima_d_pred = np.clip(arima_d_pred, 0, 90)
    mae_arima_d = np.mean(np.abs(test_actual - arima_d_pred))
    rmse_arima_d = np.sqrt(np.mean((test_actual - arima_d_pred)**2))
    medae_arima_d = np.median(np.abs(test_actual - arima_d_pred))
    results["arima_daily"] = {"mae": mae_arima_d, "rmse": rmse_arima_d, "medae": medae_arima_d,
                               "method": "ARIMA(3,1,3) daily median"}
    pf(f"  ARIMA daily (median):  MAE={mae_arima_d:.3f}d  RMSE={rmse_arima_d:.3f}d  MedAE={medae_arima_d:.3f}d")

# --- Best ML models (from saved results) ---
v4_results = pd.read_csv(RESULTS_DIR / "final_model_comparison_v4.csv")
best_model = v4_results.iloc[0]
best_model_name = best_model["model"]
results[best_model_name] = {
    "mae": best_model["mae_days"], "rmse": best_model["rmse_days"],
    "medae": best_model["median_ae_days"],
    "method": f"Best ML: {best_model_name} (99 features)"
}
# Add quantile model
quantile_row = v4_results[v4_results["model"] == "lgb_quantile_v3ref"].iloc[0]
results["lgb_quantile_v4"] = {
    "mae": quantile_row["mae_days"], "rmse": quantile_row["rmse_days"],
    "medae": quantile_row["median_ae_days"], "method": "LGB Quantile (99 features)"
}

# Linear regression (from baseline results)
full_results = pd.read_csv(RESULTS_DIR / "baseline_results.csv")
lr_row = full_results[full_results["model"] == "linear_regression"]
if len(lr_row) > 0:
    lr_row = lr_row.iloc[0]
    results["linear_regression"] = {
        "mae": lr_row["mae_days"], "rmse": lr_row["rmse_days"],
        "medae": lr_row["median_ae_days"], "method": "Linear Regression (99 features)"
    }

# ===========================================================================
# STEP 5: RESULTS TABLE & FIGURE
# ===========================================================================
pf("\n[5/5] Generating comparison table and figure...")

# Build comparison table
comp_rows = []
for name, r in results.items():
    comp_rows.append({"model": name, "method": r["method"], "mae": r["mae"], "rmse": r["rmse"], "medae": r["medae"]})
comp_df = pd.DataFrame(comp_rows).sort_values("mae", ascending=True)
comp_df.to_csv(RESULTS_DIR / "arima_comparison.csv", index=False)

pf("\n" + "=" * 90)
pf(f"  {'Model':30s} | {'Method':45s} | {'MAE':>7s} | {'RMSE':>8s} | {'MedAE':>7s}")
pf("  " + "-" * 86)
for _, row in comp_df.iterrows():
    pf(f"  {row['model']:30s} | {row['method']:45s} | {row['mae']:7.3f} | {row['rmse']:8.3f} | {row['medae']:7.3f}")
pf("=" * 90)

# --- Figure: SWIFT Comparison ---
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Assign colors based on method type
def get_color(name):
    if "arima" in name or "sarima" in name:
        return "#ED7D31"  # Orange for time-series
    elif "baseline" in name:
        return "#A5A5A5"  # Gray for baselines
    elif "linear" in name:
        return "#FFC000"  # Yellow for linear regression
    else:
        return "#4472C4"  # Blue for ML models

comp_sorted = comp_df.sort_values("mae", ascending=True)
colors = [get_color(n) for n in comp_sorted["model"]]

axes[0].barh(range(len(comp_sorted)), comp_sorted["mae"], color=colors, edgecolor="white")
axes[0].set_yticks(range(len(comp_sorted)))
axes[0].set_yticklabels(comp_sorted["method"], fontsize=8)
axes[0].set_xlabel("MAE (days)")
axes[0].set_title("Mean Absolute Error Comparison", fontweight="bold")
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis="x")

axes[1].barh(range(len(comp_sorted)), comp_sorted["medae"], color=colors, edgecolor="white")
axes[1].set_yticks(range(len(comp_sorted)))
axes[1].set_yticklabels(comp_sorted["method"], fontsize=8)
axes[1].set_xlabel("Median AE (days)")
axes[1].set_title("Median Absolute Error Comparison", fontweight="bold")
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis="x")

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#4472C4", label="Per-request ML models"),
    Patch(facecolor="#ED7D31", label="ARIMA/SARIMA (time-series)"),
    Patch(facecolor="#FFC000", label="Linear Regression"),
    Patch(facecolor="#A5A5A5", label="Constant baselines"),
]
axes[1].legend(handles=legend_elements, loc="lower right", fontsize=8)

plt.suptitle("Per-Request ML Models vs. ARIMA/SARIMA Baselines (SWIFT Comparison)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "arima_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

pf(f"\n  Results saved: results/arima_comparison.csv")
pf(f"  Figure saved:  figures/arima_comparison.png")

# Compute improvement of best ML over best ARIMA
best_ml_mae = comp_df[~comp_df["model"].str.contains("arima|sarima|baseline|linear")]["mae"].min()
arima_maes = comp_df[comp_df["model"].str.contains("arima|sarima")]["mae"]
if len(arima_maes) > 0:
    best_arima_mae = arima_maes.min()
    improvement = ((best_arima_mae - best_ml_mae) / best_arima_mae) * 100
    pf(f"\n  Best ML model MAE:    {best_ml_mae:.3f} days")
    pf(f"  Best ARIMA/SARIMA MAE: {best_arima_mae:.3f} days")
    pf(f"  Improvement:           {improvement:.1f}%")

pf("\n" + "=" * 74)
pf("  ARIMA/SARIMA COMPARISON COMPLETE")
pf("=" * 74)
