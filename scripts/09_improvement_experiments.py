
"""
Step 9: Late-Stage Improvement Experiments

Tests whether any approach can beat the blended ensemble baseline (MAE~2.704d).
The baseline is a Huber/Quantile blend (51.4%/48.6%) rebuilt from scratch in this script.
Experiments:
  1. CatBoost with GPU (native categoricals)
  2. XGBoost with CUDA
  3. Walk-forward validation (temporal stability check)
  4. Recency-weighted training
  5. Quantile prediction intervals
  6. Deeper Optuna tuning with more trials

All results compared against the frozen blended-ensemble baseline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os, warnings, json, time, gc
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import load_config, set_seed, MODELS_DIR, FIGURES_DIR, RESULTS_DIR, DATA_PROCESSED, setup_plotting, get_device
from src.evaluation import regression_metrics, compute_baseline_improvement
from src.models import load_feature_data
from src.features import build_monotonic_constraints

config = load_config()
SEED = config["random_seed"]
set_seed(SEED)
setup_plotting()

EXP_DIR = RESULTS_DIR  # Save all experiment outputs alongside other results

def pf(msg): print(msg, flush=True)

# ================================================================
# ENVIRONMENT CHECK
# ================================================================
pf("=" * 74)
pf("  IMPROVEMENT EXPERIMENTS - ENVIRONMENT CHECK")
pf("=" * 74)

# Device detection (cached, safe fallback)
XGB_DEVICE = get_device("xgb")
CB_DEVICE = get_device("catboost")
LGB_DEVICE = get_device("lgb")
gpu_info = {"xgboost": XGB_DEVICE.upper(), "catboost": CB_DEVICE, "lightgbm": LGB_DEVICE}
pf(f"  XGBoost: {XGB_DEVICE}, CatBoost: {CB_DEVICE}, LightGBM: {LGB_DEVICE}")

# ================================================================
# LOAD FEATURE-ENGINEERED DATA (from Step 4)
# ================================================================
pf("\n" + "=" * 74)
pf("  LOADING FEATURE-ENGINEERED DATA")
pf("=" * 74)

t0 = time.time()
train, val, test, feature_cols = load_feature_data()
pf(f"  Loaded in {time.time()-t0:.0f}s: Train={train.shape} Val={val.shape} Test={test.shape}")

X_train = train[feature_cols].fillna(0); y_train_log = train["resolution_days_log"]
X_val = val[feature_cols].fillna(0); y_val_log = val["resolution_days_log"]
X_test = test[feature_cols].fillna(0); y_test_log = test["resolution_days_log"]
y_train_raw = train["resolution_days"]
y_test_raw = test["resolution_days"]

pf(f"  Features: {len(feature_cols)}")
pf(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

# ================================================================
# FROZEN BASELINE (blended ensemble)
# ================================================================
pf("\n" + "=" * 74)
pf("  FROZEN BASELINE: blended ensemble")
pf("=" * 74)

from sklearn.dummy import DummyRegressor
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_absolute_error

mean_bl = DummyRegressor(strategy="mean").fit(X_train, y_train_log)
mean_bl_test = regression_metrics(y_test_log, mean_bl.predict(X_test), log_transformed=True)

# Rebuild blended ensemble from scratch
mono = build_monotonic_constraints(feature_cols)

# Huber
lgb_huber = LGBMRegressor(
    objective="huber", huber_delta=0.5,
    n_estimators=500, num_leaves=63, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, monotone_constraints=mono,
    device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)
lgb_huber.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)],
              callbacks=[early_stopping(50, verbose=False)])

# Quantile
lgb_quantile = LGBMRegressor(
    objective="quantile", alpha=0.5,
    n_estimators=500, num_leaves=63, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)
lgb_quantile.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)],
                 callbacks=[early_stopping(50, verbose=False)])

# Blend (frozen weights: 51.4% Huber + 48.6% Quantile)
y_blend_test = 0.514 * lgb_huber.predict(X_test) + 0.486 * lgb_quantile.predict(X_test)
baseline_m = regression_metrics(y_test_log, y_blend_test, log_transformed=True)
baseline_m["improvement_%"] = round(compute_baseline_improvement(baseline_m, mean_bl_test), 2)

pf(f"  Baseline MAE:   {baseline_m['mae_days']:.3f} days")
pf(f"  Baseline MedAE: {baseline_m['median_ae_days']:.3f} days")
pf(f"  Baseline RMSE:  {baseline_m['rmse_days']:.2f} days")
pf(f"  Baseline R2:    {baseline_m['r2_days']:.4f}")
pf(f"  Improvement vs mean: {baseline_m['improvement_%']:.1f}%")

FROZEN_MAE = baseline_m["mae_days"]
all_experiment_results = {"v4_blended_baseline": baseline_m}

# ================================================================
# EXPERIMENT 1: CatBoost with GPU
# ================================================================
pf("\n" + "=" * 74)
pf("  EXPERIMENT 1: CatBoost with GPU")
pf("=" * 74)

from catboost import CatBoostRegressor

# Note: Quantile(alpha=0.5) is mathematically equivalent to MAE (L1 loss) for median
# regression, so it produces identical results with the same seed and eval_metric.
# Only MAE and RMSE are tested to avoid reporting duplicate results.
for loss_name, loss_fn in [("MAE", "MAE"), ("RMSE", "RMSE")]:
    t0 = time.time()
    cb_params = {
        "iterations": 1000, "depth": 8, "learning_rate": 0.05,
        "l2_leaf_reg": 3.0, "random_seed": SEED, "verbose": 0,
        "loss_function": loss_fn, "eval_metric": "MAE",
        "early_stopping_rounds": 50,
    }
    cb_params["task_type"] = CB_DEVICE

    cb_model = CatBoostRegressor(**cb_params)
    cb_model.fit(X_train, y_train_log, eval_set=(X_val, y_val_log))
    m = regression_metrics(y_test_log, cb_model.predict(X_test), log_transformed=True)
    m["improvement_%"] = round(compute_baseline_improvement(m, mean_bl_test), 2)
    m["train_time_s"] = round(time.time()-t0, 2)
    m["device"] = gpu_info.get("catboost", "CPU")
    key = f"catboost_{loss_name.split(':')[0].lower()}"
    all_experiment_results[key] = m
    beat = "BEATS" if m["mae_days"] < FROZEN_MAE else "DOES NOT BEAT"
    pf(f"  CatBoost ({loss_name}): MAE={m['mae_days']:.3f}d R2={m['r2_days']:.4f} "
       f"({m['train_time_s']}s, {m['device']}) -- {beat} baseline")

# ================================================================
# EXPERIMENT 2: XGBoost with CUDA
# ================================================================
pf("\n" + "=" * 74)
pf("  EXPERIMENT 2: XGBoost with CUDA")
pf("=" * 74)

from xgboost import XGBRegressor

xgb_params = {
    "n_estimators": 1000, "max_depth": 8, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "tree_method": "hist", "random_state": SEED, "verbosity": 0,
    "early_stopping_rounds": 50,
}
xgb_params["device"] = XGB_DEVICE

for obj_name in ["reg:squarederror", "reg:absoluteerror", "reg:pseudohubererror"]:
    t0 = time.time()
    xgb_model = XGBRegressor(objective=obj_name, **xgb_params)
    xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)
    m = regression_metrics(y_test_log, xgb_model.predict(X_test), log_transformed=True)
    m["improvement_%"] = round(compute_baseline_improvement(m, mean_bl_test), 2)
    m["train_time_s"] = round(time.time()-t0, 2)
    m["device"] = gpu_info.get("xgboost", "CPU")
    short_name = obj_name.split(":")[-1]
    all_experiment_results[f"xgboost_{short_name}"] = m
    beat = "BEATS" if m["mae_days"] < FROZEN_MAE else "DOES NOT BEAT"
    pf(f"  XGBoost ({short_name}): MAE={m['mae_days']:.3f}d R2={m['r2_days']:.4f} "
       f"({m['train_time_s']}s, {m['device']}) -- {beat} baseline")

# ================================================================
# EXPERIMENT 3: Walk-Forward Validation
# ================================================================
pf("\n" + "=" * 74)
pf("  EXPERIMENT 3: Walk-Forward Validation (Temporal Stability)")
pf("=" * 74)

all_data = pd.concat([train, val, test], ignore_index=True)
wf_results = []

for test_year in [2022, 2023, 2024]:
    train_years_wf = list(range(2015, test_year))
    wf_train = all_data[all_data["OPEN_DT"].dt.year.isin(train_years_wf)]
    wf_test = all_data[all_data["OPEN_DT"].dt.year == test_year]

    X_wf_train = wf_train[feature_cols].fillna(0)
    y_wf_train = wf_train["resolution_days_log"]
    X_wf_test = wf_test[feature_cols].fillna(0)
    y_wf_test = wf_test["resolution_days_log"]

    t0 = time.time()
    wf_model = LGBMRegressor(
        objective="huber", huber_delta=0.5,
        n_estimators=500, num_leaves=63, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, monotone_constraints=mono,
        device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)
    wf_model.fit(X_wf_train, y_wf_train)

    m = regression_metrics(y_wf_test, wf_model.predict(X_wf_test), log_transformed=True)
    m["test_year"] = test_year
    m["train_size"] = len(wf_train)
    m["test_size"] = len(wf_test)
    m["train_time_s"] = round(time.time()-t0, 2)
    wf_results.append(m)
    pf(f"  Year {test_year}: MAE={m['mae_days']:.3f}d MedAE={m['median_ae_days']:.3f}d "
       f"R2={m['r2_days']:.4f} (train={m['train_size']:,}, test={m['test_size']:,})")

wf_df = pd.DataFrame(wf_results)
wf_df.to_csv(EXP_DIR / "walkforward_results.csv", index=False)
all_experiment_results["walkforward_mean_mae"] = {"mae_days": float(wf_df["mae_days"].mean())}

# ================================================================
# EXPERIMENT 4: Recency-Weighted Training
# ================================================================
pf("\n" + "=" * 74)
pf("  EXPERIMENT 4: Recency-Weighted Training")
pf("=" * 74)

train_years = train["OPEN_DT"].dt.year
year_range = train_years.max() - train_years.min() + 1
for decay in [0.5, 1.0, 2.0]:
    weights = np.exp(decay * (train_years - train_years.min()) / year_range)
    weights = weights / weights.mean()  # normalize

    t0 = time.time()
    rw_model = LGBMRegressor(
        objective="huber", huber_delta=0.5,
        n_estimators=500, num_leaves=63, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, monotone_constraints=mono,
        device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)
    rw_model.fit(X_train, y_train_log, sample_weight=weights,
                 eval_set=[(X_val, y_val_log)],
                 callbacks=[early_stopping(50, verbose=False)])

    m = regression_metrics(y_test_log, rw_model.predict(X_test), log_transformed=True)
    m["improvement_%"] = round(compute_baseline_improvement(m, mean_bl_test), 2)
    m["train_time_s"] = round(time.time()-t0, 2)
    m["decay"] = decay
    key = f"recency_decay_{decay}"
    all_experiment_results[key] = m
    beat = "BEATS" if m["mae_days"] < FROZEN_MAE else "DOES NOT BEAT"
    pf(f"  Decay={decay}: MAE={m['mae_days']:.3f}d R2={m['r2_days']:.4f} ({m['train_time_s']}s) -- {beat} baseline")

# ================================================================
# EXPERIMENT 5: Quantile Prediction Intervals
# ================================================================
pf("\n" + "=" * 74)
pf("  EXPERIMENT 5: Quantile Prediction Intervals")
pf("=" * 74)

for alpha_pair, label in [((0.1, 0.9), "80%"), ((0.05, 0.95), "90%")]:
    lo_model = LGBMRegressor(
        objective="quantile", alpha=alpha_pair[0],
        n_estimators=500, num_leaves=63, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)
    hi_model = LGBMRegressor(
        objective="quantile", alpha=alpha_pair[1],
        n_estimators=500, num_leaves=63, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)

    lo_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)],
                 callbacks=[early_stopping(50, verbose=False)])
    hi_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)],
                 callbacks=[early_stopping(50, verbose=False)])

    y_lo = np.expm1(lo_model.predict(X_test))
    y_hi = np.expm1(hi_model.predict(X_test))
    y_true = np.expm1(y_test_log.values)

    coverage = ((y_true >= y_lo) & (y_true <= y_hi)).mean() * 100
    interval_width = np.mean(y_hi - y_lo)

    pf(f"  {label} interval: Coverage={coverage:.1f}% (target={int(alpha_pair[1]*100 - alpha_pair[0]*100)}%) "
       f"Width={interval_width:.2f}d")
    all_experiment_results[f"quantile_interval_{label}"] = {
        "coverage_pct": round(coverage, 2),
        "target_coverage": int(alpha_pair[1]*100 - alpha_pair[0]*100),
        "mean_interval_width_days": round(interval_width, 3),
    }

# ================================================================
# EXPERIMENT 6: Deeper Optuna Tuning (50 trials)
# ================================================================
pf("\n" + "=" * 74)
pf("  EXPERIMENT 6: Deeper Optuna Tuning (50 trials)")
pf("=" * 74)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import TimeSeriesSplit

np.random.seed(SEED)
n_tune = min(300000, len(X_train))
tune_idx = np.sort(np.random.choice(len(X_train), n_tune, replace=False))
X_tune = X_train.iloc[tune_idx]; y_tune = y_train_log.iloc[tune_idx]
tscv = TimeSeriesSplit(n_splits=3)

def obj_deep(trial):
    p = {
        "objective": "huber",
        "huber_delta": trial.suggest_float("huber_delta", 0.1, 10.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
        "max_depth": trial.suggest_int("max_depth", 4, 14),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
        "monotone_constraints": mono,
        "device": LGB_DEVICE, "n_jobs": -1, "random_state": SEED, "verbose": -1,
    }
    scores = []
    for ti, vi in tscv.split(X_tune):
        mdl = LGBMRegressor(**p)
        mdl.fit(X_tune.iloc[ti], y_tune.iloc[ti],
                eval_set=[(X_tune.iloc[vi], y_tune.iloc[vi])],
                callbacks=[early_stopping(30, verbose=False)])
        pred = np.expm1(mdl.predict(X_tune.iloc[vi]))
        true = np.expm1(y_tune.iloc[vi])
        scores.append(mean_absolute_error(true, np.clip(pred, 0, None)))
    return np.mean(scores)

t0 = time.time()
study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(obj_deep, n_trials=config["optuna"]["n_trials_deep"], timeout=900)
pf(f"  Best CV MAE: {study.best_value:.3f}d ({time.time()-t0:.0f}s, {len(study.trials)} trials)")
pf(f"  Best params: {study.best_params}")

# Retrain with best params
bp = study.best_params.copy()
bp.update({"objective": "huber", "monotone_constraints": mono,
           "device": LGB_DEVICE, "n_jobs": -1, "random_state": SEED, "verbose": -1})
deep_model = LGBMRegressor(**bp)
deep_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)],
               callbacks=[early_stopping(50, verbose=False)])

m = regression_metrics(y_test_log, deep_model.predict(X_test), log_transformed=True)
m["improvement_%"] = round(compute_baseline_improvement(m, mean_bl_test), 2)
all_experiment_results["deep_optuna_huber"] = m
beat = "BEATS" if m["mae_days"] < FROZEN_MAE else "DOES NOT BEAT"
pf(f"  Deep Optuna: MAE={m['mae_days']:.3f}d R2={m['r2_days']:.4f} -- {beat} baseline")

# Also try blending deep tuned with quantile
y_deep_blend = 0.514 * deep_model.predict(X_test) + 0.486 * lgb_quantile.predict(X_test)
m2 = regression_metrics(y_test_log, y_deep_blend, log_transformed=True)
m2["improvement_%"] = round(compute_baseline_improvement(m2, mean_bl_test), 2)
all_experiment_results["deep_blend_ensemble"] = m2
beat2 = "BEATS" if m2["mae_days"] < FROZEN_MAE else "DOES NOT BEAT"
pf(f"  Deep Blend:  MAE={m2['mae_days']:.3f}d R2={m2['r2_days']:.4f} -- {beat2} baseline")

# ================================================================
# FINAL COMPARISON TABLE
# ================================================================
pf("\n" + "=" * 74)
pf("  FINAL IMPROVEMENT EXPERIMENT RESULTS")
pf("=" * 74)

rows = []
for name, m in all_experiment_results.items():
    if "mae_days" in m:
        row = {"experiment": name, "mae_days": m["mae_days"],
               "median_ae_days": m.get("median_ae_days", np.nan),
               "rmse_days": m.get("rmse_days", np.nan),
               "r2_days": m.get("r2_days", np.nan),
               "improvement_pct": m.get("improvement_%", np.nan),
               "beats_baseline": m["mae_days"] < FROZEN_MAE}
        rows.append(row)

results_df = pd.DataFrame(rows).sort_values("mae_days")
results_df.to_csv(EXP_DIR / "improvement_comparison.csv", index=False)

pf(f"\n  {'Experiment':35s} | {'MAE':>7s} | {'MedAE':>7s} | {'RMSE':>7s} | {'R2':>7s} | Beat?")
pf("  " + "-" * 85)
for _, row in results_df.iterrows():
    flag = "YES" if row["beats_baseline"] else "no"
    pf(f"  {row['experiment']:35s} | {row['mae_days']:7.3f} | {row['median_ae_days']:7.3f} | "
       f"{row['rmse_days']:7.2f} | {row['r2_days']:7.4f} | {flag}")

pf(f"\n  Frozen baseline MAE: {FROZEN_MAE:.3f} days")
any_beat = any(r["beats_baseline"] for r in rows if r["experiment"] != "v4_blended_baseline")
if any_beat:
    best_exp = min([r for r in rows if r["beats_baseline"] and r["experiment"] != "v4_blended_baseline"],
                   key=lambda r: r["mae_days"])
    pf(f"  BEST improvement: {best_exp['experiment']} (MAE={best_exp['mae_days']:.3f}d)")
else:
    pf("  NO experiment beat the baseline.")
    pf("  This confirms the improvement plateau is real and data-driven.")

# Save full results
with open(EXP_DIR / "all_results.json", "w") as f:
    json.dump({k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                   for kk, vv in v.items()} for k, v in all_experiment_results.items()},
              f, indent=2)

# Walk-forward stability plot
pf("\n  Generating walk-forward stability plot...")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(wf_df["test_year"], wf_df["mae_days"], "o-", color="steelblue", lw=2, markersize=8)
ax.axhline(FROZEN_MAE, color="red", ls="--", lw=1.5, label=f"Baseline ({FROZEN_MAE:.3f}d)")
ax.set_xlabel("Test Year"); ax.set_ylabel("MAE (days)")
ax.set_title("Walk-Forward Validation: Temporal Stability", fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(EXP_DIR / "walkforward_stability.png", dpi=300, bbox_inches="tight")
plt.close()

pf("\n" + "=" * 74)
pf("  IMPROVEMENT EXPERIMENTS COMPLETE")
pf(f"  Results saved to: {EXP_DIR}/")
pf("=" * 74)
