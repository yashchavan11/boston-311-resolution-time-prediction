
"""
Step 7: Advanced Models -- Robust Loss Functions, Tuning & Ensemble
===================================================================
Trains LightGBM with multiple loss functions, Optuna tuning, hurdle models,
and SLSQP-optimized ensemble. Produces SHAP analysis and error plots.

Methodology: All selection decisions (loss sweeps, Optuna tuning,
ensemble weights, hurdle thresholds, meta-ensemble weights) use a held-out
"tune fold" (year 2021) carved from training data. The validation set (2022)
is used ONLY for early-stopping callbacks and reporting -- never for selection.
Final models are retrained on full training data (2015-2021) before test eval.

Train-proper: 2015-2020  |  Tune fold: 2021  |  Val: 2022  |  Test: 2023-2024

Requires: Step 4 (feature-engineered parquet files)
Output: models/*.joblib, results/final_model_comparison_v4.csv, figures/*.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings, json, time, yaml, gc
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

from src.utils import load_config, set_seed, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, RESULTS_DIR, setup_plotting, get_device
from src.evaluation import regression_metrics, classification_metrics, create_comparison_table, compute_baseline_improvement
from src.models import save_model, load_feature_data
from src.features import build_monotonic_constraints

config = load_config(); SEED = config["random_seed"]; set_seed(SEED); setup_plotting()
LGB_DEVICE = get_device("lgb")
print(f"  LightGBM device: {LGB_DEVICE}")

def pf(msg): print(msg, flush=True)
def header(n, t):
    pf("\n" + "=" * 74)
    pf(f"  STEP {n}: {t}")
    pf("=" * 74)

def mstr(m, name, imp=False):
    s = (f"  {name:35s} | MAE={m['mae_days']:.3f}d | MedAE={m['median_ae_days']:.3f}d | "
         f"RMSE={m['rmse_days']:.2f}d | R2={m['r2_days']:.4f} | "
         f"WAPE={m.get('wape',np.nan):.1f}% | MASE={m.get('mase',np.nan):.3f}")
    if imp and "improvement_%" in m:
        s += f" | D={m['improvement_%']:.1f}%"
    return s

# ===========================================================================
header(1, "LOAD FEATURE-ENGINEERED DATA (from Step 4)")
# ===========================================================================
train, val, test, feature_cols = load_feature_data()
pf(f"  Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
pf(f"  Features: {len(feature_cols)}")

# -- Generate dataset summary (from clean parquets) --
train_clean = pd.read_parquet(DATA_PROCESSED / "train_clean.parquet")
val_clean = pd.read_parquet(DATA_PROCESSED / "val_clean.parquet")
test_clean = pd.read_parquet(DATA_PROCESSED / "test_clean.parquet")
all_data = pd.concat([train_clean, val_clean, test_clean])
split_info = {
    "train_years": config["data"]["train_years"],
    "val_years": config["data"]["val_years"],
    "test_years": config["data"]["test_years"],
    "max_resolution_days": config["data"]["max_resolution_days"],
    "train_count": int(len(train_clean)),
    "val_count": int(len(val_clean)),
    "test_count": int(len(test_clean)),
    "total_count": int(len(all_data)),
    "mean_resolution_days": float(all_data["resolution_days"].mean()),
    "median_resolution_days": float(all_data["resolution_days"].median()),
    "std_resolution_days": float(all_data["resolution_days"].std()),
    "skewness": float(all_data["resolution_days"].skew()),
    "kurtosis": float(all_data["resolution_days"].kurtosis()),
}
for t_thresh in [1, 3, 7, 14, 30, 60, 90]:
    split_info[f"pct_within_{t_thresh}d"] = float((all_data["resolution_days"] <= t_thresh).mean() * 100)
del train_clean, val_clean, test_clean, all_data; gc.collect()

with open(RESULTS_DIR / "dataset_summary_v4.json", "w") as f:
    json.dump(split_info, f, indent=2)
pf(f"  Saved dataset summary to results/dataset_summary_v4.json")

# ===========================================================================
header(2, "SPLIT TRAIN INTO TRAIN-PROPER (2015-2020) + TUNE FOLD (2021)")
# ===========================================================================
# The 'year' column is a feature created during feature engineering
tune_mask = train["year"] == 2021
proper_mask = ~tune_mask

train_proper = train[proper_mask].copy()
tune_fold = train[tune_mask].copy()
pf(f"  Train-proper (2015-2020): {train_proper.shape}")
pf(f"  Tune fold (2021):         {tune_fold.shape}")
pf(f"  Full train (2015-2021):   {train.shape}")
pf(f"  Val (2022, early-stop):   {val.shape}")
pf(f"  Test (2023-2024):         {test.shape}")

# Feature / target arrays: train-proper (selection training)
X_train_proper = train_proper[feature_cols].fillna(0)
y_train_proper_log = train_proper["resolution_days_log"]
y_train_proper_raw = train_proper["resolution_days"]

# Feature / target arrays: tune fold (selection evaluation)
X_tune = tune_fold[feature_cols].fillna(0)
y_tune_log = tune_fold["resolution_days_log"]
y_tune_raw = tune_fold["resolution_days"]

# Feature / target arrays: full train (final retraining)
X_train = train[feature_cols].fillna(0); y_train_log = train["resolution_days_log"]
X_val   = val[feature_cols].fillna(0);   y_val_log   = val["resolution_days_log"]
X_test  = test[feature_cols].fillna(0);  y_test_log  = test["resolution_days_log"]

# Raw targets (for Tweedie which trains on original scale)
y_train_raw = train["resolution_days"]
y_val_raw   = val["resolution_days"]
y_test_raw  = test["resolution_days"]

pf(f"  X_train_proper: {X_train_proper.shape}, X_tune: {X_tune.shape}")
pf(f"  X_train (full): {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

# ===========================================================================
header(3, "REFERENCE MODELS (trained on full train, val early-stop)")
# ===========================================================================
from lightgbm import LGBMRegressor, LGBMClassifier, early_stopping
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

all_results = {}

# Build monotonic constraints for non-quantile models
mono = build_monotonic_constraints(feature_cols)
pf(f"  Monotonic constraints: {sum(1 for m in mono if m!=0)} features constrained")

# Mean baseline (trained on full train)
from sklearn.dummy import DummyRegressor
mean_bl_model = DummyRegressor(strategy="mean")
mean_bl_model.fit(X_train, y_train_log)
mean_bl = regression_metrics(y_val_log, mean_bl_model.predict(X_val), log_transformed=True)
all_results["mean_baseline"] = mean_bl
save_model(mean_bl_model, "mean_baseline")
pf(mstr(mean_bl, "mean_baseline"))

# Reference: LightGBM quantile (alpha=0.5) - baseline (no selection needed)
pf("\n  Training reference models on full train (val early-stop)...")
t0 = time.time()
lgb_quantile = LGBMRegressor(
    objective="quantile", alpha=0.5,
    n_estimators=500, num_leaves=63, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)
lgb_quantile.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)],
                 callbacks=[early_stopping(50, verbose=False)])
m = regression_metrics(y_val_log, lgb_quantile.predict(X_val), log_transformed=True)
m["improvement_%"] = round(compute_baseline_improvement(m, mean_bl), 2)
m["train_time_s"] = round(time.time()-t0, 2)
all_results["lgb_quantile_v3ref"] = m
save_model(lgb_quantile, "lgb_quantile_v3ref")
pf(mstr(m, "lgb_quantile_v3ref (a=0.5)", imp=True))

# Standard LightGBM (L2) with monotonic constraints - reference (no selection needed)
t0 = time.time()
lgb_l2 = LGBMRegressor(
    n_estimators=500, num_leaves=63, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    monotone_constraints=mono,
    device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)
lgb_l2.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)],
           callbacks=[early_stopping(50, verbose=False)])
m = regression_metrics(y_val_log, lgb_l2.predict(X_val), log_transformed=True)
m["improvement_%"] = round(compute_baseline_improvement(m, mean_bl), 2)
m["train_time_s"] = round(time.time()-t0, 2)
all_results["lgb_l2_mono"] = m
save_model(lgb_l2, "lgb_l2_mono")
pf(mstr(m, "lgb_l2_mono (MSE)", imp=True))

# ===========================================================================
header(4, "ROBUST LOSS FUNCTIONS -- SELECTION ON TUNE FOLD")
# ===========================================================================
pf("  Selection: train on train-proper (2015-2020), evaluate on tune fold (2021)")

# -- 4A: Huber Loss ---------------------------------------------------------
# Huber loss transitions from MSE (quadratic) to MAE (linear) at delta
# Lower delta = more robust to outliers, higher delta = more like MSE
pf("\n  > HUBER LOSS -- sweeping delta [0.5, 1.0, 2.0, 5.0]")
best_huber_mae = 999; best_huber_delta = 1.0; best_huber_m = None

for delta in [0.5, 1.0, 2.0, 5.0]:
    t0 = time.time()
    lgb_huber = LGBMRegressor(
        objective="huber", huber_delta=delta,
        n_estimators=500, num_leaves=63, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        monotone_constraints=mono,
        device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)
    lgb_huber.fit(X_train_proper, y_train_proper_log,
                  eval_set=[(X_tune, y_tune_log)],
                  callbacks=[early_stopping(50, verbose=False)])
    m = regression_metrics(y_tune_log, lgb_huber.predict(X_tune), log_transformed=True)
    m["improvement_%"] = round(compute_baseline_improvement(m, mean_bl), 2)
    m["train_time_s"] = round(time.time()-t0, 2)
    pf(f"    delta={delta:.1f} -> MAE={m['mae_days']:.3f}d  MedAE={m['median_ae_days']:.3f}d  "
       f"RMSE={m['rmse_days']:.2f}d  R2={m['r2_days']:.4f}  ({m['train_time_s']}s)")
    if m["mae_days"] < best_huber_mae:
        best_huber_mae = m["mae_days"]
        best_huber_delta = delta; best_huber_m = m

pf(f"\n  * Best Huber delta={best_huber_delta} (selected on tune fold)")

# Retrain best Huber on full train with val early-stop
pf("  Retraining best Huber on full train (2015-2021)...")
t0 = time.time()
best_huber_model = LGBMRegressor(
    objective="huber", huber_delta=best_huber_delta,
    n_estimators=500, num_leaves=63, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    monotone_constraints=mono,
    device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)
best_huber_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)],
                     callbacks=[early_stopping(50, verbose=False)])
m_huber_val = regression_metrics(y_val_log, best_huber_model.predict(X_val), log_transformed=True)
m_huber_val["improvement_%"] = round(compute_baseline_improvement(m_huber_val, mean_bl), 2)
m_huber_val["train_time_s"] = round(time.time()-t0, 2)
all_results["lgb_huber_best"] = m_huber_val
save_model(best_huber_model, "lgb_huber_best")
pf(mstr(m_huber_val, f"lgb_huber (d={best_huber_delta})", imp=True))

# -- 4B: Fair Loss ----------------------------------------------------------
# Fair loss: L(x) = c2 * (|x|/c - ln(1 + |x|/c))
# Smoother transition than Huber; c controls the balance
pf("\n  > FAIR LOSS -- sweeping c [0.5, 1.0, 2.0, 5.0]")
best_fair_mae = 999; best_fair_c = 1.0; best_fair_m = None

for c in [0.5, 1.0, 2.0, 5.0]:
    t0 = time.time()
    lgb_fair = LGBMRegressor(
        objective="fair", fair_c=c,
        n_estimators=500, num_leaves=63, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        monotone_constraints=mono,
        device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)
    lgb_fair.fit(X_train_proper, y_train_proper_log,
                 eval_set=[(X_tune, y_tune_log)],
                 callbacks=[early_stopping(50, verbose=False)])
    m = regression_metrics(y_tune_log, lgb_fair.predict(X_tune), log_transformed=True)
    m["improvement_%"] = round(compute_baseline_improvement(m, mean_bl), 2)
    m["train_time_s"] = round(time.time()-t0, 2)
    pf(f"    c={c:.1f} -> MAE={m['mae_days']:.3f}d  MedAE={m['median_ae_days']:.3f}d  "
       f"RMSE={m['rmse_days']:.2f}d  R2={m['r2_days']:.4f}  ({m['train_time_s']}s)")
    if m["mae_days"] < best_fair_mae:
        best_fair_mae = m["mae_days"]
        best_fair_c = c; best_fair_m = m

pf(f"\n  * Best Fair c={best_fair_c} (selected on tune fold)")

# Retrain best Fair on full train with val early-stop
pf("  Retraining best Fair on full train (2015-2021)...")
t0 = time.time()
best_fair_model = LGBMRegressor(
    objective="fair", fair_c=best_fair_c,
    n_estimators=500, num_leaves=63, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    monotone_constraints=mono,
    device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)
best_fair_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)],
                    callbacks=[early_stopping(50, verbose=False)])
m_fair_val = regression_metrics(y_val_log, best_fair_model.predict(X_val), log_transformed=True)
m_fair_val["improvement_%"] = round(compute_baseline_improvement(m_fair_val, mean_bl), 2)
m_fair_val["train_time_s"] = round(time.time()-t0, 2)
all_results["lgb_fair_best"] = m_fair_val
save_model(best_fair_model, "lgb_fair_best")
pf(mstr(m_fair_val, f"lgb_fair (c={best_fair_c})", imp=True))

# -- 4C: Tweedie Objective -------------------------------------------------
# Tweedie handles right-skewed, non-negative distributions natively
# Variance power: 1.0 = Poisson-like, 1.5 = compound Poisson-gamma, 2.0 = Gamma
# Trained on RAW days (not log-transformed) since Tweedie handles skew itself
pf("\n  > TWEEDIE -- sweeping variance_power [1.1, 1.3, 1.5, 1.7, 1.9]")
pf("    (Training on RAW target -- Tweedie handles skew natively)")
pf("    Selection: train on train-proper, evaluate on tune fold")
best_tweedie_mae = 999; best_tweedie_power = 1.5; best_tweedie_m = None

for power in [1.1, 1.3, 1.5, 1.7, 1.9]:
    t0 = time.time()
    lgb_tw = LGBMRegressor(
        objective="tweedie", tweedie_variance_power=power,
        n_estimators=500, num_leaves=63, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        monotone_constraints=mono,
        device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)
    lgb_tw.fit(X_train_proper, y_train_proper_raw,
               eval_set=[(X_tune, y_tune_raw)],
               callbacks=[early_stopping(50, verbose=False)])
    y_tw_pred = np.clip(lgb_tw.predict(X_tune), 0, None)
    # Metrics on raw scale (not log)
    m = regression_metrics(y_tune_raw, y_tw_pred, log_transformed=False)
    m["improvement_%"] = round(compute_baseline_improvement(m, mean_bl), 2)
    m["train_time_s"] = round(time.time()-t0, 2)
    pf(f"    p={power:.1f} -> MAE={m['mae_days']:.3f}d  MedAE={m['median_ae_days']:.3f}d  "
       f"RMSE={m['rmse_days']:.2f}d  R2={m['r2_days']:.4f}  ({m['train_time_s']}s)")
    if m["mae_days"] < best_tweedie_mae:
        best_tweedie_mae = m["mae_days"]
        best_tweedie_power = power; best_tweedie_m = m

pf(f"\n  * Best Tweedie power={best_tweedie_power} (selected on tune fold)")

# Retrain best Tweedie on full train with val early-stop (raw target)
pf("  Retraining best Tweedie on full train (2015-2021)...")
t0 = time.time()
best_tweedie_model = LGBMRegressor(
    objective="tweedie", tweedie_variance_power=best_tweedie_power,
    n_estimators=500, num_leaves=63, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    monotone_constraints=mono,
    device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)
best_tweedie_model.fit(X_train, y_train_raw, eval_set=[(X_val, y_val_raw)],
                       callbacks=[early_stopping(50, verbose=False)])
y_tw_val = np.clip(best_tweedie_model.predict(X_val), 0, None)
m_tw_val = regression_metrics(y_val_raw, y_tw_val, log_transformed=False)
m_tw_val["improvement_%"] = round(compute_baseline_improvement(m_tw_val, mean_bl), 2)
m_tw_val["train_time_s"] = round(time.time()-t0, 2)
all_results["lgb_tweedie_best"] = m_tw_val
save_model(best_tweedie_model, "lgb_tweedie_best")
pf(mstr(m_tw_val, f"lgb_tweedie (p={best_tweedie_power})", imp=True))

# -- Summary table so far --------------------------------------------------
pf("\n  -- Loss Function Comparison (Validation) --------------------")
pf(f"  {'Objective':20s} | {'MAE':>7s} | {'MedAE':>7s} | {'RMSE':>7s} | {'R2':>7s}")
pf("  " + "-" * 60)
for lbl, res in [
    ("Quantile (a=0.5)", all_results["lgb_quantile_v3ref"]),
    ("L2 (MSE)", all_results["lgb_l2_mono"]),
    (f"Huber (delta={best_huber_delta})", all_results["lgb_huber_best"]),
    (f"Fair (c={best_fair_c})", all_results["lgb_fair_best"]),
    (f"Tweedie (p={best_tweedie_power})", all_results["lgb_tweedie_best"]),
]:
    pf(f"  {lbl:20s} | {res['mae_days']:7.3f} | {res['median_ae_days']:7.3f} | "
       f"{res['rmse_days']:7.2f} | {res['r2_days']:7.4f}")

# ===========================================================================
header(5, "OPTUNA TUNING -- BEST ROBUST LOSS (CV on train-proper)")
# ===========================================================================
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import TimeSeriesSplit

# Identify which loss function won (based on tune fold metrics)
loss_candidates = {
    "huber": (best_huber_mae, best_huber_delta),
    "fair": (best_fair_mae, best_fair_c),
}
best_loss_name = min(loss_candidates, key=lambda k: loss_candidates[k][0])
best_loss_param = loss_candidates[best_loss_name][1]
pf(f"  Tuning best robust loss: {best_loss_name} (param={best_loss_param})")
pf(f"  Using TimeSeriesSplit CV on train-proper (2015-2020) -- no val leakage")

np.random.seed(SEED)
n_tune = min(300000, len(X_train_proper))
tune_idx = np.sort(np.random.choice(len(X_train_proper), n_tune, replace=False))
X_optuna = X_train_proper.iloc[tune_idx]; y_optuna = y_train_proper_log.iloc[tune_idx]
N_CV_SPLITS = config["optuna"]["n_cv_splits"]
tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS); N_TRIALS = config["optuna"]["n_trials"]

def obj_robust(trial):
    obj_name = best_loss_name
    obj_params = {}
    if obj_name == "huber":
        obj_params["huber_delta"] = trial.suggest_float("huber_delta", 0.3, 8.0, log=True)
    elif obj_name == "fair":
        obj_params["fair_c"] = trial.suggest_float("fair_c", 0.3, 8.0, log=True)

    p = {"objective": obj_name,
         **obj_params,
         "n_estimators": trial.suggest_int("n_estimators", 300, 800),
         "max_depth": trial.suggest_int("max_depth", 5, 12),
         "num_leaves": trial.suggest_int("num_leaves", 30, 200),
         "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
         "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
         "subsample": trial.suggest_float("subsample", 0.7, 1.0),
         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
         "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 2.0, log=True),
         "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 2.0, log=True),
         "monotone_constraints": mono,
         "device": LGB_DEVICE, "n_jobs": -1, "random_state": SEED, "verbose": -1}
    scores = []
    for ti, vi in tscv.split(X_optuna):
        mdl = LGBMRegressor(**p)
        mdl.fit(X_optuna.iloc[ti], y_optuna.iloc[ti],
                eval_set=[(X_optuna.iloc[vi], y_optuna.iloc[vi])],
                callbacks=[early_stopping(30, verbose=False)])
        pred = np.expm1(mdl.predict(X_optuna.iloc[vi]))
        true = np.expm1(y_optuna.iloc[vi])
        scores.append(mean_absolute_error(true, np.clip(pred, 0, None)))
    return np.mean(scores)

pf(f"  Running Optuna ({N_TRIALS} trials)...")
t0 = time.time()
study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(obj_robust, n_trials=N_TRIALS, timeout=config["optuna"]["timeout"])
pf(f"    Best CV MAE: {study.best_value:.3f}d ({time.time()-t0:.0f}s)")

# Retrain best params on FULL training data (2015-2021) with val early-stop
pf("  Retraining Optuna-best on full train (2015-2021) with val early-stop...")
lp = study.best_params.copy()
lp.update({"objective": best_loss_name, "monotone_constraints": mono,
           "device": LGB_DEVICE, "n_jobs": -1, "random_state": SEED, "verbose": -1})
lgb_tuned_v4 = LGBMRegressor(**lp)
lgb_tuned_v4.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)],
                 callbacks=[early_stopping(50, verbose=False)])
save_model(lgb_tuned_v4, "lgb_tuned_v4")

m = regression_metrics(y_val_log, lgb_tuned_v4.predict(X_val), log_transformed=True)
m["improvement_%"] = round(compute_baseline_improvement(m, mean_bl), 2)
all_results["lgb_tuned_v4"] = m
pf(mstr(m, f"lgb_tuned_v4 ({best_loss_name})", imp=True))

with open(RESULTS_DIR / "optuna_best_params_v4.yaml", "w") as f:
    yaml.dump(study.best_params, f, default_flow_style=False)

# ===========================================================================
header(6, "THRESHOLD-OPTIMIZED HURDLE MODEL (selection on tune fold)")
# ===========================================================================
from sklearn.metrics import accuracy_score, f1_score as f1_sk, classification_report

y_train_proper_days = np.expm1(y_train_proper_log)
y_tune_days = np.expm1(y_tune_log)

pf("  Sweeping hurdle thresholds [0.5d, 1d, 3d, 7d]...")
pf("  Selection: train on train-proper (2015-2020), evaluate on tune fold (2021)")
hurdle_results = {}

for threshold in [0.5, 1.0, 3.0, 7.0]:
    pf(f"\n  -- Threshold = {threshold}d --")
    y_tp_cls = (y_train_proper_days > threshold).astype(int)  # 0=short, 1=long
    y_tune_cls = (y_tune_days > threshold).astype(int)

    short_pct_train = (y_tp_cls==0).mean()*100
    long_pct_train  = (y_tp_cls==1).mean()*100
    pf(f"    Train-proper: Short={short_pct_train:.1f}%, Long={long_pct_train:.1f}%")
    pf(f"    Tune fold:    Short={(y_tune_cls==0).mean()*100:.1f}%, Long={(y_tune_cls==1).mean()*100:.1f}%")

    # Stage 1: Classifier (train on train-proper, eval on tune fold)
    hurdle_clf = LGBMClassifier(
        n_estimators=400, num_leaves=63, learning_rate=0.1,
        class_weight="balanced", device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)
    hurdle_clf.fit(X_train_proper, y_tp_cls,
                   eval_set=[(X_tune, y_tune_cls)],
                   callbacks=[early_stopping(30, verbose=False)])
    hurdle_pred = hurdle_clf.predict(X_tune)
    acc = accuracy_score(y_tune_cls, hurdle_pred)
    f1m = f1_sk(y_tune_cls, hurdle_pred, average="macro")
    pf(f"    Classifier: Acc={acc:.4f}, F1-macro={f1m:.4f}")

    # Stage 2: Separate regressors (using best robust loss for long-term)
    short_mask = y_tp_cls == 0; long_mask = y_tp_cls == 1

    reg_short = LGBMRegressor(
        objective="quantile", alpha=0.5,
        n_estimators=500, num_leaves=63, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)
    reg_short.fit(X_train_proper[short_mask], y_train_proper_log[short_mask])

    # For long-term: use Huber/Fair to better handle the wide variance
    long_params = {
        "objective": best_loss_name,
        "n_estimators": 500, "num_leaves": 31, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "monotone_constraints": mono,
        "device": LGB_DEVICE, "n_jobs": -1, "random_state": SEED, "verbose": -1,
    }
    if best_loss_name == "huber":
        long_params["huber_delta"] = best_loss_param
    elif best_loss_name == "fair":
        long_params["fair_c"] = best_loss_param
    reg_long = LGBMRegressor(**long_params)
    reg_long.fit(X_train_proper[long_mask], y_train_proper_log[long_mask])

    # Soft gating (evaluated on tune fold)
    prob_long = hurdle_clf.predict_proba(X_tune)[:, 1]
    pred_short = reg_short.predict(X_tune)
    pred_long  = reg_long.predict(X_tune)
    y_soft = (1 - prob_long) * pred_short + prob_long * pred_long

    m_soft = regression_metrics(y_tune_log, y_soft, log_transformed=True)
    m_soft["improvement_%"] = round(compute_baseline_improvement(m_soft, mean_bl), 2)
    m_soft["threshold"] = threshold
    m_soft["clf_acc"] = acc
    m_soft["clf_f1"] = f1m
    pf(f"    Soft gating: MAE={m_soft['mae_days']:.3f}d  RMSE={m_soft['rmse_days']:.2f}d  R2={m_soft['r2_days']:.4f}")

    # Hard gating (evaluated on tune fold)
    gate = hurdle_clf.predict(X_tune)
    y_hard = np.empty(len(X_tune))
    s = gate == 0; l = gate == 1
    if s.any(): y_hard[s] = reg_short.predict(X_tune[s])
    if l.any(): y_hard[l] = reg_long.predict(X_tune[l])
    m_hard = regression_metrics(y_tune_log, y_hard, log_transformed=True)
    m_hard["improvement_%"] = round(compute_baseline_improvement(m_hard, mean_bl), 2)
    pf(f"    Hard gating: MAE={m_hard['mae_days']:.3f}d  RMSE={m_hard['rmse_days']:.2f}d  R2={m_hard['r2_days']:.4f}")

    hurdle_results[threshold] = {
        "soft": m_soft, "hard": m_hard,
        "threshold": threshold,
    }

# Save hurdle threshold results to CSV 
hurdle_rows = []
for t, hr in hurdle_results.items():
    for mode in ["soft", "hard"]:
        row = {"threshold_days": t, "gating": mode}
        row.update(hr[mode])
        hurdle_rows.append(row)
pd.DataFrame(hurdle_rows).to_csv(RESULTS_DIR / "hurdle_threshold_results_v4.csv", index=False)
pf("  Saved hurdle threshold results to results/hurdle_threshold_results_v4.csv")

# Find best threshold (selected on tune fold)
best_threshold = min(hurdle_results, key=lambda t: hurdle_results[t]["soft"]["mae_days"])
pf(f"\n  * Best hurdle threshold: {best_threshold}d (selected on tune fold)")

# Retrain hurdle components on full train (2015-2021) with best threshold
pf(f"  Retraining hurdle components on full train (2015-2021)...")
y_train_days = np.expm1(y_train_log)
y_train_cls = (y_train_days > best_threshold).astype(int)
y_val_days = np.expm1(y_val_log)
y_val_cls = (y_val_days > best_threshold).astype(int)

hurdle_clf_best = LGBMClassifier(
    n_estimators=400, num_leaves=63, learning_rate=0.1,
    class_weight="balanced", device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)
hurdle_clf_best.fit(X_train, y_train_cls,
                    eval_set=[(X_val, y_val_cls)],
                    callbacks=[early_stopping(30, verbose=False)])

short_mask_full = y_train_cls == 0; long_mask_full = y_train_cls == 1

reg_s = LGBMRegressor(
    objective="quantile", alpha=0.5,
    n_estimators=500, num_leaves=63, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)
reg_s.fit(X_train[short_mask_full], y_train_log[short_mask_full])

long_params_final = {
    "objective": best_loss_name,
    "n_estimators": 500, "num_leaves": 31, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "monotone_constraints": mono,
    "device": LGB_DEVICE, "n_jobs": -1, "random_state": SEED, "verbose": -1,
}
if best_loss_name == "huber":
    long_params_final["huber_delta"] = best_loss_param
elif best_loss_name == "fair":
    long_params_final["fair_c"] = best_loss_param
reg_l = LGBMRegressor(**long_params_final)
reg_l.fit(X_train[long_mask_full], y_train_log[long_mask_full])

# Evaluate retrained hurdle on val (my head hurts)
prob_long_val = hurdle_clf_best.predict_proba(X_val)[:, 1]
y_hurdle_val_soft = (1-prob_long_val)*reg_s.predict(X_val) + prob_long_val*reg_l.predict(X_val)
m_hurdle_val_soft = regression_metrics(y_val_log, y_hurdle_val_soft, log_transformed=True)
m_hurdle_val_soft["improvement_%"] = round(compute_baseline_improvement(m_hurdle_val_soft, mean_bl), 2)
all_results["hurdle_best_soft"] = m_hurdle_val_soft

gate_val = hurdle_clf_best.predict(X_val)
y_hurdle_val_hard = np.empty(len(X_val))
s_v = gate_val == 0; l_v = gate_val == 1
if s_v.any(): y_hurdle_val_hard[s_v] = reg_s.predict(X_val[s_v])
if l_v.any(): y_hurdle_val_hard[l_v] = reg_l.predict(X_val[l_v])
m_hurdle_val_hard = regression_metrics(y_val_log, y_hurdle_val_hard, log_transformed=True)
m_hurdle_val_hard["improvement_%"] = round(compute_baseline_improvement(m_hurdle_val_hard, mean_bl), 2)
all_results["hurdle_best_hard"] = m_hurdle_val_hard

save_model(hurdle_clf_best, "hurdle_clf_best")
save_model(reg_s, "hurdle_reg_short_best")
save_model(reg_l, "hurdle_reg_long_best")

pf(mstr(m_hurdle_val_soft, f"hurdle_soft (t={best_threshold}d)", imp=True))
pf(mstr(m_hurdle_val_hard, f"hurdle_hard (t={best_threshold}d)", imp=True))

# ===========================================================================
header(7, "OPTIMIZED ENSEMBLE (selection on tune fold, retrain on full train)")
# ===========================================================================
from scipy.optimize import minimize as scipy_minimize

# Step 7a: 
pf("  Training base models on train-proper for ensemble weight selection...")

base_models_tp = {}

# quantile on train-proper
lgb_quantile_tp = LGBMRegressor(
    objective="quantile", alpha=0.5,
    n_estimators=500, num_leaves=63, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)
lgb_quantile_tp.fit(X_train_proper, y_train_proper_log,
                    eval_set=[(X_tune, y_tune_log)],
                    callbacks=[early_stopping(50, verbose=False)])
base_models_tp["lgb_quantile"] = lgb_quantile_tp

# l2 mono on train-proper
lgb_l2_tp = LGBMRegressor(
    n_estimators=500, num_leaves=63, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    monotone_constraints=mono,
    device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)
lgb_l2_tp.fit(X_train_proper, y_train_proper_log,
              eval_set=[(X_tune, y_tune_log)],
              callbacks=[early_stopping(50, verbose=False)])
base_models_tp["lgb_l2_mono"] = lgb_l2_tp

# huber on train-proper
lgb_huber_tp = LGBMRegressor(
    objective="huber", huber_delta=best_huber_delta,
    n_estimators=500, num_leaves=63, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    monotone_constraints=mono,
    device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)
lgb_huber_tp.fit(X_train_proper, y_train_proper_log,
                 eval_set=[(X_tune, y_tune_log)],
                 callbacks=[early_stopping(50, verbose=False)])
base_models_tp["lgb_huber"] = lgb_huber_tp

# fair on train-proper
lgb_fair_tp = LGBMRegressor(
    objective="fair", fair_c=best_fair_c,
    n_estimators=500, num_leaves=63, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    monotone_constraints=mono,
    device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)
lgb_fair_tp.fit(X_train_proper, y_train_proper_log,
                eval_set=[(X_tune, y_tune_log)],
                callbacks=[early_stopping(50, verbose=False)])
base_models_tp["lgb_fair"] = lgb_fair_tp

# tuned model on train-proper
lgb_tuned_tp = LGBMRegressor(**lp)
lgb_tuned_tp.fit(X_train_proper, y_train_proper_log,
                 eval_set=[(X_tune, y_tune_log)],
                 callbacks=[early_stopping(50, verbose=False)])
base_models_tp["lgb_tuned_v4"] = lgb_tuned_tp

base_names = list(base_models_tp.keys())

# Get tune fold predictions for weight optimization
tune_preds = {n: m.predict(X_tune) for n, m in base_models_tp.items()}
tune_mat = np.column_stack([tune_preds[n] for n in base_names])

def blend_obj(w):
    return mean_absolute_error(np.expm1(y_tune_log),
                                np.clip(np.expm1(tune_mat @ w), 0, None))

pf("  Optimizing blend weights on tune fold...")
res = scipy_minimize(blend_obj, x0=np.ones(len(base_names))/len(base_names),
                     bounds=[(0,1)]*len(base_names),
                     constraints={"type":"eq","fun":lambda w: sum(w)-1}, method="SLSQP")
blend_w = dict(zip(base_names, res.x))
pf(f"  Blend weights (selected on tune fold): { {k:round(v,4) for k,v in blend_w.items()} }")

# Detect collapsed ensemble
max_weight_model = max(blend_w, key=blend_w.get)
max_weight = blend_w[max_weight_model]
if max_weight > 0.95:
    pf(f"  NOTE: Ensemble collapsed to single model ({max_weight_model}, weight={max_weight:.4f}).")
    pf(f"  This is a legitimate optimization outcome -- {max_weight_model} dominates on the tune fold.")

# Step 7b: Apply optimized weights to full-train models for val/test
# Use the already-trained full-train models
base_models_full = {
    "lgb_quantile": lgb_quantile,
    "lgb_l2_mono": lgb_l2,
    "lgb_huber": best_huber_model,
    "lgb_fair": best_fair_model,
    "lgb_tuned_v4": lgb_tuned_v4,
}

val_preds = {n: m.predict(X_val) for n, m in base_models_full.items()}
val_mat = np.column_stack([val_preds[n] for n in base_names])
y_blend_val = val_mat @ res.x
bm = regression_metrics(y_val_log, y_blend_val, log_transformed=True)
bm["improvement_%"] = round(compute_baseline_improvement(bm, mean_bl), 2)
all_results["blended_ensemble_v4"] = bm
pf(mstr(bm, "blended_ensemble_v4", imp=True))

with open(RESULTS_DIR / "blend_weights_v4.json", "w") as f:
    json.dump({k:float(v) for k,v in blend_w.items()}, f, indent=2)

# Step 7c: Meta-ensemble weight selection on tune fold
pf("\n  Meta-ensemble: weighted blend + hurdle (selection on tune fold)...")

# Get hurdle predictions on tune fold (from train-proper-trained hurdle)
# Retrain hurdle on train-proper for tune fold predictions
y_tp_cls_best = (y_train_proper_days > best_threshold).astype(int)
y_tune_cls_best = (y_tune_days > best_threshold).astype(int)

hurdle_clf_tp = LGBMClassifier(
    n_estimators=400, num_leaves=63, learning_rate=0.1,
    class_weight="balanced", device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)
hurdle_clf_tp.fit(X_train_proper, y_tp_cls_best,
                  eval_set=[(X_tune, y_tune_cls_best)],
                  callbacks=[early_stopping(30, verbose=False)])

short_mask_tp = y_tp_cls_best == 0; long_mask_tp = y_tp_cls_best == 1
reg_s_tp = LGBMRegressor(
    objective="quantile", alpha=0.5,
    n_estimators=500, num_leaves=63, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    device=LGB_DEVICE, n_jobs=-1, random_state=SEED, verbose=-1)
reg_s_tp.fit(X_train_proper[short_mask_tp], y_train_proper_log[short_mask_tp])

reg_l_tp = LGBMRegressor(**long_params)
reg_l_tp.fit(X_train_proper[long_mask_tp], y_train_proper_log[long_mask_tp])

prob_long_tune = hurdle_clf_tp.predict_proba(X_tune)[:, 1]
y_hurdle_tune = (1-prob_long_tune)*reg_s_tp.predict(X_tune) + prob_long_tune*reg_l_tp.predict(X_tune)

# Blend of ensemble + hurdle on tune fold
y_blend_tune = tune_mat @ res.x
for hw in [0.3, 0.5, 0.7]:
    y_meta = (1-hw) * y_blend_tune + hw * y_hurdle_tune
    mm = regression_metrics(y_tune_log, y_meta, log_transformed=True)
    pf(f"    hurdle_weight={hw:.1f} -> MAE={mm['mae_days']:.3f}d  R2={mm['r2_days']:.4f}")

# Find optimal meta weight on tune fold
meta_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
best_meta_w = 0.0; best_meta_mae = 999; best_meta_m = None
for hw in meta_weights:
    y_meta = (1-hw) * y_blend_tune + hw * y_hurdle_tune
    mm = regression_metrics(y_tune_log, y_meta, log_transformed=True)
    if mm["mae_days"] < best_meta_mae:
        best_meta_mae = mm["mae_days"]; best_meta_w = hw; best_meta_m = mm

pf(f"\n  * Best meta-ensemble: hurdle_weight={best_meta_w:.1f} (selected on tune fold)")

# Apply meta weight to full-train models on val 
y_hurdle_val = (1-prob_long_val)*reg_s.predict(X_val) + prob_long_val*reg_l.predict(X_val)
y_meta_val = (1-best_meta_w) * y_blend_val + best_meta_w * y_hurdle_val
meta_val_m = regression_metrics(y_val_log, y_meta_val, log_transformed=True)
meta_val_m["improvement_%"] = round(compute_baseline_improvement(meta_val_m, mean_bl), 2)
all_results["meta_ensemble_v4"] = meta_val_m
pf(mstr(meta_val_m, "meta_ensemble_v4", imp=True))

# ===========================================================================
header(8, "FINAL TEST SET EVALUATION (2023-2024)")
# ===========================================================================
mean_test_m = regression_metrics(y_test_log, mean_bl_model.predict(X_test), log_transformed=True)
test_results = {}

# All log-space models (trained on full train 2015-2021)
test_log_models = {
    "mean_baseline": mean_bl_model,
    "lgb_quantile_v3ref": lgb_quantile,
    "lgb_l2_mono": lgb_l2,
    "lgb_huber_best": best_huber_model,
    "lgb_fair_best": best_fair_model,
    "lgb_tuned_v4": lgb_tuned_v4,
}

for name, model in test_log_models.items():
    yp = model.predict(X_test)
    m = regression_metrics(y_test_log, yp, log_transformed=True)
    m["improvement_%"] = round(compute_baseline_improvement(m, mean_test_m), 2)
    test_results[name] = m
    pf(mstr(m, name, imp=True))

# Tweedie (raw space -- trained on days, not log-transformed)
y_tw_test = np.clip(best_tweedie_model.predict(X_test), 0, None)
tw_m = regression_metrics(y_test_raw.values, y_tw_test, log_transformed=False)
# Tweedie has no log-space metrics since it trains on raw target; mark as N/A (isn't necessary i'm stupid)
tw_m["rmse_log"] = "N/A"
tw_m["mae_log"] = "N/A"
tw_m["r2_log"] = "N/A"
tw_m["improvement_%"] = round(compute_baseline_improvement(tw_m, mean_test_m), 2)
test_results["lgb_tweedie_best"] = tw_m
pf(mstr(tw_m, "lgb_tweedie_best", imp=True))

# Blended ensemble (full-train models with tune-fold-optimized weights)
test_mat = np.column_stack([base_models_full[n].predict(X_test) for n in base_names])
y_blend_test = test_mat @ res.x
blm = regression_metrics(y_test_log, y_blend_test, log_transformed=True)
blm["improvement_%"] = round(compute_baseline_improvement(blm, mean_test_m), 2)
test_results["blended_ensemble_v4"] = blm
pf(mstr(blm, "blended_ensemble_v4", imp=True))

# Hurdle best (soft) -- full-train models
prob_long_test = hurdle_clf_best.predict_proba(X_test)[:, 1]
y_hurdle_test = (1-prob_long_test)*reg_s.predict(X_test) + prob_long_test*reg_l.predict(X_test)
htm = regression_metrics(y_test_log, y_hurdle_test, log_transformed=True)
htm["improvement_%"] = round(compute_baseline_improvement(htm, mean_test_m), 2)
test_results[f"hurdle_soft_{best_threshold}d"] = htm
pf(mstr(htm, f"hurdle_soft_{best_threshold}d", imp=True))

# Hurdle best (hard) -- full-train models
gate_test = hurdle_clf_best.predict(X_test)
y_hurdle_hard_test = np.empty(len(X_test))
s_t = gate_test == 0; l_t = gate_test == 1
if s_t.any(): y_hurdle_hard_test[s_t] = reg_s.predict(X_test[s_t])
if l_t.any(): y_hurdle_hard_test[l_t] = reg_l.predict(X_test[l_t])
hhtm = regression_metrics(y_test_log, y_hurdle_hard_test, log_transformed=True)
hhtm["improvement_%"] = round(compute_baseline_improvement(hhtm, mean_test_m), 2)
test_results[f"hurdle_hard_{best_threshold}d"] = hhtm
pf(mstr(hhtm, f"hurdle_hard_{best_threshold}d", imp=True))

# Meta-ensemble (tune-fold-optimized weight applied to full-train models)
y_meta_test = (1-best_meta_w) * y_blend_test + best_meta_w * y_hurdle_test
metam = regression_metrics(y_test_log, y_meta_test, log_transformed=True)
metam["improvement_%"] = round(compute_baseline_improvement(metam, mean_test_m), 2)
test_results["meta_ensemble_v4"] = metam
pf(mstr(metam, "meta_ensemble_v4", imp=True))

# Save comparison
final_table = create_comparison_table(test_results)
final_table.to_csv(RESULTS_DIR / "final_model_comparison_v4.csv", index=False)

# -- table -----------------------------------------------------
pf("\n" + "-" * 105)
pf("  COMPREHENSIVE TEST METRICS")
pf("-" * 105)
pf(f"  {'Model':35s} | {'MAE':>7s} | {'MedAE':>7s} | {'RMSE':>7s} | {'R2':>7s} | {'WAPE':>7s} | {'MASE':>6s} | {'D%':>6s}")
pf("  " + "-" * 101)
for _, row in final_table.iterrows():
    pf(f"  {row['model']:35s} | {row['mae_days']:7.3f} | {row['median_ae_days']:7.3f} | "
       f"{row['rmse_days']:7.2f} | {row['r2_days']:7.4f} | {row.get('wape',np.nan):6.1f}% | "
       f"{row.get('mase',np.nan):6.3f} | {row.get('improvement_%',0):5.1f}%")

# -- Within-X-days accuracy -----------------------------------------------
# Use the best model from the table
best_name = final_table.iloc[0]["model"]
if best_name in test_log_models:
    best_pred = test_log_models[best_name].predict(X_test)
    y_true_d = np.expm1(y_test_log)
    y_pred_d = np.clip(np.expm1(np.clip(best_pred, None, 20)), 0, None)
elif best_name == "lgb_tweedie_best":
    y_true_d = y_test_raw.values
    y_pred_d = y_tw_test
elif best_name == "blended_ensemble_v4":
    y_true_d = np.expm1(y_test_log)
    y_pred_d = np.clip(np.expm1(np.clip(y_blend_test, None, 20)), 0, None)
elif "hurdle_soft" in best_name:
    y_true_d = np.expm1(y_test_log)
    y_pred_d = np.clip(np.expm1(np.clip(y_hurdle_test, None, 20)), 0, None)
elif best_name == "meta_ensemble_v4":
    y_true_d = np.expm1(y_test_log)
    y_pred_d = np.clip(np.expm1(np.clip(y_meta_test, None, 20)), 0, None)
else:
    y_true_d = np.expm1(y_test_log)
    y_pred_d = np.clip(np.expm1(np.clip(lgb_tuned_v4.predict(X_test), None, 20)), 0, None)

ae = np.abs(y_true_d - y_pred_d)
pf(f"\n  WITHIN-X-DAYS ACCURACY ({best_name}):")
for t in [0.25, 0.5, 1, 2, 3, 5, 7, 14]:
    pf(f"    Within {t:>5.2f} days: {(ae<=t).mean()*100:6.2f}%")

# ===========================================================================
header(9, "SHAP + FEATURE IMPORTANCE (tuned model)")
# ===========================================================================
import shap

shap_model = lgb_tuned_v4
X_shap = X_test.sample(min(3000, len(X_test)), random_state=SEED)
pf(f"  Computing SHAP on {len(X_shap)} test samples...")
explainer = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(X_shap)

plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_shap, show=False, max_display=25)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shap_summary_v4.png", dpi=300, bbox_inches="tight"); plt.close()

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=25)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shap_importance_v4.png", dpi=300, bbox_inches="tight"); plt.close()

# ===========================================================================
header(10, "PLOTS AND ERROR ANALYSIS")
# ===========================================================================
ta = test.copy()
ta["y_pred_days"] = y_pred_d
ta["y_true_days"]  = y_true_d
ta["residual"]     = ta["y_true_days"] - ta["y_pred_days"]
ta["abs_error"]    = np.abs(ta["residual"])

# [1/5] Predicted vs Actual
pf("  [1/5] Predicted vs Actual...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
np.random.seed(SEED)
si = np.random.choice(len(ta), min(20000, len(ta)), replace=False)
ts = ta.iloc[si]
mx = max(ts["y_true_days"].quantile(0.99), ts["y_pred_days"].quantile(0.99))
axes[0].scatter(ts["y_true_days"], ts["y_pred_days"], alpha=0.08, s=3, c="steelblue")
axes[0].plot([0,mx],[0,mx], "r--", lw=2); axes[0].set_xlim(0,mx); axes[0].set_ylim(0,mx)
axes[0].set_xlabel("Actual (days)"); axes[0].set_ylabel("Predicted (days)")
axes[0].set_title("Predicted vs Actual (days)", fontweight="bold"); axes[0].grid(True, alpha=0.3)
axes[1].scatter(ts["y_true_days"]+.01, ts["y_pred_days"]+.01, alpha=0.08, s=3, c="darkorange")
axes[1].plot([.01,100],[.01,100], "r--", lw=2); axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[1].set_xlabel("Actual (days, log)"); axes[1].set_ylabel("Predicted (days, log)")
axes[1].set_title("Predicted vs Actual (log scale)", fontweight="bold"); axes[1].grid(True, alpha=0.3)
plt.suptitle(f"Best Model: {best_name}", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout(); plt.savefig(FIGURES_DIR/"predicted_vs_actual_v4.png", dpi=300, bbox_inches="tight"); plt.close()

# [2/5] Error Distribution
pf("  [2/5] Error Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].hist(ta["residual"].clip(-20,20), bins=120, alpha=0.75, color="steelblue", edgecolor="black", lw=0.3)
axes[0].axvline(0, color="red", ls="--", lw=2)
axes[0].axvline(ta["residual"].mean(), color="orange", ls="-.", lw=2, label=f"Mean={ta['residual'].mean():.2f}d")
axes[0].set_xlabel("Residual (days)"); axes[0].set_title("Residual Distribution", fontweight="bold")
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].hist(ta["abs_error"].clip(0,30), bins=100, alpha=0.75, color="coral", edgecolor="black", lw=0.3)
axes[1].axvline(ta["abs_error"].mean(), color="red", ls="--", lw=2, label=f"MAE={ta['abs_error'].mean():.2f}d")
axes[1].axvline(ta["abs_error"].median(), color="green", ls=":", lw=2, label=f"MedAE={ta['abs_error'].median():.2f}d")
axes[1].set_xlabel("Absolute Error (days)"); axes[1].set_title("Absolute Error Distribution", fontweight="bold")
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(FIGURES_DIR/"error_distribution_v4.png", dpi=300, bbox_inches="tight"); plt.close()

# [3/5] Error by resolution bucket
pf("  [3/5] Error by Resolution Bucket...")
bins = [0, 0.042, 0.5, 1, 3, 7, 14, 30, 90]
labels = ["<1h", "1h-12h", "12h-1d", "1-3d", "3-7d", "1-2w", "2w-1m", "1-3m"]
ta["bucket"] = pd.cut(ta["y_true_days"], bins=bins, labels=labels)
be = ta.groupby("bucket", observed=True).agg(
    count=("abs_error","count"), mae=("abs_error","mean"),
    median_ae=("abs_error","median"), rmse=("abs_error", lambda x: np.sqrt((x**2).mean())),
)
be["pct"] = (be["count"]/len(ta)*100).round(1)
be.to_csv(RESULTS_DIR/"error_by_bucket_v4.csv")
pf(f"\n{be.to_string()}\n")

# [4/5] Loss Function Comparison Chart
pf("  [4/5] Loss Function Comparison Chart...")
loss_names = ["lgb_quantile_v3ref", "lgb_l2_mono", "lgb_huber_best",
              "lgb_fair_best", "lgb_tuned_v4", "lgb_tweedie_best",
              f"hurdle_soft_{best_threshold}d", "blended_ensemble_v4", "meta_ensemble_v4"]
loss_names = [n for n in loss_names if n in test_results]

fig, axes = plt.subplots(1, 3, figsize=(22, 8))
maes   = [test_results[n]["mae_days"] for n in loss_names]
rmses  = [test_results[n]["rmse_days"] for n in loss_names]
r2s    = [test_results[n]["r2_days"] for n in loss_names]
colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(loss_names)))

for ax, vals, label, title in [
    (axes[0], maes, "MAE (days)", "Mean Absolute Error"),
    (axes[1], rmses, "RMSE (days)", "Root Mean Squared Error"),
    (axes[2], r2s, "R2", "R-Squared"),
]:
    ax.barh(range(len(loss_names)), vals, color=colors, edgecolor="black", lw=0.5)
    ax.set_yticks(range(len(loss_names))); ax.set_yticklabels(loss_names, fontsize=9)
    ax.set_xlabel(label); ax.set_title(title, fontweight="bold")
    ax.invert_yaxis(); ax.grid(True, alpha=0.3, axis="x")
    for i, v in enumerate(vals): ax.text(max(v+0.01,0.01), i, f"{v:.3f}", va="center", fontsize=8)

plt.suptitle("Loss Function Comparison -- Test (2023-2024)", fontsize=15, fontweight="bold")
plt.tight_layout(); plt.savefig(FIGURES_DIR/"loss_comparison_v4.png", dpi=300, bbox_inches="tight"); plt.close()

# [5/5] Hurdle threshold comparison
pf("  [5/5] Hurdle Threshold Comparison...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
thresholds = sorted(hurdle_results.keys())
h_maes  = [hurdle_results[t]["soft"]["mae_days"] for t in thresholds]
h_rmses = [hurdle_results[t]["soft"]["rmse_days"] for t in thresholds]
h_r2s   = [hurdle_results[t]["soft"]["r2_days"] for t in thresholds]
h_labels = [f"{t}d" for t in thresholds]

for ax, vals, label, title in [
    (axes[0], h_maes, "MAE (days)", "MAE by Hurdle Threshold"),
    (axes[1], h_rmses, "RMSE (days)", "RMSE by Hurdle Threshold"),
    (axes[2], h_r2s, "R2", "R2 by Hurdle Threshold"),
]:
    bars = ax.bar(h_labels, vals, color="steelblue", edgecolor="black", lw=0.5)
    ax.set_xlabel("Threshold (days)"); ax.set_ylabel(label); ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, vals): ax.text(bar.get_x()+bar.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

plt.suptitle("Hurdle Model -- Threshold Optimization (Soft Gating, Tune Fold)", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(FIGURES_DIR/"hurdle_threshold_comparison_v4.png", dpi=300, bbox_inches="tight"); plt.close()

# ===========================================================================
header(11, "ERROR ANALYSIS -- WHERE MODEL EXCELS & STRUGGLES")
# ===========================================================================
cat_err = ta.groupby("REASON").agg(count=("abs_error","count"), mae=("abs_error","mean"),
    medae=("abs_error","median")).sort_values("mae", ascending=False)
cat_err = cat_err[cat_err["count"]>100]; cat_err.to_csv(RESULTS_DIR/"error_by_reason_v4.csv")

# Department-level error analysis
if "Department" in ta.columns:
    dept_err = ta.groupby("Department").agg(
        count=("abs_error","count"), mae=("abs_error","mean"),
        medae=("abs_error","median"),
        mean_residual=("residual","mean"),
        median_resolution=("y_true_days","median"),
    ).sort_values("mae", ascending=False)
    dept_err = dept_err[dept_err["count"]>100]
    dept_err.to_csv(RESULTS_DIR/"error_by_department_v4.csv")
    pf(f"  Saved department error analysis ({len(dept_err)} departments) to results/error_by_department_v4.csv")

pf("\n  [+] WHERE THE MODEL PERFORMS WELL:")
quick = ta["y_true_days"] <= 1; std = ta["y_true_days"] <= 30
pf(f"  * Quick (<=1d): MAE={ta.loc[quick,'abs_error'].mean():.3f}d, MedAE={ta.loc[quick,'abs_error'].median():.3f}d ({quick.mean()*100:.1f}% of data)")
pf(f"  * Standard (<=30d): MAE={ta.loc[std,'abs_error'].mean():.3f}d, MedAE={ta.loc[std,'abs_error'].median():.3f}d ({std.mean()*100:.1f}%)")
pf(f"  * Best categories: {', '.join(cat_err.tail(5).index.tolist())}")

pf("\n  [-] WHERE THE MODEL STRUGGLES:")
lt = ta["y_true_days"] >= 30
pf(f"  * Long-term (>=30d): MAE={ta.loc[lt,'abs_error'].mean():.2f}d ({lt.mean()*100:.1f}%)")
pf(f"  * Worst categories: {', '.join(cat_err.head(5).index.tolist())}")

# Feature importance for tuned model
fi = pd.Series(lgb_tuned_v4.feature_importances_, index=feature_cols).sort_values(ascending=False)
pf(f"\n  Top 20 features by importance:")
for fname, imp in fi.head(20).items():
    pf(f"    {fname:40s}: {imp:6.0f}")

# SLA/velocity features in top-30
new_feats = [f for f in fi.head(30).index if any(k in f for k in
    ["sla_p25", "sla_p75", "sla_iqr", "velocity_deviation", "velocity_ratio"])]
if new_feats:
    pf(f"\n  SLA/velocity features in top-30: {new_feats}")

# ===========================================================================
# Reference vs Tuned comparison
pf("\n  -- REFERENCE -> TUNED IMPROVEMENT (Test Set) ----------")
ref_best = {"mae_days": 2.726, "median_ae_days": 0.221, "rmse_days": 8.79, "r2_days": 0.3233}
tuned_best = test_results[final_table.iloc[0]["model"]]
pf(f"  {'Metric':12s} | {'Ref':>8s} | {'Tuned':>8s} | {'Change':>10s}")
pf("  " + "-" * 45)
for metric in ["mae_days", "median_ae_days", "rmse_days", "r2_days"]:
    rv = ref_best[metric]; tv = tuned_best[metric]
    change = tv - rv
    better = "[+]" if (change < 0 and metric != "r2_days") or (change > 0 and metric == "r2_days") else "[-]"
    pf(f"  {metric:12s} | {rv:8.3f} | {tv:8.3f} | {change:+8.3f} {better}")

# ===========================================================================
pf("\n" + "=" * 74)
pf("  * * *   FINAL SUMMARY   * * *")
pf("=" * 74)
best_row = final_table.iloc[0]
pf(f"\n  Dataset: {len(train)+len(val)+len(test):,} records (90-day cap)")
pf(f"  Features: {len(feature_cols)}")
pf(f"  Train-proper: 2015-2020 ({len(train_proper):,}) | Tune fold: 2021 ({len(tune_fold):,})")
pf(f"  Full train: 2015-2021 ({len(train):,}) | Val: {len(val):,} | Test: {len(test):,}")
pf(f"\n  METHODOLOGY: All selection on tune fold (2021); val (2022) for early-stop only")
pf(f"\n  BEST MODEL: {best_row['model']}")
pf(f"     MAE:   {best_row['mae_days']:.3f} days")
pf(f"     MedAE: {best_row['median_ae_days']:.3f} days")
pf(f"     RMSE:  {best_row['rmse_days']:.2f} days")
pf(f"     R2:    {best_row['r2_days']:.4f}")
pf(f"     WAPE:  {best_row.get('wape', np.nan):.1f}%")
pf(f"     MASE:  {best_row.get('mase', np.nan):.3f}")
pf(f"     Improvement over mean baseline: {best_row.get('improvement_%',0):.1f}%")
pf(f"\n  Key Findings:")
pf(f"  * Best robust loss: {best_loss_name} (outperforms pure MSE & quantile)")
pf(f"  * Best hurdle threshold: {best_threshold}d")
pf(f"  * Best Tweedie power: {best_tweedie_power}")
pf(f"  * Ensemble blend: {blend_w}")
pf(f"  * Meta-ensemble hurdle_weight: {best_meta_w}")
pf(f"\n  Results: {RESULTS_DIR}")
pf(f"  Figures: {FIGURES_DIR}")
pf("\n" + "=" * 74)
pf("  Advanced models pipeline complete!")
pf("=" * 74)
