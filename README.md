# Predicting Resolution Time for City Service Requests Using Machine Learning (Boston 311)

**Author:** Yash Chavan, M.S. in Artificial Intelligence, Northeastern University

## Problem Statement

This project predicts how long a newly created Boston 311 service request will take to be resolved (in days), using only information available at the time of request creation. The 311 system handles non-emergency city service requests such as potholes, broken streetlights, and sanitation issues. Accurate resolution time predictions can help city agencies plan resource allocation and set realistic citizen expectations.

## Key Results

| Metric | Value |
|--------|-------|
| **Best Model** | CatBoost GPU (MAE loss) |
| **MAE** | 2.669 days |
| **Best Tuned LightGBM** | Fair loss, Optuna-tuned, MAE = 2.696 days |
| **Improvement over mean baseline** | 33.3% (CatBoost GPU) / 32.6% (Tuned LightGBM) |
| **Improvement over SARIMA** | 24.8% (CatBoost) / 24.1% (Tuned LightGBM) |
| **Models compared** | 22 configurations across 6 model families |

## Dataset

- **Source:** City of Boston Open Data Portal (data.boston.gov)
- **Time range:** 2015-2024 (10 years)
- **Raw records:** 2,550,536
- **After filtering:** 1,823,856 (closed cases, 0-90 day cap, excluding administrative closures)
- **Target:** `resolution_days` = (CLOSED_DT - OPEN_DT) in days
- **Target characteristics:** Heavily right-skewed (mean=4.50d, median=0.47d, skewness=4.23)

### Data Filtering

| Stage | Records | Dropped |
|-------|---------|---------|
| Raw data | 2,550,536 | -- |
| Closed cases only | 2,348,040 | 202,496 |
| Valid dates | 2,348,038 | 2 |
| Non-negative resolution | 2,347,246 | 792 |
| 90-day cap | 2,217,370 | 129,876 |
| Remove administrative closures | 1,823,856 | 393,514 |

### Data Splits (Temporal)

| Split | Years | Records | Percentage |
|-------|-------|---------|------------|
| Train | 2015-2021 | 1,190,355 | 65.3% |
| Validation | 2022 | 224,371 | 12.3% |
| Test | 2023-2024 | 409,130 | 22.4% |

> **Nested split for advanced model selection (Step 4):** Within the training block, 2021 is held out as a *tune fold* for all selection decisions (loss function sweeps, Optuna hyperparameter tuning, ensemble weight optimization, hurdle threshold selection). Train-proper (2015–2020) is used to fit candidates; the tune fold (2021) is used to compare them. The validation set (2022) is used **only** for early-stopping callbacks, never for model selection, preventing validation leakage. After selection, final models are retrained on the full training block (2015–2021) with 2022 for early stopping before test evaluation.

## Feature Engineering (99 Features)

All features respect a strict **creation-time feature boundary** -- only information available when a request is created is used. This ensures predictions can be made immediately for each incoming request. Unlike SWIFT [1], which forecasts weekly aggregate resolution times using historical time-series, this project predicts resolution time at the individual request level using per-request ML features.

| Category | Count | Examples |
|----------|-------|----------|
| Temporal | 27 | hour, day_of_week, year, cyclical encodings, is_weekend, is_holiday, season, time-of-day buckets, business_hours |
| Categorical (encoded) | 26 | Target encoding + frequency encoding for TYPE, REASON, Department, neighborhood |
| Geographic | 5 | Lat/Lon, zipcode, distance from city center, has_coordinates |
| Historical aggregates | 15 | Mean/median/std resolution by TYPE, REASON, Department (from training data only) |
| Workload proxies | 2 | Previous day/week request counts (1-day lag) |
| District backlog | 2 | Open cases per neighborhood/department (1-day lag on closes) |
| Department velocity | 6 | Rolling 7/14/30-day average resolution by TYPE and Department |
| Interaction features | 8 | type x dept, type x neighborhood, velocity x backlog, backlog ratio |
| SLA baseline | 6 | P25/P75/IQR resolution time per TYPE and REASON |
| Velocity deviation | 2 | Current vs historical velocity, velocity ratio |

## Full Model Progression (Test Set: 2023-2024)

### Step 1: Statistical Baselines

| Model | MAE (days) | R-squared | Improvement |
|-------|-----------|-----------|-------------|
| Mean baseline | 4.002 | -0.039 | 0.0% |
| Median baseline | 3.577 | -0.080 | 10.6% |

### Step 2: Linear Models

| Model | MAE (days) | R-squared | Improvement |
|-------|-----------|-----------|-------------|
| Linear Regression | 2.994 | 0.204 | 25.2% |
| Ridge | 2.994 | 0.204 | 25.2% |
| Lasso | 2.987 | 0.210 | 25.4% |
| Decision Tree | 2.933 | 0.250 | 26.7% |

### Step 3: Gradient Boosting (Default Hyperparameters)

| Model | MAE (days) | R-squared | Improvement |
|-------|-----------|-----------|-------------|
| Random Forest | 2.831 | 0.290 | 29.3% |
| XGBoost (CUDA) | 2.779 | 0.304 | 30.6% |
| CatBoost (GPU) | 2.757 | 0.305 | 31.1% |

### Step 4: Advanced LightGBM (Robust Loss Functions + Ensemble)

| Model | MAE (days) | R-squared | Improvement |
|-------|-----------|-----------|-------------|
| LGB Tuned (Optuna, Fair) | 2.696 | 0.332 | 32.6% |
| LGB Huber (delta=0.5) | 2.705 | 0.330 | 32.4% |
| LGB Quantile (alpha=0.5) | 2.737 | 0.322 | 31.6% |
| LGB Fair | 2.749 | 0.308 | 31.3% |
| Hurdle Soft (0.5d) | 2.754 | 0.294 | 31.2% |
| LGB L2 + Monotonic | 2.765 | 0.305 | 30.9% |
| Hurdle Hard (0.5d) | 2.812 | 0.348 | 29.7% |
| Blended Ensemble* | 2.749 | 0.308 | 31.3% |
| LGB Tweedie | 3.106 | 0.374 | 22.4% |

*\*The SLSQP-optimized blended ensemble collapsed to 100% weight on lgb_fair, so this row is effectively equivalent to LGB Fair. The meta-ensemble (blend + hurdle) similarly assigned 0% hurdle weight, making it also identical to LGB Fair.*

### Step 5: ARIMA/SARIMA Literature Comparison

| Model | MAE (days) | RMSE |
|-------|-----------|------|
| **Best ML (Tuned LightGBM)** | **2.696** | **8.73** |
| SARIMA(1,1,1)(1,1,1,52) | 3.551 | 11.15 |
| ARIMA(2,1,2) weekly | 3.554 | 11.14 |

**ML advantage over SARIMA: 24.1% lower MAE (LGB Tuned) / 24.8% (CatBoost MAE).**

### Step 6: Improvement Experiments

| Experiment | MAE (days) | Beats baseline? | Device |
|-----------|-----------|-----------|--------|
| **CatBoost MAE loss** | **2.669** | **Yes (-1.3%)** | **GPU** |
| XGBoost absolute error | 2.695 | Yes (-0.3%) | CUDA |
| Recency weighting | 2.706 | No | CPU |
| Deeper Optuna (50 trials) | 2.730 | No | CPU |

CatBoost with MAE loss achieves the overall best MAE of 2.669d (33.3% improvement over mean baseline). Most other experiments fail to beat the blended ensemble baseline (~2.704d), confirming the performance plateau is data-driven.

### Walk-Forward Validation (Temporal Stability)

| Test Year | MAE (days) | R-squared | Train Size |
|-----------|-----------|-----------|------------|
| 2022 | 2.426 | 0.310 | 1,190,355 |
| 2023 | 2.653 | 0.359 | 1,414,726 |
| 2024 | 2.662 | 0.330 | 1,617,961 |

## Why Further Improvement is Difficult

1. **Heavy-tailed target distribution:** 64% of requests resolve within 1 day, but the remaining 36% span up to 90 days.
2. **Inherent operational noise:** Resolution time depends on staffing, weather, equipment availability -- none in the data.
3. **Same-category variance:** Same TYPE and Department can resolve in hours or weeks depending on severity.
4. **Log retransformation bias:** Training on log1p(target) introduces systematic bias when converting back to days.
5. **Metric sensitivity:** R-squared is structurally low due to extreme target skewness.

## How to Run

Scripts are numbered in execution order. Each step saves its outputs for the next step.
Pipeline scripts in `scripts/` are **orchestration-only** -- all reusable logic lives in `src/`. Steps 5-9 load saved feature parquets from Step 4 via `src.models.load_feature_data()`.

```bash
# Setup
python -m venv venv
venv/Scripts/activate          # Windows
pip install -r requirements.txt

# Sequential pipeline
python scripts/01_data_collection.py          # Download raw data
python scripts/02_preprocessing.py            # Clean, filter, split
python scripts/03_eda.py                      # EDA plots
python scripts/04_feature_engineering.py      # Build 99 features
python scripts/05_baseline_models.py          # Mean, Median, LR, Ridge, Lasso, DT
python scripts/06_intermediate_models.py      # Random Forest, XGBoost, CatBoost
python scripts/07_advanced_models.py          # LightGBM tuning, robust losses, ensemble
python scripts/08_arima_comparison.py         # ARIMA/SARIMA comparison
python scripts/09_improvement_experiments.py  # CatBoost GPU, walk-forward, etc.
python scripts/10_generate_report.py          # PDF report
```

## Project Structure

```
.
|-- README.md                       # Project documentation
|-- requirements.txt                # Python dependencies
|-- .gitignore
|-- configs/
|   |-- model_configs.yaml          # Data and model configuration
|-- docs/
|   |-- paper.tex                  # Manuscript (LaTeX)
|   |-- references.bib             # Bibliography (21 references)
|   |-- paper.pdf                  # Compiled manuscript (10 pages)
|   |-- figures/                   # Figures used in paper
|   |   |-- arima_comparison.png
|   |   |-- eda_resolution_distribution.png
|   |   |-- predicted_vs_actual_v4.png
|   |   |-- shap_importance_v4.png
|   |   |-- walkforward_stability.png
|   |-- Project Proposal.pdf       # Academic proposal
|   |-- Lit_Review.pdf             # Literature review
|   |-- Project_Report.pdf         # Generated PDF report
|-- src/                            # Shared source modules
|   |-- __init__.py
|   |-- data_loader.py              # Data downloading and loading
|   |-- preprocessing.py            # Cleaning, filtering, splitting
|   |-- features.py                 # Feature engineering (99 features + monotonic constraints)
|   |-- models.py                   # Model definitions + data loading utilities
|   |-- evaluation.py               # Metrics computation
|   |-- utils.py                    # Paths, config, seed, plotting
|-- scripts/                        # Orchestration pipeline scripts
|   |-- 01_data_collection.py       # Step 1: Download 311 data
|   |-- 02_preprocessing.py         # Step 2: Clean, filter, temporal split
|   |-- 03_eda.py                   # Step 3: Exploratory data analysis
|   |-- 04_feature_engineering.py   # Step 4: Build 99 features
|   |-- 05_baseline_models.py       # Step 5: Statistical + linear baselines
|   |-- 06_intermediate_models.py   # Step 6: Gradient boosting (default)
|   |-- 07_advanced_models.py       # Step 7: Robust losses, tuning, ensemble
|   |-- 08_arima_comparison.py      # Step 8: ARIMA/SARIMA comparison
|   |-- 09_improvement_experiments.py # Step 9: CatBoost GPU, walk-forward
|   |-- 10_generate_report.py       # Step 10: PDF report generation
|-- data/raw/                       # Raw CSV data (2015-2024)
|-- data/processed/                 # Processed features
|-- models/                         # Saved model files
|-- results/                        # Result tables (CSV)
|-- figures/                        # Publication-quality figures
```

## References

[1] R. Raj, A. Ramesh, A. Seetharam, and D. DeFazio, "SWIFT: A non-emergency response prediction system using sparse Gaussian Conditional Random Fields," *Pervasive Mob. Comput.*, vol. 71, p. 101317, Feb. 2021.
