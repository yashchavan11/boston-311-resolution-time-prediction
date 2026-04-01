
"""
Step 3: Exploratory Data Analysis
==================================
Generates EDA plots from the preprocessed data.

Plots generated:
  1. Resolution time distribution (histogram + log scale)
  2. Resolution CDF (cumulative distribution)
  3. Top request categories by volume
  4. Category vs resolution time (box plots)
  5. Department analysis
  6. Temporal patterns (hourly/daily/monthly)
  7. Yearly trends
  8. Seasonal resolution patterns
  9. Neighborhood analysis
  10. Source channel distribution

Requires: Step 2 (preprocessed data in data/processed/)
Output: figures/eda_*.png (10 plots)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

from src.utils import load_config, set_seed, DATA_PROCESSED, FIGURES_DIR, setup_plotting

config = load_config()
set_seed(config["random_seed"])
setup_plotting()

print("=" * 70)
print("  STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Load preprocessed data
print("\nLoading preprocessed data...")
train = pd.read_parquet(DATA_PROCESSED / "train_clean.parquet")
val = pd.read_parquet(DATA_PROCESSED / "val_clean.parquet")
test = pd.read_parquet(DATA_PROCESSED / "test_clean.parquet")
df = pd.concat([train, val, test], ignore_index=True)
print(f"  Total records: {len(df):,}")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -- Plot 1: Resolution time distribution ----------------------------------
print("\n[1/10] Resolution time distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["resolution_days"], bins=100, color="steelblue", edgecolor="none", alpha=0.8)
axes[0].set_xlabel("Resolution Time (days)")
axes[0].set_ylabel("Count")
axes[0].set_title("Resolution Time Distribution")
axes[0].axvline(df["resolution_days"].mean(), color="red", linestyle="--", label=f'Mean: {df["resolution_days"].mean():.1f}d')
axes[0].axvline(df["resolution_days"].median(), color="orange", linestyle="--", label=f'Median: {df["resolution_days"].median():.2f}d')
axes[0].legend()
axes[1].hist(np.log1p(df["resolution_days"]), bins=100, color="steelblue", edgecolor="none", alpha=0.8)
axes[1].set_xlabel("log(1 + Resolution Days)")
axes[1].set_ylabel("Count")
axes[1].set_title("Log-Transformed Distribution")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "eda_resolution_distribution.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# -- Plot 2: CDF ----------------------------------------------------------
print("[2/10] Resolution CDF...")
fig, ax = plt.subplots(figsize=(10, 6))
sorted_days = np.sort(df["resolution_days"])
cdf = np.arange(1, len(sorted_days) + 1) / len(sorted_days)
ax.plot(sorted_days, cdf, color="steelblue", linewidth=2)
for threshold in [1, 3, 7, 14, 30]:
    pct = (df["resolution_days"] <= threshold).mean() * 100
    ax.axvline(threshold, color="gray", linestyle=":", alpha=0.5)
    ax.text(threshold + 0.5, 0.05, f"{threshold}d: {pct:.0f}%", fontsize=9, rotation=90)
ax.set_xlabel("Resolution Time (days)")
ax.set_ylabel("Cumulative Proportion")
ax.set_title("Cumulative Distribution of Resolution Time")
ax.grid(True, alpha=0.3)
fig.savefig(FIGURES_DIR / "eda_resolution_cdf.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# -- Plot 3: Top categories -----------------------------------------------
print("[3/10] Top request categories...")
if "TYPE" in df.columns:
    fig, ax = plt.subplots(figsize=(12, 8))
    top_types = df["TYPE"].value_counts().head(20)
    top_types.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Number of Requests")
    ax.set_title("Top 20 Request Types by Volume")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_top_categories.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# -- Plot 4: Category vs resolution ---------------------------------------
print("[4/10] Category vs resolution time...")
if "TYPE" in df.columns:
    fig, ax = plt.subplots(figsize=(14, 8))
    top10 = df["TYPE"].value_counts().head(10).index
    plot_data = df[df["TYPE"].isin(top10)]
    order = plot_data.groupby("TYPE")["resolution_days"].median().sort_values().index
    sns.boxplot(data=plot_data, y="TYPE", x="resolution_days", order=order, ax=ax,
                showfliers=False, palette="viridis")
    ax.set_xlabel("Resolution Time (days)")
    ax.set_title("Resolution Time by Top 10 Request Types")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_category_resolution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# -- Plot 5: Department analysis ------------------------------------------
print("[5/10] Department analysis...")
if "Department" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    dept_counts = df["Department"].value_counts().head(15)
    dept_counts.plot(kind="barh", ax=axes[0], color="steelblue")
    axes[0].set_xlabel("Number of Requests")
    axes[0].set_title("Requests by Department")
    axes[0].invert_yaxis()
    top_depts = df["Department"].value_counts().head(10).index
    dept_data = df[df["Department"].isin(top_depts)]
    dept_order = dept_data.groupby("Department")["resolution_days"].median().sort_values().index
    sns.boxplot(data=dept_data, y="Department", x="resolution_days", order=dept_order,
                ax=axes[1], showfliers=False, palette="coolwarm")
    axes[1].set_xlabel("Resolution Time (days)")
    axes[1].set_title("Resolution by Department")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_department_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# -- Plot 6: Temporal patterns --------------------------------------------
print("[6/10] Temporal patterns...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
if "OPEN_DT" in df.columns:
    hourly = df.groupby(df["OPEN_DT"].dt.hour)["resolution_days"].median()
    axes[0].plot(hourly.index, hourly.values, marker="o", color="steelblue")
    axes[0].set_xlabel("Hour of Day"); axes[0].set_title("Median Resolution by Hour")
    axes[0].set_ylabel("Median Resolution (days)")
    daily = df.groupby(df["OPEN_DT"].dt.dayofweek)["resolution_days"].median()
    axes[1].bar(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], daily.values, color="steelblue")
    axes[1].set_title("Median Resolution by Day of Week")
    axes[1].set_ylabel("Median Resolution (days)")
    monthly = df.groupby(df["OPEN_DT"].dt.month)["resolution_days"].median()
    axes[2].plot(monthly.index, monthly.values, marker="s", color="steelblue")
    axes[2].set_xlabel("Month"); axes[2].set_title("Median Resolution by Month")
    axes[2].set_ylabel("Median Resolution (days)")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "eda_temporal_patterns.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# -- Plot 7: Yearly trends ------------------------------------------------
print("[7/10] Yearly trends...")
if "OPEN_DT" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    yearly_vol = df.groupby(df["OPEN_DT"].dt.year).size()
    yearly_vol.plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_xlabel("Year"); axes[0].set_ylabel("Requests"); axes[0].set_title("Annual Request Volume")
    axes[0].tick_params(axis="x", rotation=45)
    yearly_med = df.groupby(df["OPEN_DT"].dt.year)["resolution_days"].median()
    yearly_med.plot(kind="bar", ax=axes[1], color="coral")
    axes[1].set_xlabel("Year"); axes[1].set_ylabel("Median Resolution (days)"); axes[1].set_title("Annual Median Resolution Time")
    axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_yearly_trends.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# -- Plot 8: Seasonal patterns --------------------------------------------
print("[8/10] Seasonal patterns...")
if "OPEN_DT" in df.columns:
    fig, ax = plt.subplots(figsize=(12, 6))
    df["_month"] = df["OPEN_DT"].dt.month
    seasonal = df.groupby("_month")["resolution_days"].agg(["mean", "median", "std"])
    ax.fill_between(seasonal.index, seasonal["mean"] - seasonal["std"],
                     seasonal["mean"] + seasonal["std"], alpha=0.2, color="steelblue")
    ax.plot(seasonal.index, seasonal["mean"], marker="o", color="steelblue", label="Mean")
    ax.plot(seasonal.index, seasonal["median"], marker="s", color="coral", label="Median")
    ax.set_xlabel("Month"); ax.set_ylabel("Resolution Time (days)")
    ax.set_title("Seasonal Resolution Patterns"); ax.legend(); ax.set_xticks(range(1, 13))
    df.drop(columns=["_month"], inplace=True)
    fig.savefig(FIGURES_DIR / "eda_seasonal_resolution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# -- Plot 9: Neighborhood analysis ----------------------------------------
print("[9/10] Neighborhood analysis...")
if "neighborhood" in df.columns:
    fig, ax = plt.subplots(figsize=(12, 10))
    nbhd_med = df.groupby("neighborhood")["resolution_days"].median().sort_values(ascending=False).head(20)
    nbhd_med.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Median Resolution Time (days)")
    ax.set_title("Top 20 Neighborhoods by Median Resolution Time")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_neighborhood_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# -- Plot 10: Source distribution ------------------------------------------
print("[10/10] Source distribution...")
if "Source" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    source_counts = df["Source"].value_counts()
    source_counts.plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_xlabel("Source"); axes[0].set_ylabel("Count"); axes[0].set_title("Requests by Source Channel")
    axes[0].tick_params(axis="x", rotation=45)
    source_res = df.groupby("Source")["resolution_days"].median().sort_values()
    source_res.plot(kind="bar", ax=axes[1], color="coral")
    axes[1].set_xlabel("Source"); axes[1].set_ylabel("Median Resolution (days)"); axes[1].set_title("Resolution by Source")
    axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_source_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

print(f"\nAll EDA plots saved to figures/")
print("Step 3 complete.")
