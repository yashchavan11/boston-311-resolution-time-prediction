# Note: LLMs were used in this section for report generation
"""
Generate PDF Report: Full Project Lifecycle

Creates a comprehensive PDF covering the entire project progression from
data collection through final model evaluation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd
import numpy as np
from fpdf import FPDF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

class ProjectReport(FPDF):
    def header(self):
        if self.page_no() == 1:
            return  # clean title page — no running header
        self.set_font("Helvetica", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "Boston 311 Resolution Time Prediction", align="L")
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 8, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(180, 180, 180)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def footer(self):
        if self.page_no() == 1:
            return  # no footer on title page
        self.set_y(-15)
        self.set_draw_color(180, 180, 180)
        self.set_line_width(0.2)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, f"{self.page_no()} / {{nb}}", align="C")

    def chapter_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 41, 102)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 71, 153)
        self.set_line_width(0.4)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(0, 41, 102)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)

        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(0, 71, 153)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()

        # Rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(0, 0, 0)
        for j, row in enumerate(rows):
            if j % 2 == 1:
                self.set_fill_color(230, 240, 255)
                fill = True
            else:
                self.set_fill_color(255, 255, 255)
                fill = True
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), border=1, fill=fill, align="C")
            self.ln()
        self.ln(3)

    def add_figure(self, path, caption="", width=170):
        if Path(path).exists():
            from PIL import Image as _Img
            _im = _Img.open(str(path))
            fig_h = width * (_im.size[1] / _im.size[0])  # mm
            needed = fig_h + 12  # figure + caption + spacing
            usable = 297 - self.get_y() - 20  # page height minus cursor minus bottom margin
            if needed > usable:
                self.add_page()
            x = (210 - width) / 2
            self.image(str(path), x=x, w=width)
            if caption:
                self.set_font("Helvetica", "I", 9)
                self.cell(0, 6, caption, align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(4)


def build_report():
    pdf = ProjectReport()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # -- Title Page --------------------------------------------------------
    pdf.add_page()

    # Top accent line
    pdf.set_draw_color(0, 71, 153)
    pdf.set_line_width(1.2)
    pdf.line(30, 55, 180, 55)

    # Title block
    pdf.set_y(62)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(0, 41, 102)
    pdf.multi_cell(0, 13, "Predicting Resolution Time\nfor City Service Requests\nUsing Machine Learning", align="C")

    # Subtitle / dataset line
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "Boston 311 Dataset  |  2015 - 2024  |  2.5 M Records", align="C",
             new_x="LMARGIN", new_y="NEXT")

    # Thin separator
    pdf.ln(8)
    pdf.set_draw_color(180, 180, 180)
    pdf.set_line_width(0.4)
    pdf.line(70, pdf.get_y(), 140, pdf.get_y())

    # Author block
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "Yash Chavan", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 7, "M.S. in Artificial Intelligence", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Northeastern University", align="C", new_x="LMARGIN", new_y="NEXT")

    # Key results box
    pdf.ln(18)
    box_y = pdf.get_y()
    box_h = 40
    pdf.set_fill_color(245, 249, 255)
    pdf.set_draw_color(0, 71, 153)
    pdf.set_line_width(0.6)
    pdf.rect(30, box_y, 150, box_h, style="DF")

    pdf.set_xy(30, box_y + 4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(0, 41, 102)
    pdf.cell(150, 6, "KEY RESULTS", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(30, 30, 30)
    results_lines = [
        "Best Model: CatBoost GPU (MAE loss)  --  MAE = 2.669 days",
        "33.3% improvement over mean baseline  |  24.1% over SARIMA",
        "21 configurations across 6 model families  |  99 features  |  1.8 M records",
    ]
    for line in results_lines:
        pdf.set_x(30)
        pdf.cell(150, 6, line, align="C", new_x="LMARGIN", new_y="NEXT")

    # Bottom accent line
    pdf.set_draw_color(0, 71, 153)
    pdf.set_line_width(1.2)
    pdf.line(30, 272, 180, 272)

    # -- 1. Introduction --------------------------------------------------
    pdf.add_page()
    pdf.chapter_title("1. Introduction")
    pdf.body_text(
        "City 311 systems receive millions of non-emergency service requests annually, "
        "covering issues such as potholes, broken streetlights, graffiti removal, and sanitation. "
        "Accurately predicting how long each request will take to resolve can help city agencies "
        "allocate resources more efficiently and provide citizens with realistic expectations.\n\n"
        "This project builds a machine learning pipeline to predict resolution time for Boston 311 "
        "service requests using only information available at the time the request is created. "
        "A strict creation-time feature boundary ensures predictions can be made immediately "
        "for each incoming request. Prior work by Raj et al. (2021) -- the SWIFT system -- "
        "addressed the same domain but forecasted weekly aggregate resolution times using "
        "sparse Gaussian CRFs. This project extends the problem to per-request prediction "
        "using request-level ML features.\n\n"
        "The project evaluates multiple modeling stages: statistical baselines, linear models, "
        "gradient boosting with default and tuned hyperparameters, robust loss functions, "
        "ensemble methods, and comparisons against time-series (ARIMA/SARIMA) approaches."
    )

    # -- 2. Dataset --------------------------------------------------------
    pdf.chapter_title("2. Dataset")

    # Load dataset summary
    try:
        with open(RESULTS_DIR / "dataset_summary_v4.json") as f:
            ds = json.load(f)
    except FileNotFoundError:
        ds = {}

    pdf.body_text(
        f"Source: City of Boston Open Data Portal (data.boston.gov)\n"
        f"Time range: 2015-2024 (10 years)\n"
        f"Raw records: 2,550,536\n"
        f"After filtering: {ds.get('total_count', '1,823,856'):,} "
        f"(closed cases, 0-90 day cap, excluding administrative closures)"
    )

    pdf.section_title("Data Filtering")
    pdf.add_table(
        ["Stage", "Records", "Dropped"],
        [
            ["Raw data", "2,550,536", "--"],
            ["Closed cases only", "2,348,040", "202,496"],
            ["Valid dates", "2,348,038", "2"],
            ["Non-negative resolution", "2,347,246", "792"],
            ["90-day cap", "2,217,370", "129,876"],
            ["Remove admin closures", "1,823,856", "393,514"],
        ],
        col_widths=[70, 50, 50]
    )

    pdf.section_title("Temporal Split")
    pdf.add_table(
        ["Split", "Years", "Records", "Percentage"],
        [
            ["Train", "2015-2021", "1,190,355", "65.3%"],
            ["Validation", "2022", "224,371", "12.3%"],
            ["Test", "2023-2024", "409,130", "22.4%"],
        ],
        col_widths=[35, 50, 50, 40]
    )
    pdf.body_text(
        "For advanced model selection (Step 4 onward), the training block is further split "
        "internally: 2015-2020 serves as train-proper for fitting candidate models, and 2021 "
        "is held out as a tune fold for all selection decisions (loss function sweeps, Optuna "
        "hyperparameter tuning, ensemble weight optimization, and hurdle threshold selection). "
        "The validation set (2022) is used only for early-stopping callbacks, never for model "
        "selection, preventing validation leakage. After selection, final models are retrained "
        "on the full training block (2015-2021) with 2022 for early stopping before test evaluation."
    )

    pdf.section_title("Target Distribution")
    pdf.body_text(
        f"Mean resolution: {ds.get('mean_resolution_days', 4.50):.2f} days\n"
        f"Median resolution: {ds.get('median_resolution_days', 0.47):.2f} days\n"
        f"Skewness: {ds.get('skewness', 4.23):.2f}\n"
        f"64% of requests resolve within 1 day, creating a heavily right-skewed distribution."
    )

    # EDA figures — smart placement (add_figure handles page breaks)
    for fig_name, caption in [
        ("eda_resolution_distribution.png", "Figure 1: Resolution time distribution"),
        ("eda_top_categories.png", "Figure 2: Top request categories"),
        ("eda_department_analysis.png", "Figure 3: Department analysis"),
    ]:
        fig_path = FIGURES_DIR / fig_name
        if fig_path.exists():
            pdf.add_figure(fig_path, caption)

    # -- 3. Feature Engineering --------------------------------------------
    pdf.add_page()
    pdf.chapter_title("3. Feature Engineering (99 Features)")
    pdf.body_text(
        "All features respect the creation-time feature boundary. "
        "Historical aggregates and velocity features use only training data or "
        "appropriately lagged values to prevent data leakage."
    )
    pdf.add_table(
        ["Category", "Count", "Key Examples"],
        [
            ["Temporal", "27", "hour, day_of_week, year, cyclical, time-of-day"],
            ["Categorical (enc.)", "26", "Target + frequency encoding"],
            ["Geographic", "5", "Lat/Lon, zipcode, dist. from center"],
            ["Historical aggs.", "15", "Mean/median/std by TYPE, REASON"],
            ["Workload proxies", "2", "Prev day/week request counts"],
            ["District backlog", "2", "Open cases (1-day lag)"],
            ["Dept. velocity", "6", "Rolling 7/14/30-day averages"],
            ["Interaction", "8", "type x dept, velocity x backlog"],
            ["SLA baseline", "6", "P25/P75/IQR per TYPE, REASON"],
            ["Velocity deviation", "2", "Current vs historical speed"],
        ],
        col_widths=[45, 20, 115]
    )

    pdf.section_title("Leakage Prevention")
    pdf.body_text(
        "- District backlog: Uses 1-day lag on case closures\n"
        "- Workload proxies: Previous day/week counts only\n"
        "- Historical features: Computed from training data only\n"
        "- Target encoding: Fitted on training set, applied to val/test\n"
        "- Velocity features: Rolling averages from training period"
    )

    # -- 4. Model Progression ---------------------------------------------
    pdf.add_page()
    pdf.chapter_title("4. Full Model Progression")
    pdf.body_text(
        "All models were trained on the same data pipeline (99 features, 90-day cap, "
        "temporal split) and evaluated on the held-out test set (2023-2024, 409,130 records). "
        "This ensures a fair, apples-to-apples comparison across all model families.\n\n"
        "Steps 1-3 (baselines, linear models, default gradient boosting) train on the full "
        "training block (2015-2021) with 2022 for early stopping, since no hyperparameter "
        "selection is involved. Step 4 (advanced LightGBM) uses the nested train-proper / "
        "tune fold split described in Section 2 to make all selection decisions without "
        "touching the validation set."
    )

    # Step 1: Statistical Baselines
    pdf.section_title("Step 1: Statistical Baselines")
    pdf.add_table(
        ["Model", "MAE (days)", "MedAE", "RMSE", "R2", "Improv."],
        [
            ["Mean baseline", "4.002", "1.431", "10.89", "-0.039", "0.0%"],
            ["Median baseline", "3.577", "0.537", "11.10", "-0.080", "10.6%"],
        ],
        col_widths=[45, 28, 25, 25, 25, 25]
    )

    # Step 2: Linear Baselines
    pdf.section_title("Step 2: Linear Models")
    pdf.add_table(
        ["Model", "MAE (days)", "MedAE", "RMSE", "R2", "Improv."],
        [
            ["Linear Regression", "2.994", "0.302", "9.53", "0.204", "25.2%"],
            ["Ridge", "2.994", "0.302", "9.53", "0.204", "25.2%"],
            ["Lasso", "2.987", "0.299", "9.49", "0.210", "25.4%"],
            ["Decision Tree", "2.933", "0.257", "9.25", "0.250", "26.7%"],
        ],
        col_widths=[45, 28, 25, 25, 25, 25]
    )
    pdf.body_text("Even simple linear models achieve ~25% improvement, showing the features carry strong signal.")

    # Step 3: Gradient Boosting
    pdf.section_title("Step 3: Gradient Boosting (Default Params)")
    pdf.add_table(
        ["Model", "MAE (days)", "MedAE", "RMSE", "R2", "Improv."],
        [
            ["Random Forest", "2.831", "0.265", "9.00", "0.290", "29.3%"],
            ["XGBoost (CUDA)", "2.779", "0.261", "8.91", "0.304", "30.6%"],
            ["CatBoost (GPU)", "2.757", "0.268", "8.90", "0.305", "31.1%"],
        ],
        col_widths=[45, 28, 25, 25, 25, 25]
    )
    pdf.body_text("Gradient boosting models gain 4-6% over linear models. CatBoost slightly leads.")

    # Step 4: Advanced LightGBM
    pdf.add_page()
    pdf.section_title("Step 4: Advanced LightGBM (Robust Loss Functions)")
    pdf.add_table(
        ["Model", "MAE (days)", "MedAE", "RMSE", "R2", "Improv."],
        [
            ["LGB Tuned (Optuna)", "2.696", "0.256", "8.73", "0.332", "32.6%"],
            ["LGB Huber", "2.705", "0.261", "8.74", "0.330", "32.4%"],
            ["LGB Quantile", "2.737", "0.219", "8.79", "0.322", "31.6%"],
            ["LGB Fair", "2.749", "0.262", "8.89", "0.308", "31.3%"],
            ["Hurdle Soft (0.5d)", "2.754", "0.256", "8.97", "0.294", "31.2%"],
            ["LGB L2 + Monotonic", "2.765", "0.270", "8.91", "0.305", "30.9%"],
            ["Hurdle Hard (0.5d)", "2.812", "0.216", "8.63", "0.348", "29.7%"],
            ["Blended Ensemble*", "2.749", "0.262", "8.89", "0.308", "31.3%"],
            ["LGB Tweedie", "3.106", "0.297", "8.45", "0.374", "22.4%"],
        ],
        col_widths=[45, 28, 25, 25, 25, 25]
    )
    pdf.body_text(
        "LGB Tuned (Optuna, Fair loss) achieves the best single-model LightGBM MAE. "
        "The SLSQP-optimized blended ensemble collapsed to 100% weight on lgb_fair, "
        "so the 'blended ensemble' row is effectively equivalent to LGB Fair. "
        "The meta-ensemble (blend + hurdle) similarly assigned 0% hurdle weight, "
        "making it also identical to LGB Fair."
    )

    # -- 5. Literature Comparison ------------------------------------------
    pdf.add_page()
    pdf.chapter_title("5. Literature Comparison (ARIMA/SARIMA)")
    pdf.body_text(
        "SWIFT (Raj et al., 2021) compared sparse GCRFs against Linear Regression, ARIMA, "
        "and SARIMA baselines for weekly aggregate forecasting. To situate the per-request "
        "approach against similar time-series methods, ARIMA and SARIMA models are trained on "
        "weekly aggregated resolution times and each test request is assigned its period forecast, "
        "enabling a direct per-request MAE comparison."
    )
    pdf.add_table(
        ["Model", "MAE (days)", "RMSE", "MedAE"],
        [
            ["Best ML (LGB Tuned)", "2.696", "8.73", "0.256"],
            ["Linear Regression", "2.994", "9.53", "0.302"],
            ["SARIMA(1,1,1)(1,1,1,52)", "3.551", "11.15", "0.387"],
            ["ARIMA(2,1,2) weekly", "3.554", "11.14", "0.413"],
        ],
        col_widths=[55, 40, 40, 40]
    )
    pdf.body_text(
        "ML advantage over SARIMA: 24.1% lower MAE (LGB Tuned vs SARIMA). The best overall "
        "model (CatBoost GPU, MAE=2.669d) achieves 24.8% improvement over SARIMA. Per-request "
        "ML prediction with creation-time features substantially outperforms aggregate "
        "time-series forecasting."
    )

    fig_path = FIGURES_DIR / "arima_comparison.png"
    if fig_path.exists():
        pdf.add_figure(fig_path, "Figure 4: ML vs ARIMA/SARIMA comparison")

    # -- 6. Improvement Experiments ----------------------------------------
    pdf.add_page()
    pdf.chapter_title("6. Late-Stage Improvement Experiments")
    pdf.body_text(
        "After reaching the best LightGBM (MAE~2.696 days), this work explored whether "
        "alternative algorithms or techniques could push performance further. A frozen "
        "Huber/Quantile blend baseline (~2.704 days) served as the comparison target for "
        "each experiment. CatBoost and XGBoost experiments used GPU acceleration; "
        "LightGBM-based experiments ran on CPU."
    )
    pdf.add_table(
        ["Experiment", "MAE (days)", "Beats baseline?", "Device"],
        [
            ["CatBoost MAE loss", "2.669", "Yes (-1.3%)", "GPU"],
            ["XGBoost abs. error", "2.695", "Yes (-0.3%)", "CUDA"],
            ["Recency weighting", "2.706", "No", "CPU"],
            ["Deeper Optuna (50)", "2.730", "No", "CPU"],
            ["Recency (decay=2.0)", "2.728", "No", "CPU"],
        ],
        col_widths=[55, 35, 40, 35]
    )
    pdf.body_text(
        "CatBoost with MAE loss achieves the best overall MAE of 2.669 days, beating the "
        "blended ensemble baseline (~2.704 days). Most experiments fail to beat the baseline, "
        "confirming the performance plateau is real and data-driven, not a tuning limitation."
    )

    pdf.section_title("Walk-Forward Validation")
    pdf.body_text(
        "Walk-forward validation retrains a LightGBM Huber model (delta=0.5, default "
        "hyperparameters) on expanding windows to assess temporal stability. Each year's "
        "model is trained on all preceding years without a separate validation set."
    )
    pdf.add_table(
        ["Test Year", "MAE (days)", "R-squared", "Train Size"],
        [
            ["2022", "2.426", "0.310", "1,190,355"],
            ["2023", "2.653", "0.359", "1,414,726"],
            ["2024", "2.662", "0.330", "1,617,961"],
        ],
        col_widths=[40, 40, 40, 50]
    )
    pdf.body_text(
        "Performance is stable across years with no evidence of catastrophic temporal drift. "
        "Note: these results use a standard LightGBM Huber model, not the final tuned model "
        "or CatBoost, so they represent a conservative stability check."
    )

    # -- 7. Error Analysis -------------------------------------------------
    pdf.add_page()
    pdf.chapter_title("7. Error Analysis")
    pdf.body_text(
        "Error analysis and SHAP explanations are generated from the best LightGBM model "
        "(lgb_tuned_v4, MAE=2.696d) rather than the CatBoost improvement experiment, because "
        "the LightGBM pipeline produces the full suite of diagnostic artifacts (SHAP values, "
        "error breakdowns, predicted-vs-actual plots) during Step 7."
    )

    for fig_name, caption in [
        ("predicted_vs_actual_v4.png", "Figure 5: Predicted vs actual (lgb_tuned_v4)"),
        ("error_distribution_v4.png", "Figure 6: Error distribution (lgb_tuned_v4)"),
        ("shap_importance_v4.png", "Figure 7: SHAP feature importance (lgb_tuned_v4)"),
    ]:
        fig_path = FIGURES_DIR / fig_name
        if fig_path.exists():
            pdf.add_figure(fig_path, caption, width=150)

    # -- 8. Why Improvement is Difficult -----------------------------------
    pdf.add_page()
    pdf.chapter_title("8. Why Further Improvement is Difficult")
    pdf.body_text(
        "1. Heavy-tailed target distribution: 64% of requests resolve within 1 day, "
        "but the remaining 36% span up to 90 days. The long tail dominates MAE/RMSE.\n\n"
        "2. Inherent operational noise: Resolution time depends on staffing, weather, "
        "equipment availability, and political priorities -- none of which are in the data.\n\n"
        "3. Same-category variance: Requests of the same TYPE and Department can resolve "
        "in hours or weeks depending on severity, which is not captured in structured fields.\n\n"
        "4. Log retransformation bias: Training on log1p(target) introduces systematic bias "
        "when converting predictions back to days.\n\n"
        "5. Metric sensitivity: R-squared is computed against the mean, so the extremely "
        "skewed distribution makes it structurally low even with good predictions."
    )

    # -- 9. Development History -----------------------------------------------
    pdf.add_page()
    pdf.chapter_title("9. Development History")
    pdf.body_text(
        "The project was developed iteratively. The final repository contains a single "
        "active pipeline producing the authoritative results reported throughout this "
        "document. Key milestones during development:"
    )
    pdf.add_table(
        ["Stage", "Features", "What Changed", "Test MAE"],
        [
            ["Early", "~65", "Initial features (had data-leakage issues)", "N/A*"],
            ["Intermediate", "~83", "Leakage-free rewrites + interaction features", "2.726"],
            ["Final pipeline", "99", "SLA baselines, velocity deviation, Optuna tuning", "2.696"],
            ["+ CatBoost MAE", "99", "CatBoost GPU with MAE loss (improvement exp.)", "2.669"],
        ],
        col_widths=[35, 22, 85, 28]
    )
    pdf.body_text(
        "* Early experiments used per-split rolling and backlog functions that had subtle "
        "data leakage. These were replaced with leakage-free implementations that compute "
        "features across all splits chronologically with appropriate lag constraints.\n\n"
        "The final best result -- CatBoost GPU with MAE loss (2.669 days) -- comes from a "
        "late-stage improvement experiment (Step 9) using the same 99-feature pipeline."
    )

    # -- 10. Conclusions ---------------------------------------------------
    pdf.chapter_title("10. Conclusions")
    pdf.body_text(
        "This project demonstrates that machine learning can meaningfully predict "
        "Boston 311 service request resolution times, achieving a 33.3% improvement "
        "over a naive mean baseline and 24.1% improvement over SARIMA forecasting.\n\n"
        "Key contributions:\n"
        "- Comprehensive model comparison across 21 configurations and 6 model families\n"
        "- Strict creation-time feature boundary ensuring operational deployability\n"
        "- 99 engineered features including novel velocity deviation and SLA baseline features\n"
        "- Honest assessment of the performance plateau and its data-driven causes\n"
        "- Walk-forward validation (LightGBM Huber) confirming temporal stability\n"
        "- GPU-accelerated training on NVIDIA RTX 5070 Ti (CUDA/CatBoost GPU)\n\n"
        "The best model (CatBoost GPU with MAE loss) achieves:\n"
        "- MAE: 2.669 days (33.3% improvement over mean baseline)\n\n"
        "Walk-forward validation using a LightGBM Huber model shows stable MAE across "
        "2022-2024 (2.43-2.66 days), confirming the feature pipeline generalizes temporally.\n\n"
        "The best tuned LightGBM (MAE = 2.696 days) is only 1.0% behind CatBoost, "
        "confirming the performance plateau is inherent to the data, not a modeling "
        "limitation. Further gains would require richer data sources (free-text "
        "descriptions, images, real-time staffing data) rather than algorithmic changes."
    )

    pdf.section_title("References")
    pdf.body_text(
        "[1] R. Raj, A. Ramesh, A. Seetharam, and D. DeFazio, \"SWIFT: A non-emergency "
        "response prediction system using sparse Gaussian Conditional Random Fields,\" "
        "Pervasive Mob. Comput., vol. 71, p. 101317, Feb. 2021."
    )

    # -- Save --------------------------------------------------------------
    output_path = DOCS_DIR / "Project_Report.pdf"
    pdf.output(str(output_path))
    print(f"Report saved to: {output_path}")
    print(f"Pages: {pdf.page_no()}")


if __name__ == "__main__":
    build_report()
