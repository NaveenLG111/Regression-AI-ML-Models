import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ==============================================================================
# 1. Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="AI Forecast Studio | Amazon & Google",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Anchor paths relative to current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "Output", "google_reviews_forecast_best_model.pkl"),
    os.path.join(BASE_DIR, "Output", "google_reviews_forecast_model.pkl"),
    os.path.join(BASE_DIR, "output", "google_reviews_forecast_best_model.pkl"),
    os.path.join(BASE_DIR, "output", "google_reviews_forecast_model.pkl"),
    os.path.join(BASE_DIR, "google_reviews_forecast_best_model.pkl"),
    os.path.join(BASE_DIR, "google_reviews_forecast_model.pkl"),
    os.path.join(BASE_DIR, "Models", "Amazon", "google_reviews_forecast_model.pkl"),
    os.path.join(BASE_DIR, "model.pkl"),
]

FEATURE_CANDIDATES = [
    os.path.join(BASE_DIR, "Output", "google_reviews_feature_columns.pkl"),
    os.path.join(BASE_DIR, "output", "google_reviews_feature_columns.pkl"),
    os.path.join(BASE_DIR, "google_reviews_feature_columns.pkl"),
    os.path.join(BASE_DIR, "Models", "Amazon", "google_reviews_feature_columns.pkl"),
]

REPORT_CANDIDATES = [
    os.path.join(BASE_DIR, "Output", "ml_model_report_optimized.xlsx"),
    os.path.join(BASE_DIR, "output", "ml_model_report_optimized.xlsx"),
    os.path.join(BASE_DIR, "Output", "ml_model_report.xlsx"),
    os.path.join(BASE_DIR, "output", "ml_model_report.xlsx"),
    os.path.join(BASE_DIR, "ml_model_report_optimized.xlsx"),
    os.path.join(BASE_DIR, "ml_model_report.xlsx"),
]

PLOTS_DIR_CANDIDATES = [
    os.path.join(BASE_DIR, "Output", "Plots"),
    os.path.join(BASE_DIR, "output", "plots"),
    os.path.join(BASE_DIR, "Output", "plots"),
    os.path.join(BASE_DIR, "Plots"),
    os.path.join(BASE_DIR, "plots"),
]

# ==============================================================================
# 2. Resource Loaders with Caching
# ==============================================================================
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    model_path = None
    feature_path = None

    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            model_path = path
            break

    for path in FEATURE_CANDIDATES:
        if os.path.exists(path):
            feature_path = path
            break

    model = None
    feature_cols = None

    if model_path and os.path.exists(model_path):
        model = joblib.load(model_path)
    if feature_path and os.path.exists(feature_path):
        feature_cols = joblib.load(feature_path)

    return model, feature_cols, model_path, feature_path

@st.cache_data(show_spinner="Reading analytics report...")
def load_excel_report():
    report_path = None
    for path in REPORT_CANDIDATES:
        if os.path.exists(path):
            report_path = path
            break

    if not report_path:
        return None, None

    try:
        excel_data = pd.read_excel(report_path, sheet_name=None)
        return excel_data, report_path
    except Exception as e:
        st.error(f"Error loading report: {e}")
        return None, None

def get_plots_directory():
    for pdir in PLOTS_DIR_CANDIDATES:
        if os.path.exists(pdir):
            return pdir
    return None

# Load global variables
model, feature_cols, resolved_model_path, resolved_feature_path = load_artifacts()
excel_sheets, resolved_report_path = load_excel_report()
plots_dir = get_plots_directory()

# ==============================================================================
# 3. Feature Engineering Transformation for Single / Batch Input
# ==============================================================================
def transform_inference_data(df_input: pd.DataFrame, feature_cols: list) -> tuple:
    data = df_input.copy()

    # Standardize column naming
    data.columns = [c.strip().lower() for c in data.columns]

    if "ds" not in data.columns or "y" not in data.columns:
        raise ValueError("Input dataset must contain at least 'ds' (datetime) and 'y' (target) columns.")

    data["ds"] = pd.to_datetime(data["ds"], errors="coerce")
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    data = data.dropna(subset=["ds", "y"])

    id_col = "unique_id" if "unique_id" in data.columns else None
    if id_col:
        data[id_col] = data[id_col].astype(str)
        data = data.sort_values([id_col, "ds"]).reset_index(drop=True)
    else:
        data = data.sort_values("ds").reset_index(drop=True)

    # Calendar features
    dt = data["ds"].dt
    data["hour"] = dt.hour
    data["day"] = dt.day
    data["dayofweek"] = dt.dayofweek
    data["dayofyear"] = dt.dayofyear
    data["weekofyear"] = dt.isocalendar().week.astype(int)
    data["month"] = dt.month
    data["quarter"] = dt.quarter
    data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(np.int8)

    # Cyclical representations
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24.0)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24.0)
    data["dow_sin"] = np.sin(2 * np.pi * data["dayofweek"] / 7.0)
    data["dow_cos"] = np.cos(2 * np.pi * data["dayofweek"] / 7.0)
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12.0)

    # Grouped base
    grouped = data.groupby(id_col)["y"] if id_col else data["y"]

    # Explicit Lags
    lag_intervals = [1, 2, 3, 6, 12, 24, 48, 72]
    for lag in lag_intervals:
        data[f"lag_{lag}"] = grouped.shift(lag)

    # Momentum
    data["diff_lag1_lag2"] = data["lag_1"] - data["lag_2"]
    data["diff_lag1_lag24"] = data["lag_1"] - data["lag_24"]

    # Rolling calculations
    shifted_target = grouped.shift(1)
    roll_base = shifted_target.groupby(data[id_col]) if id_col else shifted_target
    roll_windows = [3, 6, 12, 24]

    for w in roll_windows:
        data[f"rolling_mean_{w}"] = roll_base.transform(lambda s: s.rolling(w, min_periods=w).mean())
        data[f"rolling_std_{w}"] = roll_base.transform(lambda s: s.rolling(w, min_periods=w).std())
        data[f"rolling_min_{w}"] = roll_base.transform(lambda s: s.rolling(w, min_periods=w).min())
        data[f"rolling_max_{w}"] = roll_base.transform(lambda s: s.rolling(w, min_periods=w).max())

    eps = 1e-6
    data["ratio_lag1_rollmean24"] = data["lag_1"] / (data["rolling_mean_24"] + eps)

    if id_col:
        data["expanding_group_mean"] = roll_base.transform(lambda s: s.expanding(min_periods=24).mean())

    # Backfill or drop early window NaNs for inference
    data = data.bfill().ffill()

    # Enforce strictly the saved feature schema
    X = pd.DataFrame(index=data.index)
    for col in feature_cols:
        if col in data.columns:
            X[col] = data[col]
        else:
            X[col] = 0.0

    if id_col and id_col in X.columns:
        X[id_col] = X[id_col].astype("category")

    return data, X

# ==============================================================================
# 4. UI Sidebar Navigation
# ==============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
st.sidebar.title("Navigation Hub")
app_mode = st.sidebar.radio(
    "Select Interface Mode:",
    [
        "📊 Executive Dashboard",
        "⚡ Single Prediction Simulator",
        "📂 Batch Inference Engine",
        "📑 Model Artifacts & Diagnostics"
    ]
)

# Status indicators in sidebar
st.sidebar.markdown("---")
st.sidebar.caption("System Telemetry")
if model is not None:
    st.sidebar.success(f"Model: Loaded\n({os.path.basename(resolved_model_path)})")
else:
    st.sidebar.error("Model: Not Found in Output/")

if excel_sheets is not None:
    st.sidebar.success("Analytics Cache: Active")
else:
    st.sidebar.warning("Report: Missing in Output/")

# ==============================================================================
# View 1: Executive Dashboard
# ==============================================================================
if app_mode == "📊 Executive Dashboard":
    st.title("📈 Demand Forecasting & Review Metrics Dashboard")
    st.markdown("Real-time telemetry and validation performance from the latest automated model run.")

    if excel_sheets is not None:
        summary_key = None
        for key in ["Model_Summary", "Model_Metrics", "Test_Metrics"]:
            if key in excel_sheets:
                summary_key = key
                break

        if summary_key:
            summary_df = excel_sheets[summary_key]
            
            # Accommodate column-oriented or row-oriented metrics tables
            if "Selected_Best_Model" in summary_df.columns:
                model_name = summary_df["Selected_Best_Model"].iloc[0]
                mae_val = float(summary_df["MAE"].iloc[0])
                rmse_val = float(summary_df["RMSE"].iloc[0])
                r2_val = float(summary_df["R2_Score"].iloc[0])
            elif "metric" in summary_df.columns:
                metrics_dict = dict(zip(summary_df["metric"], summary_df["value"]))
                model_name = "LightGBM / Regressor"
                mae_val = float(metrics_dict.get("MAE", 0.0))
                rmse_val = float(metrics_dict.get("RMSE", 0.0))
                r2_val = float(metrics_dict.get("R2", 0.0))
            else:
                model_name, mae_val, rmse_val, r2_val = "N/A", 0.0, 0.0, 0.0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Production Model", str(model_name))
            c2.metric("Mean Absolute Error (MAE)", f"{mae_val:.3f}", delta="-Optimal", delta_color="inverse")
            c3.metric("Root Mean Sq. Error (RMSE)", f"{rmse_val:.3f}", delta="-Low", delta_color="inverse")
            c4.metric("Variance Explained (R²)", f"{r2_val * 100:.2f}%", delta="+High")
        else:
            st.info("Metrics summary table not found in the Excel report.")
    else:
        st.info("No cached training summary found. Please run the training script to populate metrics.")

    st.markdown("---")
    col_left, col_right = st.columns([6, 4])

    with col_left:
        st.subheader("Model Tournament Benchmark")
        tourney_key = None
        for key in ["Tournament_Comparison", "CV_Model_Comparison"]:
            if excel_sheets and key in excel_sheets:
                tourney_key = key
                break

        if tourney_key:
            tourney_df = excel_sheets[tourney_key]
            y_col = "Test_RMSE" if "Test_RMSE" in tourney_df.columns else "CV_RMSE_Mean"
            color_col = "Test_R2" if "Test_R2" in tourney_df.columns else "CV_R2_Mean"
            
            fig_tourney = px.bar(
                tourney_df,
                x="Model",
                y=y_col,
                color=color_col,
                text=y_col,
                title="Model Benchmark Comparison (RMSE)",
                color_continuous_scale="Blues_r"
            )
            fig_tourney.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig_tourney, use_container_width=True)
        else:
            st.info("Tournament comparison sheet not available.")

    with col_right:
        st.subheader("Top Predictive Drivers")
        if excel_sheets and "Feature_Importance" in excel_sheets:
            fi_df = excel_sheets["Feature_Importance"].head(10)
            imp_col = "Importance_Gain" if "Importance_Gain" in fi_df.columns else "importance"
            feat_col = "Feature" if "Feature" in fi_df.columns else "feature"

            fig_fi = px.bar(
                fi_df,
                x=imp_col,
                y=feat_col,
                orientation='h',
                title="Top 10 Feature Importance",
                color=imp_col,
                color_continuous_scale="Viridis"
            )
            fig_fi.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importance data not found in report.")

    st.subheader("Out-of-Sample Predictions vs Actual")
    pred_key = None
    for key in ["Predictions_Sample", "Actual_vs_Predicted"]:
        if excel_sheets and key in excel_sheets:
            pred_key = key
            break

    if pred_key:
        sample_preds = excel_sheets[pred_key].head(200)
        actual_col = "Actual" if "Actual" in sample_preds.columns else "actual"
        predicted_col = "Predicted" if "Predicted" in sample_preds.columns else "predicted"

        fig_actual_vs_pred = go.Figure()
        fig_actual_vs_pred.add_trace(go.Scatter(
            x=sample_preds["ds"], y=sample_preds[actual_col], mode="lines", name="Actual Target (y)", line=dict(color="#1f77b4", width=2)
        ))
        fig_actual_vs_pred.add_trace(go.Scatter(
            x=sample_preds["ds"], y=sample_preds[predicted_col], mode="lines", name="Model Predicted", line=dict(color="#ff7f0e", dash="dash", width=2)
        ))
        fig_actual_vs_pred.update_layout(
            xaxis_title="Timeline", yaxis_title="Review Volume / Target", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_actual_vs_pred, use_container_width=True)

# ==============================================================================
# View 2: Single Prediction Simulator
# ==============================================================================
elif app_mode == "⚡ Single Prediction Simulator":
    st.title("⚡ Interactive Target Forecast Simulator")
    st.markdown("Manually parameterize feature vectors to simulate the model's instant point prediction.")

    if model is None or feature_cols is None:
        st.error("Model artifacts not loaded. Ensure training has run and saved artifacts into `Output/`.")
        st.stop()

    with st.form("simulation_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            in_ds = st.date_input("Forecast Date", value=pd.to_datetime("today"))
            in_hour = st.slider("Hour of Day (0-23)", 0, 23, 14)
            in_unique_id = st.text_input("Entity ID (unique_id)", "store_loc_101")
        with col2:
            in_lag1 = st.number_input("Immediate Prior Period Target (lag_1)", min_value=0.0, value=250.0, step=1.0)
            in_lag2 = st.number_input("Two Periods Prior Target (lag_2)", min_value=0.0, value=242.0, step=1.0)
            in_lag24 = st.number_input("24-Period Lag Target (lag_24)", min_value=0.0, value=210.0, step=1.0)
        with col3:
            in_roll24 = st.number_input("24-Period Rolling Mean Target", min_value=0.1, value=230.0, step=1.0)
            in_rollstd24 = st.number_input("24-Period Rolling Standard Dev", min_value=0.0, value=18.5, step=0.5)

        submit_btn = st.form_submit_button("⚡ Compute Point Forecast", use_container_width=True)

    if submit_btn:
        synthetic_row = pd.DataFrame([{
            "ds": pd.to_datetime(f"{in_ds} {in_hour:02d}:00:00"),
            "y": in_lag1,
            "unique_id": in_unique_id
        }])

        try:
            _, X_sim = transform_inference_data(synthetic_row, feature_cols)
            
            # Inject inputs into feature frame where matching columns exist
            if "lag_1" in X_sim.columns: X_sim["lag_1"] = in_lag1
            if "lag_2" in X_sim.columns: X_sim["lag_2"] = in_lag2
            if "lag_24" in X_sim.columns: X_sim["lag_24"] = in_lag24
            if "diff_lag1_lag2" in X_sim.columns: X_sim["diff_lag1_lag2"] = in_lag1 - in_lag2
            if "diff_lag1_lag24" in X_sim.columns: X_sim["diff_lag1_lag24"] = in_lag1 - in_lag24
            if "rolling_mean_24" in X_sim.columns: X_sim["rolling_mean_24"] = in_roll24
            if "rolling_std_24" in X_sim.columns: X_sim["rolling_std_24"] = in_rollstd24
            if "ratio_lag1_rollmean24" in X_sim.columns: X_sim["ratio_lag1_rollmean24"] = in_lag1 / (in_roll24 + 1e-6)

            prediction = float(model.predict(X_sim)[0])

            st.success("### Forecast Generated Successfully")
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Target Level", f"{prediction:.2f}")
            m2.metric("Simulated Momentum (Lag1 - Lag2)", f"{in_lag1 - in_lag2:.2f}")
            m3.metric("Ratio vs 24h Mean", f"{in_lag1 / (in_roll24 + 1e-6):.2f}x")
        except Exception as err:
            st.error(f"Inference simulation failed: {err}")

# ==============================================================================
# View 3: Batch Inference Engine (Modified with Session State & Robust Download)
# ==============================================================================

elif app_mode == "📂 Batch Inference Engine":
    st.title("📂 Batch Data Inference & Validation")
    st.markdown("Upload raw time-series CSV data to execute end-to-end feature pipelines and download complete predictions.")

    if model is None or feature_cols is None:
        st.error("Model artifacts not loaded. Please train the model and save artifacts into `Output/` first.")
        st.stop()

    # Initialize session state for batch results persistence
    if "batch_results_df" not in st.session_state:
        st.session_state["batch_results_df"] = None
    if "batch_filename" not in st.session_state:
        st.session_state["batch_filename"] = ""

    uploaded_file = st.file_uploader(
        "Upload CSV dataset (Requires `ds`, `y`, optional `unique_id`)", 
        type=["csv"],
        key="batch_csv_uploader"
    )

    if uploaded_file is not None:
        # Reset state if a new file is uploaded
        if st.session_state["batch_filename"] != uploaded_file.name:
            st.session_state["batch_results_df"] = None
            st.session_state["batch_filename"] = uploaded_file.name

        try:
            batch_raw = pd.read_csv(uploaded_file)
            st.write(f"**Preview of uploaded data** (`{uploaded_file.name}` - {len(batch_raw):,} total rows):")
            st.dataframe(batch_raw.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
            st.stop()

        # Execution trigger
        if st.button("🚀 Run Batch Prediction Pipeline", type="primary"):
            with st.spinner("Processing temporal transforms and calculating predictions..."):
                try:
                    # Run feature engineering and alignment
                    df_transformed, X_batch = transform_inference_data(batch_raw, feature_cols)
                    
                    # Compute predictions
                    preds = model.predict(X_batch)
                    
                    # Assemble comprehensive results dataframe
                    df_transformed["Predicted_Target"] = np.round(preds, 4)
                    df_transformed["Residual_Error"] = np.round(df_transformed["y"] - df_transformed["Predicted_Target"], 4)
                    df_transformed["Absolute_Percentage_Error_%"] = np.round(
                        np.abs(df_transformed["Residual_Error"] / (df_transformed["y"] + 1e-6)) * 100, 2
                    )

                    # Persist to session state so download button does not disappear on click
                    st.session_state["batch_results_df"] = df_transformed
                    st.success(f"Successfully processed {len(df_transformed):,} rows!")
                except Exception as ex:
                    st.error(f"Batch pipeline execution failed: {ex}")

    # Display results and provide download if predictions exist in session state
    if st.session_state.get("batch_results_df") is not None:
        results_df = st.session_state["batch_results_df"]

        st.markdown("---")
        st.subheader("📊 Prediction Results Overview")

        # Key validation metrics summary
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Rows Evaluated", f"{len(results_df):,}")
        m2.metric("Batch Mean Actual (y)", f"{results_df['y'].mean():.2f}")
        m3.metric("Batch Mean Predicted", f"{results_df['Predicted_Target'].mean():.2f}")
        m4.metric("Mean Absolute Error (MAE)", f"{np.abs(results_df['Residual_Error']).mean():.2f}")

        # Choose specific output columns for preview
        preview_cols = ["ds", "y", "Predicted_Target", "Residual_Error", "Absolute_Percentage_Error_%"]
        if "unique_id" in results_df.columns:
            preview_cols.insert(1, "unique_id")
            
        st.write("**Top 100 Preview Table:**")
        st.dataframe(results_df[preview_cols].head(100), use_container_width=True)

        # Download handling: persistent and robust
        st.markdown("### 📥 Download Complete Results")
        col_dl1, col_dl2 = st.columns([2, 2])

        with col_dl1:
            # Download full feature matrix + predictions
            csv_full = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📄 Download Full Dataset (All Features + Predictions)",
                data=csv_full,
                file_name=f"complete_predictions_{st.session_state['batch_filename']}",
                mime="text/csv",
                use_container_width=True
            )

        with col_dl2:
            # Download concise metrics table (ds, unique_id, actual, predicted, error)
            csv_concise = results_df[preview_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📑 Download Summary CSV (Predictions & Errors Only)",
                data=csv_concise,
                file_name=f"summary_predictions_{st.session_state['batch_filename']}",
                mime="text/csv",
                use_container_width=True
            )

# ==============================================================================
# View 4: Artifact Diagnostics
# ==============================================================================
elif app_mode == "📑 Model Artifacts & Diagnostics":
    st.title("📑 Diagnostic Plots and Artifact Viewer")
    st.markdown("Static plots generated during the model verification and training pipeline.")

    if plots_dir and os.path.exists(plots_dir):
        plot_files = sorted([f for f in os.listdir(plots_dir) if f.endswith(".png")])
        if plot_files:
            selected_plot = st.selectbox("Select Visual Diagnostic:", plot_files)
            st.image(os.path.join(plots_dir, selected_plot), use_container_width=True)
        else:
            st.warning(f"No plot images found in `{plots_dir}`.")
    else:
        st.warning("Plots folder not found. Run the training script to generate visual plots.")

    st.markdown("---")
    st.subheader("Feature Dictionary Loaded in Pipeline")
    if feature_cols:
        st.write(pd.DataFrame({"Feature Name": feature_cols, "Column Index": list(range(len(feature_cols)))}))