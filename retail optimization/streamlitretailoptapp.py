import io
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================================================
# 1. PAGE SETUP & STYLING
# =========================================================
st.set_page_config(
    page_title="Retail Price Optimization Engine",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# 2. DATA DISCOVERY & LOADING HELPERS
# =========================================================
def locate_local_data():
    """Attempts to find the dataset using standard relative directory paths."""
    script_dir = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
    project_dir = script_dir.parent

    candidate_names = ["retail_price.csv", "retail.csv", "retail_prices.csv"]
    candidate_folders = [
        script_dir / "Data",
        script_dir,
        project_dir / "Data",
        project_dir,
    ]

    for folder in candidate_folders:
        for name in candidate_names:
            target = folder / name
            if target.is_file():
                return target

    # Fallback recursive search
    for name in candidate_names:
        matches = list(script_dir.glob(f"**/{name}")) or list(project_dir.glob(f"**/{name}"))
        if matches:
            return matches[0]

    return None

# =========================================================
# 3. PIPELINE & MODEL ENGINE (CACHED)
# =========================================================
@st.cache_resource(show_spinner="Training Ridge Regression Pipeline & Tuning Alpha...")
def train_model(data_df: pd.DataFrame, test_size: float = 0.2, cv_folds: int = 5):
    target_col = "unit_price"
    drop_cols = [target_col, "month_year", "product_id"]

    X = data_df.drop(columns=[col for col in drop_cols if col in data_df.columns])
    y = data_df[target_col]

    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", Ridge())
        ]
    )

    param_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_folds,
        scoring="r2",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_model = best_pipeline.named_steps["model"]
    fitted_prep = best_pipeline.named_steps["prep"]

    # Evaluation
    y_train_pred = best_pipeline.predict(X_train)
    y_test_pred = best_pipeline.predict(X_test)

    metrics_df = pd.DataFrame({
        "Metric": ["R2 Score", "RMSE", "MAE"],
        "Train": [
            r2_score(y_train, y_train_pred),
            np.sqrt(mean_squared_error(y_train, y_train_pred)),
            mean_absolute_error(y_train, y_train_pred)
        ],
        "Test": [
            r2_score(y_test, y_test_pred),
            np.sqrt(mean_squared_error(y_test, y_test_pred)),
            mean_absolute_error(y_test, y_test_pred)
        ]
    })

    test_preds_df = pd.DataFrame({
        "Actual_Unit_Price": y_test.values,
        "Predicted_Unit_Price": y_test_pred,
        "Residual": y_test.values - y_test_pred
    }, index=y_test.index)

    # Feature Importance & Equation
    encoded_cat_features = (
        fitted_prep.named_transformers_["cat"]
        .get_feature_names_out(cat_features)
        .tolist()
    )
    all_feature_names = num_features + encoded_cat_features
    coefficients = best_model.coef_
    intercept = best_model.intercept_

    coefs_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Coefficient": coefficients,
        "Abs_Coefficient": np.abs(coefficients)
    }).sort_values(by="Abs_Coefficient", ascending=False).drop(columns=["Abs_Coefficient"])

    # String formula
    eq_terms = [f"{intercept:.4f}"]
    for feat, coef in zip(all_feature_names, coefficients):
        sign = "+" if coef >= 0 else "-"
        eq_terms.append(f"{sign} ({abs(coef):.4f} * {feat})")
    final_equation = "unit_price = " + " ".join(eq_terms)

    return {
        "pipeline": best_pipeline,
        "best_params": grid_search.best_params_,      # Key without underscore
        "best_params_": grid_search.best_params_,     # Provided both to prevent KeyErrors
        "metrics_df": metrics_df,
        "test_preds_df": test_preds_df,
        "coefs_df": coefs_df,
        "final_equation": final_equation,
        "intercept": intercept,
        "num_features": num_features,
        "cat_features": cat_features,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_pred": y_test_pred
    }

def generate_excel_bytes(metrics_df, test_preds_df, coefs_df, final_equation):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="Model_Metrics", index=False)
        test_preds_df.to_excel(writer, sheet_name="Test_Predictions", index=True)
        coefs_df.to_excel(writer, sheet_name="Coefficients", index=False)
        pd.DataFrame({"Model_Equation": [final_equation]}).to_excel(
            writer, sheet_name="Final_Equation", index=False
        )
    return output.getvalue()

# =========================================================
# 4. SIDEBAR CONFIGURATION
# =========================================================
with st.sidebar:
    st.title("⚙️ Model Settings")
    st.markdown("Configure data sources and pipeline parameters.")

    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    test_ratio = st.slider("Test Size Ratio", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
    cv_k = st.slider("Cross-Validation Folds", min_value=3, max_value=10, value=5, step=1)

    st.markdown("---")
    st.caption("Engine: Scikit-Learn Ridge Regression (L2 Regularized)")

# Determine DataFrame
df_data = None
source_name = ""

if uploaded_file is not None:
    df_data = pd.read_csv(uploaded_file)
    source_name = f"Uploaded File ({uploaded_file.name})"
else:
    local_path = locate_local_data()
    if local_path and local_path.is_file():
        df_data = pd.read_csv(local_path)
        source_name = f"Local File ({local_path.name})"

if df_data is None:
    st.warning("⚠️ No dataset detected! Please upload `retail_price.csv` via the sidebar to begin.")
    st.stop()

# Validate Schema
if "unit_price" not in df_data.columns:
    st.error("❌ The dataset must contain the target column: `unit_price`.")
    st.stop()

# Run or Retrieve Model Training
artifacts = train_model(df_data, test_size=test_ratio, cv_folds=cv_k)

# =========================================================
# 5. MAIN APPLICATION LAYOUT
# =========================================================
st.title("🏷️ Retail Price Optimization & Regression Suite")
st.caption(f"Active Dataset Source: **{source_name}** | Observations: **{len(df_data):,}** rows | Features: **{df_data.shape[1] - 1}**")

tabs = st.tabs([
    "📊 Performance & Metrics",
    "📈 Diagnostic Visualizations",
    "📐 Equation & Weights",
    "🎯 Real-time Price Predictor",
    "📥 Export Center"
])

# ---------------------------------------------------------
# TAB 1: METRICS & PREVIEW
# ---------------------------------------------------------
with tabs[0]:
    st.subheader("Model Performance Summary")

    test_r2 = artifacts["metrics_df"].loc[artifacts["metrics_df"]["Metric"] == "R2 Score", "Test"].values[0]
    test_rmse = artifacts["metrics_df"].loc[artifacts["metrics_df"]["Metric"] == "RMSE", "Test"].values[0]
    test_mae = artifacts["metrics_df"].loc[artifacts["metrics_df"]["Metric"] == "MAE", "Test"].values[0]
    
    # Safely retrieve optimal alpha regardless of dictionary key naming
    best_params = artifacts.get("best_params") or artifacts.get("best_params_", {})
    best_a = best_params.get("model__alpha", "N/A")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Test R² Score", f"{test_r2:.4f}")
    with c2:
        st.metric("Test RMSE", f"${test_rmse:.2f}")
    with c3:
        st.metric("Test MAE", f"${test_mae:.2f}")
    with c4:
        st.metric("Optimal Alpha", f"{best_a}")

    st.markdown("### Detailed Scorecard")
    st.dataframe(
        artifacts["metrics_df"].style.format({"Train": "{:.4f}", "Test": "{:.4f}"}),
        use_container_width=True
    )

    with st.expander("🔍 Preview Raw Dataset (First 10 Rows)"):
        st.dataframe(df_data.head(10), use_container_width=True)

# ---------------------------------------------------------
# TAB 2: VISUALIZATIONS
# ---------------------------------------------------------
with tabs[1]:
    st.subheader("Regression Diagnostics")
    sns.set_theme(style="whitegrid")

    v1, v2 = st.columns(2)

    with v1:
        # Actual vs Predicted
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        y_test = artifacts["y_test"]
        y_test_pred = artifacts["y_test_pred"]

        ax1.scatter(y_test, y_test_pred, alpha=0.6, color="#1f77b4", edgecolor="k", s=35)
        min_p = min(y_test.min(), y_test_pred.min())
        max_p = max(y_test.max(), y_test_pred.max())
        ax1.plot([min_p, max_p], [min_p, max_p], "r--", lw=2, label="Perfect Fit Line")
        ax1.set_xlabel("Actual Unit Price ($)")
        ax1.set_ylabel("Predicted Unit Price ($)")
        ax1.set_title("Actual vs. Predicted Unit Price")
        ax1.legend()
        plt.tight_layout()
        st.pyplot(fig1)

    with v2:
        # Residual Distribution
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        residuals = artifacts["test_preds_df"]["Residual"]
        sns.histplot(residuals, kde=True, color="#2ca02c", ax=ax2)
        ax2.axvline(0, color="r", linestyle="--", lw=1.5)
        ax2.set_xlabel("Residuals (Actual - Predicted)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Residual Error Distribution")
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Top Feature Importance")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    top_12 = artifacts["coefs_df"].head(12)
    sns.barplot(data=top_12, x="Coefficient", y="Feature", palette="viridis", ax=ax3)
    ax3.set_title("Top 12 Features by Absolute Weight Magnitude")
    ax3.set_xlabel("Standardized/Encoded Coefficient Weight")
    plt.tight_layout()
    st.pyplot(fig3)

# ---------------------------------------------------------
# TAB 3: EQUATION & WEIGHTS
# ---------------------------------------------------------
with tabs[2]:
    st.subheader("Mathematical Model Specification")

    st.markdown("**Algebraic Regression Equation:**")
    st.code(artifacts["final_equation"], language="text")

    st.markdown("### Feature Coefficient Table")
    st.dataframe(
        artifacts["coefs_df"].style.format({"Coefficient": "{:.6f}"}),
        use_container_width=True,
        height=400
    )

# ---------------------------------------------------------
# TAB 4: REAL-TIME PREDICTOR
# ---------------------------------------------------------
with tabs[3]:
    st.subheader("Interactive Pricing Estimator")
    st.markdown("Adjust feature inputs below to simulate unit price predictions using the trained pipeline.")

    num_cols = artifacts["num_features"]
    cat_cols = artifacts["cat_features"]

    input_dict = {}

    with st.form("prediction_form"):
        st.markdown("#### Numerical Attributes")
        cols_grid = st.columns(3)
        for idx, col_name in enumerate(num_cols):
            with cols_grid[idx % 3]:
                mean_val = float(df_data[col_name].median())
                input_dict[col_name] = st.number_input(
                    label=col_name,
                    value=mean_val,
                    format="%.2f"
                )

        if cat_cols:
            st.markdown("#### Categorical Attributes")
            cat_grid = st.columns(2)
            for idx, col_name in enumerate(cat_cols):
                unique_opts = df_data[col_name].dropna().unique().tolist()
                with cat_grid[idx % 2]:
                    input_dict[col_name] = st.selectbox(
                        label=col_name,
                        options=unique_opts
                    )

        submit_btn = st.form_submit_button("Calculate Estimated Price", use_container_width=True)

    if submit_btn:
        input_row = pd.DataFrame([input_dict])
        predicted_val = artifacts["pipeline"].predict(input_row)[0]

        st.success(f"### 🏷️ Estimated Optimal Unit Price: **${predicted_val:.2f}**")

# ---------------------------------------------------------
# TAB 5: EXPORT HUB
# ---------------------------------------------------------
with tabs[4]:
    st.subheader("Data & Model Deliverables")
    st.markdown(
        "Download the comprehensive Excel evaluation report containing metrics, test set actual vs. predicted values, residuals, model weights, and the final equation."
    )

    excel_bytes = generate_excel_bytes(
        metrics_df=artifacts["metrics_df"],
        test_preds_df=artifacts["test_preds_df"],
        coefs_df=artifacts["coefs_df"],
        final_equation=artifacts["final_equation"]
    )

    st.download_button(
        label="📥 Download Full Excel Report (.xlsx)",
        data=excel_bytes,
        file_name="regression_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )