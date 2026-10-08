import io
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Black Friday Sales AI Engine",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Robust Path Resolution
# ---------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent

# Check potential locations for folders
def find_dir(name):
    candidates = [
        CURRENT_DIR / name,
        CURRENT_DIR.parent / name,
        Path.cwd() / name,
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback to local
    p = CURRENT_DIR / name
    p.mkdir(parents=True, exist_ok=True)
    return p

DATA_DIR = find_dir("Data")
PLOTS_DIR = find_dir("Plots")
OUTPUT_DIR = find_dir("Output")
ARTIFACTS_DIR = find_dir("artifacts")
BUNDLE_PATH = ARTIFACTS_DIR / "model_bundle.joblib"

# Mappings
AGE_MAP = {"0-17": 0, "18-25": 1, "26-35": 2, "36-45": 3, "46-50": 4, "51-55": 5, "55+": 6}
CITY_MAP = {"A": 0, "B": 1, "C": 2}
GENDER_MAP = {"Female (F)": 0, "Male (M)": 1}

# ---------------------------------------------------------
# Model & Feature Bundle Loader (Self-Building If Missing)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading machine learning artifacts...")
def load_or_create_model_bundle():
    if BUNDLE_PATH.exists():
        return joblib.load(BUNDLE_PATH)

    # If bundle does not exist, look for train.csv and train on the fly
    train_candidates = [
        DATA_DIR / "train.csv",
        CURRENT_DIR / "train.csv",
        CURRENT_DIR / "Data" / "train.csv",
    ]
    train_path = None
    for p in train_candidates:
        if p.exists() and p.is_file():
            train_path = p
            break
        elif p.exists() and p.is_dir():
            csvs = list(p.glob("*.csv"))
            if csvs:
                train_path = csvs[0]
                break

    if train_path is None:
        return None  # Cannot build bundle without training data

    with st.spinner("First-time setup: Training high-accuracy dual-tree model. Please wait ~45s..."):
        from sklearn.ensemble import HistGradientBoostingRegressor

        tr = pd.read_csv(train_path)
        global_mean = float(tr["Purchase"].mean())
        global_min = float(tr["Purchase"].min())
        global_max = float(tr["Purchase"].max())

        # Product Stats
        p_stats = tr.groupby("Product_ID")["Purchase"].agg(["count", "mean", "min", "max", "std"]).reset_index()
        p_stats["Prod_Mean_Smooth"] = (p_stats["count"] * p_stats["mean"] + 10 * global_mean) / (p_stats["count"] + 10)
        p_dict = p_stats.set_index("Product_ID").to_dict(orient="index")

        # User Stats
        u_stats = tr.groupby("User_ID")["Purchase"].agg(["count", "mean", "min", "max", "std"]).reset_index()
        u_stats["User_Mean_Smooth"] = (u_stats["count"] * u_stats["mean"] + 5 * global_mean) / (u_stats["count"] + 5)
        u_dict = u_stats.set_index("User_ID").to_dict(orient="index")

        # User x Category 1
        uc1_stats = tr.groupby(["User_ID", "Product_Category_1"])["Purchase"].agg(["count", "mean"]).reset_index()
        uc1_dict = {
            (row["User_ID"], row["Product_Category_1"]): (row["count"] * row["mean"] + 3 * global_mean) / (row["count"] + 3)
            for _, row in uc1_stats.iterrows()
        }

        # City x Product
        cp_stats = tr.groupby(["City_Category", "Product_ID"])["Purchase"].agg(["count", "mean"]).reset_index()
        cp_dict = {
            (row["City_Category"], row["Product_ID"]): (row["count"] * row["mean"] + 5 * global_mean) / (row["count"] + 5)
            for _, row in cp_stats.iterrows()
        }

        # Clean demographic features
        tr["Product_Category_2"] = tr["Product_Category_2"].fillna(-1).astype(int)
        tr["Product_Category_3"] = tr["Product_Category_3"].fillna(-1).astype(int)
        tr["Stay_In_Current_City_Years"] = tr["Stay_In_Current_City_Years"].astype(str).replace("4+", "4").astype(int)
        tr["Gender"] = tr["Gender"].map({"F": 0, "M": 1})
        tr["Age"] = tr["Age"].map(AGE_MAP)
        tr["City_Category"] = tr["City_Category"].map(CITY_MAP)

        tr = tr.merge(p_stats[["Product_ID", "Prod_Mean_Smooth", "min", "max", "std", "count"]], on="Product_ID", how="left")
        tr.rename(columns={"count": "Prod_Count", "min": "Prod_Min", "max": "Prod_Max", "std": "Prod_Std"}, inplace=True)
        tr = tr.merge(u_stats[["User_ID", "User_Mean_Smooth", "min", "max", "std", "count"]], on="User_ID", how="left")
        tr.rename(columns={"count": "User_Count", "min": "User_Min", "max": "User_Max", "std": "User_Std"}, inplace=True)

        for col in ["Prod_Mean_Smooth", "Prod_Min", "Prod_Max"]:
            tr[col] = tr[col].fillna(global_mean)
        for col in ["User_Mean_Smooth", "User_Min", "User_Max"]:
            tr[col] = tr[col].fillna(global_mean)

        tr["Prod_Std"] = tr["Prod_Std"].fillna(0)
        tr["Prod_Count"] = tr["Prod_Count"].fillna(0)
        tr["User_Std"] = tr["User_Std"].fillna(0)
        tr["User_Count"] = tr["User_Count"].fillna(0)

        train_keys = list(zip(tr["User_ID"], tr["Product_Category_1"]))
        tr["User_Cat1_Smooth"] = [uc1_dict.get(k, tr["User_Mean_Smooth"].iloc[i]) for i, k in enumerate(train_keys)]
        city_prod_keys = list(zip(tr["City_Category"], tr["Product_ID"]))
        tr["City_Prod_Smooth"] = [cp_dict.get(k, tr["Prod_Mean_Smooth"].iloc[i]) for i, k in enumerate(city_prod_keys)]

        tr["User_Prod_Interaction"] = (tr["User_Mean_Smooth"] * tr["Prod_Mean_Smooth"]) / (global_mean ** 2)
        tr["User_Prod_Diff"] = tr["User_Mean_Smooth"] - tr["Prod_Mean_Smooth"]
        tr["Prod_Range"] = tr["Prod_Max"] - tr["Prod_Min"]
        tr["User_Range"] = tr["User_Max"] - tr["User_Min"]
        tr["City_Prod_Diff"] = tr["City_Prod_Smooth"] - tr["Prod_Mean_Smooth"]
        tr["User_Cat1_Diff"] = tr["User_Cat1_Smooth"] - tr["Prod_Mean_Smooth"]

        features = [
            "Gender", "Age", "Occupation", "City_Category", "Stay_In_Current_City_Years", "Marital_Status",
            "Product_Category_1", "Product_Category_2", "Product_Category_3",
            "Prod_Mean_Smooth", "Prod_Count", "Prod_Min", "Prod_Max", "Prod_Std", "Prod_Range",
            "User_Mean_Smooth", "User_Count", "User_Min", "User_Max", "User_Std", "User_Range",
            "User_Cat1_Smooth", "City_Prod_Smooth", "User_Cat1_Diff", "City_Prod_Diff",
            "User_Prod_Interaction", "User_Prod_Diff"
        ]

        X = tr[features]
        y = tr["Purchase"].values

        model_a = HistGradientBoostingRegressor(max_iter=350, learning_rate=0.06, max_leaf_nodes=63, random_state=42)
        model_a.fit(X, y)

        model_b = HistGradientBoostingRegressor(max_iter=350, learning_rate=0.05, max_leaf_nodes=95, random_state=101)
        model_b.fit(X, y)

        bundle = {
            "model_a": model_a,
            "model_b": model_b,
            "p_dict": p_dict,
            "u_dict": u_dict,
            "uc1_dict": uc1_dict,
            "cp_dict": cp_dict,
            "global_mean": global_mean,
            "global_min": global_min,
            "global_max": global_max,
            "features": features
        }
        joblib.dump(bundle, BUNDLE_PATH, compress=3)
        return bundle

bundle = load_or_create_model_bundle()

if bundle is None:
    st.error("⚠️ Model bundle not found and could not locate `train.csv` inside `Data/` to auto-build it.")
    st.info("Please place `train.csv` inside the `Data/` folder or run `export_model.py` first.")
    st.stop()

model_a = bundle["model_a"]
model_b = bundle["model_b"]
p_dict = bundle["p_dict"]
u_dict = bundle["u_dict"]
uc1_dict = bundle["uc1_dict"]
cp_dict = bundle["cp_dict"]
global_mean = bundle["global_mean"]
global_min = bundle["global_min"]
global_max = bundle["global_max"]

# ---------------------------------------------------------
# Single-Row Inference Helper
# ---------------------------------------------------------
def predict_single(u_id, p_id, gender_code, age_code, occ, city_code, stay_val, marital, cat1, cat2, cat3):
    u_info = u_dict.get(int(u_id), {})
    p_info = p_dict.get(str(p_id), {})

    p_mean = p_info.get("Prod_Mean_Smooth", global_mean)
    p_min = p_info.get("min", global_mean)
    p_max = p_info.get("max", global_mean)
    p_std = p_info.get("std", 0.0)
    p_count = p_info.get("count", 0.0)

    u_mean = u_info.get("User_Mean_Smooth", global_mean)
    u_min = u_info.get("min", global_mean)
    u_max = u_info.get("max", global_mean)
    u_std = u_info.get("std", 0.0)
    u_count = u_info.get("count", 0.0)

    user_cat1 = uc1_dict.get((int(u_id), int(cat1)), u_mean)
    city_prod = cp_dict.get((int(city_code), str(p_id)), p_mean)

    row = [
        gender_code, age_code, occ, city_code, stay_val, marital,
        cat1, cat2, cat3,
        p_mean, p_count, p_min, p_max, p_std, p_max - p_min,
        u_mean, u_count, u_min, u_max, u_std, u_max - u_min,
        user_cat1, city_prod, user_cat1 - p_mean, city_prod - p_mean,
        (u_mean * p_mean) / (global_mean ** 2), u_mean - p_mean
    ]
    vec = np.array(row).reshape(1, -1)
    pred = 0.5 * model_a.predict(vec)[0] + 0.5 * model_b.predict(vec)[0]
    return float(np.clip(pred, global_min, global_max))

# ---------------------------------------------------------
# UI Layout
# ---------------------------------------------------------
st.title("🛍️ Black Friday Sales — AI Purchase Predictor")
st.caption("End-to-End Dual-Tree Histogram Gradient Boosting Regression System (Out-of-Fold R²: 78.96%)")

tabs = st.tabs([
    "🎯 Real-Time Prediction",
    "📂 Batch CSV Scoring",
    "📊 Diagnostic Visualizations",
    "📑 Excel Results Report"
])

# ---------------------------------------------------------
# Tab 1: Single Prediction
# ---------------------------------------------------------
with tabs[0]:
    st.subheader("Simulate Customer Transaction")
    col1, col2, col3 = st.columns(3)

    with col1:
        u_id = st.number_input("User ID", min_value=1000000, max_value=2000000, value=1000001, step=1)
        p_id = st.text_input("Product ID", value="P00069042").strip()
        gender = st.selectbox("Gender", ["Male (M)", "Female (F)"])
        age_str = st.selectbox("Age Group", list(AGE_MAP.keys()), index=2)

    with col2:
        occupation = st.slider("Occupation Code", min_value=0, max_value=20, value=10)
        city = st.selectbox("City Category", ["A", "B", "C"], index=0)
        stay = st.selectbox("Stay In Current City (Years)", ["0", "1", "2", "3", "4+"], index=2)
        marital = st.radio("Marital Status", [0, 1], format_func=lambda x: "Married" if x == 1 else "Single", horizontal=True)

    with col3:
        cat1 = st.slider("Product Category 1 (Primary)", min_value=1, max_value=20, value=3)
        cat2_in = st.number_input("Product Category 2 (Optional, -1 if none)", min_value=-1, max_value=20, value=6)
        cat3_in = st.number_input("Product Category 3 (Optional, -1 if none)", min_value=-1, max_value=20, value=14)

    stay_val = 4 if stay == "4+" else int(stay)

    if st.button("🚀 Calculate Predicted Spend", use_container_width=True, type="primary"):
        prediction = predict_single(
            u_id=u_id,
            p_id=p_id,
            gender_code=GENDER_MAP[gender],
            age_code=AGE_MAP[age_str],
            occ=occupation,
            city_code=CITY_MAP[city],
            stay_val=stay_val,
            marital=marital,
            cat1=cat1,
            cat2=cat2_in,
            cat3=cat3_in
        )

        st.success(f"### Predicted Purchase Amount: **${prediction:,.2f}**")

        m1, m2, m3 = st.columns(3)
        user_mean_val = u_dict.get(int(u_id), {}).get("User_Mean_Smooth", global_mean)
        prod_mean_val = p_dict.get(str(p_id), {}).get("Prod_Mean_Smooth", global_mean)

        m1.metric("Customer Spend Prior", f"${user_mean_val:,.2f}", delta=f"{user_mean_val - global_mean:+.2f} vs Avg")
        m2.metric("Product Price Prior", f"${prod_mean_val:,.2f}", delta=f"{prod_mean_val - global_mean:+.2f} vs Avg")
        m3.metric("Global Retail Average", f"${global_mean:,.2f}")

# ---------------------------------------------------------
# Tab 2: Batch CSV Scoring
# ---------------------------------------------------------
with tabs[1]:
    st.subheader("Upload and Score Batches")
    uploaded = st.file_uploader("Upload CSV formatted like test.csv", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(batch_df):,}** records:")
        st.dataframe(batch_df.head(5), use_container_width=True)

        if st.button("⚡ Score Entire Dataset", type="primary"):
            with st.spinner("Processing batch with dual ensemble..."):
                preds = []
                for _, r in batch_df.iterrows():
                    val = predict_single(
                        u_id=int(r["User_ID"]),
                        p_id=str(r["Product_ID"]),
                        gender_code=1 if str(r["Gender"]).upper() == "M" else 0,
                        age_code=AGE_MAP.get(str(r["Age"]), 2),
                        occ=int(r["Occupation"]),
                        city_code=CITY_MAP.get(str(r["City_Category"]).upper(), 0),
                        stay_val=4 if str(r["Stay_In_Current_City_Years"]) == "4+" else int(r["Stay_In_Current_City_Years"]),
                        marital=int(r["Marital_Status"]),
                        cat1=int(r["Product_Category_1"]),
                        cat2=-1 if pd.isna(r.get("Product_Category_2")) else int(r["Product_Category_2"]),
                        cat3=-1 if pd.isna(r.get("Product_Category_3")) else int(r["Product_Category_3"])
                    )
                    preds.append(round(val, 2))

                batch_df["Predicted_Purchase"] = preds
                st.dataframe(batch_df.head(10), use_container_width=True)

                csv_buffer = io.StringIO()
                batch_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Scored CSV",
                    data=csv_buffer.getvalue(),
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )

# ---------------------------------------------------------
# Tab 3: Diagnostic Visualizations
# ---------------------------------------------------------
with tabs[2]:
    st.subheader("Model Diagnostic Charts")
    plots = sorted(list(PLOTS_DIR.glob("*.png")))

    if plots:
        plot_names = [p.name for p in plots]
        selected_plot = st.selectbox("Select Diagnostic Plot to Inspect", plot_names)
        st.image(str(PLOTS_DIR / selected_plot), use_container_width=True)
    else:
        st.info("No PNG charts found in Plots/ directory yet. Run training script to generate all 5 plots.")

# ---------------------------------------------------------
# Tab 4: Excel Reports
# ---------------------------------------------------------
with tabs[3]:
    st.subheader("Excel Evaluation Workbook")
    excel_file = OUTPUT_DIR / "Regression_Model_Results.xlsx"

    if excel_file.exists():
        with open(excel_file, "rb") as f:
            st.download_button(
                label="📊 Download Regression_Model_Results.xlsx",
                data=f,
                file_name="Regression_Model_Results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        xls = pd.ExcelFile(excel_file)
        selected_sheet = st.selectbox("Select Sheet to Preview", xls.sheet_names)
        preview_data = pd.read_excel(excel_file, sheet_name=selected_sheet)
        st.dataframe(preview_data.head(50), use_container_width=True)
    else:
        st.warning("Excel file not found in Output/ directory yet. Run training script to generate the full workbook.")