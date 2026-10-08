import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. Directory Structure Setup
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. Resilient CSV Ingestion (Handles files & extracted folders)
# ---------------------------------------------------------
def load_csv_safely(target_path):
    target = Path(target_path)
    if target.is_dir():
        # Handle cases where macOS unzipped into a folder named 'train.csv/'
        csv_files = sorted(target.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found inside directory: {target}")
        return pd.read_csv(csv_files[0])
    if not target.exists():
        raise FileNotFoundError(f"File not found: {target}")
    return pd.read_csv(target)


print("=" * 65)
print("PHASE 1: Loading Dataset...")
train_path = (
    DATA_DIR / "train.csv"
    if (DATA_DIR / "train.csv").exists()
    else BASE_DIR / "train.csv"
)
print(f"Ingesting training data from: {train_path}")

tr = load_csv_safely(train_path)
print(f"Loaded Raw Dataset: {tr.shape} records")

# ---------------------------------------------------------
# 3. Global Statistics & Target Priors
# ---------------------------------------------------------
print("\nPHASE 2: Computing Bayesian Hierarchical Priors...")
global_mean = float(tr["Purchase"].mean())
global_min = float(tr["Purchase"].min())
global_max = float(tr["Purchase"].max())

# 1. Product Statistics (Count, Mean, Min, Max, Std) + Bayesian Smoothing
p_stats = (
    tr.groupby("Product_ID")["Purchase"]
    .agg(["count", "mean", "min", "max", "std"])
    .reset_index()
)
p_weight = 10
p_stats["Prod_Mean_Smooth"] = (
    p_stats["count"] * p_stats["mean"] + p_weight * global_mean
) / (p_stats["count"] + p_weight)
p_dict = p_stats.set_index("Product_ID").to_dict(orient="index")

# 2. User Statistics (Count, Mean, Min, Max, Std) + Bayesian Smoothing
u_stats = (
    tr.groupby("User_ID")["Purchase"]
    .agg(["count", "mean", "min", "max", "std"])
    .reset_index()
)
u_weight = 5
u_stats["User_Mean_Smooth"] = (
    u_stats["count"] * u_stats["mean"] + u_weight * global_mean
) / (u_stats["count"] + u_weight)
u_dict = u_stats.set_index("User_ID").to_dict(orient="index")

# 3. User x Category 1 Affinity Prior
uc1_stats = (
    tr.groupby(["User_ID", "Product_Category_1"])["Purchase"]
    .agg(["count", "mean"])
    .reset_index()
)
uc1_dict = {
    (int(row["User_ID"]), int(row["Product_Category_1"])): float(
        (row["count"] * row["mean"] + 3 * global_mean) / (row["count"] + 3)
    )
    for _, row in uc1_stats.iterrows()
}

# 4. Regional City x Product Price Variation Prior
cp_stats = (
    tr.groupby(["City_Category", "Product_ID"])["Purchase"]
    .agg(["count", "mean"])
    .reset_index()
)
cp_dict = {
    (str(row["City_Category"]), str(row["Product_ID"])): float(
        (row["count"] * row["mean"] + 5 * global_mean) / (row["count"] + 5)
    )
    for _, row in cp_stats.iterrows()
}

# ---------------------------------------------------------
# 4. Feature Matrix Preparation
# ---------------------------------------------------------
print("\nPHASE 3: Building Interaction Feature Matrix...")

AGE_MAP = {
    "0-17": 0,
    "18-25": 1,
    "26-35": 2,
    "36-45": 3,
    "46-50": 4,
    "51-55": 5,
    "55+": 6,
}
CITY_MAP = {"A": 0, "B": 1, "C": 2}

tr["Product_Category_2"] = tr["Product_Category_2"].fillna(-1).astype(int)
tr["Product_Category_3"] = tr["Product_Category_3"].fillna(-1).astype(int)
tr["Stay_In_Current_City_Years"] = (
    tr["Stay_In_Current_City_Years"].astype(str).replace("4+", "4").astype(int)
)
tr["Gender"] = tr["Gender"].map({"F": 0, "M": 1})
tr["Age"] = tr["Age"].map(AGE_MAP)
tr["City_Category"] = tr["City_Category"].map(CITY_MAP)

# Merge calculated Bayesian aggregates
tr = tr.merge(
    p_stats[["Product_ID", "Prod_Mean_Smooth", "min", "max", "std", "count"]],
    on="Product_ID",
    how="left",
).rename(
    columns={
        "count": "Prod_Count",
        "min": "Prod_Min",
        "max": "Prod_Max",
        "std": "Prod_Std",
    }
)

tr = tr.merge(
    u_stats[["User_ID", "User_Mean_Smooth", "min", "max", "std", "count"]],
    on="User_ID",
    how="left",
).rename(
    columns={
        "count": "User_Count",
        "min": "User_Min",
        "max": "User_Max",
        "std": "User_Std",
    }
)

# Impute missing aggregates
for col in ["Prod_Mean_Smooth", "Prod_Min", "Prod_Max"]:
    tr[col] = tr[col].fillna(global_mean)
for col in ["User_Mean_Smooth", "User_Min", "User_Max"]:
    tr[col] = tr[col].fillna(global_mean)

tr["Prod_Std"] = tr["Prod_Std"].fillna(0.0)
tr["Prod_Count"] = tr["Prod_Count"].fillna(0.0)
tr["User_Std"] = tr["User_Std"].fillna(0.0)
tr["User_Count"] = tr["User_Count"].fillna(0.0)

# Multi-index cohort lookups
train_keys = list(zip(tr["User_ID"].astype(int), tr["Product_Category_1"].astype(int)))
tr["User_Cat1_Smooth"] = [
    uc1_dict.get(k, tr["User_Mean_Smooth"].iloc[i])
    for i, k in enumerate(train_keys)
]

city_inv_map = {0: "A", 1: "B", 2: "C"}
city_prod_keys = [
    (city_inv_map.get(c, "A"), str(p))
    for c, p in zip(tr["City_Category"], tr["Product_ID"])
]
tr["City_Prod_Smooth"] = [
    cp_dict.get(k, tr["Prod_Mean_Smooth"].iloc[i])
    for i, k in enumerate(city_prod_keys)
]

# Non-linear affinity interactions
tr["User_Prod_Interaction"] = (
    tr["User_Mean_Smooth"] * tr["Prod_Mean_Smooth"]
) / (global_mean**2)
tr["User_Prod_Diff"] = tr["User_Mean_Smooth"] - tr["Prod_Mean_Smooth"]
tr["Prod_Range"] = tr["Prod_Max"] - tr["Prod_Min"]
tr["User_Range"] = tr["User_Max"] - tr["User_Min"]
tr["City_Prod_Diff"] = tr["City_Prod_Smooth"] - tr["Prod_Mean_Smooth"]
tr["User_Cat1_Diff"] = tr["User_Cat1_Smooth"] - tr["Prod_Mean_Smooth"]

features = [
    "Gender",
    "Age",
    "Occupation",
    "City_Category",
    "Stay_In_Current_City_Years",
    "Marital_Status",
    "Product_Category_1",
    "Product_Category_2",
    "Product_Category_3",
    "Prod_Mean_Smooth",
    "Prod_Count",
    "Prod_Min",
    "Prod_Max",
    "Prod_Std",
    "Prod_Range",
    "User_Mean_Smooth",
    "User_Count",
    "User_Min",
    "User_Max",
    "User_Std",
    "User_Range",
    "User_Cat1_Smooth",
    "City_Prod_Smooth",
    "User_Cat1_Diff",
    "City_Prod_Diff",
    "User_Prod_Interaction",
    "User_Prod_Diff",
]

X = tr[features]
y = tr["Purchase"].values

print(f"Feature Matrix Ready: {X.shape[0]} rows x {X.shape[1]} features")

# ---------------------------------------------------------
# 5. Dual-Tree Production Training
# ---------------------------------------------------------
print("\nPHASE 4: Training Dual-Tree Ensemble...")
t0 = time.time()

# Model A: Balanced breadth
print(" -> Training Model A (max_leaf_nodes=63, lr=0.05, iter=450)...")
model_a = HistGradientBoostingRegressor(
    max_iter=450,
    learning_rate=0.05,
    max_leaf_nodes=63,
    min_samples_leaf=20,
    l2_regularization=0.3,
    random_state=42,
)
model_a.fit(X, y)

# Model B: High capacity depth
print(" -> Training Model B (max_leaf_nodes=95, lr=0.04, iter=450)...")
model_b = HistGradientBoostingRegressor(
    max_iter=450,
    learning_rate=0.04,
    max_leaf_nodes=95,
    min_samples_leaf=25,
    l2_regularization=0.5,
    random_state=101,
)
model_b.fit(X, y)

print(f"Training completed in {time.time() - t0:.1f}s")

# ---------------------------------------------------------
# 6. Artifact Serialization
# ---------------------------------------------------------
print("\nPHASE 5: Packaging Artifacts...")
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
    "features": features,
}

output_bundle_path = ARTIFACTS_DIR / "model_bundle.joblib"
joblib.dump(bundle, output_bundle_path, compress=3)

file_size_mb = os.path.getsize(output_bundle_path) / (1024 * 1024)
print("=" * 65)
print(f"Artifact exported successfully: {output_bundle_path}")
print(f"Bundle File Size: {file_size_mb:.2f} MB")
print("Ready for deployment with FastAPI and Streamlit.")
print("=" * 65)