import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------
# 1. Directory Setup
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
PLOTS_DIR = BASE_DIR / "Plots"
OUTPUT_DIR = BASE_DIR / "Output"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 2. Resilient Data Ingestion
# ---------------------------------------------------------
def load_csv_safely(target_path):
    target = Path(target_path)
    if target.is_dir():
        csv_files = list(target.glob("*.csv"))
        if csv_files:
            return pd.read_csv(csv_files[0])
        raise FileNotFoundError(f"No CSV file found inside directory: {target}")
    return pd.read_csv(target)

print("=" * 65)
print("PHASE 1: Loading Datasets...")
train_csv_path = DATA_DIR / "train.csv" if (DATA_DIR / "train.csv").exists() else BASE_DIR / "train.csv"
test_csv_path = DATA_DIR / "test.csv" if (DATA_DIR / "test.csv").exists() else BASE_DIR / "test.csv"

train_df = load_csv_safely(train_csv_path)
test_df = load_csv_safely(test_csv_path)

print(f"Loaded Train: {train_df.shape} | Test: {test_df.shape}")

# ---------------------------------------------------------
# 3. Exploratory Data Profiling & Metadata
# ---------------------------------------------------------
print("\nPHASE 2: Generating EDA Profiles...")
eda_summary = pd.DataFrame({
    "Feature": train_df.columns,
    "Dtype": train_df.dtypes.values,
    "Missing_Count": train_df.isnull().sum().values,
    "Missing_Pct": (train_df.isnull().mean() * 100).round(2).values,
    "Unique_Values": [train_df[c].nunique() for c in train_df.columns]
})

numeric_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()
eda_statistics = train_df[numeric_columns].describe().T.reset_index().rename(columns={"index": "Feature"})

# ---------------------------------------------------------
# 4. Feature Engineering Pipeline
# ---------------------------------------------------------
print("\nPHASE 3: Engineering Matrix Affinity & Interaction Features...")

def grandmaster_feature_pipeline(train, test):
    tr = train.copy()
    te = test.copy()
    
    global_mean = tr["Purchase"].mean()
    global_min = tr["Purchase"].min()
    global_max = tr["Purchase"].max()
    
    # Missing category imputation
    tr["Product_Category_2"] = tr["Product_Category_2"].fillna(-1).astype(int)
    tr["Product_Category_3"] = tr["Product_Category_3"].fillna(-1).astype(int)
    te["Product_Category_2"] = te["Product_Category_2"].fillna(-1).astype(int)
    te["Product_Category_3"] = te["Product_Category_3"].fillna(-1).astype(int)
    
    # Stay in city
    tr["Stay_In_Current_City_Years"] = tr["Stay_In_Current_City_Years"].replace("4+", 4).astype(int)
    te["Stay_In_Current_City_Years"] = te["Stay_In_Current_City_Years"].replace("4+", 4).astype(int)
    
    # Demographic mappings
    tr["Gender"] = tr["Gender"].map({"F": 0, "M": 1})
    te["Gender"] = te["Gender"].map({"F": 0, "M": 1})
    
    age_map = {"0-17": 0, "18-25": 1, "26-35": 2, "36-45": 3, "46-50": 4, "51-55": 5, "55+": 6}
    tr["Age"] = tr["Age"].map(age_map)
    te["Age"] = te["Age"].map(age_map)
    
    city_map = {"A": 0, "B": 1, "C": 2}
    tr["City_Category"] = tr["City_Category"].map(city_map)
    te["City_Category"] = te["City_Category"].map(city_map)
    
    # 1. Product Statistics & Bayesian Smoothing
    p_stats = tr.groupby("Product_ID")["Purchase"].agg(["count", "mean", "min", "max", "std"]).reset_index()
    p_weight = 10
    p_stats["Prod_Mean_Smooth"] = (p_stats["count"] * p_stats["mean"] + p_weight * global_mean) / (p_stats["count"] + p_weight)
    
    # 2. User Statistics & Bayesian Smoothing
    u_stats = tr.groupby("User_ID")["Purchase"].agg(["count", "mean", "min", "max", "std"]).reset_index()
    u_weight = 5
    u_stats["User_Mean_Smooth"] = (u_stats["count"] * u_stats["mean"] + u_weight * global_mean) / (u_stats["count"] + u_weight)
    
    # 3. User x Category 1 Preference
    uc1_stats = tr.groupby(["User_ID", "Product_Category_1"])["Purchase"].agg(["count", "mean"]).reset_index()
    uc1_stats["User_Cat1_Smooth"] = (uc1_stats["count"] * uc1_stats["mean"] + 3 * global_mean) / (uc1_stats["count"] + 3)
    
    # 4. Regional Price Variance (City x Product)
    cp_stats = tr.groupby(["City_Category", "Product_ID"])["Purchase"].agg(["count", "mean"]).reset_index()
    cp_stats["City_Prod_Smooth"] = (cp_stats["count"] * cp_stats["mean"] + 5 * global_mean) / (cp_stats["count"] + 5)
    
    # Merge onto train
    tr = tr.merge(p_stats[["Product_ID", "Prod_Mean_Smooth", "min", "max", "std", "count"]], on="Product_ID", how="left")
    tr.rename(columns={"count": "Prod_Count", "min": "Prod_Min", "max": "Prod_Max", "std": "Prod_Std"}, inplace=True)
    
    tr = tr.merge(u_stats[["User_ID", "User_Mean_Smooth", "min", "max", "std", "count"]], on="User_ID", how="left")
    tr.rename(columns={"count": "User_Count", "min": "User_Min", "max": "User_Max", "std": "User_Std"}, inplace=True)
    
    tr = tr.merge(uc1_stats[["User_ID", "Product_Category_1", "User_Cat1_Smooth"]], on=["User_ID", "Product_Category_1"], how="left")
    tr = tr.merge(cp_stats[["City_Category", "Product_ID", "City_Prod_Smooth"]], on=["City_Category", "Product_ID"], how="left")
    
    # Merge onto test
    te = te.merge(p_stats[["Product_ID", "Prod_Mean_Smooth", "min", "max", "std", "count"]], on="Product_ID", how="left")
    te.rename(columns={"count": "Prod_Count", "min": "Prod_Min", "max": "Prod_Max", "std": "Prod_Std"}, inplace=True)
    
    te = te.merge(u_stats[["User_ID", "User_Mean_Smooth", "min", "max", "std", "count"]], on="User_ID", how="left")
    te.rename(columns={"count": "User_Count", "min": "User_Min", "max": "User_Max", "std": "User_Std"}, inplace=True)
    
    te = te.merge(uc1_stats[["User_ID", "Product_Category_1", "User_Cat1_Smooth"]], on=["User_ID", "Product_Category_1"], how="left")
    te = te.merge(cp_stats[["City_Category", "Product_ID", "City_Prod_Smooth"]], on=["City_Category", "Product_ID"], how="left")
    
    # Impute missing values for rare/unseen entries
    for col in ["Prod_Mean_Smooth", "Prod_Min", "Prod_Max"]:
        tr[col] = tr[col].fillna(global_mean)
        te[col] = te[col].fillna(global_mean)
        
    for col in ["User_Mean_Smooth", "User_Min", "User_Max"]:
        tr[col] = tr[col].fillna(global_mean)
        te[col] = te[col].fillna(global_mean)
        
    tr["Prod_Std"] = tr["Prod_Std"].fillna(0)
    te["Prod_Std"] = te["Prod_Std"].fillna(0)
    tr["User_Std"] = tr["User_Std"].fillna(0)
    te["User_Std"] = te["User_Std"].fillna(0)
    tr["Prod_Count"] = tr["Prod_Count"].fillna(0)
    te["Prod_Count"] = te["Prod_Count"].fillna(0)
    tr["User_Count"] = tr["User_Count"].fillna(0)
    te["User_Count"] = te["User_Count"].fillna(0)
    
    tr["User_Cat1_Smooth"] = tr["User_Cat1_Smooth"].fillna(tr["User_Mean_Smooth"])
    te["User_Cat1_Smooth"] = te["User_Cat1_Smooth"].fillna(te["User_Mean_Smooth"])
    tr["City_Prod_Smooth"] = tr["City_Prod_Smooth"].fillna(tr["Prod_Mean_Smooth"])
    te["City_Prod_Smooth"] = te["City_Prod_Smooth"].fillna(te["Prod_Mean_Smooth"])
    
    # Non-linear mathematical ratios and spreads
    tr["User_Prod_Interaction"] = (tr["User_Mean_Smooth"] * tr["Prod_Mean_Smooth"]) / (global_mean ** 2)
    te["User_Prod_Interaction"] = (te["User_Mean_Smooth"] * te["Prod_Mean_Smooth"]) / (global_mean ** 2)
    
    tr["User_Prod_Diff"] = tr["User_Mean_Smooth"] - tr["Prod_Mean_Smooth"]
    te["User_Prod_Diff"] = te["User_Mean_Smooth"] - te["Prod_Mean_Smooth"]
    
    tr["Prod_Range"] = tr["Prod_Max"] - tr["Prod_Min"]
    te["Prod_Range"] = te["Prod_Max"] - te["Prod_Min"]
    
    tr["User_Range"] = tr["User_Max"] - tr["User_Min"]
    te["User_Range"] = te["User_Max"] - te["User_Min"]
    
    tr["City_Prod_Diff"] = tr["City_Prod_Smooth"] - tr["Prod_Mean_Smooth"]
    te["City_Prod_Diff"] = te["City_Prod_Smooth"] - te["Prod_Mean_Smooth"]
    
    tr["User_Cat1_Diff"] = tr["User_Cat1_Smooth"] - tr["Prod_Mean_Smooth"]
    te["User_Cat1_Diff"] = te["User_Cat1_Smooth"] - te["Prod_Mean_Smooth"]
    
    return tr, te, global_min, global_max

train_fe, test_fe, global_min, global_max = grandmaster_feature_pipeline(train_df, test_df)

features = [
    "Gender", "Age", "Occupation", "City_Category", "Stay_In_Current_City_Years", "Marital_Status",
    "Product_Category_1", "Product_Category_2", "Product_Category_3",
    "Prod_Mean_Smooth", "Prod_Count", "Prod_Min", "Prod_Max", "Prod_Std", "Prod_Range",
    "User_Mean_Smooth", "User_Count", "User_Min", "User_Max", "User_Std", "User_Range",
    "User_Cat1_Smooth", "City_Prod_Smooth", "User_Cat1_Diff", "City_Prod_Diff",
    "User_Prod_Interaction", "User_Prod_Diff"
]

X_tr = train_fe[features]
y_tr = train_fe["Purchase"].values
X_te = test_fe[features]

print(f"Engineered Feature Matrix Shape: {X_tr.shape} ({len(features)} Features)")

# ---------------------------------------------------------
# 5. Dual-Model Architecture & 5-Fold Cross-Validation
# ---------------------------------------------------------
print("\nPHASE 4: Executing 5-Fold Cross-Validation on Dual-Tree Ensemble...")

# Model A: Balanced breadth
model_a = HistGradientBoostingRegressor(
    max_iter=450,
    learning_rate=0.05,
    max_leaf_nodes=63,
    min_samples_leaf=20,
    l2_regularization=0.3,
    random_state=42
)

# Model B: High-capacity depth
model_b = HistGradientBoostingRegressor(
    max_iter=450,
    learning_rate=0.04,
    max_leaf_nodes=95,
    min_samples_leaf=25,
    l2_regularization=0.5,
    random_state=101
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(y_tr))
fold_records = []

for fold, (trn_idx, val_idx) in enumerate(cv.split(X_tr, y_tr), start=1):
    t_fold = time.time()
    X_train_f, y_train_f = X_tr.iloc[trn_idx], y_tr[trn_idx]
    X_val_f, y_val_f = X_tr.iloc[val_idx], y_tr[val_idx]
    
    # Train both models
    model_a.fit(X_train_f, y_train_f)
    model_b.fit(X_train_f, y_train_f)
    
    preds_a = model_a.predict(X_val_f)
    preds_b = model_b.predict(X_val_f)
    
    # Weighted ensemble blend
    blend_preds = 0.5 * preds_a + 0.5 * preds_b
    blend_preds = np.clip(blend_preds, global_min, global_max)
    oof_predictions[val_idx] = blend_preds
    
    f_rmse = np.sqrt(mean_squared_error(y_val_f, blend_preds))
    f_mae = mean_absolute_error(y_val_f, blend_preds)
    f_r2 = r2_score(y_val_f, blend_preds)
    
    fold_records.append({
        "Fold": f"Fold_{fold}",
        "RMSE": round(f_rmse, 2),
        "MAE": round(f_mae, 2),
        "R2_Score_%": round(f_r2 * 100, 2),
        "Time_Sec": round(time.time() - t_fold, 1)
    })
    print(f" -> Fold {fold}: RMSE = {f_rmse:.2f} | MAE = {f_mae:.2f} | R2 = {f_r2*100:.2f}%")

df_folds = pd.DataFrame(fold_records)

overall_rmse = np.sqrt(mean_squared_error(y_tr, oof_predictions))
overall_mae = mean_absolute_error(y_tr, oof_predictions)
overall_r2 = r2_score(y_tr, oof_predictions)

print("\n" + "=" * 65)
print(f"Final Ensemble Out-Of-Fold R2 Score: {overall_r2*100:.2f}%")
print(f"Final Ensemble Out-Of-Fold RMSE:     ${overall_rmse:.2f}")
print(f"Final Ensemble Out-Of-Fold MAE:      ${overall_mae:.2f}")
print("=" * 65)

# Full Retraining for Test Inference
print("\nPHASE 5: Fitting Full Ensemble on Complete Data & Generating Test Predictions...")
model_a.fit(X_tr, y_tr)
model_b.fit(X_tr, y_tr)

test_preds_a = model_a.predict(X_te)
test_preds_b = model_b.predict(X_te)
test_blend = 0.5 * test_preds_a + 0.5 * test_preds_b
test_blend = np.clip(test_blend, global_min, global_max)

test_submission = pd.DataFrame({
    "User_ID": test_df["User_ID"],
    "Product_ID": test_df["Product_ID"],
    "Predicted_Purchase": np.round(test_blend, 2)
})
test_submission.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

# ---------------------------------------------------------
# 6. Diagnostic Visualizations
# ---------------------------------------------------------
print("\nPHASE 6: Rendering Diagnostic Plots in Plots/...")
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# Plot 1: Target Distribution
plt.figure(figsize=(9, 5))
sns.histplot(y_tr, bins=50, kde=True, color="#1f77b4")
plt.title("Purchase Amount Distribution (Target)", fontsize=13, fontweight="bold")
plt.xlabel("Purchase ($)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_purchase_distribution.png", dpi=200)
plt.close()

# Plot 2: Actual vs Predicted
np.random.seed(42)
sample_idx = np.random.choice(len(y_tr), size=15000, replace=False)
plt.figure(figsize=(7, 7))
plt.scatter(y_tr[sample_idx], oof_predictions[sample_idx], alpha=0.3, color="#2ca02c", edgecolors="none", s=14)
min_v = min(y_tr.min(), oof_predictions.min())
max_v = max(y_tr.max(), oof_predictions.max())
plt.plot([min_v, max_v], [min_v, max_v], "r--", lw=2, label="Ideal Fit (y = x)")
plt.title(f"Actual vs Predicted (Ensemble R2 = {overall_r2*100:.2f}%)", fontsize=13, fontweight="bold")
plt.xlabel("Actual Purchase ($)")
plt.ylabel("Predicted Purchase ($)")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_actual_vs_predicted.png", dpi=200)
plt.close()

# Plot 3: Residuals Distribution
residuals = y_tr - oof_predictions
plt.figure(figsize=(9, 5))
sns.histplot(residuals[sample_idx], bins=50, kde=True, color="#d62728")
plt.axvline(0, color="black", linestyle="--", lw=1.5)
plt.title("Residuals Distribution (Zero Centered)", fontsize=13, fontweight="bold")
plt.xlabel("Residual Error ($)")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_residuals_distribution.png", dpi=200)
plt.close()

# Plot 4: K-Fold Consistency
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()
sns.barplot(data=df_folds, x="Fold", y="RMSE", ax=ax1, color="#4575b4", alpha=0.85)
sns.lineplot(data=df_folds, x=range(len(df_folds)), y="R2_Score_%", ax=ax2, color="#d73027", marker="o", lw=2.5)
ax1.set_title("5-Fold Cross Validation Consistency (RMSE & R2)", fontsize=13, fontweight="bold")
ax1.set_ylabel("RMSE (Lower is Better)", color="#4575b4")
ax2.set_ylabel("R2 Score %", color="#d73027")
ax1.set_ylim(2200, 2350)
ax2.set_ylim(78.0, 80.0)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_kfold_consistency.png", dpi=200)
plt.close()

# Plot 5: Feature Importances
perm = permutation_importance(model_a, X_tr.iloc[sample_idx[:5000]], y_tr[sample_idx[:5000]], n_repeats=3, random_state=42, n_jobs=-1)
df_feat_imp = pd.DataFrame({
    "Feature": features,
    "Importance": perm.importances_mean
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_feat_imp.head(10), x="Importance", y="Feature", palette="mako")
plt.title("Top 10 Feature Importances", fontsize=13, fontweight="bold")
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_feature_importance.png", dpi=200)
plt.close()

# ---------------------------------------------------------
# 7. Multi-Sheet Excel Output
# ---------------------------------------------------------
print("\nPHASE 7: Compiling Multi-Sheet Results Workbook in Output/...")

df_act_pred = pd.DataFrame({
    "Actual_Purchase": y_tr,
    "Predicted_Purchase": np.round(oof_predictions, 2),
    "Residual": np.round(residuals, 2),
    "Absolute_Error": np.round(np.abs(residuals), 2),
    "Percentage_Error_%": np.round(np.abs(residuals) / (y_tr + 1e-5) * 100, 2)
})

summary_metrics = pd.DataFrame([
    {"Metric": "Overall R2 Score (%)", "Value": round(overall_r2 * 100, 2)},
    {"Metric": "Overall RMSE ($)", "Value": round(overall_rmse, 2)},
    {"Metric": "Overall MAE ($)", "Value": round(overall_mae, 2)},
    {"Metric": "Cross-Validation Protocol", "Value": "5-Fold Out-of-Fold Dual Ensemble"},
    {"Metric": "Ensemble Structure", "Value": "50% HistGradientBoosting (63 leaves) + 50% HistGradientBoosting (95 leaves)"},
    {"Metric": "Engineered Features Count", "Value": len(features)}
])

excel_file = OUTPUT_DIR / "Regression_Model_Results.xlsx"
with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
    summary_metrics.to_excel(writer, sheet_name="Model_Metrics", index=False)
    df_folds.to_excel(writer, sheet_name="Fold_Evaluations", index=False)
    df_feat_imp.to_excel(writer, sheet_name="Feature_Importance", index=False)
    df_act_pred.to_excel(writer, sheet_name="Actuals_vs_Predicted", index=False)
    test_submission.to_excel(writer, sheet_name="Test_Predictions", index=False)
    eda_summary.to_excel(writer, sheet_name="EDA_Summary", index=False)
    eda_statistics.to_excel(writer, sheet_name="EDA_Statistics", index=False)

print("\nPipeline Execution Successful:")
print(f" - Out-of-Fold R2 Score: {overall_r2*100:.2f}%")
print(f" - Out-of-Fold RMSE:     ${overall_rmse:.2f}")
print(f" - Charts Directory:     {PLOTS_DIR}/")
print(f" - Excel Workbook:       {excel_file}")
print(f" - Test Submission:      {OUTPUT_DIR / 'test_predictions.csv'}")