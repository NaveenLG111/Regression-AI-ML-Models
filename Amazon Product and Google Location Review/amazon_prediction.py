import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION & DIRECTORIES
# ==============================================================================


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(SCRIPT_DIR, "Data", "y_amazon-google-large.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Output")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "Plots")

MODEL_PATH = os.path.join(OUTPUT_DIR, "google_reviews_forecast_best_model.pkl")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "google_reviews_feature_columns.pkl")
OUTPUT_XLSX = os.path.join(OUTPUT_DIR, "ml_model_report_optimized.xlsx")

DATE_COL = "ds"
TARGET_COL = "y"
ID_COL = "unique_id"

# Ensure directories exist prior to any processing
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Source dataset not found: {DATA_PATH}")

# ==============================================================================
# 2. DATA INGESTION & DATA CLEANING
# ==============================================================================
print("[Step 1/7] Ingesting and sanitizing dataset...")
df = pd.read_csv(DATA_PATH)

# Drop redundant index/unnamed columns
drop_cols = [c for c in df.columns if "Unnamed" in c or c.strip().lower() == "column 1"]
df = df.drop(columns=drop_cols, errors="ignore")

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL, TARGET_COL])

# Ensure strictly chronological ordering within each entity
if ID_COL in df.columns:
    df[ID_COL] = df[ID_COL].astype(str)
    df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)
else:
    df = df.sort_values(DATE_COL).reset_index(drop=True)

# ==============================================================================
# 3. FEATURE ENGINEERING (LEAKAGE-FREE & MOMENTUM ENRICHED)
# ==============================================================================
print("[Step 2/7] Generating temporal, cyclical, and momentum features...")

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # Calendar & Cyclical Features
    dt = data[DATE_COL].dt
    data["hour"] = dt.hour
    data["day"] = dt.day
    data["dayofweek"] = dt.dayofweek
    data["dayofyear"] = dt.dayofyear
    data["weekofyear"] = dt.isocalendar().week.astype(int)
    data["month"] = dt.month
    data["quarter"] = dt.quarter
    data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(np.int8)

    # Cyclical Sine/Cosine Encodings
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24.0)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24.0)
    data["dow_sin"] = np.sin(2 * np.pi * data["dayofweek"] / 7.0)
    data["dow_cos"] = np.cos(2 * np.pi * data["dayofweek"] / 7.0)
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12.0)

    # Lag Intervals and Rolling Windows
    lag_intervals = [1, 2, 3, 6, 12, 24, 48, 72]
    roll_windows = [3, 6, 12, 24]

    grouped = (
        data.groupby(ID_COL)[TARGET_COL]
        if ID_COL in data.columns
        else data[TARGET_COL]
    )

    # Explicit Lags
    for lag in lag_intervals:
        data[f"lag_{lag}"] = grouped.shift(lag)

    # Differencing & Momentum Features
    data["diff_lag1_lag2"] = data["lag_1"] - data["lag_2"]
    data["diff_lag1_lag24"] = data["lag_1"] - data["lag_24"]

    # Rolling Statistics strictly on target.shift(1) within series boundaries
    shifted_target = grouped.shift(1)
    roll_base = (
        shifted_target.groupby(data[ID_COL])
        if ID_COL in data.columns
        else shifted_target
    )

    for w in roll_windows:
        data[f"rolling_mean_{w}"] = roll_base.transform(
            lambda s: s.rolling(w, min_periods=w).mean()
        )
        data[f"rolling_std_{w}"] = roll_base.transform(
            lambda s: s.rolling(w, min_periods=w).std()
        )
        data[f"rolling_min_{w}"] = roll_base.transform(
            lambda s: s.rolling(w, min_periods=w).min()
        )
        data[f"rolling_max_{w}"] = roll_base.transform(
            lambda s: s.rolling(w, min_periods=w).max()
        )

    # Interaction & Ratio Features
    eps = 1e-6
    data["ratio_lag1_rollmean24"] = data["lag_1"] / (
        data["rolling_mean_24"] + eps
    )

    # Historical Group Mean
    if ID_COL in data.columns:
        data["expanding_group_mean"] = roll_base.transform(
            lambda s: s.expanding(min_periods=24).mean()
        )

    return data

df_feat = engineer_features(df)
df_clean = df_feat.dropna().reset_index(drop=True)
print(f"Dataset shape after feature extraction & NaN cleaning: {df_clean.shape}")

# Convert ID_COL to category for LightGBM compatibility
if ID_COL in df_clean.columns:
    df_clean[ID_COL] = df_clean[ID_COL].astype("category")

# ==============================================================================
# 4. EXPLORATORY DATA ANALYSIS & VISUALIZATION ARTIFACTS
# ==============================================================================
print("[Step 3/7] Generating and saving EDA visualizations to 'plots' folder...")

# Plot 1: Target Over Time
fig1 = plt.figure(figsize=(14, 5))
sample_size = min(5000, len(df))
plt.plot(
    df[DATE_COL].iloc[:sample_size],
    df[TARGET_COL].iloc[:sample_size],
    alpha=0.4,
    label="Actual Target",
)
plt.plot(
    df[DATE_COL].iloc[:sample_size],
    df[TARGET_COL].iloc[:sample_size].rolling(24).mean(),
    color="crimson",
    linewidth=1.5,
    label="24-Step Rolling Mean",
)
plt.title("Target Trend Over Time")
plt.xlabel("Datetime")
plt.ylabel("Target (y)")
plt.legend()
plt.tight_layout()
p1_path = os.path.join(PLOTS_DIR, "01_target_trend.png")
fig1.savefig(p1_path, dpi=150)
plt.close(fig1)
print(f"Saved: {p1_path}")

# Plot 2: Seasonality by Hour and Day of Week
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
sample_clean_size = min(50000, len(df_clean))
sns.boxplot(
    data=df_clean.iloc[:sample_clean_size], x="hour", y=TARGET_COL, ax=axes2[0], color="skyblue"
)
axes2[0].set_title("Target Distribution by Hour of Day")
sns.boxplot(
    data=df_clean.iloc[:sample_clean_size],
    x="dayofweek",
    y=TARGET_COL,
    ax=axes2[1],
    color="salmon",
)
axes2[1].set_title("Target Distribution by Day of Week")
plt.tight_layout()
p2_path = os.path.join(PLOTS_DIR, "02_seasonality_boxplots.png")
fig2.savefig(p2_path, dpi=150)
plt.close(fig2)
print(f"Saved: {p2_path}")

# Plot 3: Lag Correlations
fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
lags_to_plot = [
    (1, axes3[0, 0]),
    (6, axes3[0, 1]),
    (24, axes3[1, 0]),
    (72, axes3[1, 1]),
]
sample_scatter_size = min(10000, len(df_clean))
sample_clean = df_clean.iloc[:sample_scatter_size]
for lag_val, ax in lags_to_plot:
    if f"lag_{lag_val}" in sample_clean.columns:
        sns.scatterplot(
            x=sample_clean[f"lag_{lag_val}"],
            y=sample_clean[TARGET_COL],
            ax=ax,
            alpha=0.3,
            color="teal",
        )
        ax.set_title(f"Target vs. Lag {lag_val}")
plt.tight_layout()
p3_path = os.path.join(PLOTS_DIR, "03_lag_correlations.png")
fig3.savefig(p3_path, dpi=150)
plt.close(fig3)
print(f"Saved: {p3_path}")

# Plot 4: Feature Correlation Heatmap
corr_cols = [
    TARGET_COL,
    "hour",
    "dayofweek",
    "lag_1",
    "lag_2",
    "lag_24",
    "diff_lag1_lag2",
    "rolling_mean_24",
    "rolling_std_24",
]
corr_cols = [c for c in corr_cols if c in df_clean.columns]
fig4 = plt.figure(figsize=(10, 8))
sns.heatmap(
    df_clean[corr_cols].iloc[:sample_clean_size].corr(numeric_only=True),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
)
plt.title("Key Feature Correlation Matrix")
plt.tight_layout()
p4_path = os.path.join(PLOTS_DIR, "04_correlation_heatmap.png")
fig4.savefig(p4_path, dpi=150)
plt.close(fig4)
print(f"Saved: {p4_path}")

# ==============================================================================
# 5. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ==============================================================================
print("[Step 4/7] Performing chronological train/validation/test split...")

ignore_cols = [TARGET_COL, DATE_COL]
feature_cols = [c for c in df_clean.columns if c not in ignore_cols]

# 70% Train, 15% Validation, 15% Out-of-Sample Test
n_total = len(df_clean)
train_end = int(n_total * 0.70)
val_end = int(n_total * 0.85)

train_df = df_clean.iloc[:train_end]
val_df = df_clean.iloc[train_end:val_end]
test_df = df_clean.iloc[val_end:]

X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
X_val, y_val = val_df[feature_cols], val_df[TARGET_COL]
X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

print(f"Train size: {len(X_train):,}")
print(f"Validation size: {len(X_val):,}")
print(f"Test size: {len(X_test):,}")

# Identify categorical features for modeling
cat_features = [c for c in feature_cols if str(df_clean[c].dtype) in ("category", "object")]

# ==============================================================================
# 6. MODEL TRAINING & COMPARATIVE BENCHMARKING
# ==============================================================================
print("[Step 5/7] Training candidate regression models...")

# Model 1: LightGBM Regressor
lgbm_model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    importance_type="gain",
)
lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

# Model 2: HistGradientBoostingRegressor
hgb_model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=63,
    random_state=42,
    categorical_features=cat_features if len(cat_features) > 0 else None
)
hgb_model.fit(X_train, y_train)

# Tournament Benchmark Evaluation on Test Set
candidates = {"LightGBM": lgbm_model, "HistGradientBoosting": hgb_model}

comparison_records = []
best_model = None
best_rmse = float("inf")
best_model_name = ""

for name, model_obj in candidates.items():
    preds = model_obj.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    comparison_records.append(
        {"Model": name, "Test_MAE": mae, "Test_RMSE": rmse, "Test_R2": r2}
    )

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model_obj
        best_model_name = name

comparison_df = pd.DataFrame(comparison_records).sort_values(by="Test_RMSE")
print("\n" + "=" * 50)
print("BENCHMARK TOURNAMENT RESULTS:")
print(comparison_df.to_string(index=False))
print("=" * 50)

# ==============================================================================
# 7. OUT-OF-SAMPLE EVALUATION, ARTIFACTS & EXCEL REPORT EXPORT
# ==============================================================================
print(f"\n[Step 6/7] Finalizing best model: {best_model_name} and persisting artifacts...")

final_preds = best_model.predict(X_test)
final_mae = mean_absolute_error(y_test, final_preds)
final_rmse = np.sqrt(mean_squared_error(y_test, final_preds))
final_r2 = r2_score(y_test, final_preds)

# Plot 5: Actual vs Predicted Series
fig5 = plt.figure(figsize=(14, 5))
preview_pts = min(500, len(test_df))
plt.plot(
    test_df[DATE_COL].iloc[:preview_pts].values,
    y_test.iloc[:preview_pts].values,
    label="Actual (y)",
    color="steelblue",
    alpha=0.8,
)
plt.plot(
    test_df[DATE_COL].iloc[:preview_pts].values,
    final_preds[:preview_pts],
    label=f"Predicted ({best_model_name})",
    color="darkorange",
    linestyle="--",
)
plt.title(f"Out-of-Sample Performance: Actual vs. Predicted ({best_model_name})")
plt.xlabel("Datetime")
plt.ylabel("Target")
plt.legend()
plt.tight_layout()
p5_path = os.path.join(PLOTS_DIR, "05_actual_vs_predicted.png")
fig5.savefig(p5_path, dpi=150)
plt.close(fig5)
print(f"Saved: {p5_path}")

# Explicitly serialize model and feature names to OUTPUT_DIR
joblib.dump(best_model, MODEL_PATH)
joblib.dump(feature_cols, FEATURES_PATH)

# Verify serialization
if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
    print(f"SUCCESS: Model saved at -> {os.path.abspath(MODEL_PATH)}")
    print(f"SUCCESS: Feature list saved at -> {os.path.abspath(FEATURES_PATH)}")
else:
    raise IOError("Failed to save .pkl files to the output directory!")

# Compute Feature Importance
if hasattr(best_model, "feature_importances_"):
    fi_df = pd.DataFrame(
        {
            "Feature": feature_cols,
            "Importance_Gain": best_model.feature_importances_,
        }
    ).sort_values(by="Importance_Gain", ascending=False)
else:
    fi_df = pd.DataFrame({"Feature": feature_cols, "Importance_Gain": np.nan})

# Compile Multi-Tab Production Excel Report
print("[Step 7/7] Generating multi-sheet Excel report...")

eda_summary = pd.DataFrame(
    {
        "Metric": [
            "Raw_Rows",
            "Cleaned_Rows",
            "Features_Generated",
            "Target_Mean",
            "Target_Std",
            "Target_Min",
            "Target_Max",
            "Start_Date",
            "End_Date",
        ],
        "Value": [
            len(df),
            len(df_clean),
            len(feature_cols),
            df[TARGET_COL].mean(),
            df[TARGET_COL].std(),
            df[TARGET_COL].min(),
            df[TARGET_COL].max(),
            str(df[DATE_COL].min()),
            str(df[DATE_COL].max()),
        ],
    }
)

metrics_summary = pd.DataFrame(
    {
        "Metric": ["Selected_Best_Model", "MAE", "RMSE", "R2_Score"],
        "Value": [best_model_name, final_mae, final_rmse, final_r2],
    }
)

predictions_table = test_df[[DATE_COL]].copy()
if ID_COL in test_df.columns:
    predictions_table[ID_COL] = test_df[ID_COL].values
predictions_table["Actual"] = y_test.values
predictions_table["Predicted"] = final_preds
predictions_table["Residual_Error"] = (
    predictions_table["Actual"] - predictions_table["Predicted"]
)
predictions_table["Absolute_Percentage_Error_%"] = (
    np.abs(predictions_table["Residual_Error"] / (predictions_table["Actual"] + 1e-6))
    * 100
)

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    metrics_summary.to_excel(writer, sheet_name="Model_Summary", index=False)
    comparison_df.to_excel(
        writer, sheet_name="Tournament_Comparison", index=False
    )
    fi_df.to_excel(writer, sheet_name="Feature_Importance", index=False)
    eda_summary.to_excel(writer, sheet_name="EDA_Summary", index=False)
    predictions_table.head(100000).to_excel(
        writer, sheet_name="Predictions_Sample", index=False
    )

print(f"SUCCESS: Excel report saved at -> {os.path.abspath(OUTPUT_XLSX)}")
print(
    f"\nPipeline Finished Successfully!\n"
    f"- Best Model: {best_model_name}\n"
    f"- Test RMSE : {final_rmse:.4f}\n"
    f"- Test R2   : {final_r2:.4f}\n"
    f"- Output Dir: {os.path.abspath(OUTPUT_DIR)}\n"
    f"- Plots Dir : {os.path.abspath(PLOTS_DIR)}"
)