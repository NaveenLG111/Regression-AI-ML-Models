import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------
# 1. Directory Structure Initialization
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "Data"
PLOTS_DIR = BASE_DIR / "Plots"
OUTPUT_DIR = BASE_DIR / "Output"

for folder in [DATA_DIR, PLOTS_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

data_filepath = DATA_DIR / "Walmart.csv"

# Fallback check: look in parent or script directory if not in Data/
if not data_filepath.exists():
    local_csv = BASE_DIR / "Walmart.csv"
    if local_csv.exists():
        import shutil
        shutil.copy(local_csv, data_filepath)
    else:
        raise FileNotFoundError(
            f"Please place 'Walmart.csv' inside '{DATA_DIR}' or '{BASE_DIR}'"
        )

# ---------------------------------------------------------
# 2. Data Ingestion
# ---------------------------------------------------------
df = pd.read_csv(data_filepath)

# Date feature decomposition
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
df['Day'] = df['Date'].dt.day

# Feature definitions
target_col = 'Weekly_Sales'
feature_cols = [
    'Store', 'Holiday_Flag', 'Temperature', 
    'Fuel_Price', 'CPI', 'Unemployment', 
    'Year', 'Month', 'WeekOfYear', 'Day'
]

X = df[feature_cols].copy()
y = df[target_col].copy()

# ---------------------------------------------------------
# 3. Train-Test Split & Scaling
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 4. Multi-Model Training & Benchmarking
# ---------------------------------------------------------
candidate_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=150, max_depth=16, random_state=42, n_jobs=-1),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42)
}

benchmark_results = []
trained_estimators = {}

for name, model in candidate_models.items():
    if "Regression" in name:
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
    trained_estimators[name] = model

    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

    benchmark_results.append({
        "Model": name,
        "Train R2": round(r2_score(y_train, y_pred_train), 4),
        "Test R2": round(test_r2, 4),
        "MAE": round(test_mae, 2),
        "RMSE": round(test_rmse, 2),
        "MAPE (%)": round(test_mape, 2)
    })

results_df = pd.DataFrame(benchmark_results).sort_values(by="Test R2", ascending=False)
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_estimators[best_model_name]

# Generate predictions with the best model
if "Regression" in best_model_name:
    y_test_pred = best_model.predict(X_test_scaled)
else:
    y_test_pred = best_model.predict(X_test)

residuals = y_test - y_test_pred

# Also fit interpretable Linear Regression to generate explicit equation
lin_reg = candidate_models["Linear Regression"]
lin_reg_intercept = lin_reg.intercept_
lin_reg_coefs = lin_reg.coef_

# ---------------------------------------------------------
# 5. Visualization Generation (Plots Folder)
# ---------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 1. Target Distribution
plt.figure(figsize=(9, 5))
sns.histplot(df[target_col], kde=True, color='#1f77b4', bins=40)
plt.title("Distribution of Weekly Sales", fontsize=13, fontweight='bold')
plt.xlabel("Weekly Sales ($)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "01_weekly_sales_distribution.png"), dpi=300)
plt.close()

# 2. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_matrix = df[feature_cols + [target_col]].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Feature Correlation Heatmap", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "02_correlation_heatmap.png"), dpi=300)
plt.close()

# 3. Actual vs Predicted Sales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.4, color='#2ca02c', edgecolors='k', linewidth=0.3)
max_val = max(y_test.max(), y_test_pred.max())
min_val = min(y_test.min(), y_test_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Identity Line (y=x)")
plt.title(f"Actual vs Predicted Sales ({best_model_name})", fontsize=13, fontweight='bold')
plt.xlabel("Actual Weekly Sales ($)")
plt.ylabel("Predicted Weekly Sales ($)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "03_actual_vs_predicted.png"), dpi=300)
plt.close()

# 4. Residual Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_test_pred, residuals, alpha=0.4, color='#d62728', edgecolors='k', linewidth=0.3)
axes[0].axhline(0, color='black', linestyle='--', lw=1.5)
axes[0].set_title("Residuals vs Predicted", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Predicted Weekly Sales ($)")
axes[0].set_ylabel("Residuals ($)")

sns.histplot(residuals, kde=True, ax=axes[1], color='#9467bd', bins=35)
axes[1].set_title("Residual Error Distribution", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Residual ($)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_residual_analysis.png"), dpi=300)
plt.close()

# 5. Feature Importance
if hasattr(best_model, "feature_importances_"):
    feat_importance = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    plt.figure(figsize=(9, 6))
    feat_importance.plot(kind='barh', color='#17becf', edgecolor='black')
    plt.title(f"Feature Importance ({best_model_name})", fontsize=13, fontweight='bold')
    plt.xlabel("Gini Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "05_feature_importance.png"), dpi=300)
    plt.close()

# ---------------------------------------------------------
# 6. Excel Multi-Sheet Reporting (Output Folder)
# ---------------------------------------------------------
excel_filepath = os.path.join(OUTPUT_DIR, "Walmart_Model_Results.xlsx")

test_indices = X_test.index
predictions_df = pd.DataFrame({
    "Date": df.loc[test_indices, 'Date'].dt.strftime('%Y-%m-%d'),
    "Store": df.loc[test_indices, 'Store'],
    "Actual_Sales": y_test.values,
    "Predicted_Sales": np.round(y_test_pred, 2),
    "Residual_Error": np.round(residuals.values, 2),
    "Absolute_Percentage_Error": np.round(np.abs(residuals.values / y_test.values) * 100, 2)
}).sort_values(by="Date")

eda_summary = df.describe().T.reset_index().rename(columns={"index": "Feature"})

with pd.ExcelWriter(excel_filepath, engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name="Model_Comparison", index=False)
    predictions_df.to_excel(writer, sheet_name="Test_Predictions", index=False)
    eda_summary.to_excel(writer, sheet_name="EDA_Statistics", index=False)
    
    if hasattr(best_model, "feature_importances_"):
        fi_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": best_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        fi_df.to_excel(writer, sheet_name="Feature_Importance", index=False)

print(f"Pipeline executed successfully.")
print(f"Visualizations saved to: {PLOTS_DIR}/")
print(f"Excel report generated: {excel_filepath}")
print(f"Optimal Model Selected: {best_model_name}")