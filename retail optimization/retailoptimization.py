import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================================================
# 1. ROBUST PATH CONFIGURATION (Script-Relative Anchoring)
# =========================================================
# SCRIPT_DIR will be: /Users/naveenlg/AI Models/Regression Models/retail optimization/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent  # /Users/naveenlg/AI Models/Regression Models/

# Define Output and Plots directories inside the script's folder
OUTPUT_DIR = SCRIPT_DIR / "Output"
PLOTS_DIR = SCRIPT_DIR / "Plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Candidate target filenames
candidate_names = ["retail_price.csv", "retail.csv", "retail_prices.csv"]

# Candidate locations to check
candidate_paths = [
    SCRIPT_DIR / "Data",
    SCRIPT_DIR,
    PROJECT_DIR / "Data",
    PROJECT_DIR,
]

data_file_path = None

# Step 1: Check standard paths
for folder in candidate_paths:
    for name in candidate_names:
        p = folder / name
        if p.is_file():
            data_file_path = p
            break
    if data_file_path:
        break

# Step 2: Fallback search if placed in an unexpected subfolder
if not data_file_path:
    for name in candidate_names:
        matches = list(SCRIPT_DIR.glob(f"**/{name}")) or list(PROJECT_DIR.glob(f"**/{name}"))
        if matches:
            data_file_path = matches[0]
            break

if not data_file_path or not data_file_path.exists():
    raise FileNotFoundError(
        f"\n[Error] Could not find 'retail_price.csv' or 'retail.csv'.\n"
        f"Searched in:\n"
        f" - {SCRIPT_DIR / 'Data'}\n"
        f" - {SCRIPT_DIR}\n"
        f" - {PROJECT_DIR / 'Data'}\n"
        f"Please verify where the CSV file is located or place it inside '{SCRIPT_DIR / 'Data'}'.\n"
    )

print(f"[Info] Loading dataset from: {data_file_path}")
df = pd.read_csv(data_file_path)

# =========================================================
# 2. FEATURE SEPARATION & PREPROCESSING
# =========================================================
target_col = "unit_price"
drop_cols = [target_col, "month_year", "product_id"]

X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df[target_col]

num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
    ]
)

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================================
# 3. PIPELINE SETUP & HYPERPARAMETER TUNING
# =========================================================
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
    cv=5,
    scoring="r2",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_pipeline = grid_search.best_estimator_
best_model = best_pipeline.named_steps["model"]
fitted_prep = best_pipeline.named_steps["prep"]

# =========================================================
# 4. EVALUATION & METRICS
# =========================================================
y_train_pred = best_pipeline.predict(X_train)
y_test_pred = best_pipeline.predict(X_test)

metrics_summary = {
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
}
df_metrics = pd.DataFrame(metrics_summary)

df_predictions = pd.DataFrame({
    "Actual_Unit_Price": y_test.values,
    "Predicted_Unit_Price": y_test_pred,
    "Residual": y_test.values - y_test_pred
}, index=y_test.index)

# =========================================================
# 5. FINAL REGRESSION EQUATION
# =========================================================
encoded_cat_features = (
    fitted_prep.named_transformers_["cat"]
    .get_feature_names_out(cat_features)
    .tolist()
)
feature_names = num_features + encoded_cat_features
coefficients = best_model.coef_
intercept = best_model.intercept_

df_coefs = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", key=abs, ascending=False)

equation_terms = [f"{intercept:.4f}"]
for feat, coef in zip(feature_names, coefficients):
    sign = "+" if coef >= 0 else "-"
    equation_terms.append(f"{sign} ({abs(coef):.4f} * {feat})")

final_equation = "unit_price = " + " ".join(equation_terms)

# =========================================================
# 6. EXCEL EXPORT
# =========================================================
excel_output_path = OUTPUT_DIR / "regression_results.xlsx"
with pd.ExcelWriter(excel_output_path, engine="openpyxl") as writer:
    df_metrics.to_excel(writer, sheet_name="Model_Metrics", index=False)
    df_predictions.to_excel(writer, sheet_name="Test_Predictions", index=True)
    df_coefs.to_excel(writer, sheet_name="Coefficients", index=False)
    pd.DataFrame({"Model_Equation": [final_equation]}).to_excel(
        writer, sheet_name="Final_Equation", index=False
    )

print(f"[Done] Excel saved to: {excel_output_path}")

# =========================================================
# 7. VISUALIZATIONS
# =========================================================
sns.set_theme(style="whitegrid")

# Plot 1: Actual vs Predicted
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_test_pred, alpha=0.6, color="#1f77b4", edgecolor="k")
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit")
plt.xlabel("Actual Unit Price")
plt.ylabel("Predicted Unit Price")
plt.title("Actual vs Predicted Unit Prices (Test Set)")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "actual_vs_predicted.png", dpi=300)
plt.close()

# Plot 2: Residual Distribution
plt.figure(figsize=(7, 5))
residuals = y_test - y_test_pred
sns.histplot(residuals, kde=True, color="#2ca02c")
plt.axvline(0, color="r", linestyle="--", lw=1.5)
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Density")
plt.title("Residual Distribution")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "residual_distribution.png", dpi=300)
plt.close()

# Plot 3: Top Feature Coefficients
plt.figure(figsize=(9, 6))
top_coefs = df_coefs.head(10)
sns.barplot(data=top_coefs, x="Coefficient", y="Feature", palette="viridis")
plt.title("Top 10 Feature Weights (Ridge Model)")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "top_feature_coefficients.png", dpi=300)
plt.close()

print(f"[Done] Plots saved to: {PLOTS_DIR}")

# =========================================================
# 8. PRINT RESULTS
# =========================================================
print("\n--- MODEL PERFORMANCE ---")
print(df_metrics.to_string(index=False))
print(f"\nOptimal Alpha: {grid_search.best_params_}")
print("\n--- FINAL REGRESSION EQUATION ---")
print(final_equation)