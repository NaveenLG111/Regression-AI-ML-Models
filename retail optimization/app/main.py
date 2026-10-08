import io
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================================================
# PATH CONFIGURATION & FILE DISCOVERY
# =========================================================
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
ROOT_DIR = PARENT_DIR.parent

candidate_names = ["retail_price.csv", "retail.csv", "retail_prices.csv"]
candidate_folders = [
    CURRENT_DIR / "Data",
    CURRENT_DIR,
    PARENT_DIR / "Data",
    PARENT_DIR,
    ROOT_DIR / "Data",
    ROOT_DIR,
]

DATA_PATH = None
for folder in candidate_folders:
    for name in candidate_names:
        p = folder / name
        if p.is_file():
            DATA_PATH = p
            break
    if DATA_PATH:
        break

if not DATA_PATH:
    for name in candidate_names:
        matches = list(PARENT_DIR.glob(f"**/{name}")) or list(ROOT_DIR.glob(f"**/{name}"))
        if matches:
            DATA_PATH = matches[0]
            break

if not DATA_PATH or not DATA_PATH.exists():
    raise FileNotFoundError("Could not find 'retail_price.csv' or 'retail.csv'.")

# =========================================================
# MODEL TRAINING & INITIALIZATION
# =========================================================
print(f"[Backend] Training model using: {DATA_PATH}")
raw_df = pd.read_csv(DATA_PATH)

target_col = "unit_price"
drop_cols = [target_col, "month_year", "product_id"]

X = raw_df.drop(columns=[col for col in drop_cols if col in raw_df.columns])
y = raw_df[target_col]

num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", Ridge())
])

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

# Compute Metrics
y_train_pred = best_pipeline.predict(X_train)
y_test_pred = best_pipeline.predict(X_test)

metrics_summary = [
    {
        "metric": "R2 Score",
        "train": round(float(r2_score(y_train, y_train_pred)), 4),
        "test": round(float(r2_score(y_test, y_test_pred)), 4)
    },
    {
        "metric": "RMSE",
        "train": round(float(np.sqrt(mean_squared_error(y_train, y_train_pred))), 4),
        "test": round(float(np.sqrt(mean_squared_error(y_test, y_test_pred))), 4)
    },
    {
        "metric": "MAE",
        "train": round(float(mean_absolute_error(y_train, y_train_pred)), 4),
        "test": round(float(mean_absolute_error(y_test, y_test_pred)), 4)
    }
]

encoded_cat_features = (
    fitted_prep.named_transformers_["cat"]
    .get_feature_names_out(cat_features)
    .tolist()
)
feature_names = num_features + encoded_cat_features
coefficients = best_model.coef_
intercept = float(best_model.intercept_)

coefs_list = []
equation_terms = [f"{intercept:.4f}"]

for feat, coef in zip(feature_names, coefficients):
    coef_val = float(coef)
    coefs_list.append({"feature": feat, "coefficient": round(coef_val, 6)})
    sign = "+" if coef_val >= 0 else "-"
    equation_terms.append(f"{sign} ({abs(coef_val):.4f} * {feat})")

final_equation = "unit_price = " + " ".join(equation_terms)
coefs_list.sort(key=lambda x: abs(x["coefficient"]), reverse=True)

defaults = {col: float(raw_df[col].median()) for col in num_features}
categorical_options = {col: raw_df[col].dropna().unique().tolist() for col in cat_features}

# =========================================================
# FASTAPI APP INSTANCE
# =========================================================
app = FastAPI(
    title="Retail Price Optimization API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionInput(BaseModel):
    inputs: Dict[str, Any]


@app.get("/api/model-info")
def get_model_info():
    return {
        "status": "success",
        "optimal_alpha": grid_search.best_params_.get("model__alpha"),
        "metrics": metrics_summary,
        "final_equation": final_equation,
        "top_coefficients": coefs_list[:15],
        "all_coefficients": coefs_list,
        "features": {
            "numerical": num_features,
            "categorical": cat_features,
            "defaults": defaults,
            "categorical_options": categorical_options
        }
    }


@app.post("/api/predict")
def predict_price(payload: PredictionInput):
    try:
        input_data = payload.inputs
        input_df = pd.DataFrame([input_data])
        
        for col in num_features:
            if col not in input_df.columns:
                input_df[col] = defaults[col]
        for col in cat_features:
            if col not in input_df.columns:
                input_df[col] = categorical_options[col][0]

        pred = float(best_pipeline.predict(input_df)[0])
        return {
            "status": "success",
            "predicted_unit_price": round(pred, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/export-excel")
def export_excel():
    output = io.BytesIO()

    df_metrics = pd.DataFrame(metrics_summary)
    df_preds = pd.DataFrame({
        "Actual_Unit_Price": y_test.values,
        "Predicted_Unit_Price": y_test_pred,
        "Residual": y_test.values - y_test_pred
    })
    df_coefs = pd.DataFrame(coefs_list)
    df_eq = pd.DataFrame({"Model_Equation": [final_equation]})

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_metrics.to_excel(writer, sheet_name="Model_Metrics", index=False)
        df_preds.to_excel(writer, sheet_name="Test_Predictions", index=False)
        df_coefs.to_excel(writer, sheet_name="Coefficients", index=False)
        df_eq.to_excel(writer, sheet_name="Final_Equation", index=False)

    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=regression_results.xlsx"}
    )