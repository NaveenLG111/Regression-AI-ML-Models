import os
import shutil
from pathlib import Path
from typing import Dict, Any, List
from datetime import date

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

# ---------------------------------------------------------------------------
# App & Paths Configuration
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Walmart Sales Prediction Engine",
    description="Production-grade API for Walmart Weekly Sales forecasting using Random Forest Regression",
    version="1.0.0",
)

# Enable CORS for Next.js frontend (default dev port: 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
MODEL_DIR = BASE_DIR / "Models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "walmart_rf_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.joblib"

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
model: RandomForestRegressor = None
model_metrics: Dict[str, Any] = {}
feature_columns: List[str] = [
    "Store", "Holiday_Flag", "Temperature", "Fuel_Price",
    "CPI", "Unemployment", "Year", "Month", "WeekOfYear", "Day"
]


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class SalesPredictionRequest(BaseModel):
    store: int = Field(..., ge=1, le=45, description="Walmart Store ID (1-45)", example=1)
    date_str: str = Field(..., alias="date", description="Date in YYYY-MM-DD or DD-MM-YYYY format", example="2012-10-26")
    holiday_flag: int = Field(0, ge=0, le=1, description="1 if holiday week, else 0", example=0)
    temperature: float = Field(..., description="Average temperature in Fahrenheit", example=69.16)
    fuel_price: float = Field(..., description="Regional fuel price per gallon", example=3.506)
    cpi: float = Field(..., description="Consumer Price Index", example=223.44)
    unemployment: float = Field(..., description="Unemployment rate (%)", example=6.573)

    class Config:
        populate_by_name = True


class SalesPredictionResponse(BaseModel):
    predicted_weekly_sales: float
    store: int
    date: str
    is_holiday: bool
    engineered_features: Dict[str, Any]
    model_version: str


class ModelMetricsResponse(BaseModel):
    model_name: str
    r2_score: float
    mae: float
    rmse: float
    mape: float
    features_importance: Dict[str, float]


# ---------------------------------------------------------------------------
# Helper: Data Ingestion & Model Training
# ---------------------------------------------------------------------------
def locate_dataset() -> Path:
    target = DATA_DIR / "Walmart.csv"
    if target.exists():
        return target
    
    # Check parent/sibling locations
    alternatives = [
        BASE_DIR / "Walmart.csv",
        BASE_DIR.parent / "Walmart.csv",
        BASE_DIR.parent / "Data" / "Walmart.csv",
    ]
    for alt in alternatives:
        if alt.exists():
            shutil.copy(alt, target)
            return target

    raise FileNotFoundError(
        f"Walmart.csv could not be found in {DATA_DIR} or fallback locations."
    )


def train_and_cache_model():
    global model, model_metrics
    data_path = locate_dataset()
    df = pd.read_csv(data_path)

    # Date feature engineering
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Day"] = df["Date"].dt.day

    X = df[feature_columns]
    y = df["Weekly_Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=16,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    importances = dict(zip(feature_columns, [round(float(v), 4) for v in rf.feature_importances_]))
    # Sort descending
    importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    metrics = {
        "model_name": "Random Forest Regressor",
        "r2_score": round(float(r2), 4),
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "mape": round(float(mape), 2),
        "features_importance": importances,
    }

    # Persist
    joblib.dump(rf, MODEL_PATH)
    joblib.dump(metrics, METRICS_PATH)

    model = rf
    model_metrics = metrics


# ---------------------------------------------------------------------------
# Startup Event
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    global model, model_metrics
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            model_metrics = joblib.load(METRICS_PATH)
            print("[INFO] Pre-trained model & metrics loaded successfully.")
            return
        except Exception as e:
            print(f"[WARN] Failed to load cached model ({e}), retraining...")

    try:
        print("[INFO] Training Random Forest model on Walmart dataset...")
        train_and_cache_model()
        print(f"[INFO] Model trained successfully! Test R2: {model_metrics.get('r2_score')}")
    except Exception as exc:
        print(f"[ERROR] Could not initialize model: {exc}")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health Check"])
def root():
    return {
        "status": "online",
        "service": "Walmart Weekly Sales Prediction API",
        "model_ready": model is not None,
    }


@app.get("/metrics", response_model=ModelMetricsResponse, tags=["Model Info"])
def get_metrics():
    if not model_metrics:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently training or metrics are unavailable.",
        )
    return model_metrics


@app.post("/predict", response_model=SalesPredictionResponse, tags=["Inference"])
def predict_sales(payload: SalesPredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Regression model is not loaded yet.",
        )

    # Parse user date string safely
    try:
        parsed_dt = pd.to_datetime(payload.date_str)
    except Exception:
        try:
            parsed_dt = pd.to_datetime(payload.date_str, format="%d-%m-%Y")
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Date format invalid. Use YYYY-MM-DD or DD-MM-YYYY.",
            )

    year = int(parsed_dt.year)
    month = int(parsed_dt.month)
    week_of_year = int(parsed_dt.isocalendar().week)
    day = int(parsed_dt.day)

    # Construct feature frame matching training order exactly
    input_data = pd.DataFrame([{
        "Store": payload.store,
        "Holiday_Flag": payload.holiday_flag,
        "Temperature": payload.temperature,
        "Fuel_Price": payload.fuel_price,
        "CPI": payload.cpi,
        "Unemployment": payload.unemployment,
        "Year": year,
        "Month": month,
        "WeekOfYear": week_of_year,
        "Day": day
    }])[feature_columns]

    pred = float(model.predict(input_data)[0])

    return SalesPredictionResponse(
        predicted_weekly_sales=round(pred, 2),
        store=payload.store,
        date=parsed_dt.strftime("%Y-%m-%d"),
        is_holiday=bool(payload.holiday_flag == 1),
        engineered_features={
            "Year": year,
            "Month": month,
            "WeekOfYear": week_of_year,
            "Day": day,
        },
        model_version="RandomForestRegressor-v1"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)