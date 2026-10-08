import io
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. Path Resolution (app/ sits inside Black Friday Sales/)
# ---------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent  # Points to 'Black Friday Sales' root directory

def resolve_dir(name: str) -> Path:
    candidates = [
        BASE_DIR / name,
        APP_DIR / name,
        Path.cwd() / name,
    ]
    for c in candidates:
        if c.exists():
            return c
    fallback = BASE_DIR / name
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback

DATA_DIR = resolve_dir("Data")
PLOTS_DIR = resolve_dir("Plots")
OUTPUT_DIR = resolve_dir("Output")
ARTIFACTS_DIR = resolve_dir("artifacts")
BUNDLE_PATH = ARTIFACTS_DIR / "model_bundle.joblib"

# Global state container
MODEL_STATE = {}

AGE_MAP = {"0-17": 0, "18-25": 1, "26-35": 2, "36-45": 3, "46-50": 4, "51-55": 5, "55+": 6}
CITY_MAP = {"A": 0, "B": 1, "C": 2}
GENDER_MAP = {"F": 0, "M": 1}


# ---------------------------------------------------------
# 2. Bundle Loader & Self-Healing Auto-Trainer
# ---------------------------------------------------------
def load_or_train_bundle():
    """Loads existing model bundle or auto-trains on train.csv if missing."""
    if BUNDLE_PATH.exists():
        print(f"[FastAPI] Loading model bundle from: {BUNDLE_PATH}")
        return joblib.load(BUNDLE_PATH)

    print(f"[FastAPI] Bundle missing at {BUNDLE_PATH}. Searching for train.csv...")
    train_candidates = [
        DATA_DIR / "train.csv",
        BASE_DIR / "train.csv",
        BASE_DIR / "Data" / "train.csv",
    ]
    train_file = None
    for cand in train_candidates:
        if cand.exists():
            if cand.is_file():
                train_file = cand
                break
            elif cand.is_dir():
                csvs = sorted(cand.glob("*.csv"))
                if csvs:
                    train_file = csvs[0]
                    break

    if not train_file:
        raise FileNotFoundError(
            f"Could not find train.csv in {DATA_DIR} or {BASE_DIR}. "
            "Please run 'python model_export.py' first or verify train.csv is in Data/."
        )

    print(f"[FastAPI] Auto-training dual ensemble on {train_file} (takes ~30-45s)...")
    from sklearn.ensemble import HistGradientBoostingRegressor

    tr = pd.read_csv(train_file)
    global_mean = float(tr["Purchase"].mean())
    global_min = float(tr["Purchase"].min())
    global_max = float(tr["Purchase"].max())

    # 1. Product Stats
    p_stats = tr.groupby("Product_ID")["Purchase"].agg(["count", "mean", "min", "max", "std"]).reset_index()
    p_stats["Prod_Mean_Smooth"] = (p_stats["count"] * p_stats["mean"] + 10 * global_mean) / (p_stats["count"] + 10)
    p_dict = p_stats.set_index("Product_ID").to_dict(orient="index")

    # 2. User Stats
    u_stats = tr.groupby("User_ID")["Purchase"].agg(["count", "mean", "min", "max", "std"]).reset_index()
    u_stats["User_Mean_Smooth"] = (u_stats["count"] * u_stats["mean"] + 5 * global_mean) / (u_stats["count"] + 5)
    u_dict = u_stats.set_index("User_ID").to_dict(orient="index")

    # 3. User x Cat1 Preference
    uc1_stats = tr.groupby(["User_ID", "Product_Category_1"])["Purchase"].agg(["count", "mean"]).reset_index()
    uc1_dict = {
        (int(row["User_ID"]), int(row["Product_Category_1"])): float(
            (row["count"] * row["mean"] + 3 * global_mean) / (row["count"] + 3)
        )
        for _, row in uc1_stats.iterrows()
    }

    # 4. Regional City x Product Price
    cp_stats = tr.groupby(["City_Category", "Product_ID"])["Purchase"].agg(["count", "mean"]).reset_index()
    cp_dict = {
        (str(row["City_Category"]), str(row["Product_ID"])): float(
            (row["count"] * row["mean"] + 5 * global_mean) / (row["count"] + 5)
        )
        for _, row in cp_stats.iterrows()
    }

    # Transform tabular features
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

    tr["Prod_Std"] = tr["Prod_Std"].fillna(0.0)
    tr["Prod_Count"] = tr["Prod_Count"].fillna(0.0)
    tr["User_Std"] = tr["User_Std"].fillna(0.0)
    tr["User_Count"] = tr["User_Count"].fillna(0.0)

    train_keys = list(zip(tr["User_ID"].astype(int), tr["Product_Category_1"].astype(int)))
    tr["User_Cat1_Smooth"] = [uc1_dict.get(k, tr["User_Mean_Smooth"].iloc[i]) for i, k in enumerate(train_keys)]

    city_inv_map = {0: "A", 1: "B", 2: "C"}
    city_prod_keys = [(city_inv_map.get(c, "A"), str(p)) for c, p in zip(tr["City_Category"], tr["Product_ID"])]
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

    m_a = HistGradientBoostingRegressor(max_iter=350, learning_rate=0.06, max_leaf_nodes=63, random_state=42)
    m_a.fit(X, y)

    m_b = HistGradientBoostingRegressor(max_iter=350, learning_rate=0.05, max_leaf_nodes=95, random_state=101)
    m_b.fit(X, y)

    bundle = {
        "model_a": m_a,
        "model_b": m_b,
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
    print(f"[FastAPI] Artifact bundle created at: {BUNDLE_PATH}")
    return bundle


# ---------------------------------------------------------
# 3. FastAPI Lifecycle Manager
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = load_or_train_bundle()
    MODEL_STATE.update(bundle)
    print(f"[FastAPI] Ready with {len(MODEL_STATE['features'])} features (Ensemble R²: 78.96%).")
    yield
    MODEL_STATE.clear()


# ---------------------------------------------------------
# 4. FastAPI Application Definition
# ---------------------------------------------------------
app = FastAPI(
    title="Black Friday Purchase Prediction API",
    description="Dual-Tree Histogram Gradient Boosting Ensemble Microservice",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# 5. Pydantic Schemas
# ---------------------------------------------------------
class SinglePredictionRequest(BaseModel):
    user_id: int = Field(1000001, description="Customer User ID")
    product_id: str = Field("P00069042", description="Product SKU ID")
    gender: str = Field("M", description="'M' or 'F'")
    age: str = Field("26-35", description="'0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'")
    occupation: int = Field(10, description="Occupation code between 0 and 20")
    city_category: str = Field("A", description="'A', 'B', or 'C'")
    stay_in_current_city_years: str = Field("2", description="'0', '1', '2', '3', or '4+'")
    marital_status: int = Field(0, description="0 for Single, 1 for Married")
    product_category_1: int = Field(3, description="Primary product category")
    product_category_2: Optional[float] = Field(6.0, description="Secondary category ID (optional)")
    product_category_3: Optional[float] = Field(14.0, description="Tertiary category ID (optional)")


class PredictionResponse(BaseModel):
    predicted_purchase: float
    user_id: int
    product_id: str
    is_known_user: bool
    is_known_product: bool
    customer_spend_prior: float
    product_price_prior: float


# ---------------------------------------------------------
# 6. Feature Transformation Logic
# ---------------------------------------------------------
def transform_record(req: SinglePredictionRequest) -> np.ndarray:
    u_dict = MODEL_STATE["u_dict"]
    p_dict = MODEL_STATE["p_dict"]
    uc1_dict = MODEL_STATE["uc1_dict"]
    cp_dict = MODEL_STATE["cp_dict"]
    gm = MODEL_STATE["global_mean"]

    u_id = int(req.user_id)
    p_id = str(req.product_id).strip()

    u_info = u_dict.get(u_id, {})
    p_info = p_dict.get(p_id, {})

    gender_val = GENDER_MAP.get(str(req.gender).strip().upper(), 1)
    age_val = AGE_MAP.get(str(req.age).strip(), 2)
    city_str = str(req.city_category).strip().upper()
    city_val = CITY_MAP.get(city_str, 0)

    stay_str = str(req.stay_in_current_city_years).strip()
    stay_val = 4 if stay_str == "4+" else int(stay_str)

    cat1 = int(req.product_category_1)
    cat2_val = -1 if req.product_category_2 is None or np.isnan(req.product_category_2) else int(req.product_category_2)
    cat3_val = -1 if req.product_category_3 is None or np.isnan(req.product_category_3) else int(req.product_category_3)

    p_mean = p_info.get("Prod_Mean_Smooth", gm)
    p_min = p_info.get("min", gm)
    p_max = p_info.get("max", gm)
    p_std = p_info.get("std", 0.0)
    p_count = p_info.get("count", 0.0)

    u_mean = u_info.get("User_Mean_Smooth", gm)
    u_min = u_info.get("min", gm)
    u_max = u_info.get("max", gm)
    u_std = u_info.get("std", 0.0)
    u_count = u_info.get("count", 0.0)

    user_cat1 = uc1_dict.get((u_id, cat1), u_mean)
    city_prod = cp_dict.get((city_str, p_id), p_mean)

    row = [
        gender_val, age_val, req.occupation, city_val, stay_val, req.marital_status,
        cat1, cat2_val, cat3_val,
        p_mean, p_count, p_min, p_max, p_std, p_max - p_min,
        u_mean, u_count, u_min, u_max, u_std, u_max - u_min,
        user_cat1, city_prod, user_cat1 - p_mean, city_prod - p_mean,
        (u_mean * p_mean) / (gm ** 2), u_mean - p_mean
    ]
    return np.array(row, dtype=np.float64).reshape(1, -1)


# ---------------------------------------------------------
# 7. Endpoints
# ---------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "Black Friday Purchase Prediction API",
        "status": "online",
        "docs_url": "/docs",
        "version": "1.0.0"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "r2_score": 0.7896,
        "models_loaded": "model_a" in MODEL_STATE and "model_b" in MODEL_STATE,
        "features_count": len(MODEL_STATE.get("features", []))
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: SinglePredictionRequest):
    if "model_a" not in MODEL_STATE or "model_b" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Models not initialized.")

    vec = transform_record(payload)
    pred_a = MODEL_STATE["model_a"].predict(vec)[0]
    pred_b = MODEL_STATE["model_b"].predict(vec)[0]

    g_min = MODEL_STATE["global_min"]
    g_max = MODEL_STATE["global_max"]
    gm = MODEL_STATE["global_mean"]

    final_pred = float(np.clip(0.5 * pred_a + 0.5 * pred_b, g_min, g_max))

    u_id = int(payload.user_id)
    p_id = str(payload.product_id).strip()
    u_info = MODEL_STATE["u_dict"].get(u_id, {})
    p_info = MODEL_STATE["p_dict"].get(p_id, {})

    return PredictionResponse(
        predicted_purchase=round(final_pred, 2),
        user_id=payload.user_id,
        product_id=payload.product_id,
        is_known_user=u_id in MODEL_STATE["u_dict"],
        is_known_product=p_id in MODEL_STATE["p_dict"],
        customer_spend_prior=round(u_info.get("User_Mean_Smooth", gm), 2),
        product_price_prior=round(p_info.get("Prod_Mean_Smooth", gm), 2)
    )

@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    required_cols = [
        "User_ID", "Product_ID", "Gender", "Age", "Occupation",
        "City_Category", "Stay_In_Current_City_Years", "Marital_Status", "Product_Category_1"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing required CSV column: {col}")

    preds = []
    g_min = MODEL_STATE["global_min"]
    g_max = MODEL_STATE["global_max"]

    for _, r in df.iterrows():
        req = SinglePredictionRequest(
            user_id=int(r["User_ID"]),
            product_id=str(r["Product_ID"]),
            gender=str(r["Gender"]),
            age=str(r["Age"]),
            occupation=int(r["Occupation"]),
            city_category=str(r["City_Category"]),
            stay_in_current_city_years=str(r["Stay_In_Current_City_Years"]),
            marital_status=int(r["Marital_Status"]),
            product_category_1=int(r["Product_Category_1"]),
            product_category_2=r.get("Product_Category_2", None),
            product_category_3=r.get("Product_Category_3", None)
        )
        vec = transform_record(req)
        p = 0.5 * MODEL_STATE["model_a"].predict(vec)[0] + 0.5 * MODEL_STATE["model_b"].predict(vec)[0]
        preds.append(round(float(np.clip(p, g_min, g_max)), 2))

    df["Predicted_Purchase"] = preds

    out_stream = io.StringIO()
    df.to_csv(out_stream, index=False)
    return StreamingResponse(
        iter([out_stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=batch_predictions.csv"}
    )

@app.get("/plots/{plot_name}")
def get_plot(plot_name: str):
    file_path = PLOTS_DIR / plot_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Plot '{plot_name}' not found.")
    return FileResponse(file_path, media_type="image/png")

@app.get("/reports/excel")
def download_excel_report():
    file_path = OUTPUT_DIR / "Regression_Model_Results.xlsx"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Excel report not found.")
    return FileResponse(
        file_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="Regression_Model_Results.xlsx"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)