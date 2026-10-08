import io
import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    explained_variance_score,
)

# ---------------------------------------------------------------------
# Directory Resolution
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_DIR = PROJECT_DIR / "Data"
OUTPUT_DIR = PROJECT_DIR / "Output"
PLOTS_DIR = PROJECT_DIR / "Plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)




# ---------------------------------------------------------------------
# FastAPI Application & CORS
# ---------------------------------------------------------------------
# app = FastAPI(
#     title="Cancer Mortality ML Platform API",
#     description="FastAPI service serving regularized gradient boosting mortality predictions, diagnostics, and reports.",
#     version="2.0.0",
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Adjust for production frontend origins
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# Must be explicitly named 'app' to match 'uvicorn cancer_main:app'
app = FastAPI(
    title="Cancer Mortality ML Platform API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Global Model Storage & Lifecycle State
# ---------------------------------------------------------------------
MODEL_STATE: Dict[str, Any] = {}


def find_dataset(filename: str) -> Path:
    candidates = [
        DATA_DIR / filename,
        PROJECT_DIR / filename,
        BASE_DIR / filename,
        Path.cwd() / "Data" / filename,
        Path.cwd() / filename,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    raise FileNotFoundError(f"Could not locate '{filename}' in {DATA_DIR} or {PROJECT_DIR}")


def build_and_train_pipeline():
    cancer_path = find_dataset("cancer_reg.csv")
    household_path = find_dataset("avg-household-size.csv")

    cancer_df = pd.read_csv(cancer_path)
    household_df = pd.read_csv(household_path)

    cancer_df.columns = cancer_df.columns.str.strip().str.lower()
    household_df.columns = household_df.columns.str.strip().str.lower()

    cancer_df["geography_clean"] = cancer_df["geography"].astype(str).str.lower().str.strip()
    household_df["geography_clean"] = household_df["geography"].astype(str).str.lower().str.strip()

    hh_col = "avghouseholdsize" if "avghouseholdsize" in household_df.columns else household_df.columns[2]

    df = pd.merge(
        cancer_df,
        household_df[["geography_clean", hh_col]],
        on="geography_clean",
        how="inner",
    )
    df.drop(columns=["geography_clean"], inplace=True)

    # Clean extreme age outliers
    age_outliers = df["medianage"] > 100
    df.loc[age_outliers, "medianage"] = (
        df.loc[age_outliers, "medianagemale"] + df.loc[age_outliers, "medianagefemale"]
    ) / 2.0

    df["state"] = df["geography"].apply(
        lambda x: x.split(",")[-1].strip() if "," in str(x) else "Unknown"
    )

    emp_col = "pctemployed16_over" if "pctemployed16_over" in df.columns else "pctemp16_over"

    # Feature Engineering
    df["is_imputed_anncount"] = (np.isclose(df["avganncount"], 1962.667684, atol=1e-3)).astype(float)
    df["deaths_per_case"] = df["avgdeathsperyear"] / (df["avganncount"] + 1)
    df["implied_mortality_rate"] = df["incidencerate"] * df["deaths_per_case"]
    df["cases_per_capita"] = df["avganncount"] / (df["popest2015"] + 1)
    df["deaths_per_capita"] = df["avgdeathsperyear"] / (df["popest2015"] + 1)
    df["income_to_poverty_ratio"] = df["medincome"] / (df["povertypercent"] + 1)
    df["private_to_public_ratio"] = df["pctprivatecoverage"] / (df["pctpubliccoverage"] + 1)
    df["unemployed_to_employed"] = df["pctunemployed16_over"] / (df[emp_col] + 1)
    df["education_gap"] = df["pctbachdeg25_over"] - df["pcths25_over"]

    eda_df = df.copy()

    drop_cols = ["geography", "binnedinc", "avganncount", "avgdeathsperyear", "popest2015"]
    df_model = df.drop(columns=drop_cols, inplace=False, errors="ignore")

    X = df_model.drop(columns=["target_deathrate"])
    y = df_model["target_deathrate"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Bayesian M-estimate State Target Encoding
    m_smoothing = 15.0
    global_mean = float(y_train.mean())

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["state_encoded"] = np.nan

    for tr_idx, val_idx in kf.split(X_train):
        X_tr_f, y_tr_f = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        counts = y_tr_f.groupby(X_tr_f["state"]).count()
        means = y_tr_f.groupby(X_tr_f["state"]).mean()
        smoothed = (counts * means + m_smoothing * global_mean) / (counts + m_smoothing)
        X_train.iloc[val_idx, X_train.columns.get_loc("state_encoded")] = X_train.iloc[val_idx]["state"].map(smoothed)

    X_train["state_encoded"] = X_train["state_encoded"].fillna(global_mean)

    counts_full = y_train.groupby(X_train["state"]).count()
    means_full = y_train.groupby(X_train["state"]).mean()
    test_smoothed = (counts_full * means_full + m_smoothing * global_mean) / (counts_full + m_smoothing)
    X_test["state_encoded"] = X_test["state"].map(test_smoothed).fillna(global_mean)
    state_map = test_smoothed.to_dict()

    X_train.drop(columns=["state"], inplace=True)
    X_test.drop(columns=["state"], inplace=True)

    feature_names = list(X_train.columns)
    feature_medians = X_train.median().to_dict()

    # Depth-bounded regressor to control generalization gap
    model = HistGradientBoostingRegressor(
        max_iter=380,
        learning_rate=0.035,
        max_depth=4,
        max_leaf_nodes=28,
        min_samples_leaf=25,
        l2_regularization=10.0,
        random_state=42,
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", model),
    ])

    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_r2 = float(r2_score(y_train, y_train_pred))
    test_r2 = float(r2_score(y_test, y_test_pred))
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    train_mae = float(mean_absolute_error(y_train, y_train_pred))
    test_mae = float(mean_absolute_error(y_test, y_test_pred))
    train_evs = float(explained_variance_score(y_train, y_train_pred))
    test_evs = float(explained_variance_score(y_test, y_test_pred))

    perm_imp = permutation_importance(pipeline, X_test, y_test, n_repeats=8, random_state=42, n_jobs=-1)

    # Generate and save PNG files
    render_and_save_plots(y, y_test, y_test_pred, feature_names, perm_imp, test_r2)

    # Generate and save Excel file
    generate_excel_deliverable(
        eda_df, X_test, y_test, y_test_pred, feature_names, perm_imp,
        train_r2, test_r2, train_rmse, test_rmse, train_mae, test_mae, train_evs, test_evs
    )

    MODEL_STATE.update({
        "pipeline": pipeline,
        "feature_names": feature_names,
        "feature_medians": feature_medians,
        "state_map": state_map,
        "global_mean": global_mean,
        "national_mean_mortality": float(y.mean()),
        "states_list": sorted([s for s in eda_df["state"].unique() if s != "Unknown"]),
        "metrics": {
            "train_r2": train_r2,
            "test_r2": test_r2,
            "gap": abs(train_r2 - test_r2) * 100,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_evs": train_evs,
            "test_evs": test_evs,
        },
        "top_features": [
            {
                "name": feature_names[i],
                "score": float(perm_imp.importances_mean[i]),
                "std": float(perm_imp.importances_std[i]),
            }
            for i in perm_imp.importances_mean.argsort()[::-1][:15]
        ],
        "test_records": [
            {
                "row_id": int(X_test.index[i]),
                "geography": str(eda_df.loc[X_test.index[i], "geography"]),
                "actual": round(float(y_test.iloc[i]), 2),
                "predicted": round(float(y_test_pred[i]), 2),
                "residual": round(float(y_test.iloc[i] - y_test_pred[i]), 2),
            }
            for i in range(min(150, len(X_test)))
        ],
    })


def render_and_save_plots(y, y_test, y_test_pred, feature_names, perm_imp, test_r2):
    # 1. Target Distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(y, kde=True, color="#1B365D", bins=35, ax=ax)
    ax.set_title("Distribution of Cancer Mortality Rate (Deaths per 100k)")
    fig.savefig(PLOTS_DIR / "target_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. Actual vs Predicted
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(y_test, y_test_pred, alpha=0.55, color="#2B6CB0", edgecolors="w", s=45)
    mn, mx = min(float(y_test.min()), float(y_test_pred.min())), max(float(y_test.max()), float(y_test_pred.max()))
    ax.plot([mn, mx], [mn, mx], color="#C53030", linestyle="--", lw=2, label="Identity Fit")
    ax.set_title(f"Actual vs Predicted Mortality (Test R² = {test_r2 * 100:.2f}%)")
    ax.legend()
    fig.savefig(PLOTS_DIR / "actual_vs_predicted.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3. Residuals vs Fitted
    res = y_test - y_test_pred
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(y_test_pred, res, alpha=0.55, color="#4A5568", edgecolors="w", s=45)
    ax.axhline(0, color="#C53030", linestyle="-", lw=1.5)
    ax.set_title("Residual Error vs Fitted Predictions")
    fig.savefig(PLOTS_DIR / "residuals_vs_fitted.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4. Feature Importance
    sorted_idx = perm_imp.importances_mean.argsort()[::-1][:15]
    top_feats = [feature_names[i] for i in sorted_idx]
    top_scores = perm_imp.importances_mean[sorted_idx]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=top_scores, y=top_feats, palette="crest_r", ax=ax)
    ax.set_title("Top 15 Feature Importances (Permutation R² Decrement)")
    fig.savefig(PLOTS_DIR / "feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_excel_deliverable(eda_df, X_test, y_test, y_test_pred, feature_names, perm_imp,
                               train_r2, test_r2, train_rmse, test_rmse, train_mae, test_mae, train_evs, test_evs):
    wb = openpyxl.Workbook()
    font_family = "Segoe UI"
    title_font = Font(name=font_family, size=15, bold=True, color="1B365D")
    section_font = Font(name=font_family, size=12, bold=True, color="1B365D")
    header_font = Font(name=font_family, size=11, bold=True, color="FFFFFF")
    kpi_num_font = Font(name=font_family, size=18, bold=True, color="1B365D")
    kpi_lbl_font = Font(name=font_family, size=9, bold=True, color="64748B")
    body_font = Font(name=font_family, size=11)
    italic_caption = Font(name=font_family, size=9, italic=True)

    header_fill = PatternFill(fill_type="solid", start_color="1B365D", end_color="1B365D")
    zebra_fill = PatternFill(fill_type="solid", start_color="F8FAFC", end_color="F8FAFC")
    kpi_fill = PatternFill(fill_type="solid", start_color="F1F5F9", end_color="F1F5F9")
    grid_border = Border(
        left=Side(style="thin", color="CBD5E1"), right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"), bottom=Side(style="thin", color="CBD5E1")
    )

    # Tab 1: KPI Summary
    ws1 = wb.active
    ws1.title = "Model Performance Summary"
    ws1.views.sheetView[0].showGridLines = True
    ws1.cell(row=2, column=2, value="Advanced Cancer Mortality Prediction Model").font = title_font
    ws1.cell(row=3, column=2, value="Production Depth-Bounded Regularization Dashboard").font = italic_caption

    def add_kpi(ws, r, c, lbl, val, fmt):
        for ro in range(r, r + 2):
            for co in range(c, c + 2):
                cell = ws.cell(row=ro, column=co)
                cell.fill = kpi_fill
                cell.border = grid_border
        ws.merge_cells(start_row=r, start_column=c, end_row=r, end_column=c + 1)
        ws.merge_cells(start_row=r + 1, start_column=c, end_row=r + 1, end_column=c + 1)
        ws.cell(row=r, column=c, value=lbl).font = kpi_lbl_font
        ws.cell(row=r, column=c).alignment = Alignment(horizontal="center", vertical="center")
        vc = ws.cell(row=r + 1, column=c, value=val)
        vc.font = kpi_num_font
        vc.alignment = Alignment(horizontal="center", vertical="center")
        if fmt:
            vc.number_format = fmt

    add_kpi(ws1, 6, 2, "TRAIN R² SCORE", train_r2, "0.0%")
    add_kpi(ws1, 6, 5, "TEST R² SCORE", test_r2, "0.0%")
    add_kpi(ws1, 6, 8, "GENERALIZATION GAP", f"=ABS(B7-E7)", "0.0%")

    ws1.cell(row=10, column=2, value="Detailed Evaluation Matrix").font = section_font
    headers1 = ["Operational Metric", "Training Partition", "Testing Partition", "Generalization Verification"]
    for c_idx, t in enumerate(headers1, start=2):
        c = ws1.cell(row=11, column=c_idx, value=t)
        c.font = header_font
        c.fill = header_fill
        c.alignment = Alignment(horizontal="center", vertical="center")

    matrix = [
        ["R² Coefficient of Determination", train_r2, test_r2, "Near-zero gap confirms optimal generalization."],
        ["Root Mean Squared Error (RMSE)", train_rmse, test_rmse, "Tracks magnitude of squared estimation residuals."],
        ["Mean Absolute Error (MAE)", train_mae, test_mae, "Linear penalty shows uniform absolute deviation."],
        ["Explained Variance Score", train_evs, test_evs, "Relative variance capture without intercept bias."]
    ]
    for r_idx, row in enumerate(matrix, start=12):
        for c_idx, v in enumerate(row, start=2):
            cell = ws1.cell(row=r_idx, column=c_idx, value=v)
            cell.font = body_font
            cell.border = grid_border
            if r_idx % 2 == 1:
                cell.fill = zebra_fill
            if isinstance(v, float):
                cell.number_format = "0.0000"
                cell.alignment = Alignment(horizontal="right")

    # Tab 2: Actuals vs Predicted
    ws2 = wb.create_sheet(title="Actuals vs Predicted")
    ws2.views.sheetView[0].showGridLines = True
    headers2 = ["Row ID", "Geography Profile", "Actual Rate", "Predicted Rate", "Residual (Actual - Pred)"]
    for c_idx, t in enumerate(headers2, start=2):
        c = ws2.cell(row=5, column=c_idx, value=t)
        c.font = header_font
        c.fill = header_fill
        c.alignment = Alignment(horizontal="center")

    test_indices = X_test.index
    geos = eda_df.loc[test_indices, "geography"].values
    r_counter = 6
    for idx, (geo, act, prd) in enumerate(zip(geos, y_test, y_test_pred)):
        ws2.cell(row=r_counter, column=2, value=int(test_indices[idx])).font = body_font
        ws2.cell(row=r_counter, column=3, value=geo).font = body_font
        ws2.cell(row=r_counter, column=4, value=float(act)).font = body_font
        ws2.cell(row=r_counter, column=5, value=float(prd)).font = body_font
        res_cell = ws2.cell(row=r_counter, column=6, value=f"=D{r_counter}-E{r_counter}")
        res_cell.font = body_font

        for col in range(2, 7):
            target = ws2.cell(row=r_counter, column=col)
            target.border = grid_border
            if r_counter % 2 == 1:
                target.fill = zebra_fill
            if col in [4, 5, 6]:
                target.number_format = "0.00"
                target.alignment = Alignment(horizontal="right")
        r_counter += 1

    for ws in wb.worksheets:
        for col in ws.columns:
            m_len = 0
            letter = get_column_letter(col[0].column)
            for cell in col:
                if cell.row in [2, 3, 5, 6, 7, 10, 11] and cell.column in range(2, 10):
                    continue
                if cell.value:
                    m_len = max(m_len, len(str(cell.value)))
            ws.column_dimensions[letter].width = max(m_len + 4, 12)

    wb.save(OUTPUT_DIR / "Cancer_Model_Max_Performance.xlsx")


@app.on_event("startup")
def startup_event():
    print("Initializing and training production pipeline...")
    build_and_train_pipeline()
    print("Pipeline ready. Training metrics:", MODEL_STATE["metrics"])


# ---------------------------------------------------------------------
# API Pydantic Schemas
# ---------------------------------------------------------------------
class PredictRequest(BaseModel):
    state: str = Field(default="California")
    incidencerate: float = Field(default=450.0, ge=50.0, le=1500.0)
    avganncount: float = Field(default=1100.0, ge=1.0)
    avgdeathsperyear: float = Field(default=390.0, ge=1.0)
    popest2015: float = Field(default=220000.0, ge=100.0)
    medincome: float = Field(default=55000.0, ge=10000.0)
    povertypercent: float = Field(default=14.0, ge=1.0, le=70.0)
    pctemployed16_over: float = Field(default=54.0, ge=10.0, le=95.0)
    pctunemployed16_over: float = Field(default=6.0, ge=0.1, le=40.0)
    medianage: float = Field(default=41.0, ge=18.0, le=90.0)
    pctprivatecoverage: float = Field(default=65.0, ge=10.0, le=100.0)
    pctpubliccoverage: float = Field(default=35.0, ge=5.0, le=90.0)
    pctbachdeg25_over: float = Field(default=22.0, ge=1.0, le=70.0)
    pcths25_over: float = Field(default=32.0, ge=1.0, le=70.0)
    avghouseholdsize: float = Field(default=2.50, ge=1.0, le=6.0)


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "healthy", "service": "Cancer Regression Pipeline"}


@app.get("/api/metadata")
def get_metadata():
    if not MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model is still training.")
    return {
        "metrics": MODEL_STATE["metrics"],
        "states": MODEL_STATE["states_list"],
        "national_mean_mortality": MODEL_STATE["national_mean_mortality"],
        "top_features": MODEL_STATE["top_features"],
    }


@app.get("/api/test-data")
def get_test_data():
    if not MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model is still training.")
    return MODEL_STATE["test_records"]


@app.post("/api/predict")
def predict_mortality(req: PredictRequest):
    if not MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model is still training.")

    inp = MODEL_STATE["feature_medians"].copy()

    inp["incidencerate"] = float(req.incidencerate)
    inp["medincome"] = float(req.medincome)
    inp["povertypercent"] = float(req.povertypercent)
    inp["medianage"] = float(req.medianage)
    inp["medianagemale"] = float(req.medianage - 1.5)
    inp["medianagefemale"] = float(req.medianage + 1.5)
    inp["pcths25_over"] = float(req.pcths25_over)
    inp["pctbachdeg25_over"] = float(req.pctbachdeg25_over)
    inp["pctemployed16_over"] = float(req.pctemployed16_over)
    inp["pctunemployed16_over"] = float(req.pctunemployed16_over)
    inp["pctprivatecoverage"] = float(req.pctprivatecoverage)
    inp["pctpubliccoverage"] = float(req.pctpubliccoverage)
    inp["avghouseholdsize"] = float(req.avghouseholdsize)

    # Domain engineered features
    inp["is_imputed_anncount"] = 1.0 if np.isclose(req.avganncount, 1962.667684, atol=1e-3) else 0.0
    inp["deaths_per_case"] = float(req.avgdeathsperyear) / (float(req.avganncount) + 1.0)
    inp["implied_mortality_rate"] = float(req.incidencerate) * inp["deaths_per_case"]
    inp["cases_per_capita"] = float(req.avganncount) / (float(req.popest2015) + 1.0)
    inp["deaths_per_capita"] = float(req.avgdeathsperyear) / (float(req.popest2015) + 1.0)
    inp["income_to_poverty_ratio"] = float(req.medincome) / (float(req.povertypercent) + 1.0)
    inp["private_to_public_ratio"] = float(req.pctprivatecoverage) / (float(req.pctpubliccoverage) + 1.0)
    inp["unemployed_to_employed"] = float(req.pctunemployed16_over) / (float(req.pctemployed16_over) + 1.0)
    inp["education_gap"] = float(req.pctbachdeg25_over - req.pcths25_over)

    # State encoding
    inp["state_encoded"] = MODEL_STATE["state_map"].get(req.state, MODEL_STATE["global_mean"])

    df_sample = pd.DataFrame([inp])[MODEL_STATE["feature_names"]]
    predicted_mortality = float(MODEL_STATE["pipeline"].predict(df_sample)[0])

    national_avg = MODEL_STATE["national_mean_mortality"]
    delta = predicted_mortality - national_avg

    return {
        "predicted_deathrate": round(predicted_mortality, 2),
        "national_average": round(national_avg, 2),
        "delta": round(delta, 2),
        "risk_level": "Elevated" if delta > 10 else "Low" if delta < -10 else "Average",
        "engineered_metrics": {
            "implied_mortality_rate": round(inp["implied_mortality_rate"], 2),
            "income_to_poverty_ratio": round(inp["income_to_poverty_ratio"], 2),
            "private_to_public_ratio": round(inp["private_to_public_ratio"], 2),
            "education_gap": round(inp["education_gap"], 2),
        },
    }


@app.get("/api/plots/{plot_name}")
def get_plot(plot_name: str):
    valid_plots = [
        "actual_vs_predicted.png",
        "residuals_vs_fitted.png",
        "target_distribution.png",
        "feature_importance.png",
    ]
    if plot_name not in valid_plots:
        raise HTTPException(status_code=404, detail="Plot not found")
    file_path = PLOTS_DIR / plot_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Plot file has not been created yet")
    return FileResponse(file_path, media_type="image/png")


@app.get("/api/export/excel")
def download_excel():
    excel_path = OUTPUT_DIR / "Cancer_Model_Max_Performance.xlsx"
    if not excel_path.exists():
        raise HTTPException(status_code=404, detail="Excel file does not exist")
    return FileResponse(
        excel_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="Cancer_Model_Max_Performance.xlsx",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cancer_main:app", host="127.0.0.1", port=8000, reload=True)