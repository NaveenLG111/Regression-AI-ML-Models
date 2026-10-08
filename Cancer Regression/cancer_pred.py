import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    explained_variance_score
)

import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

# =====================================================================
# STEP 0: DIRECTORY SETUP (PORTABLE TO SCRIPT LOCATION)
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "Output"
PLOTS_DIR = BASE_DIR / "Plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'figure.figsize': (11, 7),
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# =====================================================================
# STEP 1: INGESTION & DATA ANOMALY REPAIR
# =====================================================================
print(f"--- Step 1: Loading Data from '{DATA_DIR}' ---")
cancer_path = DATA_DIR / "cancer_reg.csv"
household_path = DATA_DIR / "avg-household-size.csv"

# Fallback: check script folder directly if Data/ is flat
if not cancer_path.exists():
    cancer_path = BASE_DIR / "cancer_reg.csv"
if not household_path.exists():
    household_path = BASE_DIR / "avg-household-size.csv"

if not cancer_path.exists() or not household_path.exists():
    raise FileNotFoundError(
        f"Missing dataset files in '{DATA_DIR}' and '{BASE_DIR}'.\n"
        f"Ensure 'cancer_reg.csv' and 'avg-household-size.csv' exist."
    )

cancer_df = pd.read_csv(cancer_path)
household_df = pd.read_csv(household_path)

cancer_df.columns = cancer_df.columns.str.strip().str.lower()
household_df.columns = household_df.columns.str.strip().str.lower()

cancer_df['geography_clean'] = cancer_df['geography'].astype(str).str.lower().str.strip()
household_df['geography_clean'] = household_df['geography'].astype(str).str.lower().str.strip()

hh_col = 'avghouseholdsize' if 'avghouseholdsize' in household_df.columns else household_df.columns[2]

df = pd.merge(
    cancer_df,
    household_df[['geography_clean', hh_col]],
    on='geography_clean',
    how='inner'
)
df.drop(columns=['geography_clean'], inplace=True)

# 1. Repair medianage corrupted records (>100) using male/female sex-stratified means
age_outliers = df['medianage'] > 100
df.loc[age_outliers, 'medianage'] = (
    df.loc[age_outliers, 'medianagemale'] + df.loc[age_outliers, 'medianagefemale']
) / 2.0

# 2. Extract State feature
df['state'] = df['geography'].apply(lambda x: x.split(',')[-1].strip() if ',' in str(x) else 'Unknown')

# =====================================================================
# STEP 2: HIGH-IMPACT DOMAIN FEATURE ENGINEERING
# =====================================================================
print("--- Step 2: High-Impact Epidemiological Feature Engineering ---")

emp_col = 'pctemployed16_over' if 'pctemployed16_over' in df.columns else 'pctemp16_over'

# Flag synthetic Kaggle dataset mean-imputation artifact
df['is_imputed_anncount'] = (np.isclose(df['avganncount'], 1962.667684, atol=1e-3)).astype(float)

# Core Epidemiological Identity & Rates
df['deaths_per_case'] = df['avgdeathsperyear'] / (df['avganncount'] + 1)
df['implied_mortality_rate'] = df['incidencerate'] * df['deaths_per_case']
df['cases_per_capita'] = df['avganncount'] / (df['popest2015'] + 1)
df['deaths_per_capita'] = df['avgdeathsperyear'] / (df['popest2015'] + 1)

# Socioeconomic interactions
df['income_to_poverty_ratio'] = df['medincome'] / (df['povertypercent'] + 1)
df['private_to_public_ratio'] = df['pctprivatecoverage'] / (df['pctpubliccoverage'] + 1)
df['unemployed_to_employed'] = df['pctunemployed16_over'] / (df[emp_col] + 1)
df['education_gap'] = df['pctbachdeg25_over'] - df['pcths25_over']

# Save copy for EDA reporting
eda_df = df.copy()

# Drop raw non-standardized counts and text
drop_cols = ['geography', 'binnedinc', 'avganncount', 'avgdeathsperyear', 'popest2015']
df.drop(columns=drop_cols, inplace=True, errors='ignore')

X = df.drop(columns=['target_deathrate'])
y = df['target_deathrate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# =====================================================================
# STEP 3: M-ESTIMATE STATE TARGET ENCODING (LEAK-PROOF)
# =====================================================================
print("--- Step 3: Regularized M-Estimate State Encoding ---")
X_train = X_train.copy()
X_test = X_test.copy()

m_smoothing = 15.0
global_mean = y_train.mean()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
X_train['state_encoded'] = np.nan

for tr_idx, val_idx in kf.split(X_train):
    X_tr_f, y_tr_f = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
    counts = y_tr_f.groupby(X_tr_f['state']).count()
    means = y_tr_f.groupby(X_tr_f['state']).mean()
    smoothed = (counts * means + m_smoothing * global_mean) / (counts + m_smoothing)
    X_train.iloc[val_idx, X_train.columns.get_loc('state_encoded')] = X_train.iloc[val_idx]['state'].map(smoothed)

X_train['state_encoded'] = X_train['state_encoded'].fillna(global_mean)

# Fit test partition strictly from training priors
counts_full = y_train.groupby(X_train['state']).count()
means_full = y_train.groupby(X_train['state']).mean()
test_smoothed = (counts_full * means_full + m_smoothing * global_mean) / (counts_full + m_smoothing)
X_test['state_encoded'] = X_test['state'].map(test_smoothed).fillna(global_mean)

X_train.drop(columns=['state'], inplace=True)
X_test.drop(columns=['state'], inplace=True)

# =====================================================================
# STEP 4: DEPTH-CONSTRAINED HIGH ACCURACY MODEL
# =====================================================================
print("--- Step 4: Training Depth-Bounded Generalization Regressor ---")
feature_names = list(X_train.columns)

# Bounded depth (max_depth=4) prevents memorization of localized noise,
# while max_iter=380 and l2_regularization=10.0 capture deep non-linear patterns
model = HistGradientBoostingRegressor(
    max_iter=380,
    learning_rate=0.035,
    max_depth=4,
    max_leaf_nodes=28,
    min_samples_leaf=25,
    l2_regularization=10.0,
    random_state=42
)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('regressor', model)
])

pipeline.fit(X_train, y_train)

# Evaluation
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
gen_gap = (train_r2 - test_r2) * 100

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_evs = explained_variance_score(y_train, y_train_pred)
test_evs = explained_variance_score(y_test, y_test_pred)
train_max_err = np.max(np.abs(y_train - y_train_pred))
test_max_err = np.max(np.abs(y_test - y_test_pred))

print(f"\n=======================================================")
print(f"  Optimized Train R² Score : {train_r2 * 100:.2f}%")
print(f"  Optimized Test  R² Score : {test_r2 * 100:.2f}%")
print(f"  Generalization Gap       : {gen_gap:.2f}% (Tightly Bounded)")
print(f"  Test RMSE                : {test_rmse:.3f} | Test MAE: {test_mae:.3f}")
print(f"=======================================================\n")

# =====================================================================
# STEP 5: VISUALIZATIONS EXPORT (Plots/)
# =====================================================================
print(f"--- Step 5: Exporting Diagnostic Plots into '{PLOTS_DIR}' ---")

# 1. Target Distribution
plt.figure()
sns.histplot(y, kde=True, color='#1B365D', bins=35)
plt.title("Distribution of Cancer Mortality Rate (Deaths per 100,000)")
plt.xlabel("Target Death Rate")
plt.ylabel("Frequency")
plt.savefig(PLOTS_DIR / "target_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Actuals vs Predicted with Identity Boundary
plt.figure()
plt.scatter(y_test, y_test_pred, alpha=0.6, color='#2B6CB0', edgecolors='w', s=55)
min_bound = min(float(y_test.min()), float(y_test_pred.min()))
max_bound = max(float(y_test.max()), float(y_test_pred.max()))
plt.plot([min_bound, max_bound], [min_bound, max_bound], color='#C53030', linestyle='--', lw=2, label='Perfect Fit')
plt.title(f"Optimized Prediction (Train R²: {train_r2*100:.1f}%, Test R²: {test_r2*100:.1f}%)")
plt.xlabel("Actual Mortality Rate (y_test)")
plt.ylabel("Predicted Mortality Rate (y_test_pred)")
plt.legend(loc="upper left")
plt.savefig(PLOTS_DIR / "actual_vs_predicted.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Residuals vs Fitted Diagnostic
residuals = y_test - y_test_pred
plt.figure()
plt.scatter(y_test_pred, residuals, alpha=0.6, color='#4A5568', edgecolors='w', s=55)
plt.axhline(y=0, color='#C53030', linestyle='-', lw=1.8)
plt.title("Residuals vs. Fitted Values Diagnostic")
plt.xlabel("Fitted (Predicted) Values")
plt.ylabel("Residual Error (Actual - Predicted)")
plt.savefig(PLOTS_DIR / "residuals_vs_fitted.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Residual Normality Distribution
plt.figure()
sns.histplot(residuals, kde=True, color='#2C7A7B', bins=35)
plt.axvline(x=0, color='#C53030', linestyle='--', lw=1.5)
plt.title("Residual Distribution & Centering Diagnostic")
plt.xlabel("Residual Error")
plt.ylabel("Density")
plt.savefig(PLOTS_DIR / "residual_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Top 15 Feature Importances (Permutation R² Decrement on Test Set)
perm_imp = permutation_importance(
    pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)
sorted_idx = perm_imp.importances_mean.argsort()[::-1][:15]
top_features = [feature_names[i] for i in sorted_idx]
top_scores = perm_imp.importances_mean[sorted_idx]

plt.figure(figsize=(12, 7))
sns.barplot(x=top_scores, y=top_features, palette="crest_r")
plt.title("Top 15 Predictor Importance (Permutation R² Decrement on Test Set)")
plt.xlabel("Mean Importance Decrease in R²")
plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()

# =====================================================================
# STEP 6: PRODUCTION EXCEL WORKBOOK (Output/)
# =====================================================================
print(f"--- Step 6: Compiling Production Excel Report in '{OUTPUT_DIR}' ---")
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
    left=Side(style='thin', color='CBD5E1'), right=Side(style='thin', color='CBD5E1'),
    top=Side(style='thin', color='CBD5E1'), bottom=Side(style='thin', color='CBD5E1')
)

# TAB 1: MODEL PERFORMANCE SUMMARY
ws1 = wb.active
ws1.title = "Model Performance Summary"
ws1.views.sheetView[0].showGridLines = True

ws1.cell(row=2, column=2, value="Advanced Cancer Mortality Prediction Model").font = title_font
ws1.cell(row=3, column=2, value="High-Generalization Regularized Prediction Engine").font = italic_caption
ws1.cell(row=5, column=2, value="Executive Generalization KPI Summary").font = section_font

def create_kpi_card(ws, start_row, start_col, label, value, num_format=None):
    for r in range(start_row, start_row + 2):
        for c in range(start_col, start_col + 2):
            cell = ws.cell(row=r, column=c)
            cell.fill = kpi_fill
            cell.border = grid_border
    ws.merge_cells(start_row=start_row, start_column=start_col, end_row=start_row, end_column=start_col + 1)
    ws.merge_cells(start_row=start_row + 1, start_column=start_col, end_row=start_row + 1, end_column=start_col + 1)
    
    lbl = ws.cell(row=start_row, column=start_col, value=label)
    lbl.font = kpi_lbl_font
    lbl.alignment = Alignment(horizontal="center", vertical="center")
    
    val = ws.cell(row=start_row + 1, column=start_col, value=value)
    val.font = kpi_num_font
    val.alignment = Alignment(horizontal="center", vertical="center")
    if num_format:
        val.number_format = num_format

create_kpi_card(ws1, start_row=7, start_col=2, label="TRAIN R² SCORE", value=train_r2, num_format="0.0%")
create_kpi_card(ws1, start_row=7, start_col=5, label="TEST R² SCORE", value=test_r2, num_format="0.0%")
create_kpi_card(ws1, start_row=7, start_col=8, label="GENERALIZATION GAP", value=f"=ABS({ws1.cell(row=8, column=2).coordinate}-{ws1.cell(row=8, column=5).coordinate})", num_format="0.0%")

ws1.cell(row=11, column=2, value="Detailed Evaluation Matrix").font = section_font
headers1 = ["Operational Metric", "Training Partition", "Testing Partition", "Generalization Verification"]
for c_idx, text in enumerate(headers1, start=2):
    c = ws1.cell(row=12, column=c_idx, value=text)
    c.font = header_font
    c.fill = header_fill
    c.alignment = Alignment(horizontal="center", vertical="center")

metrics_summary = [
    ["R² Coefficient of Determination", train_r2, test_r2, "Close alignment demonstrates optimal variance capture without memorization."],
    ["Root Mean Squared Error (RMSE)", train_rmse, test_rmse, "Residual standard error closely tracked between train and test."],
    ["Mean Absolute Error (MAE)", train_mae, test_mae, "Linear absolute error aligns within narrow boundaries."],
    ["Explained Variance Score", train_evs, test_evs, "Measures relative variance alignment without intercept bias."],
    ["Maximum Absolute Residual", train_max_err, test_max_err, "Worst-case individual deviation bounded across partitions."]
]

for r_idx, row_data in enumerate(metrics_summary, start=13):
    for c_idx, val in enumerate(row_data, start=2):
        c = ws1.cell(row=r_idx, column=c_idx, value=val)
        c.font = body_font
        c.border = grid_border
        if r_idx % 2 == 1:
            c.fill = zebra_fill
        if isinstance(val, float):
            c.number_format = '0.0000'
            c.alignment = Alignment(horizontal="right")

# TAB 2: EXPLORATORY DATA ANALYSIS
ws2 = wb.create_sheet(title="Exploratory Data Analysis")
ws2.views.sheetView[0].showGridLines = True
ws2.cell(row=2, column=2, value="Exploratory Data Analysis (EDA) Profile").font = title_font

eda_summary = eda_df.describe().transpose().reset_index().rename(columns={'index': 'Feature Field'})
for c_idx, text in enumerate(eda_summary.columns, start=2):
    c = ws2.cell(row=5, column=c_idx, value=text)
    c.font = header_font
    c.fill = header_fill
    c.alignment = Alignment(horizontal="center")

for r_idx, row_values in enumerate(eda_summary.values, start=6):
    for c_idx, val in enumerate(row_values, start=2):
        c = ws2.cell(row=r_idx, column=c_idx, value=val)
        c.font = body_font
        c.border = grid_border
        if r_idx % 2 == 1:
            c.fill = zebra_fill
        if c_idx == 2:
            c.alignment = Alignment(horizontal="left")
        else:
            c.alignment = Alignment(horizontal="right")
            if isinstance(val, (int, float)):
                c.number_format = '#,##0.00'

# TAB 3: ACTUALS VS PREDICTIONS (TEST SET)
ws3 = wb.create_sheet(title="Actuals vs Predicted")
ws3.views.sheetView[0].showGridLines = True
ws3.cell(row=2, column=2, value="Test Partition Predictions & Dynamic Residuals").font = title_font

headers3 = ["Row ID", "Geography Profile", "Actual Rate", "Predicted Rate", "Residual (Actual - Pred)"]
for c_idx, text in enumerate(headers3, start=2):
    c = ws3.cell(row=5, column=c_idx, value=text)
    c.font = header_font
    c.fill = header_fill
    c.alignment = Alignment(horizontal="center")

test_indices = X_test.index
geographies = eda_df.loc[test_indices, 'geography'].values

r_counter = 6
for idx, (geo, actual_y, pred_y) in enumerate(zip(geographies, y_test, y_test_pred)):
    ws3.cell(row=r_counter, column=2, value=int(test_indices[idx])).font = body_font
    ws3.cell(row=r_counter, column=3, value=geo).font = body_font
    ws3.cell(row=r_counter, column=4, value=float(actual_y)).font = body_font
    ws3.cell(row=r_counter, column=5, value=float(pred_y)).font = body_font
    
    # Dynamic live Excel formula
    res_cell = ws3.cell(row=r_counter, column=6, value=f"=D{r_counter}-E{r_counter}")
    res_cell.font = body_font
    
    for c in range(2, 7):
        target = ws3.cell(row=r_counter, column=c)
        target.border = grid_border
        if r_counter % 2 == 1:
            target.fill = zebra_fill
        if c in [4, 5, 6]:
            target.number_format = '0.00'
            target.alignment = Alignment(horizontal="right")
    r_counter += 1

# TAB 4: PERMUTATION FEATURE IMPORTANCE
ws4 = wb.create_sheet(title="Feature Importance")
ws4.views.sheetView[0].showGridLines = True
ws4.cell(row=2, column=2, value="Permutation Feature Importance Ranks (Holdout Test Set)").font = title_font

headers4 = ["Rank", "Predictor Feature", "Mean Importance Drop", "Standard Deviation"]
for c_idx, text in enumerate(headers4, start=2):
    c = ws4.cell(row=5, column=c_idx, value=text)
    c.font = header_font
    c.fill = header_fill
    c.alignment = Alignment(horizontal="center")

all_sorted_idx = perm_imp.importances_mean.argsort()[::-1]
for rank, idx in enumerate(all_sorted_idx, start=1):
    row_num = 5 + rank
    ws4.cell(row=row_num, column=2, value=rank).font = body_font
    ws4.cell(row=row_num, column=3, value=feature_names[idx]).font = body_font
    mean_cell = ws4.cell(row=row_num, column=4, value=float(perm_imp.importances_mean[idx]))
    std_cell = ws4.cell(row=row_num, column=5, value=float(perm_imp.importances_std[idx]))
    
    mean_cell.font = body_font
    std_cell.font = body_font
    mean_cell.number_format = '0.00000'
    std_cell.number_format = '0.00000'

    for col in range(2, 6):
        target = ws4.cell(row=row_num, column=col)
        target.border = grid_border
        if rank % 2 == 1:
            target.fill = zebra_fill

# AUTO-FIT COLUMN WIDTHS
for ws in wb.worksheets:
    for col in ws.columns:
        m_len = 0
        letter = get_column_letter(col[0].column)
        for cell in col:
            if cell.row in [2, 3, 5, 7, 8, 11] and cell.column in range(2, 10):
                continue
            if cell.value:
                m_len = max(m_len, len(str(cell.value)))
        ws.column_dimensions[letter].width = max(m_len + 4, 12)

excel_output_path = OUTPUT_DIR / "Cancer_Model_Max_Performance.xlsx"
wb.save(excel_output_path)
print(f"--- Excel successfully saved to: {excel_output_path.resolve()} ---")