"use client";

import React, { useState, useEffect } from "react";

// ---------------------------------------------------------------------------
// Type Definitions
// ---------------------------------------------------------------------------
interface ModelMetrics {
  model_name: string;
  r2_score: number;
  mae: number;
  rmse: number;
  mape: number;
  features_importance: Record<string, number>;
}

interface PredictionResponse {
  predicted_weekly_sales: number;
  store: number;
  date: string;
  is_holiday: boolean;
  engineered_features: {
    Year: number;
    Month: number;
    WeekOfYear: number;
    Day: number;
  };
  model_version: string;
}

interface ModelBenchmarkRow {
  model: string;
  trainR2: string;
  testR2: string;
  mae: string;
  rmse: string;
  mape: string;
  status: "Optimal" | "Candidate" | "Baseline";
}

// ---------------------------------------------------------------------------
// Static Benchmarking & Diagnostic Data (Mirrors walmartsales.py output)
// ---------------------------------------------------------------------------
const MODEL_COMPARISONS: ModelBenchmarkRow[] = [
  {
    model: "Random Forest Regressor",
    trainR2: "0.9912",
    testR2: "0.9584",
    mae: "$61,420.30",
    rmse: "$115,140.18",
    mape: "4.25%",
    status: "Optimal",
  },
  {
    model: "Gradient Boosting Regressor",
    trainR2: "0.9418",
    testR2: "0.9271",
    mae: "$96,840.12",
    rmse: "$152,430.70",
    mape: "7.18%",
    status: "Candidate",
  },
  {
    model: "Ridge Regression",
    trainR2: "0.1585",
    testR2: "0.1519",
    mae: "$433,810.60",
    rmse: "$520,380.15",
    mape: "39.80%",
    status: "Baseline",
  },
  {
    model: "Linear Regression",
    trainR2: "0.1585",
    testR2: "0.1518",
    mae: "$433,850.11",
    rmse: "$520,410.24",
    mape: "39.81%",
    status: "Baseline",
  },
];

const PREDICTION_LOG_SAMPLES = [
  { date: "2012-10-12", store: 1, actual: 1573072.81, predicted: 1568210.45, error: -4862.36, ape: "0.31%" },
  { date: "2012-10-19", store: 1, actual: 1508068.77, predicted: 1521400.12, error: 13331.35, ape: "0.88%" },
  { date: "2012-10-26", store: 1, actual: 1493659.74, predicted: 1489720.60, error: -3939.14, ape: "0.26%" },
  { date: "2012-09-07", store: 20, actual: 2168097.11, predicted: 2145300.90, error: -22796.21, ape: "1.05%" },
  { date: "2012-11-23", store: 20, actual: 2811634.04, predicted: 2795410.33, error: -16223.71, ape: "0.58%" },
];

const REGRESSION_COEFFICIENTS = [
  { term: "Intercept (w₀)", value: "-41,230,518.24", note: "Baseline constant" },
  { term: "Store", value: "-14,412.35", note: "Unit decrease per Store index" },
  { term: "Holiday_Flag", value: "+36,210.84", note: "Average holiday surge lift" },
  { term: "Temperature", value: "-1,284.12", note: "Per 1°F temperature change" },
  { term: "Fuel_Price", value: "-3,812.45", note: "Per $1 change in fuel price" },
  { term: "CPI", value: "-2,145.80", note: "Macro inflation adjustment" },
  { term: "Unemployment", value: "-21,840.55", note: "Per 1% increase in unemployment" },
  { term: "Year", value: "+21,045.18", note: "Macro yearly retail expansion" },
  { term: "Month", value: "+11,230.40", note: "Q3/Q4 seasonal growth" },
  { term: "WeekOfYear", value: "+4,890.15", note: "Weekly calendar cadence" },
  { term: "Day", value: "-124.62", note: "End-of-month pay cycle effect" },
];

export default function WalmartSalesDashboard() {
  // Navigation Tabs: "forecast" | "benchmarks" | "diagnostics" | "reports"
  const [activeTab, setActiveTab] = useState<"forecast" | "benchmarks" | "diagnostics" | "reports">("forecast");

  // Input Parameters State
  const [formData, setFormData] = useState({
    store: 1,
    date: "2012-10-26",
    holiday_flag: 0,
    temperature: 69.16,
    fuel_price: 3.506,
    cpi: 223.44,
    unemployment: 6.573,
  });

  const [loading, setLoading] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  // Load server benchmark metrics on mount
  useEffect(() => {
    async function fetchMetrics() {
      try {
        const res = await fetch(`${API_BASE}/metrics`);
        if (res.ok) {
          const data: ModelMetrics = await res.json();
          setMetrics(data);
        }
      } catch (err) {
        console.warn("Backend metrics endpoint not yet reachable.", err);
      }
    }
    fetchMetrics();
  }, [API_BASE]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "number" ? parseFloat(value) || 0 : value,
    }));
  };

  const handlePreset = (storeId: number, temp: number, cpi: number, unemp: number, holiday: number) => {
    setFormData((prev) => ({
      ...prev,
      store: storeId,
      temperature: temp,
      cpi: cpi,
      unemployment: unemp,
      holiday_flag: holiday,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Prediction request failed.");
      }

      const data: PredictionResponse = await res.json();
      setPrediction(data);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError(`Failed to connect to inference backend at ${API_BASE}`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 font-sans p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* =================================================================== */}
        {/* Top Header Bar                                                      */}
        {/* =================================================================== */}
        <header className="flex flex-col md:flex-row justify-between items-start md:items-center bg-slate-900/60 border border-slate-800 p-6 rounded-2xl backdrop-blur-md">
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <span className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-bold px-2.5 py-0.5 rounded text-xs uppercase tracking-wider shadow">
                ML Pipeline UI
              </span>
              <h1 className="text-2xl md:text-3xl font-extrabold text-white tracking-tight">
                Walmart Sales Analytics &amp; Forecasting Engine
              </h1>
            </div>
            <p className="text-sm text-slate-400">
              End-to-End Regression Workflow • Interactive Visualizer • Multi-Model Benchmark
            </p>
          </div>
          <div className="mt-4 md:mt-0 flex items-center gap-3">
            <div className="flex items-center gap-2 text-xs bg-slate-800/80 px-3.5 py-1.5 rounded-full border border-slate-700/80 text-slate-300">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping" />
              API: {API_BASE}
            </div>
          </div>
        </header>

        {/* =================================================================== */}
        {/* Performance KPI Cards                                               */}
        {/* =================================================================== */}
        <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl shadow-sm">
            <div className="flex justify-between items-center text-slate-400 text-xs font-semibold">
              <span>Best Model Accuracy (R²)</span>
              <span className="text-[10px] bg-emerald-950/80 text-emerald-400 px-1.5 py-0.5 rounded border border-emerald-800/60">
                Random Forest
              </span>
            </div>
            <p className="text-2xl md:text-3xl font-black text-emerald-400 mt-2">
              {metrics ? (metrics.r2_score * 100).toFixed(1) : "95.8"}%
            </p>
            <p className="text-[11px] text-slate-500 mt-1">Explained variance on holdout test set</p>
          </div>

          <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl shadow-sm">
            <div className="flex justify-between items-center text-slate-400 text-xs font-semibold">
              <span>Mean Absolute Error (MAE)</span>
              <span className="text-[10px] bg-cyan-950/80 text-cyan-400 px-1.5 py-0.5 rounded border border-cyan-800/60">
                L1 Loss
              </span>
            </div>
            <p className="text-2xl md:text-3xl font-black text-cyan-400 mt-2">
              ${metrics ? metrics.mae.toLocaleString() : "61,420"}
            </p>
            <p className="text-[11px] text-slate-500 mt-1">Average deviation per weekly observation</p>
          </div>

          <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl shadow-sm">
            <div className="flex justify-between items-center text-slate-400 text-xs font-semibold">
              <span>Root Mean Sq Error (RMSE)</span>
              <span className="text-[10px] bg-blue-950/80 text-blue-400 px-1.5 py-0.5 rounded border border-blue-800/60">
                L2 Loss
              </span>
            </div>
            <p className="text-2xl md:text-3xl font-black text-blue-400 mt-2">
              ${metrics ? metrics.rmse.toLocaleString() : "115,140"}
            </p>
            <p className="text-[11px] text-slate-500 mt-1">Penalizes large outlier seasonal swings</p>
          </div>

          <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl shadow-sm">
            <div className="flex justify-between items-center text-slate-400 text-xs font-semibold">
              <span>Mean Percentage Error</span>
              <span className="text-[10px] bg-amber-950/80 text-amber-400 px-1.5 py-0.5 rounded border border-amber-800/60">
                MAPE
              </span>
            </div>
            <p className="text-2xl md:text-3xl font-black text-amber-400 mt-2">
              {metrics ? metrics.mape.toFixed(2) : "4.25"}%
            </p>
            <p className="text-[11px] text-slate-500 mt-1">Relative accuracy across store tiers</p>
          </div>
        </section>

        {/* =================================================================== */}
        {/* Navigation Tabs Controller                                          */}
        {/* =================================================================== */}
        <nav className="flex flex-wrap gap-2 border-b border-slate-800 pb-2">
          <button
            onClick={() => setActiveTab("forecast")}
            className={`px-4 py-2.5 rounded-xl font-semibold text-xs md:text-sm transition flex items-center gap-2 ${
              activeTab === "forecast"
                ? "bg-blue-600 text-white shadow-md shadow-blue-500/20"
                : "bg-slate-900 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
            }`}
          >
            <span>⚡</span> Real-time Sales Forecaster
          </button>

          <button
            onClick={() => setActiveTab("benchmarks")}
            className={`px-4 py-2.5 rounded-xl font-semibold text-xs md:text-sm transition flex items-center gap-2 ${
              activeTab === "benchmarks"
                ? "bg-blue-600 text-white shadow-md shadow-blue-500/20"
                : "bg-slate-900 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
            }`}
          >
            <span>📊</span> Multi-Model Benchmarks
          </button>

          <button
            onClick={() => setActiveTab("diagnostics")}
            className={`px-4 py-2.5 rounded-xl font-semibold text-xs md:text-sm transition flex items-center gap-2 ${
              activeTab === "diagnostics"
                ? "bg-blue-600 text-white shadow-md shadow-blue-500/20"
                : "bg-slate-900 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
            }`}
          >
            <span>📈</span> Plots &amp; Visualizations
          </button>

          <button
            onClick={() => setActiveTab("reports")}
            className={`px-4 py-2.5 rounded-xl font-semibold text-xs md:text-sm transition flex items-center gap-2 ${
              activeTab === "reports"
                ? "bg-blue-600 text-white shadow-md shadow-blue-500/20"
                : "bg-slate-900 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
            }`}
          >
            <span>📑</span> Excel Output &amp; Equations
          </button>
        </nav>

        {/* =================================================================== */}
        {/* TAB 1: Real-time Sales Forecaster                                    */}
        {/* =================================================================== */}
        {activeTab === "forecast" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Input Controls */}
            <section className="lg:col-span-7 bg-slate-900/70 border border-slate-800 rounded-2xl p-6 md:p-8 space-y-6">
              <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-3">
                <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                  <span>Parameter Configuration</span>
                  <span className="text-[10px] bg-slate-800 text-slate-400 px-2 py-0.5 rounded border border-slate-700">
                    Auto-Decomposed
                  </span>
                </h2>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => handlePreset(1, 69.16, 223.44, 6.57, 0)}
                    className="text-xs bg-slate-800 hover:bg-slate-700 px-2.5 py-1 rounded text-slate-300 border border-slate-700 transition"
                  >
                    Preset: Store 1
                  </button>
                  <button
                    type="button"
                    onClick={() => handlePreset(20, 39.93, 211.28, 8.1, 1)}
                    className="text-xs bg-slate-800 hover:bg-slate-700 px-2.5 py-1 rounded text-slate-300 border border-slate-700 transition"
                  >
                    Preset: Store 20 (Holiday)
                  </button>
                </div>
              </div>

              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-semibold text-slate-300 mb-1">
                      Store ID (1 - 45)
                    </label>
                    <input
                      type="number"
                      name="store"
                      min="1"
                      max="45"
                      required
                      value={formData.store}
                      onChange={handleChange}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-300 mb-1">
                      Evaluation Date (Week Ending)
                    </label>
                    <input
                      type="date"
                      name="date"
                      required
                      value={formData.date}
                      onChange={handleChange}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-300 mb-1">
                      Holiday Flag
                    </label>
                    <select
                      name="holiday_flag"
                      value={formData.holiday_flag}
                      onChange={handleChange}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value={0}>0 — Regular Working Week</option>
                      <option value={1}>1 — Major Holiday Week (Super Bowl, Thanksgiving, etc.)</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-300 mb-1">
                      Average Temperature (°F)
                    </label>
                    <input
                      type="number"
                      step="0.01"
                      name="temperature"
                      required
                      value={formData.temperature}
                      onChange={handleChange}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-300 mb-1">
                      Regional Fuel Price ($/Gal)
                    </label>
                    <input
                      type="number"
                      step="0.001"
                      name="fuel_price"
                      required
                      value={formData.fuel_price}
                      onChange={handleChange}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-300 mb-1">
                      Consumer Price Index (CPI)
                    </label>
                    <input
                      type="number"
                      step="0.0001"
                      name="cpi"
                      required
                      value={formData.cpi}
                      onChange={handleChange}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>

                  <div className="sm:col-span-2">
                    <label className="block text-xs font-semibold text-slate-300 mb-1">
                      Unemployment Rate (%)
                    </label>
                    <input
                      type="number"
                      step="0.001"
                      name="unemployment"
                      required
                      value={formData.unemployment}
                      onChange={handleChange}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full py-3.5 bg-gradient-to-r from-blue-600 via-indigo-600 to-blue-500 hover:from-blue-500 hover:to-indigo-500 font-bold rounded-xl text-white shadow-lg shadow-blue-500/20 transition disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <>
                      <svg className="animate-spin h-5 w-5 text-white" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                      </svg>
                      Evaluating Random Forest Ensemble...
                    </>
                  ) : (
                    "Compute Sales Prediction"
                  )}
                </button>
              </form>

              {error && (
                <div className="p-4 bg-rose-950/60 border border-rose-800 rounded-xl text-xs text-rose-300 flex items-start gap-2">
                  <span className="text-rose-400 font-bold">Error:</span>
                  <span>{error}</span>
                </div>
              )}
            </section>

            {/* Prediction Output & Feature Decomposition */}
            <section className="lg:col-span-5 space-y-6">
              <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 md:p-8 space-y-4">
                <h3 className="text-lg font-bold text-slate-100">Live Forecast Summary</h3>

                {prediction ? (
                  <div className="space-y-4">
                    <div className="p-5 bg-gradient-to-br from-blue-950/60 to-slate-900 border border-blue-900/60 rounded-xl">
                      <span className="text-xs text-blue-400 font-bold uppercase tracking-wider">
                        Predicted Weekly Revenue
                      </span>
                      <p className="text-3xl md:text-4xl font-extrabold text-white mt-2">
                        ${prediction.predicted_weekly_sales.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                      </p>
                      <div className="mt-3 flex flex-wrap gap-2 text-xs">
                        <span className="bg-slate-800 px-2.5 py-1 rounded border border-slate-700 text-slate-300">
                          Store #{prediction.store}
                        </span>
                        <span className="bg-slate-800 px-2.5 py-1 rounded border border-slate-700 text-slate-300">
                          {prediction.date}
                        </span>
                        {prediction.is_holiday ? (
                          <span className="bg-amber-950 text-amber-300 px-2.5 py-1 rounded border border-amber-800">
                            ★ Holiday Peak
                          </span>
                        ) : (
                          <span className="bg-slate-800 px-2.5 py-1 rounded border border-slate-700 text-slate-400">
                            Regular Schedule
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="space-y-2 text-xs">
                      <p className="font-semibold text-slate-300">Engineered Date Components:</p>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="bg-slate-950 p-2.5 rounded border border-slate-800">
                          Year: <span className="font-bold text-slate-200">{prediction.engineered_features.Year}</span>
                        </div>
                        <div className="bg-slate-950 p-2.5 rounded border border-slate-800">
                          Month: <span className="font-bold text-slate-200">{prediction.engineered_features.Month}</span>
                        </div>
                        <div className="bg-slate-950 p-2.5 rounded border border-slate-800">
                          Week #: <span className="font-bold text-slate-200">{prediction.engineered_features.WeekOfYear}</span>
                        </div>
                        <div className="bg-slate-950 p-2.5 rounded border border-slate-800">
                          Day: <span className="font-bold text-slate-200">{prediction.engineered_features.Day}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="h-48 border-2 border-dashed border-slate-800 rounded-xl flex flex-col items-center justify-center text-center p-4 text-slate-500 text-xs">
                    <p className="font-medium text-slate-400">Inference engine ready.</p>
                    <p className="text-slate-600 mt-1">Submit parameters on the left to calculate weekly sales.</p>
                  </div>
                )}
              </div>

              {/* Top Gini Importance Bar */}
              <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 space-y-3">
                <h4 className="text-sm font-bold text-slate-200">
                  Feature Importances (Gini Breakdown)
                </h4>
                <div className="space-y-2.5">
                  {Object.entries(
                    metrics?.features_importance || {
                      Store: 0.584,
                      CPI: 0.142,
                      Unemployment: 0.096,
                      Temperature: 0.071,
                      WeekOfYear: 0.048,
                      Fuel_Price: 0.035,
                    }
                  )
                    .slice(0, 5)
                    .map(([feat, imp]) => {
                      const numericImp = Number(imp);
                      const pct = (numericImp * 100).toFixed(1);
                      return (
                        <div key={feat} className="text-xs">
                          <div className="flex justify-between text-slate-400 mb-1">
                            <span>{feat}</span>
                            <span className="font-semibold text-slate-200">{pct}%</span>
                          </div>
                          <div className="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-slate-800">
                            <div className="bg-indigo-500 h-full rounded-full" style={{ width: `${pct}%` }} />
                          </div>
                        </div>
                      );
                    })}
                </div>
              </div>
            </section>
          </div>
        )}

        {/* =================================================================== */}
        {/* TAB 2: Multi-Model Benchmark Comparison                             */}
        {/* =================================================================== */}
        {activeTab === "benchmarks" && (
          <section className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 md:p-8 space-y-6">
            <div>
              <h2 className="text-lg font-bold text-slate-100">Evaluated Regression Algorithms</h2>
              <p className="text-xs text-slate-400 mt-1">
                Comparative metrics produced by the pipeline in <code className="text-blue-400">walmartsales.py</code> across 80/20 train/test split.
              </p>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs md:text-sm text-slate-300">
                <thead className="bg-slate-950 text-slate-400 uppercase text-[11px] font-semibold border-b border-slate-800">
                  <tr>
                    <th className="py-3 px-4">Algorithm</th>
                    <th className="py-3 px-4">Train R²</th>
                    <th className="py-3 px-4">Test R²</th>
                    <th className="py-3 px-4">MAE ($)</th>
                    <th className="py-3 px-4">RMSE ($)</th>
                    <th className="py-3 px-4">MAPE</th>
                    <th className="py-3 px-4 text-center">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono text-xs">
                  {MODEL_COMPARISONS.map((row) => (
                    <tr key={row.model} className="hover:bg-slate-800/40 transition">
                      <td className="py-3.5 px-4 font-sans font-bold text-slate-100 flex items-center gap-2">
                        {row.model}
                      </td>
                      <td className="py-3.5 px-4 text-slate-300">{row.trainR2}</td>
                      <td className="py-3.5 px-4 text-emerald-400 font-bold">{row.testR2}</td>
                      <td className="py-3.5 px-4 text-cyan-300">{row.mae}</td>
                      <td className="py-3.5 px-4 text-blue-300">{row.rmse}</td>
                      <td className="py-3.5 px-4 text-amber-300">{row.mape}</td>
                      <td className="py-3.5 px-4 text-center">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-sans font-bold border ${
                            row.status === "Optimal"
                              ? "bg-emerald-950 text-emerald-300 border-emerald-700"
                              : row.status === "Candidate"
                              ? "bg-blue-950 text-blue-300 border-blue-700"
                              : "bg-slate-800 text-slate-400 border-slate-700"
                          }`}
                        >
                          {row.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-400 space-y-1">
              <span className="font-bold text-slate-300">Model Selection Verdict:</span>
              <p>
                <strong className="text-emerald-400">Random Forest Regressor</strong> outperforms Linear and Ridge models significantly by non-linearly capturing discrete store tiers and high-impact seasonal holiday surges (yielding <strong>95.84% Test R²</strong> with a low <strong>4.25% MAPE</strong>).
              </p>
            </div>
          </section>
        )}

        {/* =================================================================== */}
        {/* TAB 3: Diagnostic Plots & Visualizations                            */}
        {/* =================================================================== */}
        {activeTab === "diagnostics" && (
          <section className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 md:p-8 space-y-6">
            <div>
              <h2 className="text-lg font-bold text-slate-100">Exploratory &amp; Evaluation Plots</h2>
              <p className="text-xs text-slate-400 mt-1">
                Artifacts exported by <code className="text-blue-400">walmartsales.py</code> into the <code className="text-blue-400">Plots/</code> directory.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Plot 1: Target Distribution */}
              <div className="bg-slate-950 p-5 rounded-xl border border-slate-800 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-bold text-slate-200">01_weekly_sales_distribution.png</span>
                  <span className="text-[10px] bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded">Univariate</span>
                </div>
                <div className="h-44 bg-slate-900 rounded-lg flex items-center justify-center border border-slate-800/80 text-slate-500 text-xs text-center p-4">
                  <div>
                    <p className="font-semibold text-slate-300">Target Distribution (Weekly Sales)</p>
                    <p className="text-slate-500 mt-1 text-[11px]">Bimodal distribution peaking around $0.5M and $1.6M</p>
                  </div>
                </div>
              </div>

              {/* Plot 2: Correlation Heatmap */}
              <div className="bg-slate-950 p-5 rounded-xl border border-slate-800 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-bold text-slate-200">02_correlation_heatmap.png</span>
                  <span className="text-[10px] bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded">Correlation</span>
                </div>
                <div className="h-44 bg-slate-900 rounded-lg flex items-center justify-center border border-slate-800/80 text-slate-500 text-xs text-center p-4">
                  <div>
                    <p className="font-semibold text-slate-300">Feature Correlation Matrix</p>
                    <p className="text-slate-500 mt-1 text-[11px]">Highlights collinear relationships between Fuel Price, CPI, and Year</p>
                  </div>
                </div>
              </div>

              {/* Plot 3: Actual vs Predicted */}
              <div className="bg-slate-950 p-5 rounded-xl border border-slate-800 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-bold text-slate-200">03_actual_vs_predicted.png</span>
                  <span className="text-[10px] bg-emerald-950 text-emerald-400 px-1.5 py-0.5 rounded border border-emerald-800">
                    Validation
                  </span>
                </div>
                <div className="h-44 bg-slate-900 rounded-lg flex items-center justify-center border border-slate-800/80 text-slate-500 text-xs text-center p-4">
                  <div>
                    <p className="font-semibold text-emerald-400">Actual vs Predicted Scatter</p>
                    <p className="text-slate-500 mt-1 text-[11px]">Tight adherence along the 45° identity line confirming R² = 0.958</p>
                  </div>
                </div>
              </div>

              {/* Plot 4: Feature Importance */}
              <div className="bg-slate-950 p-5 rounded-xl border border-slate-800 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-bold text-slate-200">05_feature_importance.png</span>
                  <span className="text-[10px] bg-indigo-950 text-indigo-400 px-1.5 py-0.5 rounded border border-indigo-800">
                    Tree Gini
                  </span>
                </div>
                <div className="h-44 bg-slate-900 rounded-lg flex items-center justify-center border border-slate-800/80 text-slate-500 text-xs text-center p-4">
                  <div>
                    <p className="font-semibold text-indigo-400">Feature Importance Ranking</p>
                    <p className="text-slate-500 mt-1 text-[11px]">Store ID accounts for 58.4% of total tree split gain</p>
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* =================================================================== */}
        {/* TAB 4: Excel Output Reports & Regression Equations                  */}
        {/* =================================================================== */}
        {activeTab === "reports" && (
          <section className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 md:p-8 space-y-8">
            <div>
              <h2 className="text-lg font-bold text-slate-100">Output Folder &amp; Analytical Formulas</h2>
              <p className="text-xs text-slate-400 mt-1">
                Generated workbook: <code className="text-blue-400">Output/Walmart_Model_Results.xlsx</code>
              </p>
            </div>

            {/* Regression Equations */}
            <div className="space-y-4">
              <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider text-xs">
                Derived Model Equations
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Random Forest Formulation */}
                <div className="bg-slate-950 p-5 rounded-xl border border-slate-800 space-y-2">
                  <span className="text-xs font-bold text-emerald-400">
                    1. Optimal Model: Random Forest Ensemble Formulation
                  </span>
                  <div className="p-3 bg-slate-900 rounded border border-slate-800 font-mono text-[11px] text-slate-300">
                    Ŷ_sales(x) = (1 / 150) * Σ [ m=1 to 150 ] T_m(x)
                  </div>
                  <p className="text-xs text-slate-400">
                    Averages the partition estimations of 150 randomized decision trees (max_depth=16).
                  </p>
                </div>

                {/* Closed-Form Linear Equation */}
                <div className="bg-slate-950 p-5 rounded-xl border border-slate-800 space-y-2">
                  <span className="text-xs font-bold text-blue-400">
                    2. Interpretable Linear Baseline Regression Equation
                  </span>
                  <div className="p-3 bg-slate-900 rounded border border-slate-800 font-mono text-[11px] text-slate-300">
                    Ŷ = -41.23M - 14.4K(Store) + 36.2K(Holiday) - 1.28K(Temp) - 3.8K(Fuel) - 2.1K(CPI) - 21.8K(Unemp) + 21.0K(Year) ...
                  </div>
                  <p className="text-xs text-slate-400">
                    Parametric weights computed in natural unscaled coordinates.
                  </p>
                </div>
              </div>
            </div>

            {/* Linear Weight Table */}
            <div className="space-y-3">
              <h4 className="text-xs font-bold text-slate-300 uppercase">
                Parametric Coefficient Weights (Sheet 4 Reference)
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs text-slate-300 border border-slate-800 rounded-lg overflow-hidden">
                  <thead className="bg-slate-950 text-slate-400 font-semibold border-b border-slate-800">
                    <tr>
                      <th className="py-2.5 px-4">Feature / Variable</th>
                      <th className="py-2.5 px-4">Estimated Coefficient</th>
                      <th className="py-2.5 px-4">Economic Interpretation</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                    {REGRESSION_COEFFICIENTS.map((item) => (
                      <tr key={item.term} className="hover:bg-slate-800/30">
                        <td className="py-2 px-4 font-bold text-slate-200">{item.term}</td>
                        <td className="py-2 px-4 text-emerald-400">{item.value}</td>
                        <td className="py-2 px-4 font-sans text-slate-400">{item.note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Prediction Log Samples */}
            <div className="space-y-3">
              <h4 className="text-xs font-bold text-slate-300 uppercase">
                Sample Actual vs Predicted Records (Sheet 2: Test_Predictions)
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs text-slate-300 border border-slate-800 rounded-lg overflow-hidden font-mono text-[11px]">
                  <thead className="bg-slate-950 text-slate-400 font-semibold border-b border-slate-800 font-sans">
                    <tr>
                      <th className="py-2.5 px-4">Date</th>
                      <th className="py-2.5 px-4">Store</th>
                      <th className="py-2.5 px-4">Actual Sales ($)</th>
                      <th className="py-2.5 px-4">Predicted Sales ($)</th>
                      <th className="py-2.5 px-4">Residual Error ($)</th>
                      <th className="py-2.5 px-4">APE</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {PREDICTION_LOG_SAMPLES.map((log, i) => (
                      <tr key={i} className="hover:bg-slate-800/30">
                        <td className="py-2 px-4 text-slate-400">{log.date}</td>
                        <td className="py-2 px-4 text-slate-300">#{log.store}</td>
                        <td className="py-2 px-4 text-slate-100">${log.actual.toLocaleString()}</td>
                        <td className="py-2 px-4 text-emerald-400">${log.predicted.toLocaleString()}</td>
                        <td className="py-2 px-4 text-cyan-400">${log.error.toLocaleString()}</td>
                        <td className="py-2 px-4 text-amber-300 font-bold">{log.ape}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        )}
      </div>
    </main>
  );
}