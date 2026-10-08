"use client";

import React, { useEffect, useState } from "react";

interface MetricRow {
  metric: string;
  train: number;
  test: number;
}

interface CoefRow {
  feature: string;
  coefficient: number;
}

interface ModelInfo {
  optimal_alpha: number;
  metrics: MetricRow[];
  final_equation: string;
  top_coefficients: CoefRow[];
  features: {
    numerical: string[];
    categorical: string[];
    defaults: Record<string, number>;
    categorical_options: Record<string, string[]>;
  };
}

// Browser-safe fallback check for process.env
const BACKEND_URL =
  (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_API_URL) ||
  "http://localhost:8000";

export default function RetailOptimizationDashboard() {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [formData, setFormData] = useState<Record<string, any>>({});
  const [prediction, setPrediction] = useState<number | null>(null);
  const [predicting, setPredicting] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<"overview" | "predict" | "equation">("overview");

  useEffect(() => {
    fetch(`${BACKEND_URL}/api/model-info`)
      .then((res) => res.json())
      .then((data: ModelInfo) => {
        setModelInfo(data);
        const initialForm: Record<string, any> = { ...data.features.defaults };
        Object.keys(data.features.categorical_options).forEach((cat) => {
          initialForm[cat] = data.features.categorical_options[cat][0] || "";
        });
        setFormData(initialForm);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to fetch model info:", err);
        setLoading(false);
      });
  }, []);

  const handleInputChange = (field: string, value: any) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setPredicting(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ inputs: formData }),
      });
      const data = await res.json();
      if (data.status === "success") {
        setPrediction(data.predicted_unit_price);
      }
    } catch (err) {
      console.error("Prediction failed:", err);
    } finally {
      setPredicting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50 text-gray-700">
        <div className="text-center font-medium text-lg animate-pulse">
          Loading Model Artifacts & Architecture...
        </div>
      </div>
    );
  }

  if (!modelInfo) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50 text-red-600">
        <p className="font-semibold text-lg">
          Failed to load model from FastAPI backend. Ensure backend is running on port 8000.
        </p>
      </div>
    );
  }

  const testR2 = modelInfo.metrics.find((m) => m.metric === "R2 Score")?.test ?? 0;
  const testRMSE = modelInfo.metrics.find((m) => m.metric === "RMSE")?.test ?? 0;
  const testMAE = modelInfo.metrics.find((m) => m.metric === "MAE")?.test ?? 0;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800">
      {/* Header */}
      <header className="border-b bg-white px-8 py-5 shadow-sm flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">
            🏷️ Retail Price Optimization Engine
          </h1>
          <p className="text-sm text-slate-500">
            Ridge Regression Pipeline | Regularization Alpha: {modelInfo.optimal_alpha}
          </p>
        </div>
        <a
          href={`${BACKEND_URL}/api/export-excel`}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center rounded-lg bg-emerald-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-emerald-500 transition-colors"
        >
          📥 Download Excel Report
        </a>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-7xl px-6 py-8">
        {/* KPI Summary Cards */}
        <section className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4 mb-8">
          <div className="rounded-xl border bg-white p-5 shadow-sm">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Test R² Score
            </span>
            <div className="mt-2 text-3xl font-bold text-blue-600">{testR2.toFixed(4)}</div>
            <p className="mt-1 text-xs text-slate-500">Variance explained on test data</p>
          </div>
          <div className="rounded-xl border bg-white p-5 shadow-sm">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Test RMSE
            </span>
            <div className="mt-2 text-3xl font-bold text-slate-800">${testRMSE.toFixed(2)}</div>
            <p className="mt-1 text-xs text-slate-500">Root Mean Squared Error</p>
          </div>
          <div className="rounded-xl border bg-white p-5 shadow-sm">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Test MAE
            </span>
            <div className="mt-2 text-3xl font-bold text-slate-800">${testMAE.toFixed(2)}</div>
            <p className="mt-1 text-xs text-slate-500">Mean Absolute Error</p>
          </div>
          <div className="rounded-xl border bg-white p-5 shadow-sm">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Optimal Regularization
            </span>
            <div className="mt-2 text-3xl font-bold text-violet-600">
              α = {modelInfo.optimal_alpha}
            </div>
            <p className="mt-1 text-xs text-slate-500">5-Fold Cross Validation</p>
          </div>
        </section>

        {/* Tab Selection */}
        <div className="border-b border-slate-200 mb-6">
          <nav className="flex space-x-8">
            <button
              onClick={() => setActiveTab("overview")}
              className={`pb-4 text-sm font-medium border-b-2 transition-colors ${
                activeTab === "overview"
                  ? "border-blue-600 text-blue-600 font-semibold"
                  : "border-transparent text-slate-500 hover:text-slate-700"
              }`}
            >
              📊 Performance Scorecard
            </button>
            <button
              onClick={() => setActiveTab("predict")}
              className={`pb-4 text-sm font-medium border-b-2 transition-colors ${
                activeTab === "predict"
                  ? "border-blue-600 text-blue-600 font-semibold"
                  : "border-transparent text-slate-500 hover:text-slate-700"
              }`}
            >
              🎯 Price Simulator
            </button>
            <button
              onClick={() => setActiveTab("equation")}
              className={`pb-4 text-sm font-medium border-b-2 transition-colors ${
                activeTab === "equation"
                  ? "border-blue-600 text-blue-600 font-semibold"
                  : "border-transparent text-slate-500 hover:text-slate-700"
              }`}
            >
              📐 Model Equation & Weights
            </button>
          </nav>
        </div>

        {/* Tab 1: Overview */}
        {activeTab === "overview" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="rounded-xl border bg-white p-6 shadow-sm">
              <h3 className="text-lg font-bold text-slate-900 mb-4">Detailed Metrics Table</h3>
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b text-slate-400 font-medium uppercase text-xs">
                    <th className="py-3">Metric</th>
                    <th className="py-3">Train</th>
                    <th className="py-3">Test</th>
                  </tr>
                </thead>
                <tbody className="divide-y text-slate-700">
                  {modelInfo.metrics.map((row) => (
                    <tr key={row.metric}>
                      <td className="py-3 font-medium text-slate-900">{row.metric}</td>
                      <td className="py-3">{row.train}</td>
                      <td className="py-3 font-semibold text-blue-600">{row.test}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="rounded-xl border bg-white p-6 shadow-sm">
              <h3 className="text-lg font-bold text-slate-900 mb-4">Top Feature Weights</h3>
              <div className="space-y-3">
                {modelInfo.top_coefficients.slice(0, 8).map((coef) => (
                  <div key={coef.feature} className="flex justify-between items-center text-sm">
                    <span className="font-mono text-xs text-slate-600 truncate max-w-xs">
                      {coef.feature}
                    </span>
                    <span
                      className={`font-semibold font-mono ${
                        coef.coefficient >= 0 ? "text-emerald-600" : "text-rose-600"
                      }`}
                    >
                      {coef.coefficient > 0 ? `+${coef.coefficient}` : coef.coefficient}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Tab 2: Predictor Form */}
        {activeTab === "predict" && (
          <div className="rounded-xl border bg-white p-8 shadow-sm">
            <h3 className="text-lg font-bold text-slate-900 mb-2">Simulate Unit Price</h3>
            <p className="text-sm text-slate-500 mb-6">
              Adjust numerical attributes or product categories to compute predicted unit prices.
            </p>

            <form onSubmit={handlePredict}>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
                {modelInfo.features.numerical.map((feat) => (
                  <div key={feat}>
                    <label className="block text-xs font-semibold text-slate-600 uppercase mb-1">
                      {feat}
                    </label>
                    <input
                      type="number"
                      step="any"
                      value={formData[feat] ?? ""}
                      onChange={(e) => handleInputChange(feat, parseFloat(e.target.value) || 0)}
                      className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-900 focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                ))}

                {modelInfo.features.categorical.map((feat) => (
                  <div key={feat}>
                    <label className="block text-xs font-semibold text-slate-600 uppercase mb-1">
                      {feat}
                    </label>
                    <select
                      value={formData[feat] ?? ""}
                      onChange={(e) => handleInputChange(feat, e.target.value)}
                      className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-900 focus:border-blue-500 focus:outline-none"
                    >
                      {modelInfo.features.categorical_options[feat]?.map((opt) => (
                        <option key={opt} value={opt}>
                          {opt}
                        </option>
                      ))}
                    </select>
                  </div>
                ))}
              </div>

              <div className="flex items-center gap-4 pt-4 border-t">
                <button
                  type="submit"
                  disabled={predicting}
                  className="rounded-lg bg-blue-600 px-6 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-blue-500 disabled:opacity-50"
                >
                  {predicting ? "Computing..." : "Estimate Unit Price"}
                </button>

                {prediction !== null && (
                  <div className="text-lg font-bold text-emerald-600 bg-emerald-50 px-4 py-2 rounded-lg border border-emerald-200">
                    Optimal Estimated Price: ${prediction.toFixed(2)}
                  </div>
                )}
              </div>
            </form>
          </div>
        )}

        {/* Tab 3: Equation */}
        {activeTab === "equation" && (
          <div className="rounded-xl border bg-white p-8 shadow-sm">
            <h3 className="text-lg font-bold text-slate-900 mb-2">Final Regression Equation</h3>
            <p className="text-sm text-slate-500 mb-4">
              Mathematical representation of the trained Ridge regression model:
            </p>
            <div className="bg-slate-900 text-slate-100 p-4 rounded-lg font-mono text-xs overflow-x-auto mb-8 whitespace-pre-wrap leading-relaxed">
              {modelInfo.final_equation}
            </div>

            <h3 className="text-lg font-bold text-slate-900 mb-4">Full Feature Weights</h3>
            <div className="overflow-x-auto max-h-96 border rounded-lg">
              <table className="w-full text-left text-sm">
                <thead className="bg-slate-50 text-slate-400 font-medium uppercase text-xs border-b">
                  <tr>
                    <th className="py-3 px-4">Feature</th>
                    <th className="py-3 px-4">Standardized Weight</th>
                  </tr>
                </thead>
                <tbody className="divide-y text-slate-700">
                  {modelInfo.top_coefficients.map((row) => (
                    <tr key={row.feature} className="hover:bg-slate-50 font-mono text-xs">
                      <td className="py-2.5 px-4">{row.feature}</td>
                      <td
                        className={`py-2.5 px-4 font-semibold ${
                          row.coefficient >= 0 ? "text-emerald-600" : "text-rose-600"
                        }`}
                      >
                        {row.coefficient}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}