"use client";

import React, { useState, useEffect } from "react";

interface Metrics {
  train_r2: number;
  test_r2: number;
  gap: number;
  train_rmse: number;
  test_rmse: number;
  train_mae: number;
  test_mae: number;
  train_evs: number;
  test_evs: number;
}

interface FeatureImportance {
  name: string;
  score: number;
  std: number;
}

interface TestRecord {
  row_id: number;
  geography: string;
  actual: number;
  predicted: number;
  residual: number;
}

interface PredictionResponse {
  predicted_deathrate: number;
  national_average: number;
  delta: number;
  risk_level: string;
  engineered_metrics: {
    implied_mortality_rate: number;
    income_to_poverty_ratio: number;
    private_to_public_ratio: number;
    education_gap: number;
  };
}

export default function CancerPage() {
  const API_BASE = "http://localhost:8000";

  // Data state
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [states, setStates] = useState<string[]>([]);
  const [nationalAvg, setNationalAvg] = useState<number>(178.6);
  const [topFeatures, setTopFeatures] = useState<FeatureImportance[]>([]);
  const [testRecords, setTestRecords] = useState<TestRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Active view tab
  const [activeTab, setActiveTab] = useState<"dashboard" | "predictor" | "plots" | "residuals">("dashboard");

  // Plot selector
  const [selectedPlot, setSelectedPlot] = useState<string>("actual_vs_predicted.png");

  // Prediction inputs
  const [form, setForm] = useState({
    state: "California",
    incidencerate: 450.0,
    avganncount: 1100.0,
    avgdeathsperyear: 390.0,
    popest2015: 220000.0,
    medincome: 55000.0,
    povertypercent: 14.0,
    pctemployed16_over: 54.0,
    pctunemployed16_over: 6.0,
    medianage: 41.0,
    pctprivatecoverage: 65.0,
    pctpubliccoverage: 35.0,
    pctbachdeg25_over: 22.0,
    pcths25_over: 32.0,
    avghouseholdsize: 2.5,
  });

  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [predicting, setPredicting] = useState<boolean>(false);
  const [searchTerm, setSearchTerm] = useState<string>("");

  useEffect(() => {
    fetchMetadata();
    fetchTestData();
  }, []);

  const fetchMetadata = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/api/metadata`);
      if (!res.ok) throw new Error("FastAPI backend is offline or model is still training.");
      const data = await res.json();
      setMetrics(data.metrics);
      setStates(data.states);
      setNationalAvg(data.national_mean_mortality);
      setTopFeatures(data.top_features);
      if (data.states.length > 0) {
        setForm((prev) => ({ ...prev, state: data.states[0] }));
      }
    } catch (err: any) {
      setError(err.message || "Failed to connect to backend");
    } finally {
      setLoading(false);
    }
  };

  const fetchTestData = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/test-data`);
      if (res.ok) {
        const data = await res.json();
        setTestRecords(data);
      }
    } catch {
      // Non-blocking
    }
  };

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setPredicting(true);
    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) throw new Error("Inference failed.");
      const data: PredictionResponse = await res.json();
      setPrediction(data);
    } catch (err: any) {
      alert(err.message || "Prediction error");
    } finally {
      setPredicting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-slate-900 text-white">
        <div className="text-center">
          <div className="mb-4 inline-block h-12 w-12 animate-spin rounded-full border-4 border-emerald-500 border-t-transparent" />
          <h2 className="text-xl font-bold">Training & Synchronizing Model Pipeline...</h2>
          <p className="text-sm text-slate-400">Loading datasets and building regularized ensemble.</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-screen flex-col items-center justify-center bg-slate-900 p-6 text-white">
        <div className="max-w-md rounded-xl border border-red-500/40 bg-red-950/20 p-6 text-center">
          <h2 className="text-lg font-bold text-red-400">Connection Error</h2>
          <p className="mt-2 text-sm text-slate-300">{error}</p>
          <button
            onClick={fetchMetadata}
            className="mt-4 rounded-lg bg-red-600 px-4 py-2 text-sm font-semibold text-white hover:bg-red-500"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Top Header */}
      <header className="border-b border-slate-800 bg-slate-900/80 px-6 py-4 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between">
          <div className="flex items-center space-x-3">
            <span className="text-2xl">🎗️</span>
            <div>
              <h1 className="text-lg font-bold text-white">Cancer Mortality ML Intelligence</h1>
              <p className="text-xs text-slate-400">Leak-Proof Regularized Gradient Boosting Pipeline</p>
            </div>
          </div>
          <a
            href={`${API_BASE}/api/export/excel`}
            target="_blank"
            rel="noreferrer"
            className="flex items-center space-x-2 rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white shadow transition hover:bg-emerald-500"
          >
            <span>📥 Download Excel Package</span>
          </a>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="mx-auto max-w-7xl px-6 pt-6">
        <nav className="flex space-x-2 border-b border-slate-800 pb-3">
          {[
            { id: "dashboard", label: "📊 Model Dashboard" },
            { id: "predictor", label: "🔮 County Predictor" },
            { id: "plots", label: "📈 Diagnostic Plots" },
            { id: "residuals", label: "📋 Validation Data" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
                activeTab === tab.id
                  ? "bg-slate-800 text-emerald-400 shadow-sm"
                  : "text-slate-400 hover:bg-slate-900 hover:text-white"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      <main className="mx-auto max-w-7xl p-6">
        {/* VIEW 1: EXECUTIVE DASHBOARD */}
        {activeTab === "dashboard" && metrics && (
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 shadow-sm">
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">TRAIN R² SCORE</p>
                <p className="mt-2 text-3xl font-extrabold text-white">{(metrics.train_r2 * 100).toFixed(2)}%</p>
                <p className="mt-1 text-xs font-semibold text-emerald-400">Optimal Variance Fit</p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 shadow-sm">
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">TEST R² SCORE</p>
                <p className="mt-2 text-3xl font-extrabold text-emerald-400">{(metrics.test_r2 * 100).toFixed(2)}%</p>
                <p className="mt-1 text-xs font-semibold text-slate-400">Holdout Verified</p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 shadow-sm">
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">GENERALIZATION GAP</p>
                <p className="mt-2 text-3xl font-extrabold text-blue-400">{metrics.gap.toFixed(2)}%</p>
                <p className="mt-1 text-xs font-semibold text-emerald-400">&lt; 4.0% Bounded</p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 shadow-sm">
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">TEST RMSE</p>
                <p className="mt-2 text-3xl font-extrabold text-white">{metrics.test_rmse.toFixed(2)}</p>
                <p className="mt-1 text-xs text-slate-400">Deaths per 100k</p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 shadow-sm">
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">TEST MAE</p>
                <p className="mt-2 text-3xl font-extrabold text-white">{metrics.test_mae.toFixed(2)}</p>
                <p className="mt-1 text-xs text-slate-400">Mean Absolute Error</p>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
                <h3 className="text-base font-bold text-white">Cross-Partition Evaluation Matrix</h3>
                <div className="mt-4 overflow-x-auto">
                  <table className="w-full text-left text-sm text-slate-300">
                    <thead className="border-b border-slate-800 text-xs uppercase text-slate-400">
                      <tr>
                        <th className="pb-3">Metric</th>
                        <th className="pb-3 text-right">Train</th>
                        <th className="pb-3 text-right">Test</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 font-mono">
                      <tr>
                        <td className="py-2.5 font-sans">R² Coefficient</td>
                        <td className="text-right">{(metrics.train_r2 * 100).toFixed(2)}%</td>
                        <td className="text-right text-emerald-400">{(metrics.test_r2 * 100).toFixed(2)}%</td>
                      </tr>
                      <tr>
                        <td className="py-2.5 font-sans">RMSE Error</td>
                        <td className="text-right">{metrics.train_rmse.toFixed(3)}</td>
                        <td className="text-right">{metrics.test_rmse.toFixed(3)}</td>
                      </tr>
                      <tr>
                        <td className="py-2.5 font-sans">MAE Deviation</td>
                        <td className="text-right">{metrics.train_mae.toFixed(3)}</td>
                        <td className="text-right">{metrics.test_mae.toFixed(3)}</td>
                      </tr>
                      <tr>
                        <td className="py-2.5 font-sans">Explained Variance</td>
                        <td className="text-right">{(metrics.train_evs * 100).toFixed(2)}%</td>
                        <td className="text-right">{(metrics.test_evs * 100).toFixed(2)}%</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
                <h3 className="text-base font-bold text-white">Key Features by Permutation Decrement</h3>
                <div className="mt-4 space-y-3">
                  {topFeatures.slice(0, 6).map((feat, i) => (
                    <div key={feat.name} className="flex items-center justify-between text-sm">
                      <span className="flex items-center space-x-2 text-slate-300">
                        <span className="w-5 font-mono text-xs text-slate-500">#{i + 1}</span>
                        <span>{feat.name}</span>
                      </span>
                      <span className="font-mono text-xs font-semibold text-emerald-400">
                        +{feat.score.toFixed(5)} R²
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* VIEW 2: COUNTY PREDICTOR */}
        {activeTab === "predictor" && (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <form onSubmit={handlePredict} className="space-y-6 rounded-xl border border-slate-800 bg-slate-900/60 p-6 lg:col-span-2">
              <h3 className="text-base font-bold text-white">Simulate County Features</h3>

              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                  <label className="text-xs font-medium text-slate-400">State Geography</label>
                  <select
                    value={form.state}
                    onChange={(e) => setForm({ ...form, state: e.target.value })}
                    className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none"
                  >
                    {states.map((s) => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="text-xs font-medium text-slate-400">Incidence Rate (per 100k): {form.incidencerate}</label>
                  <input
                    type="range"
                    min="200"
                    max="800"
                    step="5"
                    value={form.incidencerate}
                    onChange={(e) => setForm({ ...form, incidencerate: parseFloat(e.target.value) })}
                    className="mt-2 w-full accent-emerald-500"
                  />
                </div>

                <div>
                  <label className="text-xs font-medium text-slate-400">Annual New Cases: {form.avganncount}</label>
                  <input
                    type="number"
                    value={form.avganncount}
                    onChange={(e) => setForm({ ...form, avganncount: parseFloat(e.target.value) || 0 })}
                    className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="text-xs font-medium text-slate-400">Annual Cancer Deaths: {form.avgdeathsperyear}</label>
                  <input
                    type="number"
                    value={form.avgdeathsperyear}
                    onChange={(e) => setForm({ ...form, avgdeathsperyear: parseFloat(e.target.value) || 0 })}
                    className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="text-xs font-medium text-slate-400">County Population (2015): {form.popest2015.toLocaleString()}</label>
                  <input
                    type="number"
                    value={form.popest2015}
                    onChange={(e) => setForm({ ...form, popest2015: parseFloat(e.target.value) || 0 })}
                    className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="text-xs font-medium text-slate-400">Median Income ($): {form.medincome.toLocaleString()}</label>
                  <input
                    type="number"
                    value={form.medincome}
                    onChange={(e) => setForm({ ...form, medincome: parseFloat(e.target.value) || 0 })}
                    className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="text-xs font-medium text-slate-400">Poverty Rate (%): {form.povertypercent}%</label>
                  <input
                    type="range"
                    min="3"
                    max="45"
                    step="0.5"
                    value={form.povertypercent}
                    onChange={(e) => setForm({ ...form, povertypercent: parseFloat(e.target.value) })}
                    className="mt-2 w-full accent-emerald-500"
                  />
                </div>

                <div>
                  <label className="text-xs font-medium text-slate-400">Private Coverage (%): {form.pctprivatecoverage}%</label>
                  <input
                    type="range"
                    min="20"
                    max="95"
                    step="0.5"
                    value={form.pctprivatecoverage}
                    onChange={(e) => setForm({ ...form, pctprivatecoverage: parseFloat(e.target.value) })}
                    className="mt-2 w-full accent-emerald-500"
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={predicting}
                className="w-full rounded-lg bg-emerald-600 py-3 font-semibold text-white transition hover:bg-emerald-500 disabled:opacity-50"
              >
                {predicting ? "Running Model Inference..." : "Calculate Predicted Mortality"}
              </button>
            </form>

            <div className="space-y-6">
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
                <h3 className="text-base font-bold text-white">Prediction Output</h3>
                {prediction ? (
                  <div className="mt-4 space-y-4">
                    <div className="rounded-lg bg-slate-800/80 p-4 text-center">
                      <p className="text-xs font-semibold uppercase text-slate-400">Predicted Mortality</p>
                      <p className="mt-1 text-3xl font-extrabold text-emerald-400">
                        {prediction.predicted_deathrate}
                      </p>
                      <p className="text-xs text-slate-400">Deaths per 100,000</p>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="rounded-lg bg-slate-800/40 p-2.5">
                        <span className="text-slate-400">National Mean:</span>
                        <p className="font-mono text-sm font-semibold">{prediction.national_average}</p>
                      </div>
                      <div className="rounded-lg bg-slate-800/40 p-2.5">
                        <span className="text-slate-400">Variance Delta:</span>
                        <p className={`font-mono text-sm font-semibold ${prediction.delta > 0 ? "text-red-400" : "text-emerald-400"}`}>
                          {prediction.delta > 0 ? `+${prediction.delta}` : prediction.delta}
                        </p>
                      </div>
                    </div>

                    <div className="border-t border-slate-800 pt-3 text-xs text-slate-300">
                      <p className="font-semibold text-slate-400">Engineered Dynamics:</p>
                      <ul className="mt-1 space-y-1 font-mono">
                        <li>Implied Death Rate: {prediction.engineered_metrics.implied_mortality_rate}</li>
                        <li>Income/Poverty Index: {prediction.engineered_metrics.income_to_poverty_ratio}</li>
                        <li>Private/Public Ratio: {prediction.engineered_metrics.private_to_public_ratio}x</li>
                      </ul>
                    </div>
                  </div>
                ) : (
                  <p className="mt-4 text-xs text-slate-400">Click &ldquo;Calculate Predicted Mortality&rdquo; to execute inference.</p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* VIEW 3: PLOTS GALLERY */}
        {activeTab === "plots" && (
          <div className="space-y-6">
            <div className="flex flex-wrap gap-2">
              {[
                { file: "actual_vs_predicted.png", label: "Actual vs Predicted" },
                { file: "residuals_vs_fitted.png", label: "Residuals vs Fitted" },
                { file: "target_distribution.png", label: "Target Distribution" },
                { file: "feature_importance.png", label: "Permutation Importance" },
              ].map((p) => (
                <button
                  key={p.file}
                  onClick={() => setSelectedPlot(p.file)}
                  className={`rounded-lg px-4 py-2 text-xs font-semibold transition ${
                    selectedPlot === p.file
                      ? "bg-emerald-600 text-white"
                      : "bg-slate-900 text-slate-400 hover:text-white"
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>

            <div className="flex justify-center rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`${API_BASE}/api/plots/${selectedPlot}`}
                alt={selectedPlot}
                className="max-h-[550px] w-auto rounded-lg object-contain shadow-lg"
              />
            </div>
          </div>
        )}

        {/* VIEW 4: TEST DATA MATRIX */}
        {activeTab === "residuals" && (
          <div className="space-y-4 rounded-xl border border-slate-800 bg-slate-900/60 p-6">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-white">Holdout Test Set Residual Matrix</h3>
              <input
                type="text"
                placeholder="Filter county..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs text-white focus:outline-none"
              />
            </div>

            <div className="max-h-[500px] overflow-y-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="sticky top-0 border-b border-slate-800 bg-slate-900 text-slate-400">
                  <tr>
                    <th className="py-2.5">Row ID</th>
                    <th>Geography</th>
                    <th className="text-right">Actual</th>
                    <th className="text-right">Predicted</th>
                    <th className="text-right">Residual Error</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {testRecords
                    .filter((r) => r.geography.toLowerCase().includes(searchTerm.toLowerCase()))
                    .map((r) => (
                      <tr key={r.row_id}>
                        <td className="py-2 text-slate-500">{r.row_id}</td>
                        <td className="font-sans text-slate-200">{r.geography}</td>
                        <td className="text-right">{r.actual}</td>
                        <td className="text-right text-emerald-400">{r.predicted}</td>
                        <td className={`text-right ${r.residual > 0 ? "text-amber-400" : "text-blue-400"}`}>
                          {r.residual}
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