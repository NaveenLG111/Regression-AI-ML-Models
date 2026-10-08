"use client";

import React, { useState, useEffect } from "react";

interface InspectionData {
  filename: string;
  total_rows: number;
  columns: string[];
  numeric_targets: string[];
  preview: Record<string, any>[];
}

interface SummaryItem {
  Item: string;
  Value: string | number;
}

interface MetricItem {
  Split: string;
  Model: string;
  MAE: number | string;
  RMSE: number | string;
  R2: number | string;
}

interface FeatureImportanceItem {
  Feature: string;
  Importance: number;
}

interface TrainResponse {
  status: string;
  summary: SummaryItem[];
  metrics: MetricItem[];
  feature_importance?: FeatureImportanceItem[];
  plots: string[];
  excel_ready: boolean;
}

export default function MLReportDashboard() {
  const [file, setFile] = useState<File | null>(null);
  const [inspection, setInspection] = useState<InspectionData | null>(null);
  const [selectedTarget, setSelectedTarget] = useState<string>("");
  const [testSize, setTestSize] = useState<number>(0.2);
  const [randomState, setRandomState] = useState<number>(42);
  const [loading, setLoading] = useState<boolean>(false);
  const [results, setResults] = useState<TrainResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [activeTab, setActiveTab] = useState<"overview" | "metrics" | "plots" | "preview">("overview");

  useEffect(() => {
    inspectDataset();
  }, []);

  const inspectDataset = async (uploadedFile?: File) => {
    setErrorMsg("");
    const formData = new FormData();
    if (uploadedFile) {
      formData.append("file", uploadedFile);
    }

    try {
      const res = await fetch("http://localhost:8000/api/inspect", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Dataset inspection failed.");
      }

      const data: InspectionData = await res.json();
      setInspection(data);

      if (data.numeric_targets && data.numeric_targets.length > 0) {
        const defaultMatch =
          data.numeric_targets.find((c) =>
            c.toLowerCase().includes("nursery") || c.toLowerCase().includes("cents")
          ) || data.numeric_targets[0];
        setSelectedTarget(defaultMatch);
      }
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to connect to the backend API.");
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      setFile(selected);
      inspectDataset(selected);
    }
  };

  const handleTrain = async () => {
    if (!selectedTarget) {
      setErrorMsg("Please select a valid numerical target column.");
      return;
    }

    setLoading(true);
    setErrorMsg("");

    const formData = new FormData();
    formData.append("filename", inspection?.filename || "paddydataset.csv");
    formData.append("target_column", selectedTarget);
    formData.append("test_size", testSize.toString());
    formData.append("random_state", randomState.toString());

    try {
      const res = await fetch("http://localhost:8000/api/train", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Model execution failed.");
      }

      const data: TrainResponse = await res.json();
      setResults(data);
      setActiveTab("overview");
    } catch (err: any) {
      setErrorMsg(err.message || "Execution error during training.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-100 text-slate-800 font-sans p-6 md:p-10">
      {/* Top Header */}
      <header className="max-w-7xl mx-auto mb-8 flex flex-col md:flex-row md:items-center justify-between gap-4 bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
        <div>
          <h1 className="text-2xl md:text-3xl font-extrabold text-[#1F4E78] flex items-center gap-2">
            🌾 Paddy Prediction & ML Report Engine
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Ridge Regression Pipeline with OpenPyXL Reporting, EDA, & Diagnostic Plots
          </p>
        </div>

        {results?.excel_ready && (
          <a
            href="http://localhost:8000/api/download-excel"
            download="ml_model_report.xlsx"
            className="inline-flex items-center justify-center gap-2 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold px-5 py-2.5 rounded-xl shadow-sm transition-colors text-sm"
          >
            <span>📥</span> Download Excel Report (.xlsx)
          </a>
        )}
      </header>

      {/* Main Grid Content */}
      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Settings Sidebar (4 Cols) */}
        <aside className="lg:col-span-4 bg-white p-6 rounded-2xl shadow-sm border border-slate-200 h-fit space-y-6">
          <h2 className="text-base font-bold text-slate-800 border-b pb-3 flex items-center gap-2">
            <span>⚙️</span> Model Configuration
          </h2>

          {/* Dataset Upload */}
          <div>
            <label className="block text-xs font-semibold text-slate-600 mb-1.5">
              Upload Dataset (CSV)
            </label>
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="w-full text-xs text-slate-500 file:mr-3 file:py-2 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-blue-50 file:text-[#1F4E78] hover:file:bg-blue-100 cursor-pointer border border-slate-200 rounded-lg p-1"
            />
            {inspection?.filename && (
              <p className="text-[11px] text-emerald-600 font-medium mt-1">
                Active: {inspection.filename} ({inspection.total_rows} rows)
              </p>
            )}
          </div>

          {/* Target Column Selection */}
          <div>
            <label className="block text-xs font-semibold text-slate-600 mb-1.5">
              Target Variable (Regression Target)
            </label>
            <select
              value={selectedTarget}
              onChange={(e) => setSelectedTarget(e.target.value)}
              className="w-full text-sm border border-slate-200 rounded-lg p-2.5 bg-slate-50 focus:outline-none focus:ring-2 focus:ring-[#1F4E78]"
            >
              {inspection?.numeric_targets && inspection.numeric_targets.length > 0 ? (
                inspection.numeric_targets.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))
              ) : (
                <option value="">No numeric targets found</option>
              )}
            </select>
          </div>

          {/* Test Size Slider */}
          <div>
            <div className="flex justify-between text-xs font-semibold text-slate-600 mb-1">
              <span>Test Split Ratio</span>
              <span className="font-mono text-[#1F4E78]">{Math.round(testSize * 100)}%</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.4"
              step="0.05"
              value={testSize}
              onChange={(e) => setTestSize(parseFloat(e.target.value))}
              className="w-full accent-[#1F4E78] cursor-pointer"
            />
          </div>

          {/* Random Seed */}
          <div>
            <label className="block text-xs font-semibold text-slate-600 mb-1.5">
              Random Seed (Reproducibility)
            </label>
            <input
              type="number"
              value={randomState}
              onChange={(e) => setRandomState(parseInt(e.target.value) || 0)}
              className="w-full text-sm border border-slate-200 rounded-lg p-2 bg-slate-50 focus:outline-none focus:ring-2 focus:ring-[#1F4E78]"
            />
          </div>

          {/* Execute Button */}
          <button
            onClick={handleTrain}
            disabled={loading || !selectedTarget}
            className="w-full bg-[#1F4E78] hover:bg-[#183d5e] text-white font-semibold py-3 rounded-xl shadow transition duration-150 disabled:opacity-50 text-sm flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <svg className="animate-spin h-4 w-4 text-white" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
                </svg>
                Training Pipeline...
              </>
            ) : (
              "🚀 Run Model & Build Report"
            )}
          </button>

          {/* Error Container */}
          {errorMsg && (
            <div className="p-3 bg-red-50 border border-red-200 text-red-700 text-xs rounded-lg">
              {errorMsg}
            </div>
          )}
        </aside>

        {/* Right Dynamic Analytics Panel (8 Cols) */}
        <section className="lg:col-span-8 space-y-6">
          {/* Navigation Tabs */}
          <div className="flex border-b border-slate-200 bg-white px-4 rounded-xl shadow-sm">
            {[
              { id: "overview", label: "📊 Overview" },
              { id: "metrics", label: "📈 Performance Metrics" },
              { id: "plots", label: "🖼️ Diagnostics & Plots" },
              { id: "preview", label: "📋 Raw Data Preview" },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`py-3.5 px-4 text-sm font-semibold border-b-2 transition-colors ${
                  activeTab === tab.id
                    ? "border-[#1F4E78] text-[#1F4E78]"
                    : "border-transparent text-slate-500 hover:text-slate-700"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab 1: Overview */}
          {activeTab === "overview" && (
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-6">
              {results ? (
                <>
                  <h3 className="text-base font-bold text-slate-800">Pipeline Execution Summary</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs text-left">
                      <tbody>
                        {results.summary.map((row, idx) => (
                          <tr key={idx} className="border-b border-slate-100 last:border-0 hover:bg-slate-50">
                            <td className="py-2.5 px-3 text-slate-500 font-medium">{row.Item}</td>
                            <td className="py-2.5 px-3 font-semibold text-slate-800 text-right">{row.Value}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {results.feature_importance && results.feature_importance.length > 0 && (
                    <div className="pt-4 border-t border-slate-100">
                      <h4 className="text-sm font-bold text-slate-800 mb-3">Top Influential Features</h4>
                      <div className="space-y-2">
                        {results.feature_importance.slice(0, 5).map((f, i) => (
                          <div key={i} className="flex items-center justify-between text-xs">
                            <span className="text-slate-600">{f.Feature}</span>
                            <span className="font-mono font-semibold text-[#1F4E78]">
                              {Number(f.Importance).toFixed(4)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="text-center py-12 text-slate-400">
                  <p className="text-4xl mb-2">📊</p>
                  <p className="text-sm">Click "Run Model & Build Report" to train the model and generate metrics.</p>
                </div>
              )}
            </div>
          )}

          {/* Tab 2: Performance Metrics */}
          {activeTab === "metrics" && (
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
              <h3 className="text-base font-bold text-slate-800 mb-4">Train, Test & Cross-Validation Metrics</h3>
              {results ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left border-collapse">
                    <thead>
                      <tr className="bg-slate-50 text-slate-500 font-semibold border-b border-slate-200">
                        <th className="py-2.5 px-3">Split</th>
                        <th className="py-2.5 px-3">Model</th>
                        <th className="py-2.5 px-3">MAE</th>
                        <th className="py-2.5 px-3">RMSE</th>
                        <th className="py-2.5 px-3">R² Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.metrics.map((m, idx) => (
                        <tr key={idx} className="border-b border-slate-100 last:border-0 hover:bg-slate-50">
                          <td className="py-3 px-3 font-semibold text-slate-800">{m.Split}</td>
                          <td className="py-3 px-3 text-slate-600">{m.Model}</td>
                          <td className="py-3 px-3 font-mono">
                            {typeof m.MAE === "number" ? m.MAE.toFixed(4) : "-"}
                          </td>
                          <td className="py-3 px-3 font-mono">
                            {typeof m.RMSE === "number" ? m.RMSE.toFixed(4) : "-"}
                          </td>
                          <td className="py-3 px-3 font-mono font-bold text-[#1F4E78]">
                            {typeof m.R2 === "number" ? m.R2.toFixed(4) : "-"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-center py-12 text-slate-400">
                  <p className="text-sm">Metrics will be displayed here once model training executes.</p>
                </div>
              )}
            </div>
          )}

          {/* Tab 3: Generated Visual Plots */}
          {activeTab === "plots" && (
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
              <h3 className="text-base font-bold text-slate-800">Generated Seaborn/Matplotlib Plots</h3>
              {results?.plots && results.plots.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {results.plots.map((url, i) => {
                    const plotName = url.split("/").pop()?.replace(".png", "").replace(/_/g, " ").toUpperCase();
                    return (
                      <div key={i} className="border border-slate-200 rounded-xl overflow-hidden bg-slate-50 p-2">
                        <p className="text-[11px] font-semibold text-slate-500 mb-1 px-1">{plotName}</p>
                        <img
                          src={`${url}?t=${Date.now()}`}
                          alt="Diagnostic Plot"
                          className="w-full h-auto rounded-lg object-contain bg-white"
                        />
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="text-center py-12 text-slate-400">
                  <p className="text-sm">No plot artifacts generated yet. Run the training pipeline to generate charts.</p>
                </div>
              )}
            </div>
          )}

          {/* Tab 4: Raw Data Preview */}
          {activeTab === "preview" && (
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-base font-bold text-slate-800">Dataset Preview</h3>
                <span className="text-xs text-slate-500">
                  Showing top 5 rows of {inspection?.total_rows || 0}
                </span>
              </div>
              {inspection?.preview && inspection.preview.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left border-collapse">
                    <thead>
                      <tr className="bg-slate-50 border-b border-slate-200">
                        {inspection.columns.map((c) => (
                          <th key={c} className="py-2 px-3 font-semibold text-slate-700 whitespace-nowrap">
                            {c}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {inspection.preview.map((row, idx) => (
                        <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50">
                          {inspection.columns.map((col) => (
                            <td key={col} className="py-2 px-3 text-slate-600 whitespace-nowrap">
                              {row[col] !== null && row[col] !== undefined ? String(row[col]) : ""}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-center py-12 text-slate-400">
                  <p className="text-sm">No preview records available.</p>
                </div>
              )}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}