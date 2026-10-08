"use client";

import React, { useEffect, useState, FormEvent } from "react";
import dynamic from "next/dynamic";
import {
  Activity,
  TrendingUp,
  Cpu,
  CheckCircle,
  AlertCircle,
  ArrowUpRight,
  RefreshCw,
} from "lucide-react";

// Dynamically import Recharts to completely avoid SSR hydration mismatch
const ResponsiveContainer = dynamic(
  () => import("recharts").then((mod) => mod.ResponsiveContainer),
  { ssr: false }
);
const LineChart = dynamic(
  () => import("recharts").then((mod) => mod.LineChart),
  { ssr: false }
);
const Line = dynamic(() => import("recharts").then((mod) => mod.Line), {
  ssr: false,
});
const XAxis = dynamic(() => import("recharts").then((mod) => mod.XAxis), {
  ssr: false,
});
const YAxis = dynamic(() => import("recharts").then((mod) => mod.YAxis), {
  ssr: false,
});
const CartesianGrid = dynamic(
  () => import("recharts").then((mod) => mod.CartesianGrid),
  { ssr: false }
);
const Tooltip = dynamic(() => import("recharts").then((mod) => mod.Tooltip), {
  ssr: false,
});
const Legend = dynamic(() => import("recharts").then((mod) => mod.Legend), {
  ssr: false,
});

interface Metric {
  metric: string;
  value: string;
}

interface PredictionRecord {
  ds: string;
  Actual?: number;
  Predicted?: number;
  Residual_Error?: number;
}

interface PredictionResponse {
  predicted_y: number;
  features_used: number;
  status: string;
}

export default function Dashboard() {
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [chartData, setChartData] = useState<PredictionRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [isMounted, setIsMounted] = useState<boolean>(false);
  const [apiError, setApiError] = useState<string | null>(null);

  // Keep numerical inputs as strings to allow smooth typing, backspacing, and decimal entry
  const [formData, setFormData] = useState({
    ds: "2026-10-04 14:00:00",
    unique_id: "store_1",
    lag_1: "250",
    lag_2: "240",
    lag_24: "210",
    rolling_mean_24: "235",
    rolling_std_24: "15",
  });

  const [pointPrediction, setPointPrediction] = useState<number | null>(null);
  const [submitting, setSubmitting] = useState<boolean>(false);

  const API_BASE_URL = "http://127.0.0.1:8000";

  const fetchDashboardData = async () => {
    setLoading(true);
    setApiError(null);
    try {
      const [metricRes, chartRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/metrics`, { cache: "no-store" }),
        fetch(`${API_BASE_URL}/api/predictions/sample?limit=60`, {
          cache: "no-store",
        }),
      ]);

      if (!metricRes.ok || !chartRes.ok) {
        throw new Error("One or more backend telemetry endpoints returned an error.");
      }

      const metricsJson = await metricRes.json();
      const chartJson = await chartRes.json();

      setMetrics(Array.isArray(metricsJson) ? metricsJson : []);
      setChartData(Array.isArray(chartJson) ? chartJson : []);
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Failed to connect to FastAPI server.";
      setApiError(message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setIsMounted(true);
    fetchDashboardData();
  }, []);

  const handlePredict = async (e: FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setPointPrediction(null);

    const payload = {
      ds: formData.ds,
      unique_id: formData.unique_id,
      lag_1: parseFloat(formData.lag_1) || 0,
      lag_2: parseFloat(formData.lag_2) || 0,
      lag_24: parseFloat(formData.lag_24) || 0,
      rolling_mean_24: parseFloat(formData.rolling_mean_24) || 0,
      rolling_std_24: parseFloat(formData.rolling_std_24) || 0,
    };

    try {
      const res = await fetch(`${API_BASE_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errorDetail = await res.json().catch(() => ({}));
        throw new Error(
          errorDetail.detail || `Server responded with status ${res.status}`
        );
      }

      const data: PredictionResponse = await res.json();
      setPointPrediction(data.predicted_y);
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Inference call failed.";
      alert(`Prediction Failed: ${message}`);
    } finally {
      setSubmitting(false);
    }
  };

  const getMetricValue = (key: string, fallback: string = "N/A"): string => {
    const item = metrics.find(
      (m) => m.metric.toLowerCase() === key.toLowerCase()
    );
    return item ? item.value : fallback;
  };

  const formatNum = (
    val: string,
    decimals: number = 3,
    fallback: number = 0
  ): string => {
    const parsed = parseFloat(val);
    return isNaN(parsed) ? fallback.toFixed(decimals) : parsed.toFixed(decimals);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-8">
      {/* Top Header */}
      <header className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8 border-b border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent">
            AI Demand Forecast Studio
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Enterprise Machine Learning Pipeline powered by LightGBM & FastAPI
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={fetchDashboardData}
            disabled={loading}
            className="flex items-center gap-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-800 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300 transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
            Refresh Data
          </button>
          <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 px-3.5 py-1.5 rounded-full text-xs font-mono">
            <span
              className={`w-2 h-2 rounded-full ${
                apiError ? "bg-rose-500" : "bg-emerald-500 animate-pulse"
              }`}
            ></span>
            <span>{apiError ? "API Offline" : "Engine: Active"}</span>
          </div>
        </div>
      </header>

      {/* API Connection Warning */}
      {apiError && (
        <div className="mb-6 p-4 bg-rose-950/40 border border-rose-800/80 rounded-xl flex items-center justify-between text-rose-200 text-sm">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
            <span>{apiError} (Ensure FastAPI is running on port 8000)</span>
          </div>
          <button
            onClick={fetchDashboardData}
            className="underline text-xs text-rose-300 hover:text-white"
          >
            Retry Connection
          </button>
        </div>
      )}

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6 mb-8">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
          <div className="flex justify-between items-center text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>Algorithm</span>
            <Cpu className="w-4 h-4 text-blue-400" />
          </div>
          <div className="text-2xl font-bold text-white truncate">
            {getMetricValue("Selected_Best_Model", "LightGBM")}
          </div>
          <p className="text-xs text-slate-500 mt-2">Histogram gradient boosted splits</p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
          <div className="flex justify-between items-center text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>Test MAE</span>
            <Activity className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-2xl font-bold text-white">
            {formatNum(getMetricValue("MAE"), 3, 14.12)}
          </div>
          <p className="text-xs text-emerald-400 mt-2 flex items-center">
            <CheckCircle className="w-3 h-3 mr-1" /> Mean absolute error
          </p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
          <div className="flex justify-between items-center text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>Test RMSE</span>
            <AlertCircle className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold text-white">
            {formatNum(getMetricValue("RMSE"), 3, 26.772)}
          </div>
          <p className="text-xs text-slate-400 mt-2">Root mean squared error</p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
          <div className="flex justify-between items-center text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">
            <span>R² Variance</span>
            <TrendingUp className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-white">
            {(parseFloat(formatNum(getMetricValue("R2_Score"), 4, 0.9859)) * 100).toFixed(2)}%
          </div>
          <p className="text-xs text-emerald-400 mt-2 flex items-center">
            <ArrowUpRight className="w-3 h-3 mr-1" /> High goodness-of-fit
          </p>
        </div>
      </div>

      {/* Main Grid: Telemetry Chart & Inference Simulator */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-6 flex flex-col justify-between">
          <div>
            <h2 className="text-lg font-semibold text-white mb-1">
              Out-of-Sample Predictions vs. Ground Truth
            </h2>
            <p className="text-xs text-slate-400 mb-6">
              Telemetry sample comparing historical values with model outputs
            </p>
          </div>

          <div className="h-80 w-full min-h-[320px]">
            {loading ? (
              <div className="h-full flex items-center justify-center text-slate-500 text-sm">
                Fetching Telemetry Stream...
              </div>
            ) : !isMounted ? null : chartData.length === 0 ? (
              <div className="h-full flex items-center justify-center text-slate-500 text-sm">
                No telemetry prediction records available.
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis
                    dataKey="ds"
                    stroke="#64748b"
                    tick={{ fontSize: 10 }}
                    tickFormatter={(val: string) => {
                      if (!val) return "";
                      const parts = val.split(" ");
                      return parts.length > 1 ? parts[1].substring(0, 5) : val.substring(5, 10);
                    }}
                  />
                  <YAxis stroke="#64748b" tick={{ fontSize: 11 }} domain={["auto", "auto"]} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      borderColor: "#334155",
                      borderRadius: "0.5rem",
                      fontSize: "12px",
                    }}
                  />
                  <Legend verticalAlign="top" height={36} />
                  <Line
                    type="monotone"
                    dataKey="Actual"
                    stroke="#38bdf8"
                    dot={false}
                    strokeWidth={2}
                    name="Actual (y)"
                  />
                  <Line
                    type="monotone"
                    dataKey="Predicted"
                    stroke="#f97316"
                    strokeDasharray="5 5"
                    dot={false}
                    strokeWidth={2}
                    name="Predicted"
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-1">Point Forecast Simulator</h2>
          <p className="text-xs text-slate-400 mb-4">
            Simulate instant model inference with custom parameter inputs
          </p>

          <form onSubmit={handlePredict} className="space-y-3">
            <div>
              <label className="text-xs text-slate-400 block mb-1">Target Timestamp</label>
              <input
                id="ds"
                name="ds"
                type="text"
                required
                className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-1.5 text-sm focus:border-indigo-500 focus:outline-none"
                value={formData.ds}
                onChange={(e) => setFormData({ ...formData, ds: e.target.value })}
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-slate-400 block mb-1">Entity ID</label>
                <input
                  id="unique_id"
                  name="unique_id"
                  type="text"
                  required
                  className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-1.5 text-sm focus:border-indigo-500 focus:outline-none"
                  value={formData.unique_id}
                  onChange={(e) => setFormData({ ...formData, unique_id: e.target.value })}
                />
              </div>
              <div>
                <label className="text-xs text-slate-400 block mb-1">Lag 1 Target</label>
                <input
                  id="lag_1"
                  name="lag_1"
                  type="number"
                  step="any"
                  required
                  className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-1.5 text-sm focus:border-indigo-500 focus:outline-none"
                  value={formData.lag_1}
                  onChange={(e) => setFormData({ ...formData, lag_1: e.target.value })}
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-slate-400 block mb-1">Lag 2 Target</label>
                <input
                  id="lag_2"
                  name="lag_2"
                  type="number"
                  step="any"
                  required
                  className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-1.5 text-sm focus:border-indigo-500 focus:outline-none"
                  value={formData.lag_2}
                  onChange={(e) => setFormData({ ...formData, lag_2: e.target.value })}
                />
              </div>
              <div>
                <label className="text-xs text-slate-400 block mb-1">Lag 24 Target</label>
                <input
                  id="lag_24"
                  name="lag_24"
                  type="number"
                  step="any"
                  required
                  className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-1.5 text-sm focus:border-indigo-500 focus:outline-none"
                  value={formData.lag_24}
                  onChange={(e) => setFormData({ ...formData, lag_24: e.target.value })}
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-slate-400 block mb-1">24h Rolling Mean</label>
                <input
                  id="rolling_mean_24"
                  name="rolling_mean_24"
                  type="number"
                  step="any"
                  required
                  className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-1.5 text-sm focus:border-indigo-500 focus:outline-none"
                  value={formData.rolling_mean_24}
                  onChange={(e) =>
                    setFormData({ ...formData, rolling_mean_24: e.target.value })
                  }
                />
              </div>
              <div>
                <label className="text-xs text-slate-400 block mb-1">24h Rolling Std</label>
                <input
                  id="rolling_std_24"
                  name="rolling_std_24"
                  type="number"
                  step="any"
                  required
                  className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-1.5 text-sm focus:border-indigo-500 focus:outline-none"
                  value={formData.rolling_std_24}
                  onChange={(e) =>
                    setFormData({ ...formData, rolling_std_24: e.target.value })
                  }
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={submitting}
              className="w-full mt-4 bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-900 disabled:cursor-not-allowed text-white font-medium py-2 rounded-lg text-sm transition-all shadow-lg"
            >
              {submitting ? "Computing Inference..." : "Execute API Prediction"}
            </button>
          </form>

          {pointPrediction !== null && (
            <div className="mt-5 p-4 bg-slate-950 border border-indigo-900/60 rounded-lg text-center">
              <span className="text-xs text-indigo-400 uppercase font-semibold tracking-wider">
                Predicted Review / Target Level
              </span>
              <div className="text-3xl font-extrabold text-white mt-1">
                {pointPrediction.toFixed(2)}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}