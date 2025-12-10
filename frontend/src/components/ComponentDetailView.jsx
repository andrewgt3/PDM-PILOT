import React from "react";
import {
  ArrowLeft,
  Activity,
  Bell,
  FileText,
  Settings,
  CheckCircle,
} from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const data = [
  { time: "00:00", vibration: 12, baseline: 10 },
  { time: "04:00", vibration: 18, baseline: 12 },
  { time: "08:00", vibration: 45, baseline: 15 },
  { time: "12:00", vibration: 85, baseline: 18 },
  { time: "16:00", vibration: 55, baseline: 16 },
  { time: "20:00", vibration: 30, baseline: 14 },
  { time: "24:00", vibration: 20, baseline: 12 },
];

const ComponentDetailView = ({ factor, robotName, onBack }) => {
  // Dynamic Recommendation Logic
  let recommendation = "Perform comprehensive diagnostics.";
  if (factor.name.includes("Temp") || factor.category === "Thermal") {
    recommendation =
      "Inspect cooling fans, clean vents, and check axis lubrication.";
  } else if (
    factor.name.includes("Vibration") ||
    factor.category === "Vibration"
  ) {
    recommendation =
      "Perform dynamic balance test and check for mounting looseness.";
  } else if (
    factor.name.includes("Maint") ||
    factor.category === "Maintenance"
  ) {
    recommendation =
      "Schedule full preventive maintenance service immediately.";
  } else if (
    factor.name.includes("Error") ||
    factor.category === "Electrical"
  ) {
    recommendation = "Check encoder grounding and signal cable integrity.";
  }

  return (
    <div className="animate-zoom-in" style={{ width: "100%" }}>
      {/* Header */}
      <div
        className="flex-center"
        style={{ gap: "1rem", marginBottom: "2rem" }}
      >
        <button
          onClick={onBack}
          className="clickable"
          style={{
            padding: "0.5rem",
            borderRadius: "50%",
            border: "1px solid var(--border-color)",
            background: "white",
          }}
        >
          <ArrowLeft size={20} />
        </button>
        <div>
          <h2 className="section-title" style={{ marginBottom: 0 }}>
            {factor.name.toUpperCase()} ANALYSIS
          </h2>
          <span className="text-sm text-gray">
            {robotName} • {factor.value} • {factor.trend}
          </span>
        </div>
      </div>

      <div className="dashboard-grid">
        <div className="main-col">
          <div className="card">
            <div
              className="flex-center"
              style={{ gap: "1rem", marginBottom: "1rem" }}
            >
              <div
                style={{
                  padding: "0.75rem",
                  background: "rgba(239, 68, 68, 0.1)",
                  borderRadius: "8px",
                }}
              >
                <Activity size={24} color="var(--accent-red)" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-red">
                  Critical Anomaly Detected
                </h3>
                <div className="text-sm text-gray">
                  {/* Dynamic Root Cause Display */}
                  {factor.rootCause ? (
                    <>
                      AI Identification:{" "}
                      <span className="font-bold">{factor.rootCause}</span>.{" "}
                      {factor.name} exceeds safety threshold.
                    </>
                  ) : (
                    <>
                      {factor.name} exceeds safety threshold. Failure imminent.
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Top Stats Row */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr",
          gap: "1rem",
          marginBottom: "1.5rem",
        }}
      >
        <div
          className="card flex-center"
          style={{ flexDirection: "column", gap: "0.5rem" }}
        >
          <span
            className="text-xs text-gray"
            style={{ letterSpacing: "0.05em" }}
          >
            REMAINING USEFUL LIFE
          </span>
          <span
            style={{ fontSize: "2rem", fontWeight: "bold", color: "#c2410c" }}
          >
            72h
          </span>
          <span className="text-xs text-gray">
            Time until predicted failure
          </span>
        </div>
        <div
          className="card flex-center"
          style={{ flexDirection: "column", gap: "0.5rem" }}
        >
          <span
            className="text-xs text-gray"
            style={{ letterSpacing: "0.05em" }}
          >
            PREDICTION CONFIDENCE
          </span>
          <span
            style={{ fontSize: "2rem", fontWeight: "bold", color: "#c2410c" }}
          >
            94%
          </span>
          <span className="text-xs text-gray">
            Likelihood of failure in &lt; 7 days
          </span>
        </div>
        <div
          className="card flex-center"
          style={{ flexDirection: "column", gap: "0.5rem" }}
        >
          <span
            className="text-xs text-gray"
            style={{ letterSpacing: "0.05em" }}
          >
            CRITICALITY SCORE
          </span>
          <span
            style={{ fontSize: "2rem", fontWeight: "bold", color: "#c2410c" }}
          >
            High
          </span>
          <span className="text-xs text-gray">Impact on Production</span>
        </div>
      </div>

      {/* Actions Row */}
      <div
        className="card"
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "1.5rem",
        }}
      >
        <div style={{ display: "flex", gap: "1rem" }}>
          <button
            className="flex-center"
            style={{
              gap: "0.5rem",
              padding: "0.5rem 1rem",
              background: "white",
              border: "1px solid var(--border-color)",
              borderRadius: "6px",
              cursor: "pointer",
            }}
          >
            <CheckCircle size={16} /> Acknowledge
          </button>
          <button
            className="flex-center"
            style={{
              gap: "0.5rem",
              padding: "0.5rem 1rem",
              background: "#c2410c",
              color: "white",
              border: "none",
              borderRadius: "6px",
              cursor: "pointer",
            }}
          >
            <Bell size={16} /> Notify Maintenance
          </button>
          <button
            className="flex-center"
            style={{
              gap: "0.5rem",
              padding: "0.5rem 1rem",
              background: "white",
              border: "1px solid var(--border-color)",
              borderRadius: "6px",
              cursor: "pointer",
            }}
          >
            <FileText size={16} /> View SOP
          </button>
        </div>

        <div
          style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}
        >
          <span className="text-sm font-bold">ENGINEERING ACTIONS</span>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <Settings size={16} color="var(--accent-blue)" />
            <span className="text-sm">AI Recommendation: {recommendation}</span>
            <button
              style={{
                padding: "0.25rem 0.75rem",
                background: "var(--accent-blue)",
                color: "white",
                border: "none",
                borderRadius: "4px",
                fontSize: "0.75rem",
                marginLeft: "1rem",
                cursor: "pointer",
              }}
            >
              Create Work Order
            </button>
          </div>
        </div>
      </div>

      {/* Charts Section */}
      <div className="card">
        <h3
          className="text-sm font-bold text-gray"
          style={{ marginBottom: "1rem" }}
        >
          AI REASONING & DIAGNOSTICS
        </h3>
        <div style={{ height: "300px", width: "100%", marginTop: "1rem" }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={data}
              margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
            >
              <defs>
                <linearGradient id="colorVib" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.1} />
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorBase" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#22c55e" stopOpacity={0.1} />
                  <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                vertical={false}
                stroke="#e5e7eb"
              />
              <XAxis dataKey="time" tick={{ fontSize: 12 }} stroke="#9ca3af" />
              <YAxis tick={{ fontSize: 12 }} stroke="#9ca3af" />
              <Tooltip
                contentStyle={{
                  borderRadius: "8px",
                  border: "none",
                  boxShadow: "0 4px 6px -1px rgba(0,0,0,0.1)",
                }}
              />
              <Area
                type="monotone"
                dataKey="baseline"
                stroke="#22c55e"
                strokeWidth={2}
                strokeDasharray="5 5"
                fillOpacity={1}
                fill="url(#colorBase)"
                name="Healthy Baseline"
              />
              <Area
                type="monotone"
                dataKey="vibration"
                stroke="#ef4444"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#colorVib)"
                name="Anomalous Spectrum"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div
          style={{
            display: "flex",
            gap: "2rem",
            marginTop: "1rem",
            justifyContent: "center",
          }}
        >
          <div className="flex-center" style={{ gap: "0.5rem" }}>
            <div
              style={{ width: "12px", height: "2px", background: "#ef4444" }}
            ></div>
            <span className="text-xs text-gray">Anomalous Spectrum (Live)</span>
          </div>
          <div className="flex-center" style={{ gap: "0.5rem" }}>
            <div
              style={{
                width: "12px",
                height: "2px",
                background: "#22c55e",
                borderTop: "1px dashed #22c55e",
              }}
            ></div>
            <span className="text-xs text-gray">Healthy Baseline</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ComponentDetailView;
