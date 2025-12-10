import React from "react";
import { ArrowLeft, AlertTriangle, CheckCircle, Activity } from "lucide-react";

const ProcessView = ({ processName, onBack, onSelectAsset }) => {
  const assets = [
    {
      id: "R-208",
      name: "Robot Arm R-208",
      status: "warning",
      rul: "8.0 days",
      confidence: "81%",
    },
    {
      id: "C-206",
      name: "Conveyor C-206",
      status: "warning",
      rul: "8.3 days",
      confidence: "81%",
    },
    {
      id: "W-233",
      name: "Welder W-233",
      status: "warning",
      rul: "8.3 days",
      confidence: "73%",
    },
    {
      id: "P-214",
      name: "Paint Pump P-214",
      status: "warning",
      rul: "9.8 days",
      confidence: "73%",
    },
    {
      id: "H-200",
      name: "Oven H-200",
      status: "warning",
      rul: "10.1 days",
      confidence: "84%",
    },
    {
      id: "S-215",
      name: "Separator S-215",
      status: "warning",
      rul: "10.7 days",
      confidence: "85%",
    },
    {
      id: "C-208",
      name: "Cooler C-208",
      status: "warning",
      rul: "11.0 days",
      confidence: "77%",
    },
    {
      id: "T-219",
      name: "Turbine T-219",
      status: "healthy",
      rul: "16.2 days",
      confidence: "84%",
    },
  ];

  return (
    <div className="animate-zoom-in" style={{ width: "100%" }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          marginBottom: "1.5rem",
        }}
      >
        <button
          onClick={onBack}
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
            color: "var(--text-secondary)",
            fontSize: "0.875rem",
            marginRight: "1rem",
          }}
        >
          <ArrowLeft size={16} /> Back to Overview
        </button>
        <h2
          className="text-sm font-bold text-gray"
          style={{ letterSpacing: "0.05em", textTransform: "uppercase" }}
        >
          ASSETS IN {processName} ({assets.length} MACHINES)
        </h2>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
          gap: "1rem",
        }}
      >
        {assets.map((asset) => (
          <div
            key={asset.id}
            className="card clickable"
            onClick={() => onSelectAsset(asset)}
            style={{
              borderTop: `4px solid ${asset.status === "warning" ? "var(--status-warning)" : "var(--status-healthy)"}`,
              display: "flex",
              flexDirection: "column",
              gap: "0.5rem",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "flex-start",
              }}
            >
              <span className="text-xs text-gray">CONFIDENCE</span>
              {asset.status === "warning" ? (
                <AlertTriangle size={16} color="var(--status-warning)" />
              ) : (
                <CheckCircle size={16} color="var(--status-healthy)" />
              )}
            </div>
            <span
              style={{
                fontSize: "1.5rem",
                fontWeight: "bold",
                color:
                  asset.status === "warning"
                    ? "var(--status-warning)"
                    : "var(--status-healthy)",
              }}
            >
              {asset.confidence}
            </span>

            <div style={{ marginTop: "0.5rem" }}>
              <span className="text-xs text-gray">RUL</span>
              <div className="font-bold">{asset.rul}</div>
            </div>

            <div
              style={{
                marginTop: "0.5rem",
                paddingTop: "0.5rem",
                borderTop: "1px solid var(--border-color)",
              }}
            >
              <span className="text-xs text-gray">{asset.id}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ProcessView;
