import React from "react";
import {
  Hammer,
  Car,
  PaintBucket,
  Wrench,
  CheckCircle,
  AlertTriangle,
  Activity,
} from "lucide-react";

const ProcessNode = ({ title, icon: Icon, status = "healthy", onClick }) => {
  let borderColor = "var(--accent-green)";
  let shadowColor = "rgba(34, 197, 94, 0.2)";

  if (status === "warning") {
    borderColor = "var(--accent-orange)";
    shadowColor = "rgba(249, 115, 22, 0.2)";
  }
  if (status === "critical") {
    borderColor = "var(--accent-red)";
    shadowColor = "rgba(239, 68, 68, 0.2)";
  }

  return (
    <div
      className="clickable animate-zoom-in"
      onClick={onClick}
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "1rem",
        position: "relative",
        zIndex: 1,
      }}
    >
      <div
        style={{
          width: "100px",
          height: "100px",
          borderRadius: "50%",
          border: `3px solid ${borderColor}`,
          background: "white",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          boxShadow: `0 10px 25px ${shadowColor}`,
          transition: "transform 0.3s ease",
        }}
        className="hover:scale-110"
      >
        <Icon
          size={40}
          className="pulse"
          color={status === "healthy" ? "var(--text-primary)" : borderColor}
        />
      </div>

      <div
        className="glass-panel"
        style={{
          padding: "0.5rem 1rem",
          borderRadius: "20px",
          textAlign: "center",
        }}
      >
        <span className="text-sm font-bold" style={{ display: "block" }}>
          {title}
        </span>
        <span
          className="text-xs text-gray"
          style={{ textTransform: "uppercase" }}
        >
          {status}
        </span>
      </div>
    </div>
  );
};

const PlantOverview = ({ onSelectProcess }) => {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "2rem",
        width: "100%",
      }}
    >
      <div className="flex-center" style={{ justifyContent: "space-between" }}>
        <div>
          <h2 className="section-title" style={{ marginBottom: "0.25rem" }}>
            PLANT OVERVIEW
          </h2>
          <div className="text-sm text-gray">
            Real-time Production Line Status
          </div>
        </div>
        <div className="glass-panel" style={{ padding: "0.5rem 1rem" }}>
          <span
            className="text-xs font-bold text-success flex-center"
            style={{ gap: "0.5rem" }}
          >
            <Activity size={14} /> SYSTEM OPTIMAL: 98.2%
          </span>
        </div>
      </div>

      <div
        className="card"
        style={{
          position: "relative",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "4rem 2rem",
          minHeight: "400px",
          overflow: "hidden",
        }}
      >
        {/* Background Trace Line */}
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "60px",
            right: "60px",
            height: "4px",
            background: "var(--border-color)",
            zIndex: 0,
            backgroundImage:
              "linear-gradient(to right, var(--accent-green) 50%, var(--border-color) 50%)",
            backgroundSize: "20px 100%",
          }}
        ></div>

        <ProcessNode
          title="STAMPING"
          icon={Hammer}
          status="healthy"
          onClick={() => onSelectProcess("Stamping Press")}
        />
        <ProcessNode
          title="BODY SHOP"
          icon={Car}
          status="warning"
          onClick={() => onSelectProcess("Body Shop")}
        />
        <ProcessNode
          title="PAINT SHOP"
          icon={PaintBucket}
          status="healthy"
          onClick={() => onSelectProcess("Paint Shop")}
        />
        <ProcessNode
          title="ASSEMBLY"
          icon={Wrench}
          status="healthy"
          onClick={() => onSelectProcess("Assembly")}
        />
      </div>

      <div
        className="dashboard-grid"
        style={{ gridTemplateColumns: "repeat(3, 1fr)", gap: "1.5rem" }}
      >
        <div className="glass-panel" style={{ padding: "1.5rem" }}>
          <h3 className="text-sm font-bold text-gray mb-2">OUTPUT RATE</h3>
          <div className="text-2xl font-bold">
            482 <span className="text-sm text-gray font-normal">units/hr</span>
          </div>
        </div>
        <div className="glass-panel" style={{ padding: "1.5rem" }}>
          <h3 className="text-sm font-bold text-gray mb-2">ACTIVE ALERTS</h3>
          <div className="text-2xl font-bold text-orange">
            2 <span className="text-sm text-gray font-normal">Warnings</span>
          </div>
        </div>
        <div className="glass-panel" style={{ padding: "1.5rem" }}>
          <h3 className="text-sm font-bold text-gray mb-2">
            ENERGY EFFICIENCY
          </h3>
          <div className="text-2xl font-bold text-green">
            94%{" "}
            <span className="text-sm text-gray font-normal">Target Met</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PlantOverview;
