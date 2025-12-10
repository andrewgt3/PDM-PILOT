import React from "react";
import { Loader2 } from "lucide-react";

const LoadingOverlay = ({ message = "AI Analyzing Sensor Data..." }) => {
  return (
    <div
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: "rgba(255, 255, 255, 0.85)",
        backdropFilter: "blur(4px)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 50,
        animation: "fadeIn 0.3s ease",
      }}
    >
      <div style={{ position: "relative" }}>
        <div
          style={{
            position: "absolute",
            top: "-10px",
            left: "-10px",
            right: "-10px",
            bottom: "-10px",
            borderRadius: "50%",
            border: "2px solid var(--accent-blue)",
            opacity: 0.2,
            animation: "ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite",
          }}
        ></div>
        <Loader2 size={48} className="spin-slow" color="var(--accent-blue)" />
      </div>
      <span
        className="text-sm font-bold text-gray"
        style={{ marginTop: "1rem", letterSpacing: "0.05em" }}
      >
        {message}
      </span>
    </div>
  );
};

export default LoadingOverlay;
