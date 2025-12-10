import React from "react";

const Header = () => {
  return (
    <header
      style={{
        height: "64px",
        background: "white",
        borderBottom: "1px solid var(--border-color)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 2rem",
      }}
    >
      <div style={{ display: "flex", flexDirection: "column" }}>
        <h1
          style={{ fontSize: "1.125rem", fontWeight: "600", color: "#111827" }}
        >
          Predictive Logic Systems
        </h1>
        <span style={{ fontSize: "0.75rem", color: "#6b7280" }}>
          AI-Powered Predictive Maintenance Intelligence
        </span>
      </div>

      <div style={{ flex: 1, maxWidth: "600px", margin: "0 2rem" }}>
        <input
          type="text"
          placeholder="Search for asset by name or ID..."
          style={{
            width: "100%",
            padding: "0.5rem 1rem",
            borderRadius: "6px",
            border: "1px solid var(--border-color)",
            background: "#f9fafb",
            fontSize: "0.875rem",
          }}
        />
      </div>

      <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
        <button
          style={{
            padding: "0.5rem",
            background: "none",
            border: "none",
            cursor: "pointer",
          }}
        >
          {/* Sun icon placeholder */}
          ☀️
        </button>
        <button
          style={{
            padding: "0.5rem 1rem",
            background: "white",
            border: "1px solid var(--border-color)",
            borderRadius: "6px",
            fontWeight: "500",
            fontSize: "0.875rem",
            cursor: "pointer",
          }}
        >
          Executive View
        </button>
      </div>
    </header>
  );
};

export default Header;
