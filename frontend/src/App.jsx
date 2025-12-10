import React, { useState, useEffect, Suspense } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./components/Dashboard";
import AssetMonitor from "./components/AssetMonitor";
import LoadingOverlay from "./components/LoadingOverlay";
import "./App.css";

const AssetView = React.lazy(() => import("./components/AssetView"));
const ModelAudit = React.lazy(() => import("./components/ModelAudit"));

const MACHINE_IDS = [
  "M14860", "L47181", "L47182", "L47183", "L47184", "M14865",
];

function App() {
  const [robotsData, setRobotsData] = useState([]);
  const [isDataLoading, setIsDataLoading] = useState(true);

  // Fetch Data
  useEffect(() => {
    const fetchData = async () => {
      try {
        const promises = MACHINE_IDS.map((id) =>
          fetch(`http://localhost:8000/api/v1/predict/machine/${id}`).then(
            (res) => res.json(),
          ),
        );
        const results = await Promise.all(promises);

        const formattedData = results.map((r, i) => {
          let status = "healthy";
          if (r.status.includes("Risk") || r.failure_probability > 0.5) status = "critical";
          else if (r.status.includes("Warn")) status = "warning";

          return {
            id: r.machine_id,
            name: `Robot ${i + 1}`,
            status: status,
            risk: r.failure_probability
              ? Math.round(r.failure_probability * 100)
              : 0,
            prediction: r.rul_prediction
              ? `${Math.max(1, Math.round((r.rul_prediction < 1 ? r.rul_prediction * 4000 : r.rul_prediction) / 24))} Days`
              : "--",
            details: r.status,
            sensors: r.sensor_data,
          };
        });

        setRobotsData(formattedData);
      } catch (err) {
        console.error("API Fetch Error:", err);
      } finally {
        setIsDataLoading(false);
      }
    };

    fetchData();
    // Polling 5s for "Live" feel
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Router>
      <Layout>
        {isDataLoading && <LoadingOverlay message="Syncing Telemetry..." />}
        <Routes>
          <Route path="/" element={<Dashboard robots={robotsData} />} />
          <Route path="/assets" element={<AssetMonitor robots={robotsData} />} />
          <Route path="/assets/:id" element={
            <Suspense fallback={<LoadingOverlay message="Loading Asset Context..." />}>
              <AssetView robots={robotsData} />
            </Suspense>
          } />
          <Route path="/audit" element={
            <Suspense fallback={<LoadingOverlay message="Auditing Models..." />}>
              <ModelAudit />
            </Suspense>
          } />
          {/* Fallback */}
          <Route path="*" element={<Dashboard robots={robotsData} />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
