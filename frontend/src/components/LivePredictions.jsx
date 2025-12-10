import { FixedSizeList as List } from 'react-window';

const LivePredictions = ({ robots }) => {
    const Row = ({ index, style }) => {
        const r = robots[index];
        return (
            <div
                style={{
                    ...style,
                    display: "flex",
                    borderBottom: "1px solid rgba(0,0,0,0.05)",
                    alignItems: "center",
                }}
            >
                <div
                    className="text-sm font-bold"
                    style={{ flex: 1, padding: "0.5rem" }}
                >
                    {r.name}
                </div>
                <div style={{ flex: 1, padding: "0.5rem" }}>
                    <span
                        className={`status-badge ${r.status === "critical" ? "bg-critical" : r.status === "warning" ? "bg-warning" : "bg-success"}`}
                        style={{
                            padding: "0.25rem 0.5rem",
                            borderRadius: "4px",
                            color: "white",
                            fontSize: "0.75rem",
                            fontWeight: "bold",
                        }}
                    >
                        {r.status.toUpperCase()}
                    </span>
                </div>
                <div
                    className="text-sm font-mono"
                    style={{ flex: 1, padding: "0.5rem" }}
                >
                    {r.risk}%
                </div>
                <div
                    className="text-sm text-gray"
                    style={{ flex: 1, padding: "0.5rem" }}
                >
                    {r.prediction}
                </div>
                <div
                    className="text-sm text-gray"
                    style={{ flex: 1, padding: "0.5rem" }}
                >
                    {r.details}
                </div>
            </div>
        );
    };

    return (
        <div className="glass-panel" style={{ padding: "1rem", marginTop: "2rem" }}>
            <div
                className="flex-center"
                style={{ justifyContent: "space-between", marginBottom: "1rem" }}
            >
                <h3
                    className="section-title"
                    style={{ marginBottom: 0, fontSize: "1rem" }}
                >
                    LIVE PREDICTION LOG (24H)
                </h3>
                <button className="text-xs font-bold text-blue button-ghost">
                    EXPORT LOGS
                </button>
            </div>

            <div
                style={{
                    display: "flex",
                    borderBottom: "1px solid var(--border-color)",
                    marginBottom: "0.5rem",
                }}
            >
                <div
                    className="text-xs text-gray"
                    style={{ flex: 1, padding: "0.5rem" }}
                >
                    ROBOT ID
                </div>
                <div
                    className="text-xs text-gray"
                    style={{ flex: 1, padding: "0.5rem" }}
                >
                    STATUS
                </div>
                <div
                    className="text-xs text-gray"
                    style={{ flex: 1, padding: "0.5rem" }}
                >
                    FAILURE PROBABILITY
                </div>
                <div
                    className="text-xs text-gray"
                    style={{ flex: 1, padding: "0.5rem" }}
                >
                    PREDICTED WINDOW
                </div>
                <div
                    className="text-xs text-gray"
                    style={{ flex: 1, padding: "0.5rem" }}
                >
                    TOP FACTOR
                </div>
            </div>

            <List height={300} itemCount={robots.length} itemSize={50} width={"100%"}>
                {Row}
            </List>
        </div>
    );
};

export default LivePredictions;
