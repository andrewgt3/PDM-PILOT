import React, { useMemo } from 'react';

/**
 * FleetTopology Component - Automotive Plant Edition
 * Displays machine network as a visual topology diagram organized by Shop
 * 
 * Layout Strategy:
 * - Body Shop & Stamping: Tree topology (manufacturing flow)
 * - Paint Shop & Final Assembly: Star topology (centralized operations)
 */
export function FleetTopology({ machines, onSelectMachine }) {
    // Group machines by shop
    const groupedByShop = useMemo(() => {
        const shops = {};
        machines.forEach(m => {
            const shop = m.shop || "Unassigned";
            if (!shops[shop]) {
                shops[shop] = [];
            }
            shops[shop].push(m);
        });
        return shops;
    }, [machines]);

    const getStatusColor = (machine) => {
        const prob = (machine?.failure_probability || 0) * 100;
        if (prob > 80) return { bg: '#ef4444', border: '#dc2626', text: 'white', pulse: true }; // red
        if (prob > 50) return { bg: '#fbbf24', border: '#f59e0b', text: '#78350f', pulse: false }; // amber  
        return { bg: '#34d399', border: '#10b981', text: '#065f46', pulse: false }; // emerald
    };

    const MachineNode = ({ machine, x, y, isHub = false }) => {
        const styles = getStatusColor(machine);
        const size = isHub ? 50 : 42;
        const prob = ((machine?.failure_probability || 0) * 100).toFixed(0);
        const displayId = machine?.machine_id || '?';

        const handleClick = () => {
            console.log('[FleetTopology] Node clicked:', machine?.machine_id);
            if (machine && onSelectMachine) {
                console.log('[FleetTopology] Calling onSelectMachine');
                onSelectMachine(machine.machine_id);
            } else {
                console.log('[FleetTopology] Click ignored - machine:', machine, 'onSelectMachine:', !!onSelectMachine);
            }
        };

        return (
            <g
                onClick={handleClick}
                style={{ cursor: 'pointer', pointerEvents: 'all' }}
            >
                {/* Pulsing circle for critical nodes */}
                {styles.pulse && (
                    <circle
                        cx={x}
                        cy={y}
                        r={size / 2 + 4}
                        fill="none"
                        stroke={styles.bg}
                        strokeWidth="2"
                        opacity="0.4"
                    >
                        <animate
                            attributeName="r"
                            from={size / 2 + 2}
                            to={size / 2 + 8}
                            dur="1.5s"
                            repeatCount="indefinite"
                        />
                        <animate
                            attributeName="opacity"
                            from="0.4"
                            to="0"
                            dur="1.5s"
                            repeatCount="indefinite"
                        />
                    </circle>
                )}

                {/* Hover highlight circle (invisible until hover) */}
                <circle
                    cx={x}
                    cy={y}
                    r={size / 2 + 3}
                    fill="none"
                    stroke="#3b82f6"
                    strokeWidth="2"
                    opacity="0"
                    className="transition-opacity duration-200"
                    style={{ pointerEvents: 'none' }}
                >
                    <set attributeName="opacity" to="0.6" begin="mouseover" end="mouseout" />
                </circle>

                {/* Main node circle */}
                <circle
                    cx={x}
                    cy={y}
                    r={size / 2}
                    fill={styles.bg}
                    stroke={styles.border}
                    strokeWidth="3"
                    style={{ transition: 'all 0.2s ease' }}
                />

                {/* Machine ID text */}
                <text
                    x={x}
                    y={y - 1}
                    textAnchor="middle"
                    fill={styles.text}
                    fontSize="9"
                    fontWeight="bold"
                    fontFamily="monospace"
                    style={{ pointerEvents: 'none' }}
                >
                    {displayId}
                </text>

                {/* Risk percentage */}
                <text
                    x={x}
                    y={y + 9}
                    textAnchor="middle"
                    fill={styles.text}
                    fontSize="8"
                    fontFamily="monospace"
                    opacity="0.8"
                    style={{ pointerEvents: 'none' }}
                >
                    {prob}%
                </text>
            </g>
        );
    };

    const Connection = ({ x1, y1, x2, y2, animated = false }) => (
        <line
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            stroke="#cbd5e1"
            strokeWidth="2"
            strokeDasharray={animated ? "4 4" : "none"}
        >
            {animated && (
                <animate
                    attributeName="stroke-dashoffset"
                    from="8"
                    to="0"
                    dur="0.6s"
                    repeatCount="indefinite"
                />
            )}
        </line>
    );

    const svgWidth = 900;
    const svgHeight = 320;

    // Define shop sections with their positions and layout types
    const shopConfigs = [
        { name: 'Body Shop', x: 120, layout: 'tree', label: 'BODY SHOP • WELDING' },
        { name: 'Stamping', x: 320, layout: 'tree', label: 'STAMPING • PRESS' },
        { name: 'Paint Shop', x: 550, layout: 'star', label: 'PAINT SHOP' },
        { name: 'Final Assembly', x: 750, layout: 'star', label: 'FINAL ASSEMBLY' },
    ];

    const renderShopSection = (shopConfig, machines) => {
        if (!machines || machines.length === 0) return null;

        const { x, layout, label } = shopConfig;
        const startY = 55;
        const positions = [];
        const connections = [];

        if (layout === 'tree') {
            // Tree layout - first machine at top, others below
            const spacing = 50;
            const levelY = [70, 140, 210];

            if (machines[0]) positions.push({ machine: machines[0], x, y: levelY[0], isHub: true });
            if (machines[1]) {
                positions.push({ machine: machines[1], x: x - spacing / 2, y: levelY[1] });
                connections.push({ from: 0, to: positions.length - 1 });
            }
            if (machines[2]) {
                positions.push({ machine: machines[2], x: x + spacing / 2, y: levelY[1] });
                connections.push({ from: 0, to: positions.length - 1 });
            }
            // Additional machines in level 3
            machines.slice(3).forEach((m, i) => {
                positions.push({ machine: m, x: x - spacing / 2 + i * spacing, y: levelY[2] });
                if (positions.length > 3) {
                    connections.push({ from: 1, to: positions.length - 1 });
                }
            });
        } else {
            // Star layout - first machine as hub, others around
            const hubY = 150;
            const radius = 70;

            if (machines[0]) positions.push({ machine: machines[0], x, y: hubY, isHub: true });

            const satellites = machines.slice(1);
            const angleStep = (2 * Math.PI) / Math.max(satellites.length, 1);

            satellites.forEach((m, i) => {
                const angle = angleStep * i - Math.PI / 2;
                positions.push({
                    machine: m,
                    x: x + radius * Math.cos(angle),
                    y: hubY + radius * Math.sin(angle)
                });
                connections.push({ from: 0, to: positions.length - 1 });
            });
        }

        return (
            <g key={shopConfig.name}>
                {/* Section label */}
                <text
                    x={x}
                    y={28}
                    textAnchor="middle"
                    fill="#64748b"
                    fontSize="10"
                    fontWeight="600"
                    letterSpacing="1"
                >
                    {label}
                </text>

                {/* Connections */}
                {connections.map((conn, i) => {
                    const from = positions[conn.from];
                    const to = positions[conn.to];
                    if (!from || !to) return null;
                    const isCritical = (to.machine?.failure_probability || 0) > 0.8;
                    return (
                        <Connection
                            key={`${shopConfig.name}-conn-${i}`}
                            x1={from.x}
                            y1={from.y}
                            x2={to.x}
                            y2={to.y}
                            animated={isCritical}
                        />
                    );
                })}

                {/* Nodes */}
                {positions.map((pos, i) => (
                    <MachineNode
                        key={`${shopConfig.name}-node-${i}`}
                        machine={pos.machine}
                        x={pos.x}
                        y={pos.y}
                        isHub={pos.isHub}
                    />
                ))}
            </g>
        );
    };

    return (
        <div className="w-full bg-white rounded-lg border border-slate-200 overflow-hidden">
            <svg
                width="100%"
                height="320"
                viewBox={`0 0 ${svgWidth} ${svgHeight}`}
                className="bg-slate-50/30"
            >
                {/* Grid background */}
                <defs>
                    <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                        <path
                            d="M 20 0 L 0 0 0 20"
                            fill="none"
                            stroke="#e2e8f0"
                            strokeWidth="0.5"
                        />
                    </pattern>
                </defs>
                <rect width={svgWidth} height={svgHeight} fill="url(#grid)" />

                {/* Dividers between shop sections */}
                {[1, 2, 3].map(i => (
                    <line
                        key={`divider-${i}`}
                        x1={svgWidth * i / 4}
                        y1="40"
                        x2={svgWidth * i / 4}
                        y2={svgHeight - 30}
                        stroke="#e2e8f0"
                        strokeWidth="1"
                        strokeDasharray="4 4"
                    />
                ))}

                {/* Render each shop section */}
                {shopConfigs.map(config =>
                    renderShopSection(config, groupedByShop[config.name] || [])
                )}

                {/* Handle unassigned machines */}
                {groupedByShop['Unassigned Shop'] && groupedByShop['Unassigned Shop'].length > 0 && (
                    <g>
                        <text x={svgWidth / 2} y={svgHeight - 15} textAnchor="middle" fill="#94a3b8" fontSize="10">
                            + {groupedByShop['Unassigned Shop'].length} unassigned
                        </text>
                    </g>
                )}
            </svg>
        </div>
    );
}

export default FleetTopology;
