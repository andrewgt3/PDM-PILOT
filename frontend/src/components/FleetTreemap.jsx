import React, { useMemo } from 'react';
import { ResponsiveContainer, Treemap, Tooltip } from 'recharts';

const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
        const data = payload[0].payload;
        return (
            <div className="bg-white p-3 border border-slate-200 shadow-lg rounded-lg text-xs">
                <p className="font-bold text-slate-800">{data.displayName || data.name}</p>
                <p className="text-slate-500 font-mono text-[10px]">{data.name}</p>
                {data.equipmentType && (
                    <p className="text-slate-400 text-[10px] mt-1">{data.equipmentType}</p>
                )}
                <div className="mt-2 flex items-center justify-between gap-4 pt-2 border-t border-slate-100">
                    <span className="text-slate-500">Risk:</span>
                    <span className={`font-bold font-mono ${data.risk > 80 ? 'text-red-600' : data.risk > 50 ? 'text-amber-600' : 'text-emerald-600'}`}>
                        {data.risk?.toFixed(1) || 0}%
                    </span>
                </div>
                {data.shop && (
                    <p className="text-slate-400 text-[10px] mt-1">{data.shop} • {data.line}</p>
                )}
            </div>
        );
    }
    return null;
};

const CustomizedContent = (props) => {
    const { root, depth, x, y, width, height, index, name, risk, displayName } = props;

    // Depth 1 = Shop (Body Shop, Paint Shop, etc.)
    // Depth 2 = Line (Underbody Weld Cell, Press Line 1, etc.)  
    // Depth 3 = Equipment (WB-001, HP-200, etc.)

    // Shop level - subtle background with label
    if (depth === 1) {
        return (
            <g>
                <rect x={x} y={y} width={width} height={height} style={{ fill: '#f1f5f9', stroke: '#fff', strokeWidth: 3 }} />
                {width > 60 && height > 25 && (
                    <text x={x + 6} y={y + 16} style={{ fill: '#64748b', fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        {name}
                    </text>
                )}
            </g>
        );
    }

    // Line level - lighter background
    if (depth === 2) {
        return (
            <g>
                <rect x={x} y={y} width={width} height={height} style={{ fill: '#f8fafc', stroke: '#e2e8f0', strokeWidth: 1 }} />
                {width > 50 && height > 20 && (
                    <text x={x + 4} y={y + 12} style={{ fill: '#94a3b8', fontSize: 9, fontWeight: 600 }}>
                        {name}
                    </text>
                )}
            </g>
        );
    }

    const currentRisk = risk || 0;

    // Equipment level - colored tiles based on risk
    if (depth === 3) {
        let fillColor = "#d1fae5"; // emerald-100 (healthy)
        let textColor = "#047857"; // emerald-700

        if (currentRisk > 80) {
            fillColor = "#ef4444"; // red-500 (critical)
            textColor = "#ffffff";
        } else if (currentRisk > 50) {
            fillColor = "#fbbf24"; // amber-400 (warning)
            textColor = "#78350f"; // amber-900
        }

        return (
            <g>
                <rect
                    x={x}
                    y={y}
                    width={width}
                    height={height}
                    style={{ fill: fillColor, stroke: '#fff', strokeWidth: 2 }}
                    className="transition-all duration-300 cursor-pointer"
                />
                {width > 35 && height > 35 && (
                    <>
                        <text
                            x={x + width / 2}
                            y={y + height / 2 - 2}
                            textAnchor="middle"
                            style={{ fill: textColor, fontSize: 11, fontWeight: 700, fontFamily: 'monospace' }}
                        >
                            {name}
                        </text>
                        <text
                            x={x + width / 2}
                            y={y + height / 2 + 10}
                            textAnchor="middle"
                            style={{ fill: textColor, fontSize: 9, fontFamily: 'monospace', opacity: 0.8 }}
                        >
                            {currentRisk.toFixed(0)}%
                        </text>
                    </>
                )}
            </g>
        );
    }

    return null;
};

export function FleetTreemap({ machines, onSelectMachine }) {
    const treeData = useMemo(() => {
        // Group by Shop -> Line -> Machine (realistic automotive hierarchy)
        const shops = {};

        machines.forEach(m => {
            const shop = m.shop || "Unassigned Shop";
            const line = m.line_name || m.line || "Unassigned Line";

            if (!shops[shop]) {
                shops[shop] = { name: shop, children: {} };
            }

            if (!shops[shop].children[line]) {
                shops[shop].children[line] = { name: line, children: [] };
            }

            shops[shop].children[line].children.push({
                name: m.machine_id,
                displayName: m.machine_name || m.machine_id,
                size: 100,
                risk: (m.failure_probability || 0) * 100,
                line: line,
                shop: shop,
                equipmentType: m.equipment_type || "Equipment"
            });
        });

        // Convert nested objects to arrays
        const shopArray = Object.values(shops).map(shop => ({
            name: shop.name,
            children: Object.values(shop.children)
        }));

        return [
            {
                name: 'Plant',
                children: shopArray
            }
        ];
    }, [machines]);

    return (
        <div className="h-[300px] w-full bg-slate-50/50 rounded-lg border border-slate-200 overflow-hidden">
            <ResponsiveContainer width="100%" height="100%">
                <Treemap
                    data={treeData}
                    dataKey="size"
                    aspectRatio={4 / 3}
                    stroke="#fff"
                    fill="#fff"
                    content={<CustomizedContent />}
                    onClick={(node) => {
                        if (node && node.name && node.name.startsWith('M_')) {
                            onSelectMachine(node.name);
                        }
                    }}
                >
                    <Tooltip content={<CustomTooltip />} />
                </Treemap>
            </ResponsiveContainer>
        </div>
    );
}

export default FleetTreemap;
