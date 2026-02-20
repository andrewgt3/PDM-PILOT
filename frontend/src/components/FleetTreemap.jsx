import React, { useMemo } from 'react';
import { ResponsiveContainer, Treemap, Tooltip } from 'recharts';
import { useCurrentUser } from '../hooks/useCurrentUser';

const CRITICALITY_WEIGHT = { high: 3, medium: 2, low: 1 };

const SiteTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
        const data = payload[0].payload;
        const healthScore = data.healthScore ?? (1 - (data.risk ?? 0) / 100);
        return (
            <div className="bg-white p-3 border border-slate-200 shadow-lg rounded-lg text-xs">
                <p className="font-bold text-slate-800">{data.displayName || data.name}</p>
                <p className="text-slate-500 font-mono text-[10px]">Site</p>
                <div className="mt-2 space-y-1 pt-2 border-t border-slate-100">
                    <div className="flex justify-between gap-4">
                        <span className="text-slate-500">Avg health:</span>
                        <span className="font-mono font-semibold">{(healthScore * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between gap-4">
                        <span className="text-slate-500">Machines:</span>
                        <span className="font-mono">{data.machineCount ?? 0}</span>
                    </div>
                </div>
            </div>
        );
    }
    return null;
};

const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
        const data = payload[0].payload;
        const healthScore = data.healthScore ?? (1 - (data.risk ?? 0) / 100);
        const alertTier = data.active_alert_tier ?? data.status ?? '—';
        return (
            <div className="bg-white p-3 border border-slate-200 shadow-lg rounded-lg text-xs">
                <p className="font-bold text-slate-800">{data.displayName || data.name}</p>
                <p className="text-slate-500 font-mono text-[10px]">machine_id: {data.machine_id ?? data.name}</p>
                {data.equipmentType && (
                    <p className="text-slate-400 text-[10px] mt-1">{data.equipmentType}</p>
                )}
                <div className="mt-2 space-y-1 pt-2 border-t border-slate-100">
                    <div className="flex justify-between gap-4">
                        <span className="text-slate-500">Health score:</span>
                        <span className="font-mono font-semibold">{(healthScore * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between gap-4">
                        <span className="text-slate-500">RUL (days):</span>
                        <span className="font-mono">{data.rul_days != null ? Number(data.rul_days).toFixed(1) : '—'}</span>
                    </div>
                    <div className="flex justify-between gap-4">
                        <span className="text-slate-500">Alert tier:</span>
                        <span className={`font-mono font-semibold ${alertTier === 'CRITICAL' ? 'text-red-600' : alertTier === 'WARNING' ? 'text-amber-600' : 'text-emerald-600'}`}>
                            {alertTier}
                        </span>
                    </div>
                    <div className="flex justify-between gap-4">
                        <span className="text-slate-500">Risk:</span>
                        <span className={`font-bold font-mono ${data.risk > 80 ? 'text-red-600' : data.risk > 50 ? 'text-amber-600' : 'text-emerald-600'}`}>
                            {data.risk?.toFixed(1) ?? 0}%
                        </span>
                    </div>
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
    const { depth, x, y, width, height, name, risk, displayName, isSiteLevel } = props;

    // Site-level mode (reliability_engineer): depth 1 = site cell (colored by avg health)
    if (isSiteLevel && depth === 1) {
        const currentRisk = risk || 0;
        let fillColor = "#d1fae5";
        let textColor = "#047857";
        if (currentRisk > 80) {
            fillColor = "#ef4444";
            textColor = "#ffffff";
        } else if (currentRisk > 50) {
            fillColor = "#fbbf24";
            textColor = "#78350f";
        }
        return (
            <g>
                <rect x={x} y={y} width={width} height={height} style={{ fill: fillColor, stroke: '#fff', strokeWidth: 2 }} />
                {width > 50 && height > 25 && (
                    <>
                        <text x={x + width / 2} y={y + height / 2 - 4} textAnchor="middle" style={{ fill: textColor, fontSize: 11, fontWeight: 700 }}>
                            {displayName || name}
                        </text>
                        <text x={x + width / 2} y={y + height / 2 + 8} textAnchor="middle" style={{ fill: textColor, fontSize: 9, opacity: 0.9 }}>
                            {(100 - currentRisk).toFixed(0)}%
                        </text>
                    </>
                )}
            </g>
        );
    }

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

function isAzurePmReplayMode(machines) {
    if (!machines || machines.length === 0) return false;
    return machines.every(m => {
        const id = m.machine_id;
        if (typeof id !== 'string' || !/^\d+$/.test(id)) return false;
        const n = parseInt(id, 10);
        return n >= 1 && n <= 100;
    });
}

function roleLabel(role) {
    if (!role) return '—';
    return String(role).replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function FleetTreemap({ machines, onSelectMachine }) {
    const user = useCurrentUser();
    const role = user?.role ? String(user.role).toLowerCase() : null;
    const isReliabilityEngineer = role === 'reliability_engineer';
    const isPlantManager = role === 'plant_manager';

    const filteredMachines = useMemo(() => {
        if (isPlantManager && user?.siteId) {
            return machines.filter((m) => (m.site_id || m.shop || m.line_name || '') === user.siteId);
        }
        return machines;
    }, [machines, isPlantManager, user?.siteId]);

    const replayMode = useMemo(() => isAzurePmReplayMode(filteredMachines), [filteredMachines]);

    const treeData = useMemo(() => {
        if (isReliabilityEngineer) {
            const bySite = {};
            filteredMachines.forEach((m) => {
                const siteKey = m.site_id || m.shop || m.line_name || 'Default Site';
                if (!bySite[siteKey]) bySite[siteKey] = { machines: [], healthSum: 0 };
                bySite[siteKey].machines.push(m);
                bySite[siteKey].healthSum += 1 - (m.failure_probability || 0);
            });
            const siteArray = Object.entries(bySite).map(([siteName, { machines: siteMachines, healthSum }]) => {
                const count = siteMachines.length;
                const avgHealth = count ? healthSum / count : 1;
                const risk = (1 - avgHealth) * 100;
                const size = Math.max(40, 60 + count * 8);
                return {
                    name: siteName,
                    displayName: siteName,
                    size,
                    risk,
                    healthScore: avgHealth,
                    machineCount: count,
                };
            });
            return [{ name: 'Sites', children: siteArray }];
        }

        const shops = {};
        const baselineHealth = 1.0;
        filteredMachines.forEach((m) => {
            const shop = m.shop || 'Unassigned Shop';
            const line = m.line_name || m.line || 'Unassigned Line';
            const risk = (m.failure_probability || 0) * 100;
            const healthScore = 1 - (m.failure_probability || 0);
            const healthDelta = baselineHealth - healthScore;
            const criticalityWeight = CRITICALITY_WEIGHT[m.criticality] ?? 2;
            const size = Math.max(20, 80 + criticalityWeight * 15 + healthDelta * 40);

            if (!shops[shop]) shops[shop] = { name: shop, children: {} };
            if (!shops[shop].children[line]) shops[shop].children[line] = { name: line, children: [] };
            shops[shop].children[line].children.push({
                name: m.machine_id,
                displayName: m.machine_name || m.machine_id,
                size,
                risk,
                healthScore,
                rul_days: m.rul_days,
                active_alert_tier: m.status,
                status: m.status,
                machine_id: m.machine_id,
                line,
                shop,
                equipmentType: m.equipment_type || 'Equipment',
            });
        });

        const shopArray = Object.values(shops).map((shop) => ({
            name: shop.name,
            children: Object.values(shop.children),
        }));
        return [{ name: 'Plant', children: shopArray }];
    }, [filteredMachines, isReliabilityEngineer]);

    return (
        <div className="h-[300px] w-full bg-slate-50/50 rounded-lg border border-slate-200 overflow-hidden relative">
            {role && (
                <div className="absolute top-2 right-2 z-10 px-2 py-0.5 rounded bg-slate-700/90 text-white text-xs font-medium">
                    {roleLabel(role)}
                </div>
            )}
            {replayMode && (
                <div className="absolute top-0 left-0 right-0 z-10 bg-amber-500/90 text-amber-900 text-center py-1 text-xs font-semibold">
                    REPLAY MODE — Azure PM dataset
                </div>
            )}
            <ResponsiveContainer width="100%" height="100%">
                <Treemap
                    data={treeData}
                    dataKey="size"
                    aspectRatio={4 / 3}
                    stroke="#fff"
                    fill="#fff"
                    content={<CustomizedContent isSiteLevel={isReliabilityEngineer} />}
                    onClick={(node) => {
                        if (!node || !node.name || isReliabilityEngineer) return;
                        const isEquipment = node.depth === 3 || filteredMachines.some((m) => m.machine_id === node.name);
                        if (isEquipment && onSelectMachine) onSelectMachine(node.name);
                    }}
                >
                    <Tooltip content={isReliabilityEngineer ? <SiteTooltip /> : <CustomTooltip />} />
                </Treemap>
            </ResponsiveContainer>
        </div>
    );
}

export default FleetTreemap;
