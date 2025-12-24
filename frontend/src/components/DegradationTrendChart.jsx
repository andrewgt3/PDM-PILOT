import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Area, ComposedChart } from 'recharts';
import { TrendingUp } from 'lucide-react';

/**
 * DegradationTrendChart Component
 * Shows degradation score over time with forecast projection
 */
export function DegradationTrendChart({ data }) {
    console.log('[DegradationTrendChart] Received data:', data?.length, 'items');

    if (!data || data.length === 0) {
        return (
            <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                <h3 className="text-sm font-semibold text-slate-900 uppercase tracking-wider mb-2">
                    Degradation Trend
                </h3>
                <p className="text-sm text-slate-400">No data available</p>
            </div>
        );
    }

    // Prepare chart data
    const chartData = data.map((d, idx) => ({
        index: idx,
        timestamp: new Date(d.timestamp).toLocaleTimeString(),
        degradation: (d.degradation_score_smoothed || d.degradation_score || 0) * 100,
        degradation_raw: (d.degradation_score || 0) * 100
    }));

    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            const d = payload[0].payload;
            return (
                <div className="bg-white border border-slate-200 shadow-lg rounded-lg p-3 text-sm">
                    <p className="text-slate-500 font-mono text-xs mb-2">{d.timestamp}</p>
                    <div className="space-y-1">
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-0.5 bg-indigo-600" />
                            <span className="text-slate-700 font-medium">
                                Smoothed: {d.degradation?.toFixed(1)}%
                            </span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-0.5 bg-slate-300" />
                            <span className="text-slate-500 text-xs">
                                Raw: {d.degradation_raw?.toFixed(1)}%
                            </span>
                        </div>
                    </div>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
            <div className="flex items-center justify-between mb-4">
                <div>
                    <h3 className="text-sm font-semibold text-slate-900 uppercase tracking-wider">
                        Degradation Trend
                    </h3>
                    <p className="text-xs text-slate-500 mt-1">
                        Machine wear progression over time (smoothed)
                    </p>
                </div>
                <TrendingUp className="w-5 h-5 text-slate-400" />
            </div>

            <ResponsiveContainer width="100%" height={280}>
                <ComposedChart data={chartData}>
                    <defs>
                        <linearGradient id="degradationGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#6366f1" stopOpacity={0.2} />
                            <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                        </linearGradient>
                    </defs>

                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />

                    <XAxis
                        dataKey="timestamp"
                        stroke="#94a3b8"
                        tick={{ fill: '#64748b', fontSize: 11 }}
                        tickLine={false}
                        axisLine={false}
                        interval="preserveStartEnd"
                    />

                    <YAxis
                        domain={[0, 100]}
                        stroke="#94a3b8"
                        tick={{ fill: '#64748b', fontSize: 11 }}
                        tickLine={false}
                        axisLine={false}
                        label={{ value: 'Degradation (%)', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#64748b' } }}
                    />

                    <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#cbd5e1', strokeDasharray: '4 4' }} />

                    {/* Warning threshold */}
                    <ReferenceLine y={50} stroke="#f59e0b" strokeDasharray="5 5" strokeWidth={1.5}>
                        <text x="95%" y={47} fill="#f59e0b" fontSize={10} textAnchor="end">Warning</text>
                    </ReferenceLine>

                    {/* Critical threshold */}
                    <ReferenceLine y={80} stroke="#ef4444" strokeDasharray="5 5" strokeWidth={1.5}>
                        <text x="95%" y={77} fill="#ef4444" fontSize={10} textAnchor="end">Critical</text>
                    </ReferenceLine>

                    {/* Area under smoothed line */}
                    <Area
                        type="monotone"
                        dataKey="degradation"
                        fill="url(#degradationGradient)"
                        stroke="none"
                    />

                    {/* Raw data (faint) */}
                    <Line
                        type="monotone"
                        dataKey="degradation_raw"
                        stroke="#cbd5e1"
                        strokeWidth={1}
                        dot={false}
                        name="Raw"
                    />

                    {/* Smoothed line (prominent) */}
                    <Line
                        type="monotone"
                        dataKey="degradation"
                        stroke="#6366f1"
                        strokeWidth={2.5}
                        dot={false}
                        name="Smoothed"
                        activeDot={{ r: 5, fill: '#6366f1', stroke: '#fff', strokeWidth: 2 }}
                    />
                </ComposedChart>
            </ResponsiveContainer>

            <div className="mt-3 p-3 bg-slate-50 rounded-lg border border-slate-200">
                <p className="text-xs text-slate-600">
                    <span className="font-semibold">Trend Analysis:</span> Smoothed line reduces noise for clearer progression tracking.
                    Values above 50% warrant increased monitoring; above 80% require immediate action.
                </p>
            </div>
        </div>
    );
}

export default DegradationTrendChart;
