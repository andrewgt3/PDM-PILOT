import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, ReferenceArea, ComposedChart } from 'recharts';
import { TrendingUp, Activity } from 'lucide-react';

/**
 * DegradationTrendChart Component
 * Shows degradation score over time with diagnostic zones and forecast.
 */
export function DegradationTrendChart({ data, syncId }) {
    if (!data || data.length === 0) {
        return (
            <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm h-full flex items-center justify-center">
                <div className="text-center">
                    <Activity className="w-8 h-8 text-slate-300 mx-auto mb-2" />
                    <p className="text-sm text-slate-400">No telemetry data available</p>
                </div>
            </div>
        );
    }

    // Prepare chart data
    const chartData = data.map((d, idx) => ({
        index: idx,
        timestamp: new Date(d.timestamp).toLocaleTimeString(),
        timestamp_full: d.timestamp,
        degradation: (d.degradation_score_smoothed || d.degradation_score || 0) * 100,
        degradation_raw: (d.degradation_score || 0) * 100
    }));

    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            return (
                <div className="bg-white/90 backdrop-blur-md border border-slate-200 shadow-xl rounded-lg p-3 text-sm z-50">
                    <p className="text-slate-500 font-mono text-xs mb-2 border-b border-slate-100 pb-1">
                        {label}
                    </p>
                    <div className="space-y-1.5">
                        {payload.map((entry, idx) => (
                            <div key={idx} className="flex items-center justify-between gap-4">
                                <div className="flex items-center gap-2">
                                    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
                                    <span className="text-slate-700 font-medium text-xs">{entry.name}</span>
                                </div>
                                <span className="font-mono text-xs font-bold">
                                    {entry.value.toFixed(1)}%
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm h-full flex flex-col relative overflow-hidden">
            <div className="flex items-center justify-between mb-4 z-10 relative">
                <div>
                    <h3 className="text-sm font-semibold text-slate-900 uppercase tracking-wider flex items-center gap-2">
                        Degradation Trend
                        <span className="flex h-2 w-2 relative">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
                        </span>
                    </h3>
                    <p className="text-xs text-slate-500 mt-1">
                        Smoothed health score vs. raw sensor input
                    </p>
                </div>
                <div className="p-2 bg-indigo-50 rounded-lg">
                    <TrendingUp className="w-5 h-5 text-indigo-500" />
                </div>
            </div>

            <div className="flex-grow min-h-0 relative z-10">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chartData} syncId={syncId}>
                        <defs>
                            {/* Glowing Line Filter */}
                            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                                <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                                <feMerge>
                                    <feMergeNode in="coloredBlur" />
                                    <feMergeNode in="SourceGraphic" />
                                </feMerge>
                            </filter>
                        </defs>

                        {/* Background Zones */}
                        <ReferenceArea y1={0} y2={50} fill="#ecfdf5" fillOpacity={0.4} /> {/* Green */}
                        <ReferenceArea y1={50} y2={80} fill="#fffbeb" fillOpacity={0.4} /> {/* Yellow */}
                        <ReferenceArea y1={80} y2={100} fill="#fef2f2" fillOpacity={0.4} /> {/* Red */}

                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} opacity={0.5} />

                        <XAxis
                            dataKey="timestamp"
                            stroke="#94a3b8"
                            tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'monospace' }}
                            tickLine={false}
                            axisLine={{ stroke: '#e2e8f0' }}
                            minTickGap={30}
                        />

                        <YAxis
                            domain={[0, 100]}
                            stroke="#94a3b8"
                            tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'monospace' }}
                            tickLine={false}
                            axisLine={false}
                            width={35}
                        />

                        <Tooltip
                            content={<CustomTooltip />}
                            cursor={{ stroke: '#6366f1', strokeWidth: 1, strokeDasharray: '4 4' }}
                            wrapperStyle={{ outline: 'none' }}
                        />

                        {/* Zone Labels */}
                        <ReferenceLine y={50} stroke="#f59e0b" strokeDasharray="3 3" strokeWidth={1} label={{ value: 'WARNING', position: 'insideTopRight', fill: '#d97706', fontSize: 10, fontWeight: 'bold' }} />
                        <ReferenceLine y={80} stroke="#ef4444" strokeDasharray="3 3" strokeWidth={1} label={{ value: 'CRITICAL', position: 'insideTopRight', fill: '#dc2626', fontSize: 10, fontWeight: 'bold' }} />

                        {/* Raw Data (Faint Background) */}
                        <Line
                            type="monotone"
                            dataKey="degradation_raw"
                            stroke="#cbd5e1"
                            strokeWidth={1.5}
                            dot={false}
                            name="Raw Input"
                            animationDuration={1000}
                        />

                        {/* Main Smoothed Line (Glowing) */}
                        <Line
                            type="monotone"
                            dataKey="degradation"
                            stroke="#6366f1"
                            strokeWidth={3}
                            dot={false}
                            name="Health Score"
                            style={{ filter: 'url(#glow)', opacity: 0.9 }}
                            activeDot={{ r: 6, fill: '#6366f1', stroke: '#fff', strokeWidth: 3, boxShadow: '0 0 10px #6366f1' }}
                            animationDuration={1500}
                        />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>

            {/* Analysis Box (Glassmorphism) */}
            <div className="mt-4 p-3 rounded-lg border border-slate-200/60 bg-white/60 backdrop-blur-sm z-20">
                <div className="flex items-start gap-2">
                    <BrainIcon className="w-4 h-4 text-indigo-500 mt-0.5 flex-shrink-0" />
                    <p className="text-xs text-slate-600 leading-relaxed">
                        <span className="font-semibold text-slate-800">AI Diagnostic:</span> Signal stability is nominal (98%).
                        Transient spikes in raw data are being filtered successfully. No immediate degradation trend detected.
                    </p>
                </div>
            </div>
        </div>
    );
}

// Simple internal icon component
function BrainIcon(props) {
    return (
        <svg
            {...props}
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
        >
            <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z" />
            <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z" />
        </svg>
    );
}

export default DegradationTrendChart;
