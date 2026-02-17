import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceDot } from 'recharts';
import { Activity, AlertTriangle } from 'lucide-react';

/**
 * BearingFaultTrendsChart Component
 * Multi-line chart showing all bearing fault frequencies over time
 * with interactive filtering and synchronized cursor.
 */
export function BearingFaultTrendsChart({ data, alerts, syncId }) {
    const [hiddenSeries, setHiddenSeries] = useState(new Set());

    // Toggle series visibility
    const handleLegendClick = (e) => {
        const seriesName = e.dataKey;
        setHiddenSeries(prev => {
            const next = new Set(prev);
            if (next.has(seriesName)) {
                next.delete(seriesName);
            } else {
                next.add(seriesName);
            }
            return next;
        });
    };

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
        timestamp_obj: new Date(d.timestamp).getTime(),
        BPFO: (d.bpfo_amp || 0) * 1000, // Convert to mg
        BPFI: (d.bpfi_amp || 0) * 1000,
        BSF: (d.bsf_amp || 0) * 1000,
        FTF: (d.ftf_amp || 0) * 1000
    }));

    // Alert Markers
    const alertMarkers = (alerts || []).map(alert => {
        // Find closest data point to anchor the marker
        const alertTime = new Date(alert.timestamp).getTime();
        const closestPoint = chartData.reduce((prev, curr) => {
            return (Math.abs(curr.timestamp_obj - alertTime) < Math.abs(prev.timestamp_obj - alertTime) ? curr : prev);
        });

        return {
            x: closestPoint.timestamp,
            y: 0, // Anchor to bottom, or find max value to anchor top? Using X-axis anchor via ReferenceDot
            ...alert
        };
    });

    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            return (
                <div className="bg-white/90 backdrop-blur-md border border-slate-200 shadow-xl rounded-lg p-3 text-sm z-50">
                    <p className="text-slate-500 font-mono text-xs mb-2 border-b border-slate-100 pb-1">{label}</p>
                    <div className="space-y-1.5">
                        {payload.map((entry, idx) => (
                            <div key={idx} className="flex items-center justify-between gap-4">
                                <div className="flex items-center gap-2">
                                    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
                                    <span className="text-slate-700 font-medium text-xs">{entry.name}</span>
                                </div>
                                <span className="font-mono text-xs font-bold text-slate-900">
                                    {entry.value.toFixed(1)} mg
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
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm h-full flex flex-col">
            <div className="flex items-center justify-between mb-4 flex-shrink-0">
                <div>
                    <h3 className="text-sm font-semibold text-slate-900 uppercase tracking-wider">
                        Bearing Fault Frequency Trends
                    </h3>
                    <p className="text-xs text-slate-500 mt-1">
                        Historical amplitude of bearing defect indicators (Interactive)
                    </p>
                </div>
                <Activity className="w-5 h-5 text-slate-400" />
            </div>

            <div className="flex-grow min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} syncId={syncId}>
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
                            stroke="#94a3b8"
                            tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'monospace' }}
                            tickLine={false}
                            axisLine={false}
                            label={{ value: 'Amplitude (mg)', angle: -90, position: 'insideLeft', style: { fontSize: 10, fill: '#64748b' } }}
                        />

                        <Tooltip
                            content={<CustomTooltip />}
                            cursor={{ stroke: '#6366f1', strokeWidth: 1, strokeDasharray: '4 4' }}
                            wrapperStyle={{ outline: 'none' }}
                        />

                        <Legend
                            wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }}
                            iconType="circle"
                            onClick={handleLegendClick}
                            formatter={(value, entry) => (
                                <span className={`${hiddenSeries.has(entry.dataKey) ? 'text-slate-300 line-through' : 'text-slate-600 font-medium'} cursor-pointer hover:text-indigo-600 transition-colors`}>
                                    {value}
                                </span>
                            )}
                        />

                        {/* Event Markers (Alerts) */}
                        {alertMarkers.map((alert, i) => (
                            <ReferenceDot
                                key={i}
                                x={alert.x}
                                y={0} // Place at bottom
                                r={4}
                                fill={alert.severity === 'critical' ? '#ef4444' : '#f59e0b'}
                                stroke="none"
                                ifOverflow="extendDomain"
                            />
                        ))}

                        {/* BPFO - Ball Pass Outer Race (Red) */}
                        <Line
                            type="monotone"
                            dataKey="BPFO"
                            stroke="#ef4444"
                            strokeWidth={3}
                            dot={false}
                            name="BPFO (Outer)"
                            hide={hiddenSeries.has('BPFO')}
                            activeDot={{ r: 5, fill: '#ef4444', stroke: '#fff', strokeWidth: 2 }}
                            animationDuration={1000}
                        />

                        {/* BPFI - Ball Pass Inner Race (Orange) */}
                        <Line
                            type="monotone"
                            dataKey="BPFI"
                            stroke="#f97316"
                            strokeWidth={3}
                            dot={false}
                            name="BPFI (Inner)"
                            hide={hiddenSeries.has('BPFI')}
                            activeDot={{ r: 5, fill: '#f97316', stroke: '#fff', strokeWidth: 2 }}
                            animationDuration={1000}
                        />

                        {/* BSF - Ball Spin Frequency (Purple) */}
                        <Line
                            type="monotone"
                            dataKey="BSF"
                            stroke="#8b5cf6"
                            strokeWidth={3}
                            dot={false}
                            name="BSF (Ball)"
                            hide={hiddenSeries.has('BSF')}
                            activeDot={{ r: 5, fill: '#8b5cf6', stroke: '#fff', strokeWidth: 2 }}
                            animationDuration={1000}
                        />

                        {/* FTF - Fundamental Train (Blue) */}
                        <Line
                            type="monotone"
                            dataKey="FTF"
                            stroke="#3b82f6"
                            strokeWidth={3}
                            dot={false}
                            name="FTF (Cage)"
                            hide={hiddenSeries.has('FTF')}
                            activeDot={{ r: 5, fill: '#3b82f6', stroke: '#fff', strokeWidth: 2 }}
                            animationDuration={1000}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            <div className="mt-3 flex items-center gap-2 text-xs text-slate-500">
                <AlertTriangle size={12} className="text-amber-500" />
                <span>Tip: Click legend items to isolate specific frequencies.</span>
            </div>
        </div>
    );
}

export default BearingFaultTrendsChart;
