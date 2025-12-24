import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Activity } from 'lucide-react';

/**
 * BearingFaultTrendsChart Component
 * Multi-line chart showing all bearing fault frequencies over time
 */
export function BearingFaultTrendsChart({ data }) {
    if (!data || data.length === 0) {
        return (
            <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                <h3 className="text-sm font-semibold text-slate-900 uppercase tracking-wider mb-2">
                    Bearing Fault Trends
                </h3>
                <p className="text-sm text-slate-400">No data available</p>
            </div>
        );
    }

    // Prepare chart data
    const chartData = data.map((d, idx) => ({
        index: idx,
        timestamp: new Date(d.timestamp).toLocaleTimeString(),
        BPFO: (d.bpfo_amp || 0) * 1000, // Convert to mg for better scale
        BPFI: (d.bpfi_amp || 0) * 1000,
        BSF: (d.bsf_amp || 0) * 1000,
        FTF: (d.ftf_amp || 0) * 1000
    }));

    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            const d = payload[0].payload;
            return (
                <div className="bg-white border border-slate-200 shadow-lg rounded-lg p-3 text-sm">
                    <p className="text-slate-500 font-mono text-xs mb-2">{d.timestamp}</p>
                    <div className="space-y-1">
                        {payload.map((entry, idx) => (
                            <div key={idx} className="flex items-center justify-between gap-4">
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-0.5" style={{ backgroundColor: entry.color }} />
                                    <span className="text-slate-700 font-medium">{entry.name}</span>
                                </div>
                                <span className="text-slate-900 font-mono text-xs">
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
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
            <div className="flex items-center justify-between mb-4">
                <div>
                    <h3 className="text-sm font-semibold text-slate-900 uppercase tracking-wider">
                        Bearing Fault Frequency Trends
                    </h3>
                    <p className="text-xs text-slate-500 mt-1">
                        Historical amplitude of bearing defect indicators
                    </p>
                </div>
                <Activity className="w-5 h-5 text-slate-400" />
            </div>

            <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
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
                        stroke="#94a3b8"
                        tick={{ fill: '#64748b', fontSize: 11 }}
                        tickLine={false}
                        axisLine={false}
                        label={{ value: 'Amplitude (mg)', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#64748b' } }}
                    />

                    <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#cbd5e1', strokeDasharray: '4 4' }} />

                    <Legend
                        wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }}
                        iconType="line"
                    />

                    {/* BPFO - Ball Pass Outer Race (Red) */}
                    <Line
                        type="monotone"
                        dataKey="BPFO"
                        stroke="#ef4444"
                        strokeWidth={2}
                        dot={false}
                        name="BPFO (Outer)"
                        activeDot={{ r: 4, fill: '#ef4444', stroke: '#fff', strokeWidth: 2 }}
                    />

                    {/* BPFI - Ball Pass Inner Race (Orange) */}
                    <Line
                        type="monotone"
                        dataKey="BPFI"
                        stroke="#f97316"
                        strokeWidth={2}
                        dot={false}
                        name="BPFI (Inner)"
                        activeDot={{ r: 4, fill: '#f97316', stroke: '#fff', strokeWidth: 2 }}
                    />

                    {/* BSF - Ball Spin Frequency (Purple) */}
                    <Line
                        type="monotone"
                        dataKey="BSF"
                        stroke="#8b5cf6"
                        strokeWidth={2}
                        dot={false}
                        name="BSF (Ball)"
                        activeDot={{ r: 4, fill: '#8b5cf6', stroke: '#fff', strokeWidth: 2 }}
                    />

                    {/* FTF - Fundamental Train (Blue) */}
                    <Line
                        type="monotone"
                        dataKey="FTF"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={false}
                        name="FTF (Cage)"
                        activeDot={{ r: 4, fill: '#3b82f6', stroke: '#fff', strokeWidth: 2 }}
                    />
                </LineChart>
            </ResponsiveContainer>

            <div className="mt-3 p-3 bg-slate-50 rounded-lg border border-slate-200">
                <p className="text-xs text-slate-600">
                    <span className="font-semibold">Pattern Recognition:</span> Rising trends indicate progressive bearing degradation.
                    Sudden spikes may indicate impact events or intermittent faults requiring investigation.
                </p>
            </div>
        </div>
    );
}

export default BearingFaultTrendsChart;
