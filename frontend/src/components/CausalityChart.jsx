import React from 'react';
import {
    ComposedChart,
    Line,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer
} from 'recharts';

/**
 * CausalityChart Component (Light Mode)
 * 
 * Visualizes the relationship between Physical Faults (Blue) and AI Predictions (Red).
 * Adapted for white background.
 */
function CausalityChart({ data }) {
    // Data is already filtered by parent component
    const chartData = [...data].reverse().map((d, i) => ({
        ...d,
        index: i,
        time: new Date(d.timestamp).toLocaleTimeString(),
        ai_prob: d.failure_probability * 100,
        fault_energy: d.bpfi_amp * 1000
    }));

    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            const d = payload[0].payload;
            return (
                <div className="bg-white border border-slate-200 shadow-lg rounded-lg p-3 text-sm z-50">
                    <p className="text-slate-500 font-mono text-xs mb-1">{d.time}</p>
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-red-500" />
                        <span className="text-slate-700 font-medium">AI Confidence: {d.ai_prob.toFixed(1)}%</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-indigo-500" />
                        <span className="text-slate-700 font-medium">Fault Energy: {d.fault_energy.toFixed(2)}</span>
                    </div>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h3 className="text-lg font-semibold text-slate-900">
                        Causality Analysis
                    </h3>
                    <p className="text-sm text-slate-500">
                        Correlating physical vibration energy with AI failure probability
                    </p>
                </div>
                <div className="flex gap-4 text-xs font-medium">
                    <div className="flex items-center gap-1 text-red-600">
                        <div className="w-2 h-2 rounded-full bg-red-600" />
                        AI Probability (Effect)
                    </div>
                    <div className="flex items-center gap-1 text-indigo-600">
                        <div className="w-2 h-2 rounded-full bg-indigo-600" />
                        Fault Energy (Cause)
                    </div>
                </div>
            </div>

            <ResponsiveContainer width="100%" height={350}>
                <ComposedChart data={chartData}>
                    <defs>
                        <linearGradient id="faultGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#6366f1" stopOpacity={0.1} />
                            <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                        </linearGradient>
                    </defs>

                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />

                    <XAxis
                        dataKey="time"
                        stroke="#94a3b8"
                        tick={{ fill: '#64748b', fontSize: 10 }}
                        tickLine={false}
                        axisLine={false}
                        interval="preserveStartEnd"
                        dy={10}
                    />

                    <YAxis
                        yAxisId="left"
                        domain={[0, 100]}
                        stroke="#ef4444"
                        tick={{ fill: '#ef4444', fontSize: 11 }}
                        tickLine={false}
                        axisLine={false}
                        width={40}
                    />

                    <YAxis
                        yAxisId="right"
                        orientation="right"
                        stroke="#6366f1"
                        tick={{ fill: '#6366f1', fontSize: 11 }}
                        tickLine={false}
                        axisLine={false}
                        width={40}
                    />

                    <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#cbd5e1', strokeDasharray: '4 4' }} />

                    <Area
                        yAxisId="right"
                        type="monotone"
                        dataKey="fault_energy"
                        stroke="#6366f1"
                        fill="url(#faultGradient)"
                        strokeWidth={2}
                    />

                    <Line
                        yAxisId="left"
                        type="monotone"
                        dataKey="ai_prob"
                        stroke="#ef4444"
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4, fill: '#ef4444', stroke: '#fff', strokeWidth: 2 }}
                    />
                </ComposedChart>
            </ResponsiveContainer>
        </div>
    );
}

export default CausalityChart;
