import React, { useMemo } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    ReferenceLine, Area, AreaChart, ComposedChart
} from 'recharts';
import { Card, Typography, Box } from '@mui/material';
import { Activity } from 'lucide-react';

const RUL_DAYS_FORMULA = (deg) => Math.max(0, (2000 * Math.pow(1 - (deg ?? 0), 2)) / 24);
const SEVEN_DAYS_MS = 7 * 24 * 60 * 60 * 1000;

export function RULTimeSeriesChart({ data, syncId }) {
    const { chartData, alertTransitions } = useMemo(() => {
        if (!data || data.length === 0) return { chartData: [], alertTransitions: [] };
        const cutoff = Date.now() - SEVEN_DAYS_MS;
        const filtered = data.filter((d) => new Date(d.timestamp).getTime() >= cutoff);
        const series = (filtered.length > 0 ? filtered : data.slice(-Math.max(1, data.length))).map((d) => {
            const deg = d.degradation_score ?? d.degradation_score_smoothed ?? 0;
            const rulDays = d.rul_days != null ? d.rul_days : RUL_DAYS_FORMULA(deg);
            const lower = d.rul_lower_80 != null ? d.rul_lower_80 : rulDays * 0.85;
            const upper = d.rul_upper_80 != null ? d.rul_upper_80 : rulDays * 1.15;
            const fp = d.failure_probability ?? d.failure_prediction ?? 0;
            return {
                timestamp: d.timestamp,
                timeLabel: new Date(d.timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }),
                rulDays,
                rulLower: Math.max(0, lower),
                rulUpper: upper,
                healthScore: 1 - fp,
                failureProbability: fp
            };
        });
        const transitions = [];
        for (let i = 1; i < series.length; i++) {
            const prev = series[i - 1].failureProbability;
            const curr = series[i].failureProbability;
            if ((prev <= 0.5 && curr > 0.5) || (prev <= 0.8 && curr > 0.8)) {
                transitions.push(series[i].timestamp);
            }
        }
        return { chartData: series, alertTransitions: transitions };
    }, [data]);

    if (chartData.length === 0) {
        return (
            <Card variant="outlined" sx={{ height: '100%', borderRadius: 2, display: 'flex', alignItems: 'center', justifyContent: 'center', p: 2 }}>
                <Box sx={{ textAlign: 'center' }}>
                    <Activity sx={{ fontSize: 32, color: 'text.disabled', mb: 1 }} />
                    <Typography variant="body2" color="text.secondary">No telemetry for last 7 days</Typography>
                </Box>
            </Card>
        );
    }

    const CustomTooltip = ({ active, payload, label }) => {
        if (!active || !payload?.length) return null;
        const p = payload[0]?.payload;
        if (!p) return null;
        return (
            <Box sx={{ bgcolor: 'background.paper', border: 1, borderColor: 'divider', borderRadius: 1, p: 1.5, boxShadow: 2, minWidth: 180 }}>
                <Typography variant="caption" display="block" color="text.secondary">{p.timeLabel}</Typography>
                <Typography variant="body2"><strong>RUL:</strong> {p.rulDays.toFixed(0)} days</Typography>
                <Typography variant="caption">80% band: {p.rulLower.toFixed(0)}–{p.rulUpper.toFixed(0)} days</Typography>
                <Typography variant="caption" display="block">Health: {(p.healthScore * 100).toFixed(0)}%</Typography>
            </Box>
        );
    };

    return (
        <Card variant="outlined" sx={{ height: '100%', borderRadius: 2, overflow: 'hidden', display: 'flex', flexDirection: 'column', p: 2 }}>
            <Typography variant="subtitle2" fontWeight="bold" sx={{ mb: 1 }}>RUL Time Series (7 days)</Typography>
            <Box sx={{ flex: 1, minHeight: 200 }}>
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }} syncId={syncId}>
                        <defs>
                            <linearGradient id="rulBandFill" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#6366f1" stopOpacity={0.3} />
                                <stop offset="100%" stopColor="#6366f1" stopOpacity={0.05} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis dataKey="timeLabel" tick={{ fontSize: 10 }} />
                        <YAxis tick={{ fontSize: 10 }} label={{ value: 'RUL (days)', angle: -90, position: 'insideLeft', style: { fontSize: 10 } }} />
                        <Tooltip content={<CustomTooltip />} />
                        {alertTransitions.map((ts, i) => (
                            <ReferenceLine key={i} x={chartData.find((d) => d.timestamp === ts)?.timeLabel} stroke="#ef4444" strokeDasharray="3 3" />
                        ))}
                        <Area type="monotone" dataKey="rulUpper" fill="#6366f1" fillOpacity={0.25} stroke="none" baseValue={0} />
                        <Area type="monotone" dataKey="rulLower" fill="#ffffff" stroke="none" baseValue={0} />
                        <Line type="monotone" dataKey="rulDays" stroke="#6366f1" strokeWidth={2} dot={false} name="RUL (days)" />
                    </ComposedChart>
                </ResponsiveContainer>
            </Box>
        </Card>
    );
}
