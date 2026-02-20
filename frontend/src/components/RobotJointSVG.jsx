import React from 'react';
import { Card, Typography, Box } from '@mui/material';

const JOINT_NAMES = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6'];

function jointColor(score) {
    if (score == null) return '#94a3b8';
    if (score > 0.7) return '#22c55e';
    if (score > 0.4) return '#f59e0b';
    return '#ef4444';
}

function shouldPulse(score) {
    return score != null && score <= 0.7;
}

export function RobotJointSVG({ jointHealthScores, healthScore, selectedJointIndex, onSelectJoint }) {
    const scores = Array.isArray(jointHealthScores) && jointHealthScores.length >= 6
        ? jointHealthScores.slice(0, 6)
        : Array(6).fill(healthScore ?? 1);
    const anyPulse = scores.some(shouldPulse);

    return (
        <Card variant="outlined" sx={{ height: '100%', borderRadius: 2, overflow: 'hidden', display: 'flex', flexDirection: 'column', p: 2 }}>
            <Typography variant="subtitle2" fontWeight="bold" sx={{ mb: 1 }}>Joint Health</Typography>
            <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 120 }}>
                <svg viewBox="0 0 320 80" width="100%" height="100%" style={{ maxHeight: 160 }}>
                    {JOINT_NAMES.map((label, i) => {
                        const cx = 40 + i * 48;
                        const cy = 40;
                        const score = scores[i];
                        const fill = jointColor(score);
                        const pulse = shouldPulse(score);
                        const selected = selectedJointIndex === i;
                        return (
                            <g key={i} onClick={() => onSelectJoint?.(i)} style={{ cursor: onSelectJoint ? 'pointer' : 'default' }}>
                                <circle
                                    cx={cx}
                                    cy={cy}
                                    r={18}
                                    fill={fill}
                                    stroke={selected ? '#1e293b' : '#e2e8f0'}
                                    strokeWidth={selected ? 3 : 1}
                                    opacity={pulse && anyPulse ? 0.9 : 1}
                                    style={{
                                        animation: pulse && anyPulse ? 'joint-pulse 1.5s ease-in-out infinite' : undefined
                                    }}
                                />
                                <text x={cx} y={cy + 5} textAnchor="middle" fontSize="10" fill="#fff" fontWeight="bold">{label}</text>
                            </g>
                        );
                    })}
                </svg>
            </Box>
            <style>{`
                @keyframes joint-pulse {
                    0%, 100% { opacity: 0.9; transform: scale(1); }
                    50% { opacity: 1; transform: scale(1.08); }
                }
                svg g circle[style*="animation"] { transform-origin: center; }
            `}</style>
        </Card>
    );
}
