import React, { useState, useEffect } from 'react';
import { Calendar, Sun, Moon, Clock, TrendingDown, RefreshCw, Zap, PauseCircle, PlayCircle, BarChart2 } from 'lucide-react';
import { Card, Box, Typography, IconButton, Chip, Stack, CircularProgress, Divider, Grid, Button, ButtonGroup, Tooltip } from '@mui/material';

/**
 * ShiftAwareRUL Component
 * 
 * Dynamic Operational Forecasting tile allowing "What-If" simulation of shift changes.
 */
function ShiftAwareRUL({ baseRulDays = 30, machineId }) {
    const [schedule, setSchedule] = useState(null);
    const [loading, setLoading] = useState(true);
    const [simulatedShift, setSimulatedShift] = useState(null); // 'Day', 'Afternoon', 'Night' or null

    const fetchSchedule = async () => {
        setLoading(true);
        try {
            // Simulate API latency for "Refresh" feel
            await new Promise(r => setTimeout(r, 600));

            // Mock data incase API fails/is missing (common in this env)
            setSchedule({
                current_shift: 'Day',
                shifts: [
                    { shift_name: 'Day', start_time: '06:00:00', end_time: '14:00:00', intensity: [65, 78, 82, 90, 85, 70, 60, 55] },
                    { shift_name: 'Afternoon', start_time: '14:00:00', end_time: '22:00:00', intensity: [50, 55, 60, 65, 60, 55, 50, 45] },
                    { shift_name: 'Night', start_time: '22:00:00', end_time: '06:00:00', intensity: [30, 35, 30, 25, 30, 35, 30, 25] }
                ],
                production_mode: 'normal',
                weekly_hours: 110,
                wear_factor: 1.0
            });
        } catch (err) {
            console.error('[Schedule Error]', err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchSchedule();
    }, []);

    // Simulation Logic
    const activeShiftName = simulatedShift || schedule?.current_shift || 'Day';

    // Define wear factors for simulation
    const shiftWearFactors = {
        'Day': 1.2,       // Heavy load
        'Afternoon': 1.0, // Normal load
        'Night': 0.8      // Light load
    };

    // Calculate Adjusted RUL
    const calculateAdjustedRUL = () => {
        if (!schedule) return baseRulDays;

        // Use simulated wear factor if simulating, else actual
        const currentWearFactor = shiftWearFactors[activeShiftName];
        const normalWeeklyHours = 120;
        const hoursRatio = (schedule.weekly_hours || 120) / normalWeeklyHours;

        // RUL Adjustment Formula: Base * (1/Wear) * (1/HoursRatio)
        const adjustmentFactor = (1 / currentWearFactor) * (1 / hoursRatio);
        return baseRulDays * adjustmentFactor;
    };

    const adjustedRul = calculateAdjustedRUL();
    const rulDifference = adjustedRul - baseRulDays;

    const shiftConfig = {
        Day: { icon: Sun, color: 'warning', time: '06:00 - 14:00', label: 'Heavy Load' },
        Afternoon: { icon: Clock, color: 'info', time: '14:00 - 22:00', label: 'Std Load' },
        Night: { icon: Moon, color: 'primary', time: '22:00 - 06:00', label: 'Light Load' }
    };

    if (loading && !schedule) {
        return (
            <Card variant="outlined" sx={{ height: '100%', borderRadius: 2, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <CircularProgress size={30} />
            </Card>
        );
    }

    const currentShift = schedule?.current_shift || 'Day';
    const weeklyHours = schedule?.weekly_hours || 0;
    const isOverCapacity = weeklyHours > 120;

    // SVG Gauge Calculations
    const maxRul = baseRulDays * 2;
    const baseAngle = (baseRulDays / maxRul) * 180;
    const adjAngle = Math.min((adjustedRul / maxRul) * 180, 180);

    return (
        <Card variant="outlined" sx={{ height: '100%', borderRadius: 2, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            {/* Header */}
            <Box sx={{ px: 2, py: 1.5, borderBottom: 1, borderColor: 'divider', bgcolor: 'grey.50', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Stack direction="row" spacing={1} alignItems="center">
                    <Calendar size={18} className="text-slate-600" />
                    <Typography variant="subtitle2" fontWeight="bold" color="text.secondary" textTransform="uppercase">
                        Operational Forecasting
                    </Typography>
                </Stack>
                {simulatedShift && (
                    <Chip label="SIMULATION ACTIVE" size="small" color="secondary" sx={{ height: 20, fontSize: '0.65rem', fontWeight: 'bold' }} />
                )}
            </Box>

            <Box sx={{ p: 2, flex: 1, display: 'flex', flexDirection: 'column', gap: 2 }}>

                {/* RUL Comparison Gauge */}
                <Box sx={{ position: 'relative', height: 140, display: 'flex', justifyContent: 'center', alignItems: 'flex-end', pb: 0 }}>
                    <svg width="240" height="120" viewBox="0 0 240 120">
                        {/* Background Arc */}
                        <path d="M 20 120 A 100 100 0 0 1 220 120" fill="none" stroke="#e2e8f0" strokeWidth="12" strokeLinecap="round" />

                        {/* Base RUL Arc (Static) - Slate */}
                        <path
                            d={`M 20 120 A 100 100 0 0 1 ${120 - 100 * Math.cos(baseAngle * Math.PI / 180)} ${120 - 100 * Math.sin(baseAngle * Math.PI / 180)}`}
                            fill="none" stroke="#94a3b8" strokeWidth="12" strokeLinecap="round" opacity="0.3"
                        />

                        {/* Adjusted RUL Arc (Dynamic) - Gradient */}
                        <defs>
                            <linearGradient id="heatGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" stopColor="#f87171" />
                                <stop offset="50%" stopColor="#facc15" />
                                <stop offset="100%" stopColor="#4ade80" />
                            </linearGradient>
                        </defs>
                        <path
                            d={`M 20 120 A 100 100 0 0 1 ${120 - 100 * Math.cos(adjAngle * Math.PI / 180)} ${120 - 100 * Math.sin(adjAngle * Math.PI / 180)}`}
                            fill="none" stroke="url(#heatGradient)" strokeWidth="12" strokeLinecap="round"
                            style={{ transition: 'd 0.5s ease-out' }}
                        />

                        {/* Needle */}
                        <line
                            x1="120" y1="120"
                            x2={120 - 90 * Math.cos(adjAngle * Math.PI / 180)}
                            y2={120 - 90 * Math.sin(adjAngle * Math.PI / 180)}
                            stroke="#1e293b" strokeWidth="3" markerEnd="url(#arrowhead)"
                            style={{ transition: 'all 0.5s ease-out', transformOrigin: '120px 120px' }}
                        />
                        <circle cx="120" cy="120" r="6" fill="#1e293b" />
                    </svg>

                    {/* Gauge Labels */}
                    <Box sx={{ position: 'absolute', bottom: 0, left: 0, right: 0, display: 'flex', justifyContent: 'space-between', px: 4 }}>
                        <Box sx={{ textAlign: 'center' }}>
                            <Typography variant="caption" color="text.secondary" display="block">Base</Typography>
                            <Typography variant="body2" fontWeight="bold" fontFamily="monospace">{Math.round(baseRulDays)}d</Typography>
                        </Box>
                        <Box sx={{ textAlign: 'center' }}>
                            <Typography variant="caption" color={rulDifference < 0 ? 'error.main' : 'success.main'} fontWeight="bold" display="block">
                                {rulDifference > 0 ? '+' : ''}{rulDifference.toFixed(1)}d
                            </Typography>
                            <Typography variant="h6" fontWeight="bold" fontFamily="monospace" sx={{ lineHeight: 1 }}>
                                {Math.round(adjustedRul)}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">Forecast</Typography>
                        </Box>
                    </Box>
                </Box>

                {/* Simulation Controls */}
                <Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                        <Typography variant="caption" fontWeight="bold" color="text.secondary" textTransform="uppercase">
                            Simulate Shift Impact
                        </Typography>
                        {simulatedShift && (
                            <IconButton size="small" onClick={() => setSimulatedShift(null)} sx={{ p: 0.5 }}>
                                <RefreshCw size={12} />
                            </IconButton>
                        )}
                    </Box>
                    <Grid container spacing={1}>
                        {Object.keys(shiftConfig).map((shift) => {
                            const config = shiftConfig[shift];
                            const Icon = config.icon;
                            // Check against simulated OR actual current if no simulation
                            const isActive = shift === activeShiftName;
                            const isLive = !simulatedShift && shift === currentShift;

                            return (
                                <Grid item xs={4} key={shift}>
                                    <Box
                                        onClick={() => setSimulatedShift(shift)}
                                        sx={{
                                            p: 1, borderRadius: 2, cursor: 'pointer',
                                            border: 1,
                                            borderColor: isActive ? `${config.color}.main` : 'divider',
                                            bgcolor: isActive ? `${config.color}.lighter` : 'background.paper',
                                            boxShadow: isLive ? `0 0 0 2px rgba(255,255,255,1), 0 0 0 4px ${config.color === 'warning' ? '#f59e0b' : config.color === 'info' ? '#3b82f6' : '#6366f1'}` : 'none',
                                            opacity: simulatedShift && !isActive ? 0.5 : 1,
                                            transition: 'all 0.2s',
                                            position: 'relative', overflow: 'hidden'
                                        }}
                                    >
                                        {/* Sparkline Mock */}
                                        <Box sx={{ fontStyle: 'italic', position: 'absolute', bottom: 2, right: 2, opacity: 0.1 }}>
                                            <BarChart2 size={32} />
                                        </Box>

                                        <Stack spacing={0.5}>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                                <Icon size={14} className={`text-${config.color}-main`} />
                                                {isLive && <Box sx={{ width: 6, height: 6, borderRadius: '50%', bgcolor: 'success.main', boxShadow: '0 0 6px #4ade80' }} />}
                                            </Box>
                                            <Typography variant="caption" fontWeight="bold">{shift}</Typography>
                                            <Typography variant="caption" sx={{ fontSize: '0.6rem', color: 'text.secondary' }}>{config.label}</Typography>
                                        </Stack>
                                    </Box>
                                </Grid>
                            );
                        })}
                    </Grid>
                </Box>

                {/* Capacity Bar */}
                <Box>
                    <Stack direction="row" justifyContent="space-between" mb={0.5}>
                        <Typography variant="caption" fontWeight="bold" color="text.secondary" textTransform="uppercase">Weekly Capacity</Typography>
                        <Stack direction="row" alignItems="baseline" spacing={0.5}>
                            <Typography variant="caption" fontFamily="monospace" fontWeight="bold" color={isOverCapacity ? 'error.main' : 'text.primary'}>
                                {weeklyHours}h
                            </Typography>
                            <Typography variant="caption" color="text.secondary">/ 120h</Typography>
                        </Stack>
                    </Stack>

                    <Box sx={{ height: 6, width: '100%', bgcolor: 'grey.200', borderRadius: 3, position: 'relative', overflow: 'hidden' }}>
                        {/* 120h Marker Line */}
                        <Box sx={{ position: 'absolute', left: `${(120 / 168) * 100}%`, top: 0, bottom: 0, width: 2, bgcolor: 'text.primary', zIndex: 10 }} />

                        {/* Fill */}
                        <Box sx={{
                            height: '100%',
                            width: `${Math.min((weeklyHours / 168) * 100, 100)}%`,
                            bgcolor: isOverCapacity ? 'error.main' : 'success.main',
                            backgroundImage: isOverCapacity ? 'repeating-linear-gradient(45deg, transparent, transparent 4px, rgba(255,255,255,0.3) 4px, rgba(255,255,255,0.3) 8px)' : 'none',
                            transition: 'width 0.5s ease'
                        }} />
                    </Box>
                </Box>

            </Box>
        </Card>
    );
}

export default ShiftAwareRUL;
