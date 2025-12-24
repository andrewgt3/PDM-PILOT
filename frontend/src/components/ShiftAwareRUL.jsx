import React, { useState, useEffect } from 'react';
import { Calendar, Sun, Moon, Clock, TrendingDown, RefreshCw } from 'lucide-react';
import { Card, Box, Typography, IconButton, Chip, Stack, CircularProgress, Divider, Grid } from '@mui/material';

/**
 * ShiftAwareRUL Component
 * 
 * Adjusts RUL predictions based on production schedule intensity.
 */
function ShiftAwareRUL({ baseRulDays = 30, machineId }) {
    const [schedule, setSchedule] = useState(null);
    const [loading, setLoading] = useState(true);

    const fetchSchedule = async () => {
        try {
            const response = await fetch('http://localhost:8000/api/enterprise/schedule');
            if (response.ok) {
                const data = await response.json();
                setSchedule(data);
            } else {
                setSchedule({
                    current_shift: 'Day',
                    shifts: [
                        { shift_name: 'Day', start_time: '06:00:00', end_time: '14:00:00' },
                        { shift_name: 'Afternoon', start_time: '14:00:00', end_time: '22:00:00' },
                        { shift_name: 'Night', start_time: '22:00:00', end_time: '06:00:00' }
                    ],
                    production_mode: 'normal',
                    weekly_hours: 120,
                    wear_factor: 1.0
                });
            }
        } catch (err) {
            console.error('[Schedule API Error]', err);
            setSchedule({
                current_shift: 'Day',
                shifts: [],
                production_mode: 'normal',
                weekly_hours: 120,
                wear_factor: 1.0
            });
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchSchedule();
    }, []);

    // Calculate adjusted RUL based on production intensity
    const calculateAdjustedRUL = () => {
        if (!schedule) return baseRulDays;
        const wearFactor = schedule.wear_factor || 1.0;
        const normalWeeklyHours = 120;
        const hoursRatio = (schedule.weekly_hours || 120) / normalWeeklyHours;
        const adjustmentFactor = (1 / wearFactor) * (1 / hoursRatio);
        return baseRulDays * adjustmentFactor;
    };

    const adjustedRul = calculateAdjustedRUL();
    const rulDifference = adjustedRul - baseRulDays;

    const shiftConfig = {
        Day: { icon: Sun, color: 'warning', time: '6:00 - 14:00' },
        Afternoon: { icon: Clock, color: 'warning', time: '14:00 - 22:00' }, // Orange mapped to warning
        Night: { icon: Moon, color: 'primary', time: '22:00 - 6:00' }
    };

    const productionModeConfig = {
        overtime: { label: 'Overtime', color: 'error', impact: 'Accelerated wear (+15%)' },
        normal: { label: 'Normal', color: 'success', impact: 'Standard wear' },
        reduced: { label: 'Reduced', color: 'info', impact: 'Slower wear (-15%)' }
    };

    if (loading) {
        return (
            <Card variant="outlined" sx={{ height: '100%', borderRadius: 2, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <CircularProgress />
            </Card>
        );
    }

    const modeConfig = productionModeConfig[schedule?.production_mode] || productionModeConfig.normal;
    const currentShift = schedule?.current_shift || 'Day';

    return (
        <Card variant="outlined" sx={{ height: '100%', borderRadius: 2, display: 'flex', flexDirection: 'column' }}>
            {/* Header */}
            <Box sx={{ px: 2, py: 1.5, borderBottom: 1, borderColor: 'divider', bgcolor: 'grey.50', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Stack direction="row" spacing={1} alignItems="center">
                    <Calendar size={20} className="text-slate-600" />
                    <Typography variant="subtitle2" fontWeight="bold">Shift-Aware RUL</Typography>
                </Stack>
                <Stack direction="row" spacing={1} alignItems="center">
                    <IconButton size="small" onClick={fetchSchedule} title="Refresh">
                        <RefreshCw size={16} />
                    </IconButton>
                    <Chip
                        label={`${modeConfig.label} Production`}
                        size="small"
                        color={modeConfig.color}
                        sx={{ fontWeight: 'bold' }}
                    />
                </Stack>
            </Box>

            <Box sx={{ p: 3, flex: 1, display: 'flex', flexDirection: 'column', gap: 3 }}>
                {/* RUL Comparison */}
                <Grid container spacing={2}>
                    <Grid item xs={6}>
                        <Box sx={{ p: 2, bgcolor: 'grey.50', borderRadius: 2, height: '100%' }}>
                            <Typography variant="caption" color="text.secondary" gutterBottom>Base RUL</Typography>
                            <Stack direction="row" alignItems="baseline" spacing={0.5}>
                                <Typography variant="h5" fontWeight="bold" color="text.primary">{Math.round(baseRulDays)}</Typography>
                                <Typography variant="body2" color="text.secondary">days</Typography>
                            </Stack>
                            <Typography variant="caption" color="text.disabled" display="block" sx={{ mt: 0.5 }}>Standard Model</Typography>
                        </Box>
                    </Grid>
                    <Grid item xs={6}>
                        <Box sx={{ p: 2, bgcolor: rulDifference < 0 ? 'error.lighter' : 'success.lighter', borderRadius: 2, height: '100%' }}>
                            <Typography variant="caption" color="text.secondary" gutterBottom>Adjusted RUL</Typography>
                            <Stack direction="row" alignItems="baseline" spacing={0.5}>
                                <Typography variant="h5" fontWeight="bold" sx={{ color: rulDifference < 0 ? 'error.main' : 'success.main' }}>
                                    {Math.round(adjustedRul)}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">days</Typography>
                            </Stack>
                            <Stack direction="row" alignItems="center" spacing={0.5} sx={{ mt: 0.5, color: rulDifference < 0 ? 'error.main' : 'success.main' }}>
                                <TrendingDown size={12} />
                                <Typography variant="caption" fontWeight="medium">
                                    {rulDifference > 0 ? '+' : ''}{rulDifference.toFixed(1)} days ({modeConfig.impact})
                                </Typography>
                            </Stack>
                        </Box>
                    </Grid>
                </Grid>

                {/* Current Shift Indicator */}
                <Box>
                    <Typography variant="caption" fontWeight="bold" color="text.secondary" sx={{ textTransform: 'uppercase', mb: 1.5, display: 'block' }}>Current Shift</Typography>
                    <Stack direction="row" spacing={2}>
                        {['Day', 'Afternoon', 'Night'].map((shift) => {
                            const config = shiftConfig[shift];
                            const ShiftIcon = config.icon;
                            const isCurrent = shift === currentShift;
                            return (
                                <Box
                                    key={shift}
                                    sx={{
                                        flex: 1, p: 1.5, borderRadius: 2, border: 1,
                                        borderColor: isCurrent ? `${config.color}.main` : 'divider',
                                        bgcolor: isCurrent ? `${config.color}.lighter` : 'background.paper',
                                        position: 'relative'
                                    }}
                                >
                                    <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                                        <ShiftIcon size={16} className={isCurrent ? `text-${config.color}-main` : 'text-slate-400'} style={{ color: isCurrent ? undefined : '#94a3b8' }} />
                                        <Typography variant="caption" fontWeight={isCurrent ? 'bold' : 'medium'} color={isCurrent ? 'text.primary' : 'text.secondary'}>
                                            {shift}
                                        </Typography>
                                    </Stack>
                                    <Typography variant="caption" color="text.disabled" sx={{ fontSize: '0.6rem' }}>{config.time}</Typography>
                                    {isCurrent && (
                                        <Box sx={{ position: 'absolute', top: 4, right: 4, px: 0.5, bgcolor: 'background.paper', borderRadius: 0.5, border: 1, borderColor: `${config.color}.main` }}>
                                            <Typography variant="caption" fontWeight="bold" sx={{ fontSize: '0.55rem', color: `${config.color}.main` }}>NOW</Typography>
                                        </Box>
                                    )}
                                </Box>
                            );
                        })}
                    </Stack>
                </Box>

                {/* Weekly Hours */}
                <Box sx={{ p: 2, bgcolor: 'grey.50', borderRadius: 2 }}>
                    <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                        <Typography variant="caption" color="text.secondary">Weekly Operating Hours</Typography>
                        <Stack direction="row" spacing={1} alignItems="center">
                            <Box sx={{ width: 100, height: 8, bgcolor: 'grey.300', borderRadius: 4, overflow: 'hidden' }}>
                                <Box sx={{
                                    height: '100%',
                                    width: `${Math.min(100, ((schedule?.weekly_hours || 120) / 168) * 100)}%`,
                                    bgcolor: (schedule?.weekly_hours || 120) > 120 ? 'error.main' : 'success.main'
                                }} />
                            </Box>
                            <Typography variant="body2" fontWeight="bold">{schedule?.weekly_hours || 120}h</Typography>
                        </Stack>
                    </Stack>
                    <Typography variant="caption" color="text.disabled">vs. 120h normal capacity (168h max)</Typography>
                </Box>
            </Box>

            {/* Footer */}
            <Box sx={{ px: 2, py: 1, bgcolor: 'grey.50', borderTop: 1, borderColor: 'divider', display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="caption" color="text.secondary">Wear Factor: {schedule?.wear_factor?.toFixed(2) || '1.00'}x</Typography>
                <Typography variant="caption" color="text.disabled">Source: PDM Config</Typography>
            </Box>
        </Card>
    );
}

export default ShiftAwareRUL;
