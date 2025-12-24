import React from 'react';
import {
    Clock, AlertTriangle, TrendingUp, ArrowRight, Calendar,
    Bell, Flame, AlertOctagon, Zap, Lightbulb, Wrench
} from 'lucide-react';
import {
    Card, CardContent, Typography, Box, Chip, Button,
    LinearProgress, Grid, Stack, IconButton
} from '@mui/material';

/**
 * CriticalMachineCard Component
 * Enhanced card for critical machines with actionable information
 */
export function CriticalMachineCard({ machine, onViewDetails, onSchedule, onAlert }) {
    const prob = (machine.failure_probability || 0) * 100;
    const rulDays = machine.rul_days || 0;

    // Determine urgency level
    const getUrgency = () => {
        if (prob > 95) return {
            level: 'EXTREME',
            color: 'error',
            Icon: Flame,
            gradient: 'linear-gradient(90deg, #ef4444 0%, #dc2626 100%)', // red
            bg: 'error.light'
        };
        if (prob > 85) return {
            level: 'HIGH',
            color: 'warning',
            Icon: AlertOctagon,
            gradient: 'linear-gradient(90deg, #f97316 0%, #ef4444 100%)', // orange -> red
            bg: 'warning.light'
        };
        return {
            level: 'ELEVATED',
            color: 'warning',
            Icon: Zap,
            gradient: 'linear-gradient(90deg, #f59e0b 0%, #f97316 100%)', // amber -> orange
            bg: 'warning.light'
        };
    };

    const urgency = getUrgency();

    // Get recommended action
    const getRecommendedAction = () => {
        if (prob > 95) return 'Immediate shutdown and bearing replacement required';
        if (prob > 85) return 'Schedule urgent maintenance within 24 hours';
        return 'Plan preventive maintenance in next maintenance window';
    };

    // Calculate time critical (mock - would come from backend)
    const timeCritical = '2h 15m'; // This would be calculated based on when it crossed 80% threshold

    return (
        <Card sx={{
            position: 'relative', overflow: 'hidden', borderRadius: 0,
            border: 'none',
            borderBottom: '1px solid', borderBottomColor: 'divider',
            borderLeft: 4, borderLeftColor: urgency.level === 'EXTREME' ? 'error.main' : 'warning.main',
            boxShadow: 'none',
            transition: 'background-color 0.2s ease',
            '&:hover': { bgcolor: 'grey.50' },
            width: '100%', boxSizing: 'border-box',
            bgcolor: 'background.paper',
            '&:last-child': { borderBottom: 'none' }
        }}>
            <CardContent sx={{ p: 3.5, '&:last-child': { pb: 3.5 } }}>
                {/* Header Row */}
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
                    <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                        {/* Pulsing Status Indicator */}
                        <Box sx={{ position: 'relative', display: 'flex' }}>
                            <Box sx={{ width: 16, height: 16, borderRadius: '50%', bgcolor: 'error.main', animation: 'pulse 1.5s infinite' }} />
                            <Box sx={{ position: 'absolute', inset: 0, borderRadius: '50%', bgcolor: 'error.main', animation: 'ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite', opacity: 0.75 }} />
                        </Box>

                        {/* Machine ID */}
                        <Box>
                            <Typography variant="h5" fontWeight="bold" fontFamily="monospace" lineHeight={1}>
                                {machine.machine_id}
                            </Typography>
                            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                                {machine.line_name || 'Production Line B'} • Assembly Station
                            </Typography>
                        </Box>
                    </Box>

                    {/* Urgency Badge */}
                    <Chip
                        icon={<urgency.Icon size={18} />}
                        label={urgency.level}
                        color={urgency.color}
                        size="medium"
                        sx={{ fontWeight: 'bold', fontSize: '0.875rem', py: 0.5 }}
                    />
                </Box>

                {/* Failure Probability Bar */}
                <Box sx={{ mb: 3 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                        <Typography variant="body2" fontWeight={600} color="text.secondary">Failure Probability</Typography>
                        <Stack direction="row" alignItems="center" spacing={0.5}>
                            <TrendingUp size={18} className="text-red-600" />
                            <Typography variant="h6" fontWeight="bold" color="error.main" fontFamily="monospace">
                                {prob.toFixed(1)}%
                            </Typography>
                        </Stack>
                    </Box>
                    <Box sx={{ width: '100%', height: 14, bgcolor: 'grey.100', borderRadius: 1, overflow: 'hidden' }}>
                        <Box sx={{
                            height: '100%', width: `${prob}%`,
                            background: urgency.gradient,
                            position: 'relative', transition: 'width 0.5s ease-out'
                        }}>
                            <Box sx={{ position: 'absolute', inset: 0, bgcolor: 'white', opacity: 0.2, animation: 'pulse 2s infinite' }} />
                        </Box>
                    </Box>
                </Box>

                {/* Key Metrics Grid */}
                <Box sx={{
                    display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 2, mb: 3,
                    bgcolor: 'grey.50', p: 2, borderRadius: 2, border: '1px solid', borderColor: 'divider'
                }}>
                    <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="caption" display="flex" justifyContent="center" gap={0.5} alignItems="center" color="text.secondary" fontWeight={600} fontSize={12} textTransform="uppercase">
                            <Clock size={16} /> Critical For
                        </Typography>
                        <Typography variant="body1" fontWeight="bold" color="error.main">{timeCritical}</Typography>
                    </Box>

                    <Box sx={{ textAlign: 'center', borderLeft: 1, borderRight: 1, borderColor: 'divider' }}>
                        <Typography variant="caption" display="flex" justifyContent="center" gap={0.5} alignItems="center" color="text.secondary" fontWeight={600} fontSize={12} textTransform="uppercase">
                            <Wrench size={16} /> Est. Failure
                        </Typography>
                        <Typography variant="body1" fontWeight="bold" color="warning.dark">
                            {rulDays < 1 ? '< 24h' : `${rulDays.toFixed(0)} days`}
                        </Typography>
                    </Box>

                    <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="caption" display="flex" justifyContent="center" gap={0.5} alignItems="center" color="text.secondary" fontWeight={600} fontSize={12} textTransform="uppercase">
                            <AlertTriangle size={16} /> Impact
                        </Typography>
                        <Typography variant="body1" fontWeight="bold" color="text.primary">High</Typography>
                    </Box>
                </Box>

                {/* Recommended Action */}
                <Box sx={{ mb: 3, p: 2, borderLeft: 4, borderColor: 'info.main', borderRadius: 1, bgcolor: '#eff6ff' }}>
                    <Stack direction="row" spacing={1.5} alignItems="start">
                        <Lightbulb size={20} className="text-blue-600 mt-0.5" />
                        <Box>
                            <Typography variant="body2" fontWeight="bold" color="info.dark" display="block">Recommended Action</Typography>
                            <Typography variant="body2" color="info.dark" lineHeight={1.4}>
                                {getRecommendedAction()}
                            </Typography>
                        </Box>
                    </Stack>
                </Box>

                {/* Action Buttons */}
                <Grid container spacing={1.5}>
                    <Grid item xs={4}>
                        <Button
                            variant="contained" color="inherit" fullWidth size="medium"
                            onClick={() => onViewDetails(machine.machine_id)}
                            sx={{ bgcolor: 'grey.800', color: 'white', '&:hover': { bgcolor: 'grey.900' }, fontSize: '0.875rem', py: 1 }}
                            startIcon={<ArrowRight size={18} />}
                        >
                            Details
                        </Button>
                    </Grid>
                    <Grid item xs={4}>
                        <Button
                            variant="contained" color="primary" fullWidth size="medium"
                            onClick={() => onSchedule && onSchedule(machine.machine_id)}
                            sx={{ fontSize: '0.875rem', py: 1 }}
                            startIcon={<Calendar size={18} />}
                        >
                            Schedule
                        </Button>
                    </Grid>
                    <Grid item xs={4}>
                        <Button
                            variant="contained" color="error" fullWidth size="medium"
                            onClick={() => onAlert && onAlert(machine.machine_id)}
                            sx={{ fontSize: '0.875rem', py: 1 }}
                            startIcon={<Bell size={18} />}
                        >
                            Alert
                        </Button>
                    </Grid>
                </Grid>
            </CardContent>
        </Card>
    );
}

export default CriticalMachineCard;
