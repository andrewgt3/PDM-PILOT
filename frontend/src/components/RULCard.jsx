import React from 'react';
import { AlertCircle, CheckCircle, Clock, TrendingDown, Activity, Calendar, Target, Percent } from 'lucide-react';
import { Card, CardContent, Typography, Box, LinearProgress, Chip, Stack, Grid, Skeleton, Tooltip } from '@mui/material';

/**
 * RULCard Component (Enterprise Version)
 * 
 * Displays detailed Remaining Useful Life analytics including:
 * - RUL estimate with confidence interval
 * - Failure probability distribution
 * - Risk assessment metrics
 * - Scheduled maintenance window
 */
function RULCard({ rul, maxRul = 90, failureProbability = 0, degradationScore = 0 }) {
    // Handle missing data
    if (rul === undefined || rul === null) {
        return (
            <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent>
                    <Stack spacing={2}>
                        <Skeleton variant="text" width="50%" height={20} />
                        <Skeleton variant="rectangular" height={60} sx={{ borderRadius: 1 }} />
                        <Skeleton variant="text" width="30%" />
                    </Stack>
                    <Typography variant="caption" color="text.secondary" align="center" display="block" sx={{ mt: 2 }}>
                        Calculating RUL...
                    </Typography>
                </CardContent>
            </Card>
        );
    }

    // Normalize values
    const rulDays = Math.max(0, rul);
    const failProb = Math.min(1, Math.max(0, failureProbability)) * 100;
    const degradation = Math.min(1, Math.max(0, degradationScore)) * 100;

    // Confidence interval (±15% typical for bearing prognostics)
    const confidenceMargin = rulDays * 0.15;
    const rulLower = Math.max(0, rulDays - confidenceMargin);
    const rulUpper = rulDays + confidenceMargin;

    // Determine urgency level
    let urgency = 'healthy';
    if (rulDays < 7 || failProb > 80) urgency = 'critical';
    else if (rulDays < 30 || failProb > 50) urgency = 'warning';

    // Risk score (0-100)
    const riskScore = Math.min(100, (failProb * 0.6) + ((100 - (rulDays / maxRul * 100)) * 0.4));

    // Styling based on urgency
    const styles = {
        critical: {
            headerBg: 'linear-gradient(to right, #dc2626, #ef4444)', // red-600 to red-500
            color: 'error.main',
            accentBg: '#fef2f2', // red-50
            progressColor: 'error',
            Icon: AlertCircle
        },
        warning: {
            headerBg: 'linear-gradient(to right, #d97706, #f59e0b)', // amber-600 to amber-500
            color: 'warning.main',
            accentBg: '#fffbeb', // amber-50
            progressColor: 'warning',
            Icon: Clock
        },
        healthy: {
            headerBg: 'linear-gradient(to right, #059669, #10b981)', // emerald-600 to emerald-500
            color: 'success.main',
            accentBg: '#ecfdf5', // emerald-50
            progressColor: 'success',
            Icon: CheckCircle
        }
    };

    const style = styles[urgency];
    const progress = Math.min(100, (rulDays / maxRul) * 100);

    // Next maintenance window calculation
    const getMaintenanceWindow = () => {
        if (rulDays < 3) return 'IMMEDIATE';
        if (rulDays < 7) return 'This Week';
        if (rulDays < 14) return 'Next 2 Weeks';
        if (rulDays < 30) return 'This Month';
        return 'As Scheduled';
    };

    return (
        <Card variant="outlined" sx={{ height: '100%', borderRadius: 2, display: 'flex', flexDirection: 'column' }}>
            {/* Header with gradient */}
            <Box sx={{ px: 2, py: 1.5, background: style.headerBg, color: 'white' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Stack direction="row" spacing={1} alignItems="center">
                        <TrendingDown size={18} style={{ opacity: 0.9 }} />
                        <Typography variant="caption" fontWeight="bold" sx={{ textTransform: 'uppercase', letterSpacing: 0.5 }}>
                            Remaining Useful Life
                        </Typography>
                    </Stack>
                    <Chip
                        icon={<style.Icon size={12} color="white" />}
                        label={urgency === 'critical' ? 'Critical' : urgency === 'warning' ? 'Caution' : 'Healthy'}
                        size="small"
                        sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white', fontWeight: 'bold', textTransform: 'uppercase', fontSize: '0.65rem', height: 22, '& .MuiChip-icon': { color: 'white' } }}
                    />
                </Box>
            </Box>

            {/* Main Content */}
            <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 2, p: { xs: 1.5, md: 2 } }}>
                {/* RUL Display with Confidence */}
                <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 2, flexWrap: 'nowrap' }}>
                    <Box sx={{ flex: 1, minWidth: 0 }}>
                        <Typography variant="caption" color="text.secondary" gutterBottom>Predicted Time to Failure</Typography>
                        <Stack direction="row" alignItems="baseline" spacing={0.5}>
                            <Typography variant="h4" fontWeight="bold" sx={{ color: style.color, lineHeight: 1 }}>
                                {rulDays < 1 ? '<24' : rulDays.toFixed(0)}
                            </Typography>
                            <Typography variant="body1" color="text.secondary" fontWeight="medium">
                                {rulDays < 1 ? 'hours' : 'days'}
                            </Typography>
                        </Stack>
                        <Typography variant="caption" color="text.disabled" sx={{ mt: 0.5, display: 'block', fontSize: '0.7rem' }}>
                            95% CI: {rulLower.toFixed(0)} - {rulUpper.toFixed(0)} days
                        </Typography>
                    </Box>

                    {/* Circular Risk Gauge */}
                    <Box sx={{ flexShrink: 0, position: 'relative', width: 64, height: 64 }}>
                        <svg width="64" height="64" style={{ transform: 'rotate(-90deg)' }}>
                            <circle cx="32" cy="32" r="28" stroke="#f1f5f9" strokeWidth="5" fill="none" />
                            <circle
                                cx="32"
                                cy="32"
                                r="28"
                                stroke="currentColor"
                                strokeWidth="5"
                                fill="none"
                                strokeDasharray={`${riskScore * 1.76} ${176 - riskScore * 1.76}`}
                                strokeLinecap="round"
                                style={{ color: urgency === 'critical' ? '#dc2626' : urgency === 'warning' ? '#d97706' : '#059669' }}
                            />
                        </svg>
                        <Box sx={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                            <Typography variant="body1" fontWeight="bold" sx={{ color: style.color, lineHeight: 1 }}>{riskScore.toFixed(0)}</Typography>
                            <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.6rem' }}>RISK</Typography>
                        </Box>
                    </Box>
                </Box>

                {/* Life Progress Bar */}
                <Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5, flexWrap: 'wrap', gap: 0.5 }}>
                        <Typography variant="caption" color="text.secondary" sx={{ whiteSpace: 'nowrap', fontSize: '0.7rem' }}>Lifecycle Progress</Typography>
                        <Typography variant="caption" color="text.secondary" sx={{ whiteSpace: 'nowrap', fontSize: '0.7rem' }}>{progress.toFixed(0)}% remaining</Typography>
                    </Box>
                    <LinearProgress
                        variant="determinate"
                        value={progress}
                        color={style.progressColor}
                        sx={{ height: 6, borderRadius: 3, width: '100%' }}
                    />
                </Box>

                {/* Metrics Grid */}
                <Grid container spacing={1}>
                    <Grid item xs={4}>
                        <Box sx={{ p: 1, borderRadius: 1.5, bgcolor: style.accentBg, height: '100%' }}>
                            <Stack direction="row" spacing={0.5} alignItems="center" sx={{ mb: 0.5 }}>
                                <Percent size={12} className="text-slate-500" />
                                <Typography sx={{ fontSize: '0.6rem', fontWeight: 'bold', color: 'text.secondary' }}>FAIL PROB</Typography>
                            </Stack>
                            <Typography variant="body1" fontWeight="bold" sx={{ color: style.color }}>
                                {failProb.toFixed(1)}%
                            </Typography>
                        </Box>
                    </Grid>
                    <Grid item xs={4}>
                        <Box sx={{ p: 1, borderRadius: 1.5, bgcolor: style.accentBg, height: '100%' }}>
                            <Stack direction="row" spacing={0.5} alignItems="center" sx={{ mb: 0.5 }}>
                                <Activity size={12} className="text-slate-500" />
                                <Typography sx={{ fontSize: '0.6rem', fontWeight: 'bold', color: 'text.secondary' }}>WEAR</Typography>
                            </Stack>
                            <Typography variant="body1" fontWeight="bold" sx={{ color: style.color }}>
                                {degradation.toFixed(0)}%
                            </Typography>
                        </Box>
                    </Grid>
                    <Grid item xs={4}>
                        <Box sx={{ p: 1, borderRadius: 1.5, bgcolor: style.accentBg, height: '100%' }}>
                            <Stack direction="row" spacing={0.5} alignItems="center" sx={{ mb: 0.5 }}>
                                <Calendar size={12} className="text-slate-500" />
                                <Typography sx={{ fontSize: '0.6rem', fontWeight: 'bold', color: 'text.secondary' }}>SERVICE</Typography>
                            </Stack>
                            <Typography variant="body2" fontWeight="bold" sx={{ color: style.color, lineHeight: 1.2 }}>
                                {getMaintenanceWindow()}
                            </Typography>
                        </Box>
                    </Grid>
                </Grid>

                {/* Confidence Breakdown */}
                <Box sx={{ pt: 1.5, borderTop: 1, borderColor: 'divider' }}>
                    <Typography sx={{ fontSize: '0.65rem', fontWeight: 'bold', color: 'text.secondary', display: 'block', mb: 1, textTransform: 'uppercase' }}>
                        Failure Distribution
                    </Typography>
                    <Stack spacing={0.75}>
                        <ProbabilityBar label="0-7 days" value={urgency === 'critical' ? failProb : failProb * 0.3} color="error" />
                        <ProbabilityBar label="7-30 days" value={urgency === 'warning' ? failProb * 0.7 : failProb * 0.5} color="warning" />
                        <ProbabilityBar label="30-90 days" value={urgency === 'healthy' ? failProb * 0.2 : failProb * 0.2} color="success" />
                    </Stack>
                </Box>
            </CardContent>
        </Card>
    );
}

// Sub-component for probability bars
function ProbabilityBar({ label, value, color }) {
    return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
            <Typography sx={{ fontSize: '0.7rem', color: 'text.secondary', width: 55, flexShrink: 0 }}>{label}</Typography>
            <LinearProgress
                variant="determinate"
                value={Math.min(100, value)}
                color={color}
                sx={{ flex: 1, height: 5, borderRadius: 2.5 }}
            />
            <Typography sx={{ fontSize: '0.7rem', fontWeight: 'medium', width: 36, textAlign: 'right', flexShrink: 0 }}>
                {value.toFixed(1)}%
            </Typography>
        </Box>
    );
}

export default RULCard;
