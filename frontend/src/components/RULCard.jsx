import React, { useMemo, useState } from 'react';
import { AlertCircle, CheckCircle, Clock, TrendingDown, Activity, Calendar, Percent, TriangleAlert, ChevronDown, ChevronUp } from 'lucide-react';
import { Card, CardContent, Typography, Box, Button, Chip, Stack, Grid, Skeleton, Collapse } from '@mui/material';
import ExplanationCard from './ExplanationCard';

/**
 * RULCard Component (High-Urgency Diagnostic Edition)
 * 
 * Displays detailed Remaining Useful Life analytics with industrial precision.
 * Optional: rulLower80, rulUpper80, confidence (HIGH|MEDIUM|LOW), inTrainingDistribution.
 */
function RULCard({ rul, maxRul = 90, failureProbability = 0, degradationScore = 0, onSchedule, rulLower80, rulUpper80, confidence, inTrainingDistribution = true, explanations = [], healthScore, machineName }) {
    const [whyExpanded, setWhyExpanded] = useState(false);
    const topExplanation = Array.isArray(explanations) && explanations.length > 0 ? explanations[0] : null;
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

    // Confidence interval: use provided 80% band or fallback ±15%
    const hasBand = rulLower80 != null && rulUpper80 != null && typeof rulLower80 === 'number' && typeof rulUpper80 === 'number';
    const confidenceMargin = rulDays * 0.15;
    const rulLower = hasBand ? Math.max(0, rulLower80) : Math.max(0, rulDays - confidenceMargin);
    const rulUpper = hasBand ? rulUpper80 : rulDays + confidenceMargin;
    const confidenceLevel = (confidence || '').toUpperCase();
    const confidenceColor = confidenceLevel === 'HIGH' ? '#10b981' : confidenceLevel === 'LOW' ? '#ef4444' : confidenceLevel === 'MEDIUM' ? '#f59e0b' : 'inherit';

    // Predicted Failure Date
    const failureDate = new Date();
    failureDate.setDate(failureDate.getDate() + rulDays);
    const dateString = failureDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });

    // Determine urgency level
    let urgency = 'healthy';
    if (rulDays < 7 || failProb > 80) urgency = 'critical';
    else if (rulDays < 30 || failProb > 50) urgency = 'warning';

    // Risk score (0-100)
    const riskScore = Math.min(100, (failProb * 0.6) + ((100 - (rulDays / maxRul * 100)) * 0.4));

    // Dynamic Styles
    const styles = {
        critical: {
            headerBg: 'linear-gradient(to right, #b91c1c, #dc2626)', // Dark Red
            color: '#ef4444',
            lightColor: '#fca5a5',
            glassBorder: 'rgba(239, 68, 68, 0.3)',
            glassBg: 'rgba(239, 68, 68, 0.05)',
            Icon: TriangleAlert,
            glow: '0 0 15px rgba(220, 38, 38, 0.6)'
        },
        warning: {
            headerBg: 'linear-gradient(to right, #b45309, #d97706)', // Dark Amber
            color: '#f59e0b',
            lightColor: '#fcd34d',
            glassBorder: 'rgba(245, 158, 11, 0.3)',
            glassBg: 'rgba(245, 158, 11, 0.05)',
            Icon: Clock,
            glow: '0 0 10px rgba(217, 119, 6, 0.4)'
        },
        healthy: {
            headerBg: 'linear-gradient(to right, #047857, #059669)', // Dark Emerald
            color: '#10b981',
            lightColor: '#6ee7b7',
            glassBorder: 'rgba(16, 185, 129, 0.3)',
            glassBg: 'rgba(16, 185, 129, 0.05)',
            Icon: CheckCircle,
            glow: 'none'
        }
    };

    const style = styles[urgency];
    const progress = Math.min(100, (1 - (rulDays / maxRul)) * 100); // % Consumed

    // Next maintenance window calculation
    const getMaintenanceWindow = () => {
        if (rulDays < 3) return 'IMMEDIATE';
        if (rulDays < 7) return 'This Week';
        if (rulDays < 14) return 'Next 2 Weeks';
        if (rulDays < 30) return 'This Month';
        return 'As Scheduled';
    };

    return (
        <Card variant="outlined" sx={{ height: '100%', borderRadius: 2, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            {inTrainingDistribution === false && (
                <Box sx={{ px: 2, py: 1, bgcolor: '#fef3c7', borderBottom: 1, borderColor: '#f59e0b', display: 'flex', alignItems: 'center', gap: 1 }}>
                    <TriangleAlert size={16} className="text-amber-600" />
                    <Typography variant="caption" fontWeight="bold" sx={{ color: '#b45309' }}>Operating outside trained conditions</Typography>
                </Box>
            )}
            {/* Header */}
            <Box sx={{ px: 2, py: 1.5, background: style.headerBg, color: 'white' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Stack direction="row" spacing={1} alignItems="center">
                        <Activity size={18} strokeWidth={2.5} />
                        <Typography variant="caption" fontWeight="800" sx={{ textTransform: 'uppercase', letterSpacing: 1 }}>
                            DIAGNOSTICS
                        </Typography>
                    </Stack>
                    <Chip
                        icon={<style.Icon size={14} color="white" strokeWidth={3} />}
                        label={urgency.toUpperCase()}
                        size="small"
                        sx={{
                            bgcolor: 'rgba(0,0,0,0.2)',
                            color: 'white',
                            fontWeight: 'bold',
                            fontSize: '0.7rem',
                            height: 24,
                            border: '1px solid rgba(255,255,255,0.2)'
                        }}
                    />
                </Box>
            </Box>

            <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 2.5, p: { xs: 2, md: 2.5 } }}>

                {/* Top Section: Countdown & Risk Gauge */}
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <Box>
                        <Typography variant="caption" color="text.secondary" fontWeight="600" sx={{ letterSpacing: 0.5, textTransform: 'uppercase' }}>
                            Estimated RUL
                        </Typography>
                        <Stack alignItems="flex-start" sx={{ mt: 0.5 }}>
                            <Typography
                                variant="h2"
                                fontWeight="800"
                                sx={{
                                    fontFamily: '"JetBrains Mono", monospace',
                                    color: style.color,
                                    lineHeight: 0.9,
                                    letterSpacing: -2
                                }}
                            >
                                {rulDays < 1 ? '<24' : rulDays.toFixed(0)}
                                <Box component="span" sx={{ fontSize: '1rem', ml: 1, color: 'text.secondary', fontWeight: 500, letterSpacing: 'normal' }}>
                                    {rulDays < 1 ? 'HOURS' : 'DAYS'}
                                </Box>
                            </Typography>
                            <Typography variant="caption" sx={{ mt: 0.5, color: 'text.secondary', fontWeight: '500' }}>
                                Predicted Failure: <Box component="span" sx={{ color: 'text.primary', fontWeight: 'bold' }}>{dateString}</Box>
                            </Typography>
                            <Typography variant="caption" sx={{ mt: 0.25, fontSize: '0.7rem', fontWeight: '500' }} component="span" style={{ color: confidenceColor }}>
                                80% confident: {rulLower.toFixed(0)}–{rulUpper.toFixed(0)} days
                            </Typography>
                        </Stack>
                    </Box>

                    {/* Glowing Risk Gauge */}
                    <Box sx={{ position: 'relative', width: 72, height: 72 }}>
                        <svg width="72" height="72" style={{ transform: 'rotate(-90deg)' }}>
                            {/* Track */}
                            <circle cx="36" cy="36" r="30" stroke="#f1f5f9" strokeWidth="6" fill="none" />
                            {/* Fill */}
                            <circle
                                cx="36"
                                cy="36"
                                r="30"
                                stroke={style.color}
                                strokeWidth="6"
                                fill="none"
                                strokeDasharray={`${riskScore * 1.88} ${188 - riskScore * 1.88}`}
                                strokeLinecap="round"
                                style={{
                                    filter: urgency === 'critical' ? `drop-shadow(${style.glow})` : 'none',
                                    transition: 'stroke-dasharray 1s ease-out'
                                }}
                            />
                        </svg>
                        <Box sx={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                            <Typography variant="h5" fontWeight="bold" sx={{ color: 'text.primary', lineHeight: 1 }}>
                                {riskScore.toFixed(0)}
                            </Typography>
                            <Typography variant="caption" sx={{ fontSize: '0.6rem', fontWeight: 'bold', color: 'text.secondary' }}>RISK</Typography>
                        </Box>
                    </Box>
                </Box>

                {/* Shimmering Lifecycle Bar */}
                <Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                        <Typography variant="caption" fontWeight="bold" color="text.secondary" fontSize="0.7rem">LIFECYCLE CONSUMED</Typography>
                        <Typography variant="caption" fontWeight="bold" fontSize="0.7rem">{progress.toFixed(0)}%</Typography>
                    </Box>
                    <Box sx={{
                        position: 'relative',
                        height: 12,
                        bgcolor: '#f1f5f9',
                        borderRadius: 1,
                        overflow: 'hidden',
                        boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.05)'
                    }}>
                        {/* Active Bar with Shimmer */}
                        <Box sx={{
                            position: 'absolute',
                            left: 0, top: 0, bottom: 0,
                            width: `${progress}%`,
                            background: urgency === 'critical'
                                ? `linear-gradient(45deg, ${style.color} 25%, #ef4444 50%, ${style.color} 75%)`
                                : style.color,
                            backgroundSize: '200% 100%',
                            animation: urgency === 'critical' ? 'shimmer 2s infinite linear' : 'none',
                            '@keyframes shimmer': {
                                '0%': { backgroundPosition: '100% 0' },
                                '100%': { backgroundPosition: '-100% 0' }
                            }
                        }} />

                        {/* Confidence Interval Overlay (Visual approx) */}
                        {/* Only show if we have meaningful progress to overlay on */}
                        <Box sx={{
                            position: 'absolute',
                            left: `${Math.max(0, progress - 5)}%`,
                            width: '10%', // roughly representing CI width
                            top: 0, bottom: 0,
                            bgcolor: 'rgba(255,255,255,0.3)',
                            borderLeft: '1px solid rgba(255,255,255,0.5)',
                            borderRight: '1px solid rgba(255,255,255,0.5)'
                        }} />
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                        <Typography variant="caption" color="text.disabled" fontSize="0.6rem">Install</Typography>
                        <Typography variant="caption" color="text.disabled" fontSize="0.6rem">End of Life</Typography>
                    </Box>
                </Box>

                {/* Glassmorphic Metric Tiles */}
                <Grid container spacing={1.5}>
                    <MetricTile
                        label="FAIL PROB"
                        value={`${failProb.toFixed(1)}%`}
                        icon={Percent}
                        style={style}
                        warning={failProb > 80}
                    />
                    <MetricTile
                        label="WEAR LEVEL"
                        value={`${degradation.toFixed(0)}%`}
                        icon={Activity}
                        style={style}
                        warning={degradation > 80}
                    />
                    <MetricTile
                        label="SERVICE"
                        value={getMaintenanceWindow()}
                        icon={Calendar}
                        style={style}
                        warning={getMaintenanceWindow() === 'IMMEDIATE'}
                    />
                </Grid>

                {/* Horizontal Failure Distribution */}
                <Box>
                    <Typography variant="caption" fontWeight="bold" color="text.secondary" sx={{ display: 'block', mb: 1, textTransform: 'uppercase', fontSize: '0.65rem' }}>
                        Failure Distribution (Days)
                    </Typography>
                    <Box sx={{ display: 'flex', height: 8, borderRadius: 4, overflow: 'hidden' }}>
                        {/* Red Segment (0-7d) */}
                        <Box sx={{ flex: urgency === 'critical' ? 5 : 1, bgcolor: '#ef4444', borderRight: '1px solid white' }} />
                        {/* Amber Segment (7-30d) */}
                        <Box sx={{ flex: urgency === 'warning' ? 4 : 2, bgcolor: '#f59e0b', borderRight: '1px solid white' }} />
                        {/* Green Segment (30+d) */}
                        <Box sx={{ flex: urgency === 'healthy' ? 5 : 2, bgcolor: '#10b981' }} />
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                        <Typography variant="caption" color="error.main" fontWeight="bold" fontSize="0.6rem">0-7d</Typography>
                        <Typography variant="caption" color="warning.main" fontWeight="bold" fontSize="0.6rem">7-30d</Typography>
                        <Typography variant="caption" color="success.main" fontWeight="bold" fontSize="0.6rem">30d+</Typography>
                    </Box>
                </Box>

                {/* Why? collapsed section: top driver + expand for full ExplanationCard */}
                <Box sx={{ mt: 1 }}>
                    <Button
                        fullWidth
                        size="small"
                        variant="outlined"
                        onClick={() => setWhyExpanded((e) => !e)}
                        endIcon={whyExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                        sx={{ justifyContent: 'space-between', textTransform: 'none', fontWeight: 600 }}
                    >
                        Why?
                    </Button>
                    {topExplanation && !whyExpanded && (
                        <Stack direction="row" spacing={0.5} alignItems="center" sx={{ mt: 0.5 }}>
                            <TriangleAlert size={14} color="#d97706" />
                            <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500 }}>
                                Primary driver: {topExplanation.display_name || topExplanation.feature || '—'}
                            </Typography>
                        </Stack>
                    )}
                    <Collapse in={whyExpanded}>
                        <Box sx={{ mt: 1.5 }}>
                            <ExplanationCard
                                explanations={explanations}
                                healthScore={healthScore}
                                machineName={machineName}
                            />
                        </Box>
                    </Collapse>
                </Box>

                {/* Critical Action Button - Maintenance */}
                {urgency === 'critical' && (
                    <Box sx={{ mt: 'auto', pt: 1 }}>
                        <Button
                            fullWidth
                            variant="contained"
                            color="error"
                            startIcon={<TriangleAlert size={16} />}
                            onClick={() => onSchedule && onSchedule()}
                            sx={{
                                fontWeight: 'bold',
                                boxShadow: '0 4px 6px rgba(220, 38, 38, 0.3)',
                                animation: 'pulse 2s infinite'
                            }}
                        >
                            SCHEDULE IMMEDIATE MAINTENANCE
                        </Button>
                    </Box>
                )}

            </CardContent>
        </Card>
    );
}

function MetricTile({ label, value, icon: Icon, style, warning }) {
    return (
        <Grid item xs={4}>
            <Box sx={{
                p: 1.5,
                borderRadius: 2,
                bgcolor: style.glassBg,
                border: '1px solid',
                borderColor: style.glassBorder,
                backdropFilter: 'blur(4px)',
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center'
            }}>
                <Stack direction="row" spacing={0.5} alignItems="center" sx={{ mb: 0.5, opacity: 0.8 }}>
                    <Icon size={12} color={style.color} />
                    <Typography sx={{ fontSize: '0.6rem', fontWeight: '800', color: style.color }}>{label}</Typography>
                </Stack>
                <Typography variant="body2" fontWeight="bold" sx={{ color: '#1e293b', lineHeight: 1.1 }}>
                    {value}
                </Typography>
            </Box>
        </Grid>
    );
}

export default RULCard;
