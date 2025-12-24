import React, { useState, useEffect } from 'react';
import { Wrench, Calendar, Clock, AlertTriangle, CheckCircle, Sparkles, RefreshCw, Shield } from 'lucide-react';
import { Card, CardContent, Typography, Box, Button, Chip, Stack, Alert, Grid, IconButton, CircularProgress, Divider } from '@mui/material';

/**
 * MaintenanceRecommendationCard Component
 * Provides AI-powered maintenance recommendations via LLM API
 * Falls back to rule-based logic if AI is unavailable
 */
export function MaintenanceRecommendationCard({ machine, latestData }) {
    const [recommendation, setRecommendation] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [isAI, setIsAI] = useState(false);

    const machineId = machine?.machine_id;

    // Fetch AI recommendation from API
    const fetchRecommendation = async () => {
        if (!machineId) return;

        setLoading(true);
        setError(null);

        try {
            const response = await fetch(`http://localhost:8000/api/recommendations/${machineId}`);

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            const data = await response.json();
            setRecommendation(data);
            setIsAI(data.aiGenerated !== false);
        } catch (err) {
            console.error('[AI Recommendation Error]', err);
            setError(err.message);
            // Fall back to rule-based recommendation
            setRecommendation(getRuleBasedRecommendation());
            setIsAI(false);
        } finally {
            setLoading(false);
        }
    };

    // Fetch once on machine change
    useEffect(() => {
        fetchRecommendation();
    }, [machineId]);

    // Fallback rule-based recommendation (if API fails)
    const getRuleBasedRecommendation = () => {
        const prob = (machine?.failure_probability || 0) * 100;

        if (prob > 90) {
            return {
                priority: 'CRITICAL',
                action: 'Immediate bearing assembly replacement required',
                reasoning: 'Multiple bearing defects detected with high confidence based on BPFI and BPFO amplitudes.',
                timeWindow: 'Within 24 hours',
                parts: ['Bearing assembly', 'Shaft seal', 'Lubricant'],
                estimatedDowntime: '4-6 hours',
                safetyNotes: 'Lockout/tagout required. High-energy equipment.',
            };
        } else if (prob > 70) {
            return {
                priority: 'HIGH',
                action: 'Schedule bearing inspection and replacement',
                reasoning: 'Developing bearing fault on inner race detected via envelope analysis.',
                timeWindow: 'Within 3-5 days',
                parts: ['Bearing assembly', 'Vibration sensor recalibration'],
                estimatedDowntime: '3-4 hours',
                safetyNotes: 'Standard maintenance precautions.',
            };
        } else if (prob > 50) {
            return {
                priority: 'MEDIUM',
                action: 'Monitor closely - schedule preventive maintenance',
                reasoning: 'Elevated vibration levels detected. Condition trending toward fault.',
                timeWindow: 'Next scheduled maintenance window',
                parts: ['Inspection', 'Lubrication'],
                estimatedDowntime: '2-3 hours',
                safetyNotes: 'Standard procedures.',
            };
        } else {
            return {
                priority: 'LOW',
                action: 'Continue routine monitoring',
                reasoning: 'Machine operating within normal parameters. No immediate action required.',
                timeWindow: 'Standard maintenance schedule',
                parts: [],
                estimatedDowntime: 'N/A',
                safetyNotes: 'None',
            };
        }
    };

    // Use fallback if no AI recommendation
    const rec = recommendation || getRuleBasedRecommendation();

    // Priority styling
    const getPriorityColor = (priority) => {
        switch (priority?.toUpperCase()) {
            case 'CRITICAL': return 'error';
            case 'HIGH': return 'warning';
            case 'MEDIUM': return 'warning'; // Amber mapped to warning
            default: return 'success';
        }
    };

    const color = getPriorityColor(rec.priority);
    const PriorityIcon = ['CRITICAL', 'HIGH'].includes(rec.priority?.toUpperCase()) ? AlertTriangle : CheckCircle;

    return (
        <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
            {/* Header */}
            <Box sx={{ px: 2, py: 1.5, bgcolor: `${color}.lighter`, borderBottom: 1, borderColor: `${color}.light`, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Stack direction="row" spacing={1} alignItems="center">
                    <Wrench className={`text-${color}-main`} size={20} />
                    <Typography variant="subtitle2" fontWeight="bold" sx={{ textTransform: 'uppercase', color: `${color}.main` }}>
                        Maintenance Recommendation
                    </Typography>
                    {isAI && (
                        <Chip
                            icon={<Sparkles size={12} />}
                            label="AI Generated"
                            size="small"
                            color="primary"
                            variant="outlined"
                            sx={{ height: 20, fontSize: '0.65rem' }}
                        />
                    )}
                </Stack>
                <Stack direction="row" spacing={1}>
                    <IconButton size="small" onClick={fetchRecommendation} disabled={loading}>
                        <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
                    </IconButton>
                    <Chip
                        label={rec.priority || 'UNKNOWN'}
                        color={color}
                        size="small"
                        sx={{ fontWeight: 'bold' }}
                    />
                </Stack>
            </Box>

            <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {loading && (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, color: 'text.secondary' }}>
                        <CircularProgress size={16} />
                        <Typography variant="body2">Analyzing diagnostic data...</Typography>
                    </Box>
                )}

                {/* AI Reasoning */}
                {!loading && rec.reasoning && (
                    <Box>
                        <Typography variant="caption" fontWeight="bold" color="text.secondary" textTransform="uppercase">
                            AI Analysis
                        </Typography>
                        <Typography variant="body2" sx={{ mt: 0.5, lineHeight: 1.6 }}>
                            {rec.reasoning}
                        </Typography>
                    </Box>
                )}

                {/* Recommended Action */}
                {!loading && (
                    <Alert severity={color} icon={<Wrench size={18} />} sx={{ alignItems: 'center' }}>
                        <Typography variant="body2" fontWeight="bold">
                            {rec.action}
                        </Typography>
                    </Alert>
                )}

                {/* Details Grid */}
                {!loading && (
                    <Grid container spacing={2}>
                        <Grid item xs={6}>
                            <Box>
                                <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                                    <Calendar size={14} className="text-slate-500" />
                                    <Typography variant="caption" fontWeight="bold" color="text.secondary">TIME WINDOW</Typography>
                                </Stack>
                                <Typography variant="body2" fontWeight="medium">{rec.timeWindow}</Typography>
                            </Box>
                        </Grid>
                        <Grid item xs={6}>
                            <Box>
                                <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                                    <Clock size={14} className="text-slate-500" />
                                    <Typography variant="caption" fontWeight="bold" color="text.secondary">EST. DOWNTIME</Typography>
                                </Stack>
                                <Typography variant="body2" fontWeight="medium">{rec.estimatedDowntime}</Typography>
                            </Box>
                        </Grid>
                    </Grid>
                )}

                {!loading && rec.safetyNotes && (
                    <Box>
                        <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                            <Shield size={14} className="text-slate-500" />
                            <Typography variant="caption" fontWeight="bold" color="text.secondary">SAFETY NOTES</Typography>
                        </Stack>
                        <Typography variant="body2" color="text.secondary">{rec.safetyNotes}</Typography>
                    </Box>
                )}

                {/* Parts */}
                {!loading && rec.parts && rec.parts.length > 0 && (
                    <Box>
                        <Typography variant="caption" fontWeight="bold" color="text.secondary" display="block" sx={{ mb: 1 }}>REQUIRED PARTS</Typography>
                        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                            {rec.parts.map((part, idx) => (
                                <Chip key={idx} label={part} size="small" variant="outlined" />
                            ))}
                        </Stack>
                    </Box>
                )}

                {/* Action Buttons for Critical */}
                {!loading && rec.priority === 'CRITICAL' && (
                    <Stack spacing={1} sx={{ mt: 1 }}>
                        <Button variant="contained" color="error" fullWidth startIcon={<Wrench size={16} />}>
                            Create Urgent Work Order
                        </Button>
                        <Button variant="outlined" color="inherit" fullWidth>
                            Notify Maintenance Team
                        </Button>
                    </Stack>
                )}

                {/* Provider Footer */}
                {!loading && rec.llmProvider && (
                    <Typography variant="caption" color="text.disabled" align="right" display="block">
                        Powered by {rec.llmProvider === 'azure' ? 'Azure OpenAI' : rec.llmProvider === 'ollama' ? 'Local AI' : 'OpenAI'}
                    </Typography>
                )}
            </CardContent>
        </Card>
    );
}

export default MaintenanceRecommendationCard;
