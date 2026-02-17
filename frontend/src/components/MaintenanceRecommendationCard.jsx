import React, { useState, useEffect } from 'react';
import { Wrench, Calendar, Clock, AlertTriangle, CheckCircle, Sparkles, RefreshCw, Shield, Zap, Box as BoxIcon, ExternalLink } from 'lucide-react';
import { Card, CardContent, Typography, Box, Button, Chip, Stack, Alert, Grid, IconButton, CircularProgress, Divider, Tooltip } from '@mui/material';

/**
 * MaintenanceRecommendationCard (Action Center)
 * 
 * High-utility tile for maintenance actions.
 * Features:
 * - "System Protocol" fallback UI
 * - Integrated Logistics Bar
 * - High-vis Safety Section
 * - Inventory Integration
 */
export function MaintenanceRecommendationCard({ machine, latestData }) {
    const [recommendation, setRecommendation] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [isAI, setIsAI] = useState(false);
    // Mock timestamp for "Last Analyzed"
    const [lastAnalyzed, setLastAnalyzed] = useState('Just now');

    const machineId = machine?.machine_id;

    // Fetch AI recommendation from API
    const fetchRecommendation = async () => {
        if (!machineId) return;

        setLoading(true);
        setError(null);
        setLastAnalyzed('Updating...');

        try {
            const response = await fetch(`http://localhost:8000/api/recommendations/${machineId}`);

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            const data = await response.json();
            setRecommendation(data);
            setIsAI(data.aiGenerated !== false);
            setLastAnalyzed('Just now');
        } catch (err) {
            console.error('[AI Recommendation Error]', err);
            setError(err.message);
            // Fall back to rule-based recommendation
            setRecommendation(getRuleBasedRecommendation());
            setIsAI(false);
            setLastAnalyzed('2m ago'); // Fallback mock
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
                reasoning: 'Multiple bearing defects detected with high confidence based on BPFI and BPFO amplitudes. Vibration energy exceeds ISO 10816-3 Zone D limits.',
                timeWindow: 'Within 24 hours',
                parts: ['Bearing Assembly (SKF-6205)', 'Shaft Seal', 'Synth. Lubricant'],
                estimatedDowntime: '4-6 hours',
                safetyNotes: 'LOCKOUT/TAGOUT REQUIRED. High rotational energy hazard. Ensure zero energy state before casing removal.',
            };
        } else if (prob > 70) {
            return {
                priority: 'HIGH',
                action: 'Schedule bearing inspection & replacement',
                reasoning: 'Developing bearing fault on inner race detected via envelope analysis. Harmonic cursor indicates early spalling.',
                timeWindow: 'Within 3-5 days',
                parts: ['Bearing Assembly (SKF-6205)', 'Vibration Sensor Kit'],
                estimatedDowntime: '3-4 hours',
                safetyNotes: 'Standard maintenance precautions apply. Wear hearing protection.',
            };
        } else if (prob > 50) {
            return {
                priority: 'MEDIUM',
                action: 'Monitor trend & schedule preventive maintenance',
                reasoning: 'Elevated vibration levels detected. Condition trending toward fault. Lubrication degradation suspected.',
                timeWindow: 'Next scheduled window',
                parts: ['Lubricant', 'Seal Kit'],
                estimatedDowntime: '2-3 hours',
                safetyNotes: 'Standard PPE required.',
            };
        } else {
            return {
                priority: 'LOW',
                action: 'Continue routine monitoring protocol',
                reasoning: 'Machine operating within normal parameters. No significant anomalies detected in spectral data.',
                timeWindow: 'Standard Interval',
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
            case 'MEDIUM': return 'warning'; // Amber
            default: return 'success';
        }
    };

    const color = getPriorityColor(rec.priority);
    // const PriorityIcon = ['CRITICAL', 'HIGH'].includes(rec.priority?.toUpperCase()) ? AlertTriangle : CheckCircle;

    return (
        <Card variant="outlined" sx={{ height: '100%', borderRadius: 3, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            {/* Action Center Header */}
            <Box sx={{
                px: 2, py: 1.5,
                bgcolor: 'background.paper',
                borderBottom: 1,
                borderColor: 'divider',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
            }}>
                <Stack direction="row" spacing={1.5} alignItems="center">
                    <Box sx={{
                        p: 0.5, borderRadius: 1.5,
                        bgcolor: `${color}.light`,
                        color: `${color}.main`,
                        display: 'flex', alignItems: 'center', justifyContent: 'center'
                    }}>
                        <Zap size={18} strokeWidth={2.5} fill="currentColor" fillOpacity={0.2} />
                    </Box>
                    <Typography variant="subtitle2" fontWeight="800" sx={{ textTransform: 'uppercase', letterSpacing: 0.5 }}>
                        Action Center
                    </Typography>
                </Stack>

                <Stack direction="row" spacing={1.5} alignItems="center">
                    <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 500 }}>
                        Last Analyzed: {lastAnalyzed}
                    </Typography>
                    <IconButton size="small" onClick={fetchRecommendation} disabled={loading} sx={{ p: 0.5, border: 1, borderColor: 'divider' }}>
                        <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
                    </IconButton>
                    <Chip
                        label={rec.priority || 'UNKNOWN'}
                        color={color}
                        size="small"
                        sx={{
                            fontWeight: '800',
                            height: 24,
                            boxShadow: rec.priority === 'MEDIUM' || rec.priority === 'HIGH' ? '0 0 8px rgba(245, 158, 11, 0.4)' : 'none'
                        }}
                    />
                </Stack>
            </Box>

            <CardContent sx={{ p: 0, flex: 1, display: 'flex', flexDirection: 'column' }}>
                {loading ? (
                    <Box sx={{ p: 4, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 2, color: 'text.secondary', height: '100%' }}>
                        <CircularProgress size={20} thickness={5} />
                        <Typography variant="body2" fontWeight="medium">Running protocols...</Typography>
                    </Box>
                ) : (
                    <>
                        {/* Protocol Box (Analysis) */}
                        <Box sx={{ p: 2, pb: 0 }}>
                            <Box sx={{
                                p: 2,
                                borderRadius: 2,
                                bgcolor: isAI ? 'primary.50' : 'grey.50',
                                border: '1px solid',
                                borderColor: isAI ? 'primary.100' : 'grey.200',
                                position: 'relative'
                            }}>
                                <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                                    {isAI ? <Sparkles size={14} className="text-blue-600" /> : <BoxIcon size={14} className="text-slate-500" />}
                                    <Typography variant="caption" fontWeight="bold" color="text.secondary" fontFamily="monospace">
                                        {isAI ? 'AI DIAGNOSTIC ENGINE' : 'STANDARD INSPECTION PROTOCOL ENGAGED'}
                                    </Typography>
                                </Stack>
                                <Typography variant="body2" sx={{
                                    fontFamily: '"JetBrains Mono", monospace',
                                    fontSize: '0.8rem',
                                    lineHeight: 1.5,
                                    color: 'text.primary'
                                }}>
                                    {rec.reasoning}
                                </Typography>
                            </Box>
                        </Box>

                        {/* Action Text */}
                        <Box sx={{ px: 2, py: 2 }}>
                            <Typography variant="subtitle1" fontWeight="bold" sx={{ lineHeight: 1.3 }}>
                                {rec.action}
                            </Typography>
                        </Box>

                        {/* Inventory & Logistics */}
                        <Box sx={{ mt: 'auto' }}>
                            {/* Logistics Bar */}
                            <Box sx={{
                                display: 'flex',
                                borderTop: 1,
                                borderBottom: 1,
                                borderColor: 'divider',
                                bgcolor: 'grey.50'
                            }}>
                                <Box sx={{ flex: 1, p: 1.5, borderRight: 1, borderColor: 'divider' }}>
                                    <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                                        <Calendar size={14} className="text-slate-500" />
                                        <Typography variant="caption" fontWeight="bold" color="text.secondary">TIME WINDOW</Typography>
                                    </Stack>
                                    <Typography variant="body2" fontWeight="medium">{rec.timeWindow}</Typography>
                                </Box>
                                <Box sx={{ flex: 1, p: 1.5 }}>
                                    <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                                        <Clock size={14} className="text-slate-500" />
                                        <Typography variant="caption" fontWeight="bold" color="text.secondary">EST. DOWNTIME</Typography>
                                    </Stack>
                                    <Typography variant="body2" fontWeight="medium">{rec.estimatedDowntime}</Typography>
                                </Box>
                            </Box>

                            {/* Safety Section */}
                            {rec.safetyNotes && (
                                <Box sx={{ px: 2, py: 1.5, bgcolor: '#fefce8', borderBottom: 1, borderColor: '#fde047' }}>
                                    <Stack direction="row" spacing={1.5} alignItems="flex-start">
                                        <Shield size={16} className="text-amber-600" style={{ marginTop: 2 }} />
                                        <Box>
                                            <Typography variant="caption" fontWeight="bold" sx={{ color: 'warning.dark' }}>SAFETY NOTES</Typography>
                                            <Typography variant="body2" sx={{ fontSize: '0.8rem', color: 'warning.dark', lineHeight: 1.4 }}>
                                                {rec.safetyNotes}
                                            </Typography>
                                        </Box>
                                    </Stack>
                                </Box>
                            )}

                            {/* Required Parts */}
                            {rec.parts && rec.parts.length > 0 && (
                                <Box sx={{ px: 2, py: 2 }}>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                                        <Typography variant="caption" fontWeight="bold" color="text.secondary">REQUIRED PARTS</Typography>
                                        <Stack direction="row" spacing={0.5} alignItems="center" sx={{ cursor: 'pointer', '&:hover': { textDecoration: 'underline' } }}>
                                            <ExternalLink size={12} className="text-blue-600" />
                                            <Typography variant="caption" fontWeight="bold" color="primary">View Order</Typography>
                                        </Stack>
                                    </Box>
                                    <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                                        {rec.parts.map((part, idx) => (
                                            <Chip
                                                key={idx}
                                                label={part}
                                                size="small"
                                                variant="outlined"
                                                icon={<Box sx={{ width: 6, height: 6, borderRadius: '50%', bgcolor: 'success.main', ml: 1 }} />}
                                                sx={{ pl: 0.5, borderColor: 'divider' }}
                                            />
                                        ))}
                                    </Stack>
                                </Box>
                            )}

                            {/* CTA Button */}
                            <Box sx={{ p: 2, pt: 0 }}>
                                <Button
                                    variant="contained"
                                    color={color === 'default' ? 'primary' : color}
                                    fullWidth
                                    size="large"
                                    startIcon={<Wrench size={18} />}
                                    sx={{
                                        fontWeight: 'bold',
                                        boxShadow: 2,
                                        textTransform: 'none',
                                        py: 1.2
                                    }}
                                >
                                    {rec.priority === 'CRITICAL' ? 'Create Urgent Work Order' : 'Schedule Routine Inspection'}
                                </Button>
                            </Box>
                        </Box>
                    </>
                )}
            </CardContent>
        </Card>
    );
}

export default MaintenanceRecommendationCard;
