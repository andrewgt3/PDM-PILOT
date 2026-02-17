import React, { useState } from 'react';
import { BarChart3, ChevronDown, ChevronUp, Info, Activity, Lightbulb, Zap } from 'lucide-react';
import { Card, Box, Typography, Collapse, LinearProgress, Stack, Grid, Chip } from '@mui/material';

/**
 * FeatureImportancePanel Component
 * Shows which features are driving the AI's failure prediction (Explainable AI)
 * Refactored for "Expert Analysis" suite.
 */
export function FeatureImportancePanel({ data, failureProbability }) {
    const [expandedFeature, setExpandedFeature] = useState(null);

    if (!data || data.length === 0) {
        return (
            <Card variant="outlined" sx={{ p: 3, borderRadius: 2 }}>
                <Typography variant="subtitle2" fontWeight="bold" textTransform="uppercase" color="text.secondary" gutterBottom>
                    Expert Analysis: Predictions
                </Typography>
                <Typography variant="body2" color="text.secondary">No telemetry data available</Typography>
            </Card>
        );
    }

    const latest = data[data.length - 1] || {};
    const prob = (failureProbability || latest.failure_prediction || 0) * 100;

    // Calculate feature importance based on actual values vs. thresholds
    const calculateImportance = (value, healthyBaseline, criticalThreshold) => {
        const range = criticalThreshold - healthyBaseline;
        const deviation = Math.abs(value - healthyBaseline);
        return Math.min(100, (deviation / range) * 100);
    };

    const features = [
        {
            name: 'BPFI Amplitude',
            value: latest.bpfi_amp || 0,
            importance: calculateImportance(latest.bpfi_amp || 0, 0, 0.5),
            detailedInfo: {
                whatIs: 'Ball Pass Frequency Inner - measures vibration amplitude at the frequency where rolling elements pass defects on the inner race.',
                whyImportant: 'High BPFI amplitude is a strong predictor of inner race bearing damage. The ML model weighs this heavily because inner race faults progress rapidly.',
                dataSource: 'Calculated from FFT of raw vibration signal at the bearing inner race fault frequency.',
                healthyRange: '< 0.05 g',
                warningRange: '0.05 - 0.3 g',
                criticalRange: '> 0.3 g',
                actions: ['Schedule bearing inspection', 'Check lubrication', 'Verify shaft alignment']
            },
            limit: 0.5
        },
        {
            name: 'Degradation Score',
            value: latest.degradation_score || 0,
            importance: (latest.degradation_score || 0) * 100,
            detailedInfo: {
                whatIs: 'A composite index (0-1) calculated from multiple sensor readings representing overall equipment health degradation.',
                whyImportant: 'The degradation score integrates multiple failure modes into a single metric. It is the strongest single predictor of remaining useful life.',
                dataSource: 'Computed from weighted combination of vibration statistics.',
                healthyRange: '< 0.3',
                warningRange: '0.3 - 0.7',
                criticalRange: '> 0.7',
                actions: ['Review all fault indicators', 'Plan preventive maintenance']
            },
            limit: 1.0
        },
        {
            name: 'Spectral Kurtosis',
            value: latest.spectral_kurtosis || 0,
            importance: calculateImportance(latest.spectral_kurtosis || 0, 3, 10),
            detailedInfo: {
                whatIs: 'Statistical measure of "peakedness" in the vibration signal. High kurtosis indicates impulsive events like bearing impacts.',
                whyImportant: 'Kurtosis rises before RMS vibration in early fault stages, making it a leading indicator of developing damage.',
                dataSource: 'Calculated from time-domain vibration signal.',
                healthyRange: '~3',
                warningRange: '4 - 7',
                criticalRange: '> 7',
                actions: ['Investigate source of impacts', 'Check for looseness']
            },
            limit: 10
        },
        {
            name: 'BPFO Amplitude',
            value: latest.bpfo_amp || 0,
            importance: calculateImportance(latest.bpfo_amp || 0, 0, 0.5),
            detailedInfo: {
                whatIs: 'Ball Pass Frequency Outer - measures vibration amplitude at the frequency where rolling elements pass defects on the outer race.',
                whyImportant: 'BPFO is the most common bearing fault. Outer race faults are often caused by contamination.',
                dataSource: 'Calculated from FFT of raw vibration signal.',
                healthyRange: '< 0.05 g',
                warningRange: '0.05 - 0.3 g',
                criticalRange: '> 0.3 g',
                actions: ['Inspect bearing for spalling', 'Check housing fit']
            },
            limit: 0.5
        },
        {
            name: 'High Band Power',
            value: latest.high_band_power || 0,
            importance: calculateImportance(latest.high_band_power || 0, 0, 1),
            detailedInfo: {
                whatIs: 'Total vibration energy in the 2000-6000 Hz frequency band.',
                whyImportant: 'High frequency energy often increases before lower frequency symptoms appear.',
                dataSource: 'Computed as RMS of FFT bins range.',
                healthyRange: '< 0.1',
                warningRange: '0.1 - 0.5',
                criticalRange: '> 0.5',
                actions: ['Monitor for progression', 'Check gearbox if applicable']
            },
            limit: 1.0
        }
    ];

    // Sort by importance descending
    features.sort((a, b) => b.importance - a.importance);
    const topFeatures = features.slice(0, 5);
    const maxImportance = Math.max(...topFeatures.map(f => f.importance), 1);

    const toggleExpand = (featureName) => {
        setExpandedFeature(expandedFeature === featureName ? null : featureName);
    };

    return (
        <Card variant="outlined" sx={{ p: 3, borderRadius: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Box>
                    <Typography variant="subtitle2" fontWeight="bold" textTransform="uppercase" color="text.secondary">
                        Expert Analysis: Predictions
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                        Explainable AI (XAI) Driver Analysis
                    </Typography>
                </Box>
                <BarChart3 size={20} className="text-slate-400" />
            </Box>

            <Stack spacing={2}>
                {topFeatures.map((feature, idx) => {
                    const percentage = (feature.importance / maxImportance) * 100;
                    const isHighImpact = percentage > 70;
                    const isExpanded = expandedFeature === feature.name;
                    const percOfLimit = Math.min((feature.value / feature.limit) * 100, 100);

                    // Gradient for the influence bar
                    const gradientClass = isHighImpact
                        ? 'linear-gradient(90deg, #ef4444 0%, #f97316 100%)' // Red-Orange
                        : idx === 0
                            ? 'linear-gradient(90deg, #6366f1 0%, #4f46e5 100%)' // Indigo
                            : 'linear-gradient(90deg, #94a3b8 0%, #64748b 100%)'; // Gray

                    return (
                        <Card key={feature.name} variant="outlined" sx={{ overflow: 'hidden', transition: 'box-shadow 0.2s', boxShadow: isExpanded ? 2 : 0 }}>
                            <Box
                                onClick={() => toggleExpand(feature.name)}
                                sx={{ p: 2, cursor: 'pointer', '&:hover': { bgcolor: 'action.hover' } }}
                            >
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                                    <Stack direction="row" spacing={2} alignItems="center">
                                        <Box sx={{
                                            width: 24, height: 24, borderRadius: '50%',
                                            bgcolor: 'slate.900', color: 'white',
                                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                                            fontSize: '0.75rem', fontWeight: 'bold', fontFamily: 'monospace'
                                        }}>
                                            {idx + 1}
                                        </Box>
                                        <Box>
                                            <Typography variant="body2" fontWeight="medium">{feature.name}</Typography>
                                            <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                                <Zap size={10} className={isHighImpact ? "text-amber-500" : "text-slate-400"} fill={isHighImpact ? "currentColor" : "none"} />
                                                Influence Weight
                                            </Typography>
                                        </Box>
                                    </Stack>
                                    <Stack direction="row" spacing={2} alignItems="center">
                                        <Typography variant="body2" fontWeight="bold" fontFamily="monospace" color={isHighImpact ? 'error.main' : 'text.secondary'}>
                                            {percentage.toFixed(0)}%
                                        </Typography>
                                        {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                                    </Stack>
                                </Box>
                                <Box sx={{ mt: 1 }}>
                                    {/* XAI Influence Bar */}
                                    <Box sx={{ position: 'relative', height: 8, borderRadius: 1, bgcolor: 'grey.100', overflow: 'hidden' }}>
                                        <Box sx={{
                                            height: '100%',
                                            width: `${percentage}%`,
                                            background: gradientClass,
                                            transition: 'width 0.5s ease',
                                            boxShadow: '0 0 8px rgba(99, 102, 241, 0.2)'
                                        }} />
                                    </Box>
                                </Box>
                            </Box>

                            <Collapse in={isExpanded}>
                                <Box sx={{ p: 2, bgcolor: 'grey.50', borderTop: 1, borderColor: 'divider' }}>
                                    <Grid container spacing={2}>
                                        <Grid item xs={12} md={7}>
                                            <Stack spacing={2}>
                                                <Box>
                                                    <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                                                        <Info size={14} className="text-indigo-500" />
                                                        <Typography variant="caption" fontWeight="bold" textTransform="uppercase" color="primary.main">AI Context</Typography>
                                                    </Stack>
                                                    <Typography variant="body2" color="text.secondary" fontSize="0.85rem" sx={{ lineHeight: 1.6 }}>
                                                        {feature.detailedInfo.whyImportant}
                                                    </Typography>
                                                </Box>
                                                <Box sx={{ p: 1.5, bgcolor: 'white', borderRadius: 1, border: 1, borderColor: 'divider' }}>
                                                    <Typography variant="caption" fontWeight="bold" textTransform="uppercase" display="block" gutterBottom color="text.secondary">Definition</Typography>
                                                    <Typography variant="caption" color="text.primary">{feature.detailedInfo.whatIs}</Typography>
                                                </Box>
                                            </Stack>
                                        </Grid>
                                        <Grid item xs={12} md={5}>
                                            {/* Digital Readout */}
                                            <Box sx={{ p: 2, bgcolor: 'slate.900', borderRadius: 2, color: 'white', boxShadow: 'inset 0 2px 10px rgba(0,0,0,0.5)' }}>
                                                <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1, opacity: 0.7 }}>
                                                    <Activity size={14} />
                                                    <Typography variant="caption" fontWeight="bold" textTransform="uppercase">Live Value</Typography>
                                                </Stack>

                                                <Typography variant="h4" fontFamily="monospace" fontWeight="bold" sx={{ color: percOfLimit > 80 ? '#f87171' : '#4ade80', textShadow: '0 0 10px rgba(74, 222, 128, 0.3)' }}>
                                                    {typeof feature.value === 'number' ? feature.value.toFixed(4) : feature.value}
                                                </Typography>

                                                <Box sx={{ mt: 2 }}>
                                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                                        <Typography variant="caption" sx={{ fontSize: '0.65rem', opacity: 0.7 }}>% of Limit</Typography>
                                                        <Typography variant="caption" sx={{ fontSize: '0.65rem', color: percOfLimit > 80 ? '#f87171' : '#4ade80' }}>{percOfLimit.toFixed(0)}%</Typography>
                                                    </Box>
                                                    <LinearProgress
                                                        variant="determinate"
                                                        value={percOfLimit}
                                                        sx={{
                                                            height: 4,
                                                            bgcolor: 'rgba(255,255,255,0.1)',
                                                            '& .MuiLinearProgress-bar': {
                                                                bgcolor: percOfLimit > 80 ? '#f87171' : '#4ade80'
                                                            }
                                                        }}
                                                    />
                                                </Box>
                                            </Box>

                                            <Box sx={{ mt: 2 }}>
                                                <Typography variant="caption" fontWeight="bold" textTransform="uppercase" display="block" gutterBottom color="text.secondary">Thresholds</Typography>
                                                <Stack direction="row" spacing={1}>
                                                    <Chip label={`Warn: ${feature.detailedInfo.warningRange}`} size="small" variant="outlined" sx={{ fontSize: '0.65rem', height: 20, borderColor: 'warning.main', color: 'warning.dark' }} />
                                                    <Chip label={`Crit: ${feature.detailedInfo.criticalRange}`} size="small" variant="outlined" sx={{ fontSize: '0.65rem', height: 20, borderColor: 'error.main', color: 'error.dark' }} />
                                                </Stack>
                                            </Box>
                                        </Grid>
                                    </Grid>
                                </Box>
                            </Collapse>
                        </Card>
                    );
                })}
            </Stack>
        </Card>
    );
}

export default FeatureImportancePanel;
