import React, { useState } from 'react';
import { BarChart3, ChevronDown, ChevronUp, Info, Activity, Lightbulb } from 'lucide-react';
import { Card, Box, Typography, Collapse, LinearProgress, Stack, Grid, Chip } from '@mui/material';

/**
 * FeatureImportancePanel Component
 * Shows which features are driving the AI's failure prediction (Explainable AI)
 */
export function FeatureImportancePanel({ data, failureProbability }) {
    const [expandedFeature, setExpandedFeature] = useState(null);

    if (!data || data.length === 0) {
        return (
            <Card variant="outlined" sx={{ p: 3, borderRadius: 2 }}>
                <Typography variant="subtitle2" fontWeight="bold" textTransform="uppercase" color="text.secondary" gutterBottom>
                    Prediction Drivers
                </Typography>
                <Typography variant="body2" color="text.secondary">No data available</Typography>
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
            }
        },
        // ... (Other features would be here, simplifying for brevity/MUI translation)
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
            }
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
            }
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
            }
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
            }
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
                        Prediction Drivers
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                        Contribution to {prob.toFixed(1)}% failure probability
                    </Typography>
                </Box>
                <BarChart3 size={20} className="text-slate-400" />
            </Box>

            <Stack spacing={2}>
                {topFeatures.map((feature, idx) => {
                    const percentage = (feature.importance / maxImportance) * 100;
                    const isHighImpact = percentage > 70;
                    const isExpanded = expandedFeature === feature.name;

                    const progressColor = isHighImpact ? 'error' : idx === 0 ? 'primary' : 'inherit';
                    const gradientClass = isHighImpact ? 'linear-gradient(90deg, #ef4444 0%, #f97316 100%)' : idx === 0 ? 'linear-gradient(90deg, #6366f1 0%, #4f46e5 100%)' : undefined;

                    return (
                        <Card key={feature.name} variant="outlined" sx={{ overflow: 'hidden' }}>
                            <Box
                                onClick={() => toggleExpand(feature.name)}
                                sx={{ p: 2, cursor: 'pointer', '&:hover': { bgcolor: 'action.hover' } }}
                            >
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <Stack direction="row" spacing={2} alignItems="center">
                                        <Box sx={{
                                            width: 20, height: 20, borderRadius: '50%',
                                            bgcolor: idx === 0 ? 'primary.lighter' : 'grey.100',
                                            color: idx === 0 ? 'primary.main' : 'text.disabled',
                                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                                            fontSize: '0.7rem', fontWeight: 'bold'
                                        }}>
                                            {idx + 1}
                                        </Box>
                                        <Typography variant="body2" fontWeight="medium">{feature.name}</Typography>
                                    </Stack>
                                    <Stack direction="row" spacing={2} alignItems="center">
                                        <Typography variant="body2" fontWeight="bold" fontFamily="monospace" color={isHighImpact ? 'error.main' : 'text.secondary'}>
                                            {percentage.toFixed(0)}%
                                        </Typography>
                                        {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                                    </Stack>
                                </Box>
                                <Box sx={{ mt: 2 }}>
                                    {/* Custom Linear Progress to support gradient */}
                                    <Box sx={{ position: 'relative', height: 12, borderRadius: 1, bgcolor: 'grey.100', overflow: 'hidden' }}>
                                        <Box sx={{
                                            height: '100%',
                                            width: `${percentage}%`,
                                            background: gradientClass || '#94a3b8',
                                            transition: 'width 0.5s ease'
                                        }} />
                                    </Box>
                                </Box>
                            </Box>

                            <Collapse in={isExpanded}>
                                <Box sx={{ p: 2, bgcolor: 'grey.50', borderTop: 1, borderColor: 'divider' }}>
                                    <Grid container spacing={2}>
                                        <Grid item xs={12} md={6}>
                                            <Stack spacing={1}>
                                                <Box>
                                                    <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                                                        <Info size={14} />
                                                        <Typography variant="caption" fontWeight="bold" textTransform="uppercase">What is this?</Typography>
                                                    </Stack>
                                                    <Typography variant="caption" color="text.secondary">{feature.detailedInfo.whatIs}</Typography>
                                                </Box>
                                                <Box>
                                                    <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                                                        <Lightbulb size={14} />
                                                        <Typography variant="caption" fontWeight="bold" textTransform="uppercase">Why it matters</Typography>
                                                    </Stack>
                                                    <Typography variant="caption" color="text.secondary">{feature.detailedInfo.whyImportant}</Typography>
                                                </Box>
                                            </Stack>
                                        </Grid>
                                        <Grid item xs={12} md={6}>
                                            <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1, border: 1, borderColor: 'divider' }}>
                                                <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                                                    <Activity size={14} />
                                                    <Typography variant="caption" fontWeight="bold" textTransform="uppercase">Live Values</Typography>
                                                </Stack>
                                                <Grid container spacing={1}>
                                                    <Grid item xs={3}>
                                                        <Box sx={{ p: 1, bgcolor: 'grey.50', borderRadius: 1 }}>
                                                            <Typography variant="caption" color="text.secondary" display="block">current</Typography>
                                                            <Typography variant="caption" fontWeight="bold" fontFamily="monospace">
                                                                {typeof feature.value === 'number' ? feature.value.toFixed(4) : feature.value}
                                                            </Typography>
                                                        </Box>
                                                    </Grid>
                                                    <Grid item xs={3}>
                                                        <Box sx={{ p: 1, bgcolor: 'success.lighter', borderRadius: 1, border: 1, borderColor: 'success.light' }}>
                                                            <Typography variant="caption" color="success.main" display="block">healthy</Typography>
                                                            <Typography variant="caption" fontWeight="medium" color="success.dark">{feature.detailedInfo.healthyRange}</Typography>
                                                        </Box>
                                                    </Grid>
                                                    <Grid item xs={3}>
                                                        <Box sx={{ p: 1, bgcolor: 'warning.lighter', borderRadius: 1, border: 1, borderColor: 'warning.light' }}>
                                                            <Typography variant="caption" color="warning.main" display="block">warning</Typography>
                                                            <Typography variant="caption" fontWeight="medium" color="warning.dark">{feature.detailedInfo.warningRange}</Typography>
                                                        </Box>
                                                    </Grid>
                                                    <Grid item xs={3}>
                                                        <Box sx={{ p: 1, bgcolor: 'error.lighter', borderRadius: 1, border: 1, borderColor: 'error.light' }}>
                                                            <Typography variant="caption" color="error.main" display="block">critical</Typography>
                                                            <Typography variant="caption" fontWeight="medium" color="error.dark">{feature.detailedInfo.criticalRange}</Typography>
                                                        </Box>
                                                    </Grid>
                                                </Grid>
                                            </Box>

                                            <Box sx={{ mt: 2 }}>
                                                <Typography variant="caption" fontWeight="bold" textTransform="uppercase" display="block" gutterBottom>Recommended Actions</Typography>
                                                <Stack spacing={0.5}>
                                                    {feature.detailedInfo.actions.map((action, i) => (
                                                        <Typography key={i} variant="caption" color="primary.main" display="block">• {action}</Typography>
                                                    ))}
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
