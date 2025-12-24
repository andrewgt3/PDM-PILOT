import React, { useState } from 'react';
import { AlertOctagon, AlertTriangle, CheckCircle, ChevronDown, ChevronUp, Info, Activity } from 'lucide-react';
import { Card, Box, Typography, Collapse, LinearProgress, Stack, Grid, IconButton, Chip } from '@mui/material';

/**
 * BearingFaultPanel Component
 * Displays bearing fault frequency amplitudes with expandable detail sections
 */
export function BearingFaultPanel({ data }) {
    const [expandedFault, setExpandedFault] = useState(null);

    if (!data || data.length === 0) {
        return (
            <Card variant="outlined" sx={{ p: 3, borderRadius: 2 }}>
                <Typography variant="subtitle2" fontWeight="bold" textTransform="uppercase" color="text.secondary" gutterBottom>
                    Bearing Fault Diagnostics
                </Typography>
                <Typography variant="body2" color="text.secondary">No data available</Typography>
            </Card>
        );
    }

    // Get latest reading
    const latest = data[data.length - 1] || {};

    const faults = [
        {
            name: 'BPFO',
            label: 'Ball Pass Outer Race',
            value: latest.bpfo_amp || 0,
            threshold: 0.3,
            description: 'Outer race defect indicator',
            detailedInfo: {
                whatIs: 'The frequency at which rolling elements pass over a defect on the outer race of the bearing.',
                causes: ['Contamination', 'Improper lubrication', 'Misalignment', 'Overloading', 'Fatigue'],
                actions: ['Inspect bearing for spalling', 'Check lubrication', 'Verify alignment', 'Schedule replacement if wear is advanced'],
                formula: 'BPFO = (N/2) × (1 - Bd/Pd × cos(θ)) × RPM/60'
            },
            dataFields: { bpfo_amp: latest.bpfo_amp, rpm: latest.rotational_speed }
        },
        {
            name: 'BPFI',
            label: 'Ball Pass Inner Race',
            value: latest.bpfi_amp || 0,
            threshold: 0.3,
            description: 'Inner race defect indicator',
            detailedInfo: {
                whatIs: 'The frequency at which rolling elements pass over a defect on the inner race of the bearing.',
                causes: ['Tight fit on shaft', 'Improper installation', 'Contamination', 'Fatigue', 'Excessive load'],
                actions: ['Check inner race for pitting', 'Verify proper shaft fit', 'Assess lubricant', 'Plan replacement'],
                formula: 'BPFI = (N/2) × (1 + Bd/Pd × cos(θ)) × RPM/60'
            },
            dataFields: { bpfi_amp: latest.bpfi_amp, rpm: latest.rotational_speed }
        },
        {
            name: 'BSF',
            label: 'Ball Spin Frequency',
            value: latest.bsf_amp || 0,
            threshold: 0.25,
            description: 'Ball/roller element defect indicator',
            detailedInfo: {
                whatIs: 'The frequency at which a defect on a rolling element contacts the inner or outer race.',
                causes: ['Ball fatigue', 'Contaminated lubricant', 'Manufacturing defects', 'Overloading'],
                actions: ['Inspect rolling elements', 'Check for debris', 'Replace bearing assembly'],
                formula: 'BSF = (Pd/(2×Bd)) × (1 - (Bd/Pd × cos(θ))²) × RPM/60'
            },
            dataFields: { bsf_amp: latest.bsf_amp, rpm: latest.rotational_speed }
        },
        {
            name: 'FTF',
            label: 'Fundamental Train Frequency',
            value: latest.ftf_amp || 0,
            threshold: 0.2,
            description: 'Cage/retainer defect indicator',
            detailedInfo: {
                whatIs: 'The rotational frequency of the bearing cage (retainer) that holds the rolling elements.',
                causes: ['Cage wear', 'Improper lubrication', 'Cage pocket damage', 'Bearing overheating'],
                actions: ['Check cage integrity', 'Ensure proper lubrication', 'Monitor progression', 'Replace if damaged'],
                formula: 'FTF = (1/2) × (1 - Bd/Pd × cos(θ)) × RPM/60'
            },
            dataFields: { ftf_amp: latest.ftf_amp, rpm: latest.rotational_speed }
        }
    ];

    const getStatus = (value, threshold) => {
        if (value > threshold * 1.5) return { level: 'critical', color: 'error', Icon: AlertOctagon };
        if (value > threshold) return { level: 'warning', color: 'warning', Icon: AlertTriangle };
        return { level: 'healthy', color: 'success', Icon: CheckCircle };
    };

    const maxValue = Math.max(...faults.map(f => f.value), 0.5);

    const toggleExpand = (faultName) => {
        setExpandedFault(expandedFault === faultName ? null : faultName);
    };

    return (
        <Card variant="outlined" sx={{ p: 3, borderRadius: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Box>
                    <Typography variant="subtitle2" fontWeight="bold" textTransform="uppercase" color="text.secondary">
                        Bearing Fault Diagnostics
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                        Click any fault type for detailed analysis
                    </Typography>
                </Box>
                <Stack direction="row" spacing={1}>
                    <Chip label="Healthy" size="small" color="success" sx={{ height: 20, fontSize: '0.65rem' }} />
                    <Chip label="Warning" size="small" color="warning" sx={{ height: 20, fontSize: '0.65rem' }} />
                    <Chip label="Critical" size="small" color="error" sx={{ height: 20, fontSize: '0.65rem' }} />
                </Stack>
            </Box>

            <Stack spacing={2}>
                {faults.map((fault) => {
                    const status = getStatus(fault.value, fault.threshold);
                    const percentage = (fault.value / maxValue) * 100;
                    const isExpanded = expandedFault === fault.name;

                    return (
                        <Card key={fault.name} variant="outlined" sx={{ overflow: 'hidden' }}>
                            {/* Header */}
                            <Box
                                onClick={() => toggleExpand(fault.name)}
                                sx={{ p: 2, cursor: 'pointer', '&:hover': { bgcolor: 'action.hover' } }}
                            >
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <Stack direction="row" spacing={2} alignItems="center">
                                        <status.Icon size={18} className={`text-${status.color}-main`} style={{
                                            color: status.color === 'error' ? '#d32f2f' : status.color === 'warning' ? '#ed6c02' : '#2e7d32'
                                        }} />
                                        <Box>
                                            <Typography variant="subtitle2" fontWeight="bold">{fault.name}</Typography>
                                            <Typography variant="caption" color="text.secondary">{fault.label}</Typography>
                                        </Box>
                                    </Stack>
                                    <Stack direction="row" spacing={2} alignItems="center">
                                        <Stack direction="row" alignItems="baseline" spacing={0.5}>
                                            <Typography variant="body2" fontWeight="bold" fontFamily="monospace" sx={{ color: `${status.color}.main` }}>
                                                {fault.value.toFixed(4)}
                                            </Typography>
                                            <Typography variant="caption" color="text.secondary">g</Typography>
                                        </Stack>
                                        {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                                    </Stack>
                                </Box>

                                {/* Progress */}
                                <Box sx={{ mt: 2, position: 'relative' }}>
                                    <LinearProgress
                                        variant="determinate"
                                        value={percentage}
                                        color={status.color}
                                        sx={{ height: 8, borderRadius: 1, bgcolor: 'grey.100' }}
                                    />
                                    {/* Threshold Line */}
                                    <Box sx={{ position: 'absolute', top: -2, bottom: -2, width: 2, bgcolor: 'text.secondary', zIndex: 1, left: `${(fault.threshold / maxValue) * 100}%`, opacity: 0.5 }} />
                                </Box>
                            </Box>

                            {/* Details */}
                            <Collapse in={isExpanded}>
                                <Box sx={{ p: 2, bgcolor: 'grey.50', borderTop: 1, borderColor: 'divider' }}>
                                    <Stack spacing={2}>
                                        <Box>
                                            <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                                                <Info size={14} />
                                                <Typography variant="caption" fontWeight="bold" textTransform="uppercase">What is {fault.name}?</Typography>
                                            </Stack>
                                            <Typography variant="body2" color="text.secondary" fontSize="0.85rem">
                                                {fault.detailedInfo.whatIs}
                                            </Typography>
                                        </Box>

                                        <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1, border: 1, borderColor: 'divider' }}>
                                            <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                                                <Activity size={14} />
                                                <Typography variant="caption" fontWeight="bold" textTransform="uppercase">Live Data Values</Typography>
                                            </Stack>
                                            <Grid container spacing={1}>
                                                {Object.entries(fault.dataFields).map(([key, val]) => (
                                                    <Grid item xs={6} key={key}>
                                                        <Box sx={{ display: 'flex', justifyContent: 'space-between', bgcolor: 'grey.50', p: 1, borderRadius: 1 }}>
                                                            <Typography variant="caption" fontFamily="monospace" color="text.secondary">{key}</Typography>
                                                            <Typography variant="caption" fontFamily="monospace" fontWeight="bold">
                                                                {typeof val === 'number' ? val.toFixed(4) : val || 'N/A'}
                                                            </Typography>
                                                        </Box>
                                                    </Grid>
                                                ))}
                                            </Grid>
                                        </Box>

                                        <Grid container spacing={2}>
                                            <Grid item xs={6}>
                                                <Typography variant="caption" fontWeight="bold" textTransform="uppercase" display="block" gutterBottom>Common Causes</Typography>
                                                <Stack spacing={0.5}>
                                                    {fault.detailedInfo.causes.map((cause, i) => (
                                                        <Typography key={i} variant="caption" color="text.secondary" display="block">• {cause}</Typography>
                                                    ))}
                                                </Stack>
                                            </Grid>
                                            <Grid item xs={6}>
                                                <Typography variant="caption" fontWeight="bold" textTransform="uppercase" display="block" gutterBottom>Recommended Actions</Typography>
                                                <Stack spacing={0.5}>
                                                    {fault.detailedInfo.actions.map((action, i) => (
                                                        <Typography key={i} variant="caption" color="primary.main" display="block">→ {action}</Typography>
                                                    ))}
                                                </Stack>
                                            </Grid>
                                        </Grid>

                                        <Box sx={{ p: 2, bgcolor: 'primary.lighter', borderRadius: 1, border: 1, borderColor: 'primary.light' }}>
                                            <Typography variant="caption" fontWeight="bold" color="primary.main" textTransform="uppercase" display="block">Calculation Formula</Typography>
                                            <Typography variant="caption" fontFamily="monospace" color="primary.dark" display="block" sx={{ my: 0.5 }}>{fault.detailedInfo.formula}</Typography>
                                            <Typography variant="caption" color="primary.main" fontSize="0.65rem">N = rolling elements, Bd = ball diameter, Pd = pitch diameter, θ = angle</Typography>
                                        </Box>
                                    </Stack>
                                </Box>
                            </Collapse>
                        </Card>
                    );
                })}
            </Stack>
        </Card>
    );
}

export default BearingFaultPanel;
