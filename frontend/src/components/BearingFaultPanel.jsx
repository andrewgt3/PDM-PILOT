import React, { useState } from 'react';
import { AlertOctagon, AlertTriangle, CheckCircle, ChevronDown, ChevronUp, Info, Activity, Square, CheckSquare } from 'lucide-react';
import { Card, Box, Typography, Collapse, LinearProgress, Stack, Grid, IconButton, Chip, FormGroup, FormControlLabel, Checkbox } from '@mui/material';

/**
 * BearingFaultPanel Component
 * Displays bearing fault frequency amplitudes with expandable detail sections,
 * technical diagrams, and interactive maintenance checklists.
 */
export function BearingFaultPanel({ data }) {
    const [expandedFault, setExpandedFault] = useState(null);
    const [checkedActions, setCheckedActions] = useState({});

    if (!data || data.length === 0) {
        return (
            <Card variant="outlined" sx={{ p: 3, borderRadius: 2 }}>
                <Typography variant="subtitle2" fontWeight="bold" textTransform="uppercase" color="text.secondary" gutterBottom>
                    Expert Analysis: Diagnostics
                </Typography>
                <Typography variant="body2" color="text.secondary">No telemetry data available</Typography>
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
                whatIs: 'Frequency of rolling elements passing a text on the outer race.',
                causes: ['Contamination', 'Improper lubrication', 'Misalignment', 'Overloading', 'Fatigue'],
                actions: ['Inspect bearing for spalling', 'Check lubrication quality', 'Verify shaft alignment', 'Schedule replacement'],
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
                whatIs: 'Frequency of rolling elements passing a defect on the inner race.',
                causes: ['Tight fit on shaft', 'Improper installation', 'Contamination', 'Fatigue', 'Excessive load'],
                actions: ['Check inner race for pitting', 'Verify proper shaft fit', 'Assess lubricant condition', 'Plan replacement'],
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
                whatIs: 'Frequency of a defect on a rolling element contacting the races.',
                causes: ['Ball fatigue', 'Contaminated lubricant', 'Manufacturing defects', 'Overloading'],
                actions: ['Inspect rolling elements', 'Check for metallic debris', 'Replace bearing assembly'],
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
                whatIs: 'Rotational frequency of the bearing cage (retainer).',
                causes: ['Cage wear', 'Improper lubrication', 'Cage pocket damage', 'Bearing overheating'],
                actions: ['Check cage integrity', 'Ensure proper lubrication', 'Monitor for rapid progression'],
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

    const handleActionToggle = (faultName, action) => {
        setCheckedActions(prev => ({
            ...prev,
            [`${faultName}-${action}`]: !prev[`${faultName}-${action}`]
        }));
    };

    return (
        <Card variant="outlined" sx={{ p: 3, borderRadius: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Box>
                    <Typography variant="subtitle2" fontWeight="bold" textTransform="uppercase" color="text.secondary">
                        Expert Analysis: Diagnostics
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                        Vibration spectrum decomposition & fault isolation
                    </Typography>
                </Box>
                <Activity className="w-5 h-5 text-slate-400" />
            </Box>

            <Grid container spacing={3}>
                <Grid item xs={12} md={7}>
                    <Stack spacing={2}>
                        {faults.map((fault) => {
                            const status = getStatus(fault.value, fault.threshold);
                            const percentage = Math.min((fault.value / (fault.threshold * 2)) * 100, 100);
                            const isExpanded = expandedFault === fault.name;

                            return (
                                <Card key={fault.name} variant="outlined" sx={{ overflow: 'hidden', boxShadow: isExpanded ? 2 : 0, transition: 'box-shadow 0.2s' }}>
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

                                        {/* Zone Gauge */}
                                        <Box sx={{ mt: 2, position: 'relative', height: 12, borderRadius: 1, overflow: 'hidden', bgcolor: 'grey.100', boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.1)' }}>
                                            {/* Gradient Background */}
                                            <Box sx={{ position: 'absolute', top: 0, left: 0, bottom: 0, width: '100%', background: 'linear-gradient(90deg, #4ade80 0%, #4ade80 50%, #facc15 50%, #facc15 75%, #f87171 75%, #f87171 100%)', opacity: 0.3 }} />

                                            {/* Needle */}
                                            <Box sx={{
                                                position: 'absolute',
                                                top: -2,
                                                bottom: -2,
                                                left: `${percentage}%`,
                                                width: 4,
                                                bgcolor: 'slate.800',
                                                borderRadius: 1,
                                                boxShadow: '0 0 4px rgba(0,0,0,0.3)',
                                                transition: 'left 0.5s ease-out',
                                                zIndex: 2
                                            }} />
                                        </Box>
                                    </Box>

                                    {/* Details */}
                                    <Collapse in={isExpanded}>
                                        <Box sx={{ p: 2, bgcolor: 'grey.50', borderTop: 1, borderColor: 'divider' }}>
                                            <Grid container spacing={2}>
                                                <Grid item xs={12}>
                                                    <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1, border: 1, borderColor: 'divider', boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.03)' }}>
                                                        <Typography variant="caption" fontWeight="bold" textTransform="uppercase" display="block" gutterBottom sx={{ mb: 1, color: 'text.secondary', borderBottom: 1, borderColor: 'divider', pb: 0.5 }}>
                                                            Maintenance Task List
                                                        </Typography>
                                                        <FormGroup>
                                                            {fault.detailedInfo.actions.map((action, i) => (
                                                                <FormControlLabel
                                                                    key={i}
                                                                    control={
                                                                        <Checkbox
                                                                            size="small"
                                                                            checked={checkedActions[`${fault.name}-${action}`] || false}
                                                                            onChange={() => handleActionToggle(fault.name, action)}
                                                                            icon={<Square size={16} className="text-slate-400" />}
                                                                            checkedIcon={<CheckSquare size={16} className="text-indigo-600" />}
                                                                        />
                                                                    }
                                                                    label={<Typography variant="caption" color={checkedActions[`${fault.name}-${action}`] ? 'text.disabled' : 'text.primary'} sx={{ textDecoration: checkedActions[`${fault.name}-${action}`] ? 'line-through' : 'none' }}>{action}</Typography>}
                                                                    sx={{ mb: 0.5, ml: 0 }}
                                                                />
                                                            ))}
                                                        </FormGroup>
                                                    </Box>
                                                </Grid>
                                                <Grid item xs={12}>
                                                    <Box sx={{ p: 1.5, bgcolor: 'primary.lighter', borderRadius: 1, border: 1, borderColor: 'primary.light' }}>
                                                        <Typography variant="caption" fontWeight="bold" color="primary.main" textTransform="uppercase" display="block">Formula</Typography>
                                                        <Typography variant="caption" fontFamily="monospace" color="primary.dark" display="block">{fault.detailedInfo.formula}</Typography>
                                                    </Box>
                                                </Grid>
                                            </Grid>
                                        </Box>
                                    </Collapse>
                                </Card>
                            );
                        })}
                    </Stack>
                </Grid>

                {/* Technical Diagram Column */}
                <Grid item xs={12} md={5}>
                    <Card variant="outlined" sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', bgcolor: 'grey.50' }}>
                        <Typography variant="caption" fontWeight="bold" color="text.secondary" textTransform="uppercase" sx={{ mb: 2, alignSelf: 'flex-start' }}>
                            Component Schematic
                        </Typography>

                        {/* SVG Drawing of Ball Bearing */}
                        <Box sx={{ position: 'relative', width: '100%', maxWidth: 220, aspectRatio: '1/1' }}>
                            <svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
                                {/* Outer Race */}
                                <circle cx="100" cy="100" r="90" fill="none" stroke="#94a3b8" strokeWidth="12" />
                                {/* Inner Race */}
                                <circle cx="100" cy="100" r="50" fill="none" stroke="#64748b" strokeWidth="12" />
                                {/* Rolling Elements */}
                                {[0, 45, 90, 135, 180, 225, 270, 315].map((angle, i) => (
                                    <circle
                                        key={i}
                                        cx={100 + 70 * Math.cos(angle * Math.PI / 180)}
                                        cy={100 + 70 * Math.sin(angle * Math.PI / 180)}
                                        r="18"
                                        fill="#cbd5e1"
                                        stroke="#475569"
                                        strokeWidth="2"
                                        className={expandedFault ? (expandedFault.includes('BPFO') || expandedFault.includes('BPFI') ? 'animate-pulse' : '') : ''}
                                    />
                                ))}
                                {/* Center Shaft */}
                                <circle cx="100" cy="100" r="20" fill="#e2e8f0" />
                                {/* Labels */}
                                <text x="100" y="105" textAnchor="middle" fontSize="10" fill="#64748b" fontWeight="bold">SHAFT</text>

                                {/* Annotations */}
                                <line x1="100" y1="100" x2="170" y2="100" stroke="#6366f1" strokeWidth="1" strokeDasharray="4 2" />
                                <text x="140" y="95" textAnchor="middle" fontSize="10" fill="#6366f1" fontWeight="bold">Pd</text>

                                <line x1="100" y1="100" x2="150" y2="150" stroke="#6366f1" strokeWidth="1" strokeDasharray="4 2" />
                                <text x="130" y="130" textAnchor="middle" fontSize="10" fill="#6366f1" fontWeight="bold">θ</text>
                            </svg>
                        </Box>
                        <Box sx={{ mt: 3, width: '100%' }}>
                            <Stack spacing={1}>
                                <div className="flex justify-between text-xs text-slate-500">
                                    <span><strong>Pd:</strong> Pitch Diameter</span>
                                    <span><strong>Bd:</strong> Ball Diameter</span>
                                </div>
                                <div className="flex justify-between text-xs text-slate-500">
                                    <span><strong>N:</strong> Num Elements</span>
                                    <span><strong>θ:</strong> Contact Angle</span>
                                </div>
                            </Stack>
                        </Box>
                    </Card>
                </Grid>
            </Grid>
        </Card>
    );
}

export default BearingFaultPanel;
