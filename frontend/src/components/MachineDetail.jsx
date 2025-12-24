import React, { useMemo, useState, useEffect } from 'react';
import { Brain, Gauge, Thermometer, Activity, ArrowLeft } from 'lucide-react';
import {
    Grid, Box, Card, CardContent, Typography, Button,
    Breadcrumbs, Link, Chip, Table, TableBody, TableCell,
    TableContainer, TableHead, TableRow, Paper, Stack
} from '@mui/material';
import { NavigateNext } from '@mui/icons-material';

import RULCard from './RULCard';
import BearingFaultPanel from './BearingFaultPanel';
import FeatureImportancePanel from './FeatureImportancePanel';
import MaintenanceRecommendationCard from './MaintenanceRecommendationCard';
import DegradationTrendChart from './DegradationTrendChart';
import BearingFaultTrendsChart from './BearingFaultTrendsChart';
// Enterprise CMMS Components
import ReliabilityMetricsCard from './ReliabilityMetricsCard';
import ActiveAlarmFeed from './ActiveAlarmFeed';
import WorkOrderPanel from './WorkOrderPanel';
import ShiftAwareRUL from './ShiftAwareRUL';

/**
 * MachineDetail Component (Unified Platform)
 * 
 * Merges "Operator Console" (metrics, status) with "Forensics Lab" (charts, history).
 * Fetches historical data from API to power the charts.
 */
function MachineDetail({ machine, messages, onBack }) {
    // 1. Live Message Stream
    const machineMessages = useMemo(() => {
        return messages.filter(m => m.machine_id === machine.machine_id);
    }, [messages, machine.machine_id]);

    const latest = machineMessages[0] || {};

    // 2. Historical Data State
    const [history, setHistory] = useState([]);
    const [isLoadingHistory, setIsLoadingHistory] = useState(false);

    // Fetch history when machine changes
    useEffect(() => {
        const fetchHistory = async () => {
            setIsLoadingHistory(true);
            try {
                // Fetch last 50 records for charts (balanced speed vs. detail)
                const res = await fetch(`http://localhost:8000/api/features?limit=50&machine_id=${machine.machine_id}`);

                const data = await res.json();
                // Reverse to have oldest first for charts
                if (data.data) {
                    setHistory(data.data.reverse());
                }
            } catch (err) {
                console.error("Failed to fetch history:", err);
            } finally {
                setIsLoadingHistory(false);
            }
        };

        if (machine.machine_id) {
            fetchHistory();
        }
    }, [machine.machine_id]);


    // Status Helpers
    const isCritical = machine.failure_probability > 0.8;
    const isWarning = machine.failure_probability > 0.5;
    const formatVal = (val, fixed = 2) => val !== undefined ? val.toFixed(fixed) : '-';

    return (
        <Box sx={{ pb: 8 }}>
            {/* Back Button */}
            {onBack && (
                <Button
                    startIcon={<ArrowLeft size={16} />}
                    onClick={onBack}
                    sx={{ mb: 2, color: 'text.secondary' }}
                >
                    Back to Overview
                </Button>
            )}

            {/* Header Section */}
            <Box sx={{ mb: 4, pb: 2, borderBottom: 1, borderColor: 'divider' }}>
                {/* Breadcrumbs */}
                <Breadcrumbs separator={<NavigateNext fontSize="small" />} sx={{ mb: 2 }}>
                    <Link underline="hover" color="inherit" href="#">Plant</Link>
                    <Link underline="hover" color="inherit" href="#">{machine.line_name || 'Unassigned Line'}</Link>
                    <Typography color="text.primary" fontWeight="medium">{machine.machine_id}</Typography>
                </Breadcrumbs>

                <Grid container alignItems="center" justifyContent="space-between" spacing={2}>
                    <Grid item>
                        <Stack direction="row" spacing={2} alignItems="center">
                            <Box>
                                <Typography variant="h4" fontWeight="bold">{machine.machine_id}</Typography>
                                <Typography variant="body2" color="text.secondary">
                                    {machine.model_number || 'Unknown Model'} • Installed {machine.install_date || 'N/A'}
                                </Typography>
                            </Box>
                            <Chip
                                label={machine.operational_status || 'OFFLINE'}
                                color={machine.operational_status === 'RUNNING' ? 'success' : 'default'}
                                variant="outlined"
                                sx={{ fontWeight: 'bold' }}
                            />
                        </Stack>
                    </Grid>

                    <Grid item>
                        <Chip
                            icon={<Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: isCritical ? 'error.main' : isWarning ? 'warning.main' : 'success.main', ml: 1, mr: -0.5, animation: 'pulse 2s infinite' }} />}
                            label={isCritical ? 'CRITICAL FAILURE' : isWarning ? 'WARNING' : 'HEALTHY'}
                            color={isCritical ? 'error' : isWarning ? 'warning' : 'success'}
                            sx={{ px: 1, py: 2.5, fontWeight: 'bold', fontSize: '0.875rem' }}
                        />
                    </Grid>
                </Grid>
            </Box>

            {/* KPI Cards */}
            <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={6} sm={6} md={3}>
                    <MetricCard
                        label="AI Confidence"
                        value={`${(machine.failure_probability * 100).toFixed(1)}%`}
                        trend={isCritical ? 'Risk High' : 'Stable'}
                        color={isCritical ? 'error' : 'primary'}
                        icon={Brain}
                    />
                </Grid>
                <Grid item xs={6} sm={6} md={3}>
                    <MetricCard
                        label="Rotational Speed"
                        value={history.length > 0 ? `${formatVal(history[history.length - 1].rotational_speed, 0)}` : '-'}
                        sub="Target: 1800 RPM"
                        color="default"
                        icon={Gauge}
                    />
                </Grid>
                <Grid item xs={6} sm={6} md={3}>
                    <MetricCard
                        label="Bearing Temp"
                        value={history.length > 0 ? `${formatVal(history[history.length - 1].temperature, 1)}°` : '-°'}
                        sub="Max: 100°C"
                        color="warning"
                        icon={Thermometer}
                    />
                </Grid>
                <Grid item xs={6} sm={6} md={3}>
                    <MetricCard
                        label="Fault Energy"
                        value={history.length > 0 ? formatVal(history[history.length - 1].bpfi_amp, 4) : '-'}
                        sub="BPFI Amplitude (g)"
                        color="success"
                        icon={Activity}
                    />
                </Grid>
            </Grid>

            {/* --- TOP ROW: Asset Profile & RUL Side by Side --- */}
            <Grid container spacing={2} sx={{ mb: 3 }}>
                {/* Asset Info */}
                <Grid item xs={12} md={6} lg={4}>
                    <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
                        <CardContent sx={{ p: { xs: 2, md: 2.5 } }}>
                            <Typography variant="overline" color="text.secondary" fontWeight="bold" display="block" gutterBottom>
                                Asset Profile
                            </Typography>
                            <Stack spacing={1.5} sx={{ mt: 1.5 }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', borderBottom: 1, borderColor: 'divider', pb: 1 }}>
                                    <Typography variant="body2" color="text.secondary">Model</Typography>
                                    <Typography variant="body2" fontWeight="medium">{machine.model_number || '-'}</Typography>
                                </Box>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', borderBottom: 1, borderColor: 'divider', pb: 1 }}>
                                    <Typography variant="body2" color="text.secondary">Serial No.</Typography>
                                    <Typography variant="body2" fontWeight="medium">SN-{machine.machine_id.split('_')[1] || machine.machine_id}-202X</Typography>
                                </Box>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', borderBottom: 1, borderColor: 'divider', pb: 1 }}>
                                    <Typography variant="body2" color="text.secondary">Line</Typography>
                                    <Typography variant="body2" fontWeight="medium">{machine.line_name || machine.line_id || '-'}</Typography>
                                </Box>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <Typography variant="body2" color="text.secondary">Equipment Type</Typography>
                                    <Typography variant="body2" fontWeight="medium">{machine.equipment_type || '-'}</Typography>
                                </Box>
                            </Stack>
                        </CardContent>
                    </Card>
                </Grid>

                {/* RUL Card */}
                <Grid item xs={12} md={6} lg={5}>
                    <RULCard
                        rul={machine.rul_days}
                        failureProbability={machine.failure_probability}
                        degradationScore={history.length > 0 ? history[history.length - 1].degradation_score : 0}
                    />
                </Grid>

                {/* Maintenance Recommendation */}
                <Grid item xs={12} lg={3}>
                    <MaintenanceRecommendationCard
                        machine={machine}
                        latestData={history[history.length - 1]}
                    />
                </Grid>
            </Grid>

            {/* --- FULL WIDTH: Trend Charts --- */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
                <Grid item xs={12} lg={6}>
                    <DegradationTrendChart data={history} />
                </Grid>
                <Grid item xs={12} lg={6}>
                    <BearingFaultTrendsChart data={history} />
                </Grid>
            </Grid>

            {/* --- FULL WIDTH: Diagnostic Panels --- */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
                <Grid item xs={12} lg={6}>
                    <BearingFaultPanel data={history} />
                </Grid>
                <Grid item xs={12} lg={6}>
                    <FeatureImportancePanel
                        data={history}
                        failureProbability={machine.failure_probability}
                    />
                </Grid>
            </Grid>

            {/* --- FULL WIDTH: Feature Log --- */}
            <Card variant="outlined" sx={{ mb: 4, borderRadius: 2, overflow: 'hidden' }}>
                <Box sx={{ px: 3, py: 2, borderBottom: 1, borderColor: 'divider', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Stack direction="row" spacing={2} alignItems="center">
                        <Typography variant="subtitle1" fontWeight="bold">Feature Log</Typography>
                        {machineMessages.length > 0 ? (
                            <Chip label="LIVE" color="success" size="small" variant="outlined" />
                        ) : (
                            <Chip label="HISTORICAL" color="default" size="small" variant="outlined" />
                        )}
                    </Stack>
                    <Typography variant="caption" color="text.secondary" fontFamily="monospace">
                        {machineMessages.length > 0 ? `${machineMessages.length} live events` : `${history.length} records`}
                    </Typography>
                </Box>
                <TableContainer sx={{ maxHeight: 300 }}>
                    <Table stickyHeader size="small">
                        <TableHead>
                            <TableRow>
                                <TableCell>Timestamp</TableCell>
                                <TableCell>Failure Prob</TableCell>
                                <TableCell>Degradation</TableCell>
                                <TableCell>BPFI</TableCell>
                                <TableCell>BPFO</TableCell>
                                <TableCell>Speed (RPM)</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {(machineMessages.length > 0 ? machineMessages.slice(0, 20) : [...history].reverse().slice(0, 20)).map((record, i) => {
                                const failProb = record.failure_probability || record.failure_prediction || 0;
                                const degradation = record.degradation_score || record.degradation_score_smoothed || 0;
                                return (
                                    <TableRow key={i} hover>
                                        <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                                            {new Date(record.timestamp).toLocaleString()}
                                        </TableCell>
                                        <TableCell sx={{ fontWeight: 'bold', color: failProb > 0.8 ? 'error.main' : failProb > 0.5 ? 'warning.main' : 'success.main' }}>
                                            {(failProb * 100).toFixed(1)}%
                                        </TableCell>
                                        <TableCell sx={{ fontWeight: 'bold', color: degradation > 0.7 ? 'error.main' : degradation > 0.4 ? 'warning.main' : 'success.main' }}>
                                            {(degradation * 100).toFixed(1)}%
                                        </TableCell>
                                        <TableCell>{formatVal(record.bpfi_amp, 4)}</TableCell>
                                        <TableCell>{formatVal(record.bpfo_amp, 4)}</TableCell>
                                        <TableCell>{formatVal(record.rotational_speed, 0)}</TableCell>
                                    </TableRow>
                                );
                            })}
                            {machineMessages.length === 0 && history.length === 0 && (
                                <TableRow>
                                    <TableCell colSpan={6} align="center" sx={{ py: 4, color: 'text.secondary' }}>
                                        No data available. Waiting for sensor readings...
                                    </TableCell>
                                </TableRow>
                            )}
                        </TableBody>
                    </Table>
                </TableContainer>
            </Card>

            {/* --- ENTERPRISE CMMS SECTION --- */}
            <Box sx={{ pt: 3, borderTop: 1, borderColor: 'divider' }}>
                <Typography variant="h6" fontWeight="bold" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box sx={{ width: 6, height: 24, bgcolor: 'primary.main', borderRadius: 1 }} />
                    Enterprise Integration
                </Typography>

                <Grid container spacing={3} sx={{ mb: 3 }}>
                    <Grid item xs={12} lg={6}>
                        <ReliabilityMetricsCard machineId={machine.machine_id} />
                    </Grid>
                    <Grid item xs={12} lg={6}>
                        <ShiftAwareRUL
                            baseRulDays={machine.rul_days || 30}
                            machineId={machine.machine_id}
                        />
                    </Grid>
                </Grid>

                <Grid container spacing={3}>
                    <Grid item xs={12} lg={6}>
                        <ActiveAlarmFeed machineId={machine.machine_id} />
                    </Grid>
                    <Grid item xs={12} lg={6}>
                        <WorkOrderPanel machine={machine} />
                    </Grid>
                </Grid>
            </Box>
        </Box>
    );
}

// Simple internal component for KPI cards
function MetricCard({ label, value, sub, trend, color = 'default', icon: Icon }) {
    const colorMap = {
        default: 'text.secondary',
        primary: 'primary.main',
        success: 'success.main',
        warning: 'warning.main',
        error: 'error.main',
    };

    // Slight tint backgrounds
    const bgMap = {
        default: 'background.paper',
        primary: '#eef2ff', // indigo-50
        success: '#ecfdf5', // emerald-50
        warning: '#fffbeb', // amber-50
        error: '#fef2f2'   // red-50
    };

    return (
        <Card variant="outlined" sx={{ borderRadius: 2, height: '100%', borderTop: 3, borderColor: colorMap[color] || 'grey.300', '&:hover': { boxShadow: 2 } }}>
            <CardContent sx={{ p: { xs: 1.5, md: 2 }, '&:last-child': { pb: { xs: 1.5, md: 2 } } }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography sx={{ fontSize: '0.65rem', fontWeight: 'bold', color: 'text.secondary', textTransform: 'uppercase', letterSpacing: 0.5 }}>
                        {label}
                    </Typography>
                    {Icon && (
                        <Box sx={{ p: 0.5, borderRadius: 1, bgcolor: bgMap[color], color: colorMap[color] || 'text.secondary' }}>
                            <Icon size={14} />
                        </Box>
                    )}
                </Box>

                <Typography variant="h5" fontWeight="bold" fontFamily="monospace" sx={{ mb: 0.5 }}>
                    {value}
                </Typography>

                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 0.5 }}>
                    {sub && <Typography sx={{ fontSize: '0.7rem', color: 'text.secondary' }}>{sub}</Typography>}
                    {trend && (
                        <Chip
                            label={trend}
                            size="small"
                            color={trend.includes('High') ? 'error' : 'success'}
                            variant="outlined"
                            sx={{ fontSize: '0.6rem', height: 18 }}
                        />
                    )}
                </Box>
            </CardContent>
        </Card>
    );
}

export default MachineDetail;
