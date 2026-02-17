import React, { useMemo, useState, useEffect } from 'react';
import { IndustrialTile } from './IndustrialTile';
import { Brain, Gauge, Thermometer, Activity, ArrowLeft } from 'lucide-react';
import {
    Grid, Box, Card, CardContent, Typography, Button,
    Breadcrumbs, Link, Chip, Table, TableBody, TableCell,
    TableContainer, TableHead, TableRow, Paper, Stack
} from '@mui/material';
import { NavigateNext } from '@mui/icons-material';

import { DashboardGrid } from './DashboardGrid';
import RULCard from './RULCard';
import BearingFaultPanel from './BearingFaultPanel';
import FeatureImportancePanel from './FeatureImportancePanel';
import MaintenanceRecommendationCard from './MaintenanceRecommendationCard';
import DegradationTrendChart from './DegradationTrendChart';
import BearingFaultTrendsChart from './BearingFaultTrendsChart';
import LineContextWidget from './LineContextWidget';
// Enterprise CMMS Components
import ReliabilityMetricsCard from './ReliabilityMetricsCard';
import ActiveAlarmFeed from './ActiveAlarmFeed';
import WorkOrderPanel from './WorkOrderPanel';
import ShiftAwareRUL from './ShiftAwareRUL';
import MaintenanceModal from './MaintenanceModal';
import { Snackbar, Alert } from '@mui/material';

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

    // Scheduling State
    const [isScheduling, setIsScheduling] = useState(false);
    const [showSuccess, setShowSuccess] = useState(false);

    const handleConfirmSchedule = (data) => {
        console.log("Scheduled from Detail:", data);
        setIsScheduling(false);
        setShowSuccess(true);
    };

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

    // --- Drag and Drop State Management ---
    // Layout Strategy: Canvas-based Bento Grid
    const defaultPositions = {
        // Row 1: KPIs (Total ~1350px)
        metric_confidence: { x: 0, y: 190, w: 322, h: 200 },
        metric_speed: { x: 342, y: 190, w: 322, h: 200 },
        metric_temp: { x: 684, y: 190, w: 322, h: 200 },
        metric_fault: { x: 1026, y: 190, w: 322, h: 200 },

        // Row 2: Asset & RUL & Rec
        line_context: { x: 0, y: 410, w: 322, h: 600 },
        rul_card: { x: 342, y: 410, w: 664, h: 600 },
        maintenance_rec: { x: 1026, y: 410, w: 322, h: 600 },

        // Row 3: Trend Charts (Full Width)
        trend_degradation: { x: 0, y: 1030, w: 1350, h: 500 },
        trend_faults: { x: 0, y: 1550, w: 1350, h: 500 },

        // Row 4: Diagnostic Panels (Split)
        panel_faults: { x: 0, y: 2070, w: 664, h: 600 },
        panel_features: { x: 684, y: 2070, w: 664, h: 600 },

        // Row 5: Log
        feature_log: { x: 0, y: 2690, w: 1350, h: 500 },

        // Row 6: Enterprise 1
        cmms_metrics: { x: 0, y: 3210, w: 664, h: 500 },
        cmms_rul: { x: 684, y: 3210, w: 664, h: 500 },

        // Row 7: Enterprise 2
        cmms_alarms: { x: 0, y: 3730, w: 664, h: 600 },
        cmms_orders: { x: 684, y: 3730, w: 664, h: 600 }
    };

    const [layoutPositions, setLayoutPositions] = useState(() => {
        try {
            const saved = localStorage.getItem('machine_detail_positions_v6'); // Incremented key for update
            const parsed = saved ? JSON.parse(saved) : null;
            if (parsed && typeof parsed === 'object') {
                return { ...defaultPositions, ...parsed };
            }
            return defaultPositions;
        } catch (e) {
            return defaultPositions;
        }
    });

    const handleUpdate = (id, updates) => {
        const newPositions = {
            ...layoutPositions,
            [id]: { ...layoutPositions[id], ...updates }
        };
        setLayoutPositions(newPositions);
        localStorage.setItem('machine_detail_positions_v6', JSON.stringify(newPositions));
    };

    const widgetsContent = useMemo(() => ({
        metric_confidence: {
            id: 'metric_confidence',
            content: (
                <IndustrialTile
                    title="AI Confidence"
                    value={`${(machine.failure_probability * 100).toFixed(1)}`}
                    unit="%"
                    icon={Brain}
                    trendData={history}
                    dataKey="failure_probability"
                    limit={100} // Logical max for percentage
                    isHero={true}
                    riskLevel={isCritical ? 'critical' : isWarning ? 'warning' : 'normal'}
                    onAction={() => console.log('View Diagnostics')} // Placeholder or scroll to panels
                />
            )
        },
        metric_speed: {
            id: 'metric_speed',
            content: (
                <IndustrialTile
                    title="Rotational Speed"
                    value={history.length > 0 ? `${formatVal(history[history.length - 1].rotational_speed, 0)}` : '-'}
                    unit="RPM"
                    icon={Gauge}
                    trendData={history}
                    dataKey="rotational_speed"
                    limit={1800} // Target
                    riskLevel={'normal'}
                />
            )
        },
        metric_temp: {
            id: 'metric_temp',
            content: (
                <IndustrialTile
                    title="Bearing Temp"
                    value={history.length > 0 ? `${formatVal(history[history.length - 1].temperature, 1)}` : '-'}
                    unit="°C"
                    icon={Thermometer}
                    trendData={history}
                    dataKey="temperature"
                    limit={100} // Max
                    riskLevel={history.length > 0 && history[history.length - 1].temperature > 90 ? 'critical' : 'normal'}
                />
            )
        },
        metric_fault: {
            id: 'metric_fault',
            content: (
                <IndustrialTile
                    title="Fault Energy"
                    value={history.length > 0 ? formatVal(history[history.length - 1].bpfi_amp, 4) : '-'}
                    unit="g"
                    icon={Activity}
                    trendData={history}
                    dataKey="bpfi_amp"
                    limit={0.02} // Estimated limit for small vibration
                    riskLevel={'normal'}
                />
            )
        },
        line_context: {
            id: 'line_context',
            content: <LineContextWidget machineId={machine.machine_id} />
        },
        rul_card: {
            id: 'rul_card',
            content: (
                <Box sx={{ height: '100%' }}>
                    <RULCard
                        rul={machine.rul_days}
                        timeToFailure={machine.rul_days * 24}
                        degradationScore={history.length > 0 ? history[history.length - 1].degradation_score : 0}
                        onSchedule={() => setIsScheduling(true)}
                    />
                </Box>
            )
        },
        maintenance_rec: {
            id: 'maintenance_rec',
            content: (
                <MaintenanceRecommendationCard
                    machine={machine}
                    latestData={history[history.length - 1]}
                />
            )
        },
        trend_degradation: {
            id: 'trend_degradation',
            content: <DegradationTrendChart data={history} syncId="machineSync" />
        },
        trend_faults: {
            id: 'trend_faults',
            content: (
                <BearingFaultTrendsChart
                    data={history}
                    alerts={machineMessages}
                    syncId="machineSync"
                />
            )
        },
        panel_faults: {
            id: 'panel_faults',
            content: <BearingFaultPanel data={history} />
        },
        panel_features: {
            id: 'panel_features',
            content: (
                <FeatureImportancePanel
                    data={history}
                    failureProbability={machine.failure_probability}
                />
            )
        },
        feature_log: {
            id: 'feature_log',
            content: (
                <Card variant="outlined" sx={{ height: '100%', borderRadius: 2, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
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
                    <TableContainer sx={{ flexGrow: 1, overflow: 'auto' }}>
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
            )
        },
        cmms_metrics: { id: 'cmms_metrics', content: <ReliabilityMetricsCard machineId={machine.machine_id} /> },
        cmms_rul: { id: 'cmms_rul', content: <ShiftAwareRUL baseRulDays={machine.rul_days || 30} machineId={machine.machine_id} /> },
        cmms_alarms: { id: 'cmms_alarms', content: <ActiveAlarmFeed machineId={machine.machine_id} /> },
        cmms_orders: { id: 'cmms_orders', content: <WorkOrderPanel machine={machine} /> },

    }), [machine, history, machineMessages, isCritical, isWarning]);

    // Merge content and positions
    const gridItems = useMemo(() => {
        const items = {};
        Object.keys(widgetsContent).forEach(key => {
            const pos = layoutPositions[key] || { x: 0, y: 0, w: 400, h: 400 };
            items[key] = {
                ...widgetsContent[key],
                x: pos.x,
                y: pos.y,
                width: pos.w,
                height: pos.h
            };
        });
        return items;
    }, [layoutPositions, widgetsContent]);

    return (
        <Box sx={{ flexGrow: 1 }}>
            <DashboardGrid
                items={gridItems}
                onUpdate={handleUpdate}
            >
                {/* Static Header & Navigation */}
                <Box sx={{ p: 0, pt: 2, px: 1, pointerEvents: 'none', maxWidth: '100%' }}>
                    <Box sx={{ pointerEvents: 'auto' }}>
                        {onBack && (
                            <Button
                                startIcon={<ArrowLeft size={16} />}
                                onClick={onBack}
                                sx={{ mb: 2, color: 'text.secondary' }}
                            >
                                Back to Overview
                            </Button>
                        )}

                        <Box sx={{ mb: 4, pb: 2, borderBottom: 1, borderColor: 'divider' }}>
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
                    </Box>
                </Box>
            </DashboardGrid>


            {/* Maintenance Modal */}
            <MaintenanceModal
                open={isScheduling}
                machines={[machine]}
                onClose={() => setIsScheduling(false)}
                onConfirm={handleConfirmSchedule}
            />

            {/* Success Toast */}
            <Snackbar open={showSuccess} autoHideDuration={4000} onClose={() => setShowSuccess(false)}>
                <Alert onClose={() => setShowSuccess(false)} severity="success" sx={{ width: '100%', fontWeight: 'bold' }}>
                    Maintenance Scheduled Successfully
                </Alert>
            </Snackbar>
        </Box >
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
