import React, { useMemo, useState, useEffect } from 'react';
import AlertsPanel from './AlertsPanel';
import FleetTopology from './FleetTopology';
import CriticalMachineCard from './CriticalMachineCard';
import { DashboardGrid } from './DashboardGrid';
import AnomalyDiscoveryPanel from './AnomalyDiscoveryPanel';
import MaintenanceModal from './MaintenanceModal';
import {
    Grid, Box, Card, CardContent, CardHeader, Typography,
    Chip, LinearProgress, Stack, Avatar, Button, useTheme, IconButton,
    Paper, Slide, Snackbar, Alert
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import {
    Activity, AlertOctagon, AlertTriangle, CheckCircle,
    Factory, LayoutDashboard, Bolt, ArrowRight, Check, X, Calendar, Download
} from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

/**
 * PlantOverview Component (Enterprise Edition)
 * High-density dashboard for fleet monitoring.
 */
function PlantOverview({ machines, messages, onSelectMachine }) {
    const theme = useTheme();

    // Batch Selection State
    const [selectedIds, setSelectedIds] = useState(new Set());
    const [showBatchSuccess, setShowBatchSuccess] = useState(false);
    const [batchActionType, setBatchActionType] = useState('');

    // Scheduling Modal State
    const [schedulingMachines, setSchedulingMachines] = useState(null); // Array of machines or null

    const handleToggleSelect = (id, checked) => {
        const newSet = new Set(selectedIds);
        if (checked) {
            newSet.add(id);
        } else {
            newSet.delete(id);
        }
        setSelectedIds(newSet);
    };

    const clearSelection = () => setSelectedIds(new Set());

    const handleBatchAction = (action) => {
        if (action === 'scheduled') {
            // Open Modal with selected machines
            const selectedMachines = machines.filter(m => selectedIds.has(m.machine_id));
            setSchedulingMachines(selectedMachines);
        } else {
            setBatchActionType(action);
            setShowBatchSuccess(true);
            setTimeout(() => {
                clearSelection();
            }, 1000);
        }
    };

    const handleIndividualSchedule = (machineId) => {
        const machine = machines.find(m => m.machine_id === machineId);
        if (machine) {
            setSchedulingMachines([machine]);
        }
    };

    const handleConfirmSchedule = (data) => {
        console.log("Maintenance Scheduled:", data);
        setBatchActionType('scheduled');
        setShowBatchSuccess(true);
        setSchedulingMachines(null);
        clearSelection();
    };

    // Categorize machines by health status
    const { healthy, warning, critical, alerts } = useMemo(() => {
        const h = [], w = [], c = [];
        const alertList = [];

        machines.forEach(m => {
            const prob = m.failure_probability || 0;
            if (prob > 0.8) {
                c.push(m);
                alertList.push({
                    machine_id: m.machine_id,
                    severity: 'critical',
                    probability: prob,
                    timestamp: m.last_seen || new Date().toISOString(),
                    message: `Critical failure risk detected - ${(prob * 100).toFixed(1)}% probability`
                });
            } else if (prob > 0.5) {
                w.push(m);
                alertList.push({
                    machine_id: m.machine_id,
                    severity: 'warning',
                    probability: prob,
                    timestamp: m.last_seen || new Date().toISOString(),
                    message: `Elevated failure risk - ${(prob * 100).toFixed(1)}% probability`
                });
            } else {
                h.push(m);
            }
        });

        // Sort alerts by probability
        alertList.sort((a, b) => b.probability - a.probability);
        return { healthy: h, warning: w, critical: c, alerts: alertList };
    }, [machines]);

    const overallHealth = machines.length > 0
        ? Math.round(100 - (machines.reduce((sum, m) => sum + (m.failure_probability || 0), 0) / machines.length) * 100)
        : 100;

    // Distribution Data for Donut Chart
    const distributionData = useMemo(() => [
        { name: 'Critical', value: critical.length, color: theme.palette.error.main },
        { name: 'Warning', value: warning.length, color: theme.palette.warning.main },
        { name: 'Optimal', value: healthy.length, color: theme.palette.success.main },
    ], [critical, warning, healthy, theme]);


    // --- Drag and Drop State Management (Free Form) ---
    // Layout Strategy: Flush Left "Bento Grid"
    const defaultPositions = {
        topology: { x: 0, y: 190, w: 900, h: 700 },
        priority: { x: 920, y: 190, w: 430, h: 700 }, // Total ~1350px
        discovery: { x: 0, y: 910, w: 900, h: 600 },
        alerts: { x: 920, y: 910, w: 430, h: 600 }
    };

    const [layoutPositions, setLayoutPositions] = useState(() => {
        try {
            const saved = localStorage.getItem('plant_overview_positions_v13'); // Balanced layout
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
        localStorage.setItem('plant_overview_positions_v13', JSON.stringify(newPositions));
    };

    // --- Widget Content Definitions ---
    const widgetsContent = useMemo(() => ({
        priority: {
            id: 'priority',
            content: (
                <Card variant="outlined" sx={{ borderRadius: 2, height: '100%', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                    <CardHeader
                        title={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <AlertOctagon size={18} className="text-red-600" />
                                <Typography variant="subtitle1" fontWeight="bold">Priority Attention</Typography>
                            </Box>
                        }
                        action={
                            <Chip label={`${critical.length}`} color="error" size="small" />
                        }
                        sx={{ bgcolor: 'grey.50', py: 1, borderBottom: 1, borderColor: 'divider', flexShrink: 0 }}
                    />
                    <CardContent sx={{ p: 0, flexGrow: 1, overflow: 'auto', bgcolor: 'grey.50' }}>
                        {critical.length === 0 ? (
                            <Box sx={{ p: 4, textAlign: 'center', color: 'text.secondary' }}>
                                <CheckCircle className="h-10 w-10 mx-auto mb-2 text-emerald-400 opacity-50" />
                                <Typography variant="body2">No critical anomalies detected.</Typography>
                            </Box>
                        ) : (
                            <Box sx={{ display: 'flex', flexDirection: 'column', width: '100%', gap: 0 }}>
                                {/* Setup Sorting & Grouping */}
                                {(() => {
                                    // 1. Sort by Urgency (Critical In / RUL)
                                    // Lowest RUL = Most Urgent
                                    const sortedCritical = [...critical].sort((a, b) => (a.rul_days || 0) - (b.rul_days || 0));

                                    if (sortedCritical.length === 0) return null;

                                    // 2. Extract Primary Alert (Top Item)
                                    const primaryAlert = sortedCritical[0];
                                    const otherAlerts = sortedCritical.slice(1);

                                    // 3. Group Remaining Alerts by Line
                                    const grouped = otherAlerts.reduce((acc, m) => {
                                        const line = m.line_name || 'Unassigned Line';
                                        if (!acc[line]) acc[line] = [];
                                        acc[line].push(m);
                                        return acc;
                                    }, {});

                                    return (
                                        <>
                                            {/* Primary Alert (Pinned Top) */}
                                            <Box sx={{ p: 1.5, pb: 2, bgcolor: 'white', borderBottom: 1, borderColor: 'divider' }}>
                                                <Typography variant="caption" fontWeight="bold" color="error.main" sx={{ display: 'block', mb: 1, letterSpacing: 0.5 }}>
                                                    MOST URGENT ACTION
                                                </Typography>
                                                <CriticalMachineCard
                                                    machine={primaryAlert}
                                                    isPrimary={true}
                                                    onViewDetails={onSelectMachine}
                                                    onSchedule={handleIndividualSchedule}
                                                    onAlert={(id) => console.log('Send alert for', id)}
                                                    selected={selectedIds.has(primaryAlert.machine_id)}
                                                    onToggleSelect={handleToggleSelect}
                                                />
                                            </Box>

                                            {/* Grouped Lists */}
                                            {Object.keys(grouped).sort().map(lineName => (
                                                <Box key={lineName} sx={{ p: 1.5, borderBottom: 1, borderColor: 'divider' }}>
                                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                                        <Box sx={{ width: 4, height: 16, bgcolor: 'grey.300', borderRadius: 1 }} />
                                                        <Typography variant="caption" fontWeight="bold" color="text.secondary" textTransform="uppercase">
                                                            {lineName}
                                                        </Typography>
                                                    </Box>
                                                    <Stack spacing={1.5}>
                                                        {grouped[lineName].map(m => (
                                                            <CriticalMachineCard
                                                                key={m.machine_id}
                                                                machine={m}
                                                                onViewDetails={onSelectMachine}
                                                                onSchedule={handleIndividualSchedule}
                                                                onAlert={(id) => console.log('Send alert for', id)}
                                                                selected={selectedIds.has(m.machine_id)}
                                                                onToggleSelect={handleToggleSelect}
                                                            />
                                                        ))}
                                                    </Stack>
                                                </Box>
                                            ))}
                                        </>
                                    );
                                })()}
                            </Box>
                        )}
                    </CardContent>
                </Card>
            )
        },
        topology: {
            id: 'topology',
            content: (
                <Card variant="outlined" sx={{ borderRadius: 2, height: '100%', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                    <CardHeader
                        title={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <LayoutDashboard size={18} className="text-slate-500" />
                                <Typography variant="subtitle1" fontWeight="bold">Fleet Topology</Typography>
                            </Box>
                        }
                        sx={{ bgcolor: 'grey.50', py: 1, borderBottom: 1, borderColor: 'divider', flexShrink: 0 }}
                    />
                    <CardContent sx={{ p: 0, '&:last-child': { pb: 0 }, flexGrow: 1, overflow: 'hidden' }}>
                        <Box sx={{ overflow: 'hidden', height: '100%' }}>
                            <FleetTopology
                                machines={machines}
                                onSelectMachine={onSelectMachine}
                                height={layoutPositions['topology']?.h ? layoutPositions['topology'].h - 100 : 500}
                                width={layoutPositions['topology']?.w}
                            />
                        </Box>

                        {/* Legend Footer */}
                        <Stack direction="row" spacing={2} p={1.5} borderTop={1} borderColor="divider" justifyContent="center">
                            <Stack direction="row" alignItems="center" spacing={1}>
                                <Box sx={{ width: 8, height: 8, bgcolor: 'success.light', borderRadius: '50%' }} />
                                <Typography variant="caption" color="text.secondary">Optimal</Typography>
                            </Stack>
                            <Stack direction="row" alignItems="center" spacing={1}>
                                <Box sx={{ width: 8, height: 8, bgcolor: 'warning.main', borderRadius: '50%' }} />
                                <Typography variant="caption" color="text.secondary">Warning</Typography>
                            </Stack>
                            <Stack direction="row" alignItems="center" spacing={1}>
                                <Box sx={{ width: 8, height: 8, bgcolor: 'error.main', borderRadius: '50%' }} />
                                <Typography variant="caption" color="text.secondary">Critical</Typography>
                            </Stack>
                        </Stack>
                    </CardContent>
                </Card>
            )
        },
        alerts: {
            id: 'alerts',
            content: (
                <Box sx={{ height: '100%', overflow: 'hidden' }}>
                    <AlertsPanel
                        alerts={alerts}
                        selectedIds={selectedIds}
                        onToggleSelect={handleToggleSelect}
                    />
                </Box>
            )
        },
        discovery: {
            id: 'discovery',
            content: (
                <Box sx={{ height: '100%', overflow: 'hidden' }}>
                    <AnomalyDiscoveryPanel />
                </Box>
            )
        }
    }), [critical, machines, onSelectMachine, alerts, layoutPositions, healthy.length, warning.length, theme, selectedIds]); // Added selectedIds dependency

    // Merge static content with dynamic position state
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
            {/* Draggable Dashboard Grid (Free Form) */}
            <DashboardGrid
                items={gridItems}
                onUpdate={handleUpdate}
            >
                {/* Static Header & KPI Row wrapper - sits on grid but not draggable */}
                <Box sx={{ p: 0, pt: 2, px: 1, pointerEvents: 'none', maxWidth: '100%' }}>
                    <Box sx={{ pointerEvents: 'auto' }}> {/* Re-enable pointer events for interactive elements */}
                        {/* Header */}
                        <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Box>
                                <Typography variant="h5" fontWeight="800" color="text.primary" sx={{ letterSpacing: '-0.5px' }}>
                                    Plant Overview
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    Real-time telemetry • {machines.length} active assets
                                </Typography>
                            </Box>

                            {/* Compact System Health Indicator */}
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, px: 2, py: 0.5, bgcolor: 'background.paper', borderRadius: 2, border: 1, borderColor: 'divider' }}>
                                <Box sx={{ textAlign: 'right' }}>
                                    <Typography variant="caption" fontWeight="bold" color="text.secondary" display="block" lineHeight={1}>SYSTEM HEALTH</Typography>
                                    <Typography variant="h6" fontWeight="bold" fontFamily="monospace" lineHeight={1}
                                        sx={{ color: overallHealth > 90 ? 'success.main' : overallHealth > 70 ? 'warning.main' : 'error.main' }}>
                                        {overallHealth}%
                                    </Typography>
                                </Box>
                                <Activity className={`h-6 w-6 ${overallHealth > 90 ? 'text-emerald-500' : overallHealth > 70 ? 'text-amber-500' : 'text-red-500'}`} />
                            </Box>
                        </Box>

                        {/* Stat Cards - High Density Grid */}
                        <Grid container spacing={1.5} sx={{ mb: 2.5 }}>
                            <Grid item xs={12} sm={6} md={3}>
                                <StatCard
                                    title="Total Assets"
                                    value={machines.length}
                                    icon={Factory}
                                    color="default"
                                    type="summary"
                                    donutData={distributionData}
                                />
                            </Grid>
                            <Grid item xs={12} sm={6} md={3}>
                                <StatCard
                                    title="Optimal"
                                    value={healthy.length}
                                    total={machines.length}
                                    icon={CheckCircle}
                                    color="success"
                                />
                            </Grid>
                            <Grid item xs={12} sm={6} md={3}>
                                <StatCard
                                    title="Warning"
                                    value={warning.length}
                                    total={machines.length}
                                    icon={AlertTriangle}
                                    color="warning"
                                />
                            </Grid>
                            <Grid item xs={12} sm={6} md={3}>
                                <StatCard
                                    title="Critical"
                                    value={critical.length}
                                    total={machines.length}
                                    icon={AlertOctagon}
                                    color="error"
                                />
                            </Grid>
                        </Grid>
                    </Box>
                </Box>
            </DashboardGrid>

            {/* Batch Action Bar */}
            <Slide direction="up" in={selectedIds.size > 0} mountOnEnter unmountOnExit>
                <Paper sx={{
                    position: 'fixed', bottom: 24, left: '50%', transform: 'translateX(-50%)',
                    zIndex: 1300, px: 3, py: 1.5, borderRadius: 4,
                    display: 'flex', alignItems: 'center', gap: 3,
                    boxShadow: '0 8px 32px rgba(0,0,0,0.2)', border: '1px solid', borderColor: 'divider'
                }}>
                    <Stack direction="row" alignItems="center" spacing={1}>
                        <Box sx={{ width: 24, height: 24, borderRadius: '50%', bgcolor: 'primary.main', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', fontSize: '0.8rem' }}>
                            {selectedIds.size}
                        </Box>
                        <Typography variant="body2" fontWeight="bold">Items Selected</Typography>
                    </Stack>

                    <Box sx={{ width: 1, height: 24, bgcolor: 'divider' }} />

                    <Stack direction="row" spacing={1}>
                        <Button
                            variant="contained" size="small"
                            startIcon={<Calendar size={16} />}
                            onClick={() => handleBatchAction('scheduled')}
                            sx={{ bgcolor: 'white', color: 'text.primary', '&:hover': { bgcolor: 'grey.100' }, boxShadow: 1 }}
                        >
                            Schedule Group Maintenance
                        </Button>
                        <Button
                            variant="outlined" size="small"
                            startIcon={<Download size={16} />}
                            onClick={() => handleBatchAction('exported')}
                        >
                            Export Data
                        </Button>
                    </Stack>

                    <IconButton size="small" onClick={clearSelection} sx={{ ml: 1 }}>
                        <X size={18} />
                    </IconButton>
                </Paper>
            </Slide>

            {/* Maintenance Modal */}
            <MaintenanceModal
                open={!!schedulingMachines}
                machines={schedulingMachines || []}
                onClose={() => setSchedulingMachines(null)}
                onConfirm={handleConfirmSchedule}
            />

            {/* Action Success Toast */}
            <Snackbar open={showBatchSuccess} autoHideDuration={4000} onClose={() => setShowBatchSuccess(false)}>
                <Alert onClose={() => setShowBatchSuccess(false)} severity="success" sx={{ width: '100%', fontWeight: 'bold' }}>
                    {batchActionType === 'scheduled' ? 'Maintenance Scheduled Successfully' : 'Sensor Data Export Initiated'}
                </Alert>
            </Snackbar>

        </Box>
    );
}

function StatCard({ title, value, total = 0, icon: Icon, color, type = 'status', donutData }) {
    const theme = useTheme();
    const isZero = value === 0;

    // Color mapping
    const colors = {
        default: theme.palette.text.secondary,
        success: theme.palette.success.main,
        warning: theme.palette.warning.main,
        error: theme.palette.error.main
    };

    const activeColor = colors[color] || colors.default;

    // Liquid Fill opacity calculation
    // Calculate percentage, floor at 5% so there is always a tiny bit of color if > 0
    // Max at 100%
    const percentage = total > 0 ? (value / total) * 100 : 0;
    const fillOpacity = isZero ? 0 : Math.max(5, percentage);

    // Gradient Background
    // Right side matches solid background, Left side fades to transparent/white
    // But requirement says "liquid fill background effect where color level corresponds to percentage"
    // We'll use a linear gradient stop to simulate a "bar" filling up the card
    const bgGradient = color === 'default'
        ? theme.palette.background.paper
        : `linear-gradient(90deg, ${alpha(activeColor, 0.15)} ${fillOpacity}%, ${theme.palette.background.paper} ${fillOpacity}%)`;

    // Glow effect for critical > 0
    const glowSx = (color === 'error' && !isZero) ? {
        boxShadow: `0 0 15px ${alpha(activeColor, 0.4)}`,
        borderColor: alpha(activeColor, 0.6)
    } : {};

    // Hover state styles
    const hoverSx = {
        transform: 'translateY(-4px)',
        boxShadow: (theme) => theme.shadows[4],
        '& .view-btn': {
            opacity: 1,
            transform: 'translateY(0)'
        }
    };

    // Icon logic for empty state
    const ActiveIcon = isZero && type !== 'summary' ? Check : Icon;
    const iconColor = isZero ? theme.palette.text.disabled : activeColor;

    return (
        <Card
            variant="outlined"
            sx={{
                borderRadius: 2,
                background: bgGradient,
                borderLeft: 3,
                borderColor: isZero ? theme.palette.divider : activeColor,
                height: '100%',
                position: 'relative',
                transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                opacity: isZero && type !== 'summary' ? 0.6 : 1, // Dim if 0
                ...glowSx,
                '&:hover': type !== 'summary' ? { // Only hover effect on status cards
                    ...hoverSx
                } : {}
            }}
        >
            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 }, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>

                    {/* Icon Container */}
                    {type === 'summary' && donutData ? (
                        <Box sx={{ width: 44, height: 44 }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                    <Pie
                                        data={donutData}
                                        cx="50%"
                                        cy="50%"
                                        innerRadius={14}
                                        outerRadius={22}
                                        paddingAngle={2}
                                        dataKey="value"
                                        stroke="none"
                                    >
                                        {donutData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.color} />
                                        ))}
                                    </Pie>
                                </PieChart>
                            </ResponsiveContainer>
                        </Box>
                    ) : (
                        <Box sx={{
                            p: 1.2,
                            borderRadius: 1.5,
                            bgcolor: isZero ? alpha(theme.palette.action.disabledBackground, 0.5) : 'white',
                            color: iconColor,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            boxShadow: isZero ? 'none' : 1
                        }}>
                            <ActiveIcon size={22} strokeWidth={isZero ? 2.5 : 2} />
                        </Box>
                    )}

                    <Box>
                        {/* Typography: Bold, larger font-weight for counts */}
                        <Typography variant="overline" fontWeight="800" color="text.secondary" sx={{ lineHeight: 1, letterSpacing: 0.5 }}>
                            {title}
                        </Typography>
                        <Typography variant="h4" fontWeight="800" sx={{ color: isZero ? 'text.disabled' : 'text.primary', lineHeight: 1 }}>
                            {value}
                        </Typography>
                    </Box>
                </Box>

                {/* Micro-Interaction: View Button */}
                {type !== 'summary' && !isZero && (
                    <Box
                        className="view-btn"
                        sx={{
                            position: 'absolute',
                            right: 16,
                            opacity: 0,
                            transform: 'translateY(10px)',
                            transition: 'all 0.3s ease'
                        }}
                    >
                        <Button
                            variant="contained"
                            size="small"
                            color={color === 'default' ? 'primary' : color}
                            sx={{ minWidth: 'auto', px: 1, boxShadow: 2 }}
                        >
                            <ArrowRight size={16} />
                        </Button>
                    </Box>
                )}

                {/* For Summary, maybe show total? Or just keep it clean */}
            </CardContent>
        </Card>
    );
}

export default PlantOverview;
