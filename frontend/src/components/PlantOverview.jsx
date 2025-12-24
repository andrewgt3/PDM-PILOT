import React, { useMemo } from 'react';
import AlertsPanel from './AlertsPanel';
import FleetTopology from './FleetTopology';
import CriticalMachineCard from './CriticalMachineCard';
import {
    Grid, Box, Card, CardContent, CardHeader, Typography,
    Chip, LinearProgress, Stack, Avatar
} from '@mui/material';
import {
    Activity, AlertOctagon, AlertTriangle, CheckCircle,
    Factory, LayoutDashboard, Bolt
} from 'lucide-react'; // Keeping Lucide for internal components until they are refactored
// MUI icons removed - using Lucide icons consistently

/**
 * PlantOverview Component (Enterprise Edition)
 * High-density dashboard for fleet monitoring.
 */
function PlantOverview({ machines, messages, onSelectMachine }) {
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

    return (
        <Box sx={{ flexGrow: 1 }}>
            {/* Header / KPI Row */}
            <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <LayoutDashboard className="h-8 w-8 text-slate-500" />
                    <Box>
                        <Typography variant="h5" fontWeight="bold" color="text.primary">Plant Overview</Typography>
                        <Typography variant="body2" color="text.secondary">Real-time telemetry • {machines.length} active assets</Typography>
                    </Box>
                </Box>

                <Card variant="outlined" sx={{ minWidth: 200, borderRadius: 2 }}>
                    <CardContent sx={{ p: 2, '&:last-child': { pb: 2 }, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <Box sx={{ textAlign: 'right', mr: 2 }}>
                            <Typography variant="caption" fontWeight="bold" color="text.secondary" sx={{ textTransform: 'uppercase' }}>System Health</Typography>
                            <Typography variant="h4" fontWeight="bold" fontFamily="monospace"
                                sx={{ color: overallHealth > 90 ? 'success.main' : overallHealth > 70 ? 'warning.main' : 'error.main' }}>
                                {overallHealth}%
                            </Typography>
                        </Box>
                        <Activity className={`h-8 w-8 ${overallHealth > 90 ? 'text-emerald-500' : overallHealth > 70 ? 'text-amber-500' : 'text-red-500'}`} />
                    </CardContent>
                </Card>
            </Box>

            {/* Stat Cards - High Density Grid */}
            <Grid container spacing={2} sx={{ mb: 4 }}>
                <Grid item xs={12} sm={3}>
                    <StatCard title="Total Assets" value={machines.length} icon={Factory} color="default" />
                </Grid>
                <Grid item xs={12} sm={3}>
                    <StatCard title="Optimal" value={healthy.length} icon={CheckCircle} color="success" />
                </Grid>
                <Grid item xs={12} sm={3}>
                    <StatCard title="Warning" value={warning.length} icon={AlertTriangle} color="warning" />
                </Grid>
                <Grid item xs={12} sm={3}>
                    <StatCard title="Critical" value={critical.length} icon={AlertOctagon} color="error" />
                </Grid>
            </Grid>

            {/* Main Content Split */}
            <Grid container spacing={3}>
                {/* Left Column: Priority Action Items (2/3 width) */}
                <Grid item xs={12} lg={8}>
                    <Stack spacing={3}>
                        {/* Critical Machines List */}
                        <Card variant="outlined" sx={{ borderRadius: 2 }}>
                            <CardHeader
                                title={
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <AlertOctagon size={18} className="text-red-600" />
                                        <Typography variant="subtitle1" fontWeight="bold">Priority Attention Required</Typography>
                                    </Box>
                                }
                                action={
                                    <Chip label={`${critical.length} Machines`} color="error" size="small" />
                                }
                                sx={{ bgcolor: 'grey.50', py: 1.5, borderBottom: 1, borderColor: 'divider' }}
                            />
                            <CardContent sx={{ p: 0 }}>
                                {critical.length === 0 ? (
                                    <Box sx={{ p: 4, textAlign: 'center', color: 'text.secondary' }}>
                                        <CheckCircle className="h-10 w-10 mx-auto mb-2 text-emerald-400 opacity-50" />
                                        <Typography variant="body2">No critical anomalies detected.</Typography>
                                    </Box>
                                ) : (
                                    <Box sx={{ display: 'flex', flexDirection: 'column', width: '100%' }}>
                                        {critical.map(m => (
                                            <CriticalMachineCard
                                                key={m.machine_id}
                                                machine={m}
                                                onViewDetails={onSelectMachine}
                                                onSchedule={(id) => console.log('Schedule maintenance for', id)}
                                                onAlert={(id) => console.log('Send alert for', id)}
                                            />
                                        ))}
                                    </Box>
                                )}
                            </CardContent>
                        </Card>

                        {/* Fleet Heatmap (Treemap) */}
                        <Card variant="outlined" sx={{ borderRadius: 2 }}>
                            <CardHeader
                                title={
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <LayoutDashboard size={18} className="text-slate-500" />
                                        <Typography variant="subtitle1" fontWeight="bold">Fleet Topography</Typography>
                                    </Box>
                                }
                                action={
                                    <Typography variant="caption" color="text.secondary">Size = Standard • Color = Risk</Typography>
                                }
                                sx={{ bgcolor: 'grey.50', py: 1.5, borderBottom: 1, borderColor: 'divider' }}
                            />
                            <CardContent>
                                <FleetTopology machines={machines} onSelectMachine={onSelectMachine} />

                                {/* Legend */}
                                <Stack direction="row" spacing={3} mt={2} pt={2} borderTop={1} borderColor="divider">
                                    <Stack direction="row" alignItems="center" spacing={1}>
                                        <Box sx={{ width: 12, height: 12, bgcolor: 'success.light', borderRadius: 0.5 }} />
                                        <Typography variant="caption" color="text.secondary">Healthy ({healthy.length})</Typography>
                                    </Stack>
                                    <Stack direction="row" alignItems="center" spacing={1}>
                                        <Box sx={{ width: 12, height: 12, bgcolor: 'warning.main', borderRadius: 0.5 }} />
                                        <Typography variant="caption" color="text.secondary">Warning ({warning.length})</Typography>
                                    </Stack>
                                    <Stack direction="row" alignItems="center" spacing={1}>
                                        <Box sx={{ width: 12, height: 12, bgcolor: 'error.main', borderRadius: 0.5, boxShadow: '0 0 0 2px rgba(220, 38, 38, 0.2)' }} />
                                        <Typography variant="caption" color="text.secondary">Critical ({critical.length})</Typography>
                                    </Stack>
                                </Stack>
                            </CardContent>
                        </Card>
                    </Stack>
                </Grid>

                {/* Right Column: Alerts Stream */}
                <Grid item xs={12} lg={4}>
                    <AlertsPanel alerts={alerts} />
                </Grid>
            </Grid>
        </Box>
    );
}

function StatCard({ title, value, icon: Icon, color }) {
    const colorMap = {
        default: 'text.secondary',
        success: 'success.main',
        warning: 'warning.main',
        error: 'error.main'
    };

    // Determine bg color based on color prop for slight tint often seen in dashboards
    const bgMap = {
        default: 'background.paper',
        success: '#ecfdf5', // emerald-50
        warning: '#fffbeb', // amber-50
        error: '#fef2f2'   // red-50
    };

    return (
        <Card variant="outlined" sx={{ borderRadius: 2, bgcolor: bgMap[color] || 'background.paper', borderLeft: 4, borderColor: colorMap[color] || 'grey.300' }}>
            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 }, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box>
                    <Typography variant="caption" fontWeight="bold" sx={{ textTransform: 'uppercase', opacity: 0.7, display: 'block', mb: 0.5 }}>
                        {title}
                    </Typography>
                    <Typography variant="h4" fontWeight="bold" fontFamily="monospace">
                        {value}
                    </Typography>
                </Box>
                <Box sx={{ color: colorMap[color] || 'text.secondary', opacity: 0.5 }}>
                    <Icon size={24} />
                </Box>
            </CardContent>
        </Card>
    );
}

export default PlantOverview;
