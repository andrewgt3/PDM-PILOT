import React, { useState, useEffect } from 'react';
import { AlertTriangle, AlertCircle, Info, Bell, Clock, CheckCircle, RefreshCw } from 'lucide-react';
import { Card, Box, Typography, IconButton, Chip, ToggleButton, ToggleButtonGroup, Stack, CircularProgress, Button } from '@mui/material';

/**
 * ActiveAlarmFeed Component
 * 
 * Real-time display of PDM-generated alarms.
 * Fetches from enterprise API.
 */
function ActiveAlarmFeed({ machineId }) {
    const [alarms, setAlarms] = useState([]);
    const [loading, setLoading] = useState(true);
    const [filter, setFilter] = useState('all');

    const fetchAlarms = async () => {
        try {
            const url = machineId
                ? `http://localhost:8000/api/enterprise/alarms?machine_id=${machineId}&limit=20`
                : `http://localhost:8000/api/enterprise/alarms?limit=20`;

            const response = await fetch(url);
            if (response.ok) {
                const result = await response.json();
                setAlarms(result.data || []);
            } else {
                setAlarms([]);
            }
        } catch (err) {
            console.error('[Alarms API Error]', err);
            setAlarms([]);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchAlarms();
        // Poll every 30 seconds
        const interval = setInterval(fetchAlarms, 30000);
        return () => clearInterval(interval);
    }, [machineId]);

    const handleAcknowledge = async (alarmId) => {
        try {
            await fetch(`http://localhost:8000/api/enterprise/alarms/${alarmId}/acknowledge`, {
                method: 'POST'
            });
            fetchAlarms(); // Refresh
        } catch (err) {
            console.error('[Acknowledge Error]', err);
        }
    };

    // Filter alarms
    const filteredAlarms = alarms.filter(alarm => {
        if (filter === 'active') return alarm.active && !alarm.acknowledged;
        if (filter === 'acknowledged') return alarm.acknowledged;
        return true;
    });

    // Count by severity
    const criticalCount = alarms.filter(a => a.severity === 'critical' && a.active).length;
    const warningCount = alarms.filter(a => a.severity === 'warning' && a.active).length;

    const severityConfig = {
        critical: { icon: AlertCircle, color: 'error', bg: '#fef2f2' },
        warning: { icon: AlertTriangle, color: 'warning', bg: '#fffbeb' },
        info: { icon: Info, color: 'info', bg: '#eff6ff' }
    };

    const formatTime = (timestamp) => {
        if (!timestamp) return 'Unknown';
        const diff = Date.now() - new Date(timestamp).getTime();
        const mins = Math.floor(diff / 60000);
        if (mins < 60) return `${mins}m ago`;
        const hours = Math.floor(mins / 60);
        if (hours < 24) return `${hours}h ago`;
        return `${Math.floor(hours / 24)}d ago`;
    };

    if (loading) {
        return (
            <Card variant="outlined" sx={{ height: '100%', borderRadius: 2, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <CircularProgress />
            </Card>
        );
    }

    return (
        <Card variant="outlined" sx={{ height: '100%', borderRadius: 2, display: 'flex', flexDirection: 'column' }}>
            {/* Header */}
            <Box sx={{ px: 2, py: 1.5, borderBottom: 1, borderColor: 'divider', bgcolor: 'grey.50', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Stack direction="row" spacing={1} alignItems="center">
                    <Bell size={20} className="text-slate-600" />
                    <Typography variant="subtitle2" fontWeight="bold">PDM Alarms</Typography>
                    {criticalCount > 0 && (
                        <Chip label={`${criticalCount} Critical`} color="error" size="small" sx={{ fontWeight: 'bold', height: 20 }} />
                    )}
                </Stack>
                <Stack direction="row" spacing={1} alignItems="center">
                    <IconButton size="small" onClick={fetchAlarms} title="Refresh">
                        <RefreshCw size={16} />
                    </IconButton>
                    <ToggleButtonGroup
                        value={filter}
                        exclusive
                        onChange={(e, val) => val && setFilter(val)}
                        size="small"
                        sx={{ height: 24 }}
                    >
                        <ToggleButton value="all" sx={{ px: 1, fontSize: '0.65rem' }}>All</ToggleButton>
                        <ToggleButton value="active" sx={{ px: 1, fontSize: '0.65rem' }}>Active</ToggleButton>
                        <ToggleButton value="acknowledged" sx={{ px: 1, fontSize: '0.65rem' }}>Ack</ToggleButton>
                    </ToggleButtonGroup>
                </Stack>
            </Box>

            {/* Alarm List */}
            <Box sx={{ flex: 1, overflowY: 'auto', maxHeight: 350 }}>
                {filteredAlarms.length === 0 ? (
                    <Box sx={{ p: 4, textAlign: 'center', color: 'text.secondary' }}>
                        <CheckCircle size={32} color="#10b981" style={{ opacity: 0.5, marginBottom: 8 }} />
                        <Typography variant="body2">{alarms.length === 0 ? 'No alarms recorded' : 'No alarms matching filter'}</Typography>
                        <Typography variant="caption">Alarms are generated when failure probability exceeds thresholds</Typography>
                    </Box>
                ) : (
                    <Stack divider={<Box sx={{ borderBottom: 1, borderColor: 'divider' }} />}>
                        {filteredAlarms.map(alarm => {
                            const config = severityConfig[alarm.severity] || severityConfig.info;
                            const Icon = config.icon;
                            return (
                                <Box
                                    key={alarm.alarm_id}
                                    sx={{
                                        px: 2, py: 1.5,
                                        bgcolor: config.bg,
                                        opacity: !alarm.active ? 0.6 : 1,
                                        display: 'flex', alignItems: 'flex-start', gap: 2
                                    }}
                                >
                                    <Icon size={20} className={`text-${config.color}-main`} style={{ marginTop: 2, color: config.color === 'error' ? '#d32f2f' : config.color === 'warning' ? '#ed6c02' : '#0288d1' }} />
                                    <Box sx={{ flex: 1, minWidth: 0 }}>
                                        <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                                            <Chip label={alarm.code} size="small" color={config.color} sx={{ height: 18, fontSize: '0.6rem', fontWeight: 'bold' }} />
                                            <Typography variant="caption" color="text.secondary">{alarm.source || 'PDM System'}</Typography>
                                            {alarm.acknowledged && (
                                                <Stack direction="row" spacing={0.5} alignItems="center" sx={{ color: 'text.secondary' }}>
                                                    <CheckCircle size={10} />
                                                    <Typography variant="caption" sx={{ fontSize: '0.6rem' }}>ACK</Typography>
                                                </Stack>
                                            )}
                                        </Stack>
                                        <Typography variant="body2" fontWeight="medium" sx={{ lineHeight: 1.3 }}>
                                            {alarm.message}
                                        </Typography>
                                        <Stack direction="row" spacing={0.5} alignItems="center" sx={{ mt: 0.5 }}>
                                            <Clock size={12} className="text-slate-400" />
                                            <Typography variant="caption" color="text.secondary">{formatTime(alarm.timestamp)}</Typography>
                                        </Stack>
                                    </Box>
                                    {alarm.active && !alarm.acknowledged && (
                                        <Button
                                            size="small"
                                            variant="outlined"
                                            color="inherit"
                                            onClick={() => handleAcknowledge(alarm.alarm_id)}
                                            sx={{ fontSize: '0.65rem', minWidth: 'auto', px: 1, height: 24, borderColor: 'divider' }}
                                        >
                                            ACK
                                        </Button>
                                    )}
                                </Box>
                            );
                        })}
                    </Stack>
                )}
            </Box>

            {/* Footer */}
            <Box sx={{ px: 2, py: 1, bgcolor: 'grey.50', borderTop: 1, borderColor: 'divider', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Stack direction="row" spacing={2}>
                    <Stack direction="row" spacing={0.5} alignItems="center">
                        <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: 'error.main' }} />
                        <Typography variant="caption" color="text.secondary">{criticalCount} Critical</Typography>
                    </Stack>
                    <Stack direction="row" spacing={0.5} alignItems="center">
                        <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: 'warning.main' }} />
                        <Typography variant="caption" color="text.secondary">{warningCount} Warning</Typography>
                    </Stack>
                </Stack>
                <Typography variant="caption" color="text.disabled">Auto-refresh: 30s</Typography>
            </Box>
        </Card>
    );
}

export default ActiveAlarmFeed;
