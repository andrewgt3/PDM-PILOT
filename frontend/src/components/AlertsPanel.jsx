import React from 'react';
import { Card, CardContent, CardHeader, Typography, Box, Badge, IconButton, List, ListItem, ListItemText, ListItemAvatar } from '@mui/material';
import { AlertCircle, AlertTriangle, Bell, CheckCircle } from 'lucide-react';
import { Warning, Error as ErrorIcon, CheckCircleOutline, Notifications } from '@mui/icons-material';

/**
 * AlertsPanel Component (Enterprise Edition)
 * Dense list of critical system events.
 */
function AlertsPanel({ alerts }) {
    // Deduplicate: keep only the most recent alert per machine
    const deduplicatedAlerts = React.useMemo(() => {
        const byMachine = {};
        alerts.forEach(alert => {
            const existing = byMachine[alert.machine_id];
            if (!existing || alert.probability > existing.probability) {
                byMachine[alert.machine_id] = alert;
            }
        });
        return Object.values(byMachine).sort((a, b) => b.probability - a.probability);
    }, [alerts]);

    return (
        <Card variant="outlined" sx={{ height: '100%', borderRadius: 2 }}>
            <CardHeader
                title={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Notifications fontSize="small" sx={{ color: 'text.secondary' }} />
                        <Typography variant="subtitle2" fontWeight="bold">Active Alerts</Typography>
                    </Box>
                }
                action={
                    deduplicatedAlerts.length > 0 && (
                        <Box sx={{ px: 1, py: 0.5, bgcolor: 'error.light', borderRadius: 4, color: 'error.dark', fontSize: '0.75rem', fontWeight: 'bold' }}>
                            {deduplicatedAlerts.length}
                        </Box>
                    )
                }
                sx={{ bgcolor: 'grey.50', py: 1.5, borderBottom: 1, borderColor: 'divider' }}
            />
            <CardContent sx={{ p: 0 }}>
                {deduplicatedAlerts.length === 0 ? (
                    <Box sx={{ p: 4, textAlign: 'center', color: 'text.secondary' }}>
                        <CheckCircleOutline sx={{ fontSize: 40, mb: 1, color: 'success.light', opacity: 0.5 }} />
                        <Typography variant="body2">All systems nominal.</Typography>
                    </Box>
                ) : (
                    <Box sx={{ maxHeight: 400, overflowY: 'auto' }}>
                        {deduplicatedAlerts.map((alert, index) => (
                            <Box
                                key={alert.machine_id}
                                sx={{
                                    p: 2, display: 'flex', gap: 2,
                                    borderBottom: index < deduplicatedAlerts.length - 1 ? 1 : 0,
                                    borderColor: 'divider',
                                    bgcolor: alert.severity === 'critical' ? '#fef2f2' : 'inherit', // red-50
                                    '&:hover': { bgcolor: 'action.hover' },
                                    transition: 'background-color 0.2s'
                                }}
                            >
                                <Box sx={{ mt: 0.5 }}>
                                    {alert.severity === 'critical' ? (
                                        <ErrorIcon fontSize="small" color="error" />
                                    ) : (
                                        <Warning fontSize="small" color="warning" />
                                    )}
                                </Box>
                                <Box sx={{ flex: 1, minWidth: 0 }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                                        <Typography variant="caption" fontWeight="bold" fontFamily="monospace" sx={{ mr: 1 }}>
                                            {alert.machine_id}
                                        </Typography>
                                        <Typography variant="caption" color="text.secondary" sx={{ textTransform: 'uppercase', fontSize: '0.65rem' }}>
                                            {new Date(alert.timestamp).toLocaleTimeString()}
                                        </Typography>
                                    </Box>
                                    <Typography variant="body2" color="text.secondary" lineHeight={1.3}>
                                        {alert.message}
                                    </Typography>
                                </Box>
                            </Box>
                        ))}
                    </Box>
                )}
            </CardContent>
        </Card>
    );
}

export default AlertsPanel;
