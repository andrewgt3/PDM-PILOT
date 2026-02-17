import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, Typography, Box, Badge, IconButton, Tooltip, Collapse, Button, Checkbox } from '@mui/material';
import {
    AlertCircle, AlertTriangle, Bell, CheckCircle, Clock,
    ChevronRight, ChevronDown, Check, Wrench, Activity
} from 'lucide-react';
import { alpha, useTheme } from '@mui/material/styles';

/**
 * Helper: Format Relative Time
 * e.g. "Just now", "5m ago", "1h ago"
 */
function getRelativeTime(isoString) {
    const date = new Date(isoString);
    const now = new Date();
    const diffInSeconds = Math.floor((now - date) / 1000);

    if (diffInSeconds < 60) return 'Just now';
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`;
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`;
    return date.toLocaleDateString();
}

/**
 * AlertsPanel Component (System Log Edition)
 * High-fidelity system event log with timeline view.
 */
function AlertsPanel({ alerts, selectedIds = new Set(), onToggleSelect = () => { } }) {
    const theme = useTheme();
    const [expandedGroups, setExpandedGroups] = useState({});

    const toggleGroup = (groupId) => {
        setExpandedGroups(prev => ({
            ...prev,
            [groupId]: !prev[groupId]
        }));
    };

    // 1. Process Alerts: Sort & Group by Time Block (10 min windows)
    const groupedAlerts = useMemo(() => {
        // Sort descending by time
        const sorted = [...alerts].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

        const groups = [];
        let currentGroup = null;

        sorted.forEach((alert) => {
            const alertTime = new Date(alert.timestamp).getTime();

            // Check if fits in current group (within 10 mins of group start)
            if (currentGroup && Math.abs(currentGroup.startTime - alertTime) < 10 * 60 * 1000) {
                currentGroup.items.push(alert);
                // Keep the group severity as the max of its items
                if (alert.severity === 'critical') currentGroup.severity = 'critical';
            } else {
                // Start a new group
                currentGroup = {
                    id: `group-${groups.length}`,
                    startTime: alertTime,
                    severity: alert.severity,
                    items: [alert]
                };
                groups.push(currentGroup);
            }
        });

        return groups;
    }, [alerts]);

    return (
        <Card variant="outlined" sx={{ height: '100%', borderRadius: 2, display: 'flex', flexDirection: 'column' }}>
            <CardHeader
                title={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Bell size={18} className="text-slate-500" />
                        <Typography variant="subtitle1" fontWeight="bold">System Log</Typography>
                    </Box>
                }
                action={
                    alerts.length > 0 && (
                        <Box sx={{
                            display: 'flex', alignItems: 'center', gap: 0.5,
                            pl: 1, pr: 1.5, py: 0.5, bgcolor: 'primary.50',
                            borderRadius: 4, color: 'primary.700', border: 1, borderColor: 'primary.100'
                        }}>
                            <Activity size={14} />
                            <Typography variant="caption" fontWeight="bold">LIVE</Typography>
                        </Box>
                    )
                }
                sx={{ bgcolor: 'grey.50', py: 1.5, borderBottom: 1, borderColor: 'divider', flexShrink: 0 }}
            />

            <CardContent sx={{ p: 0, flexGrow: 1, overflowY: 'auto', bgcolor: 'white' }}>
                {groupedAlerts.length === 0 ? (
                    <Box sx={{ p: 4, textAlign: 'center', color: 'text.secondary' }}>
                        <CheckCircle className="h-10 w-10 mx-auto mb-2 text-emerald-400 opacity-50" />
                        <Typography variant="body2">System Nominal. No active events.</Typography>
                    </Box>
                ) : (
                    <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                        {groupedAlerts.map((group, groupIndex) => {
                            const isGroup = group.items.length > 1;
                            const mainAlert = group.items[0];
                            const isExpanded = expandedGroups[group.id];
                            const isSelected = selectedIds.has(mainAlert.machine_id);

                            return (
                                <Box key={group.id} sx={{ position: 'relative' }}>
                                    {/* Timeline Thread */}
                                    {groupIndex !== groupedAlerts.length - 1 && (
                                        <Box sx={{
                                            position: 'absolute', top: 32, bottom: -16, left: 24 + 32, width: 2, // Shifted for checkbox
                                            bgcolor: 'grey.100', zIndex: 0
                                        }} />
                                    )}

                                    {/* System Event Row (Group Header or Single Item) */}
                                    <Box
                                        onClick={() => isGroup && toggleGroup(group.id)}
                                        sx={{
                                            display: 'flex', gap: 1, p: 1.5, pl: 1,
                                            cursor: isGroup ? 'pointer' : 'default',
                                            bgcolor: isSelected ? 'primary.lighter' : (groupIndex % 2 === 0 ? 'grey.50' : 'white'),
                                            borderLeft: 4,
                                            borderColor: isSelected ? 'primary.main' : 'transparent',
                                            '&:hover': { bgcolor: isSelected ? 'primary.lighter' : 'grey.100' },
                                            '&:hover .actions': { opacity: 1 },
                                            position: 'relative'
                                        }}
                                    >
                                        <Checkbox
                                            size="small"
                                            checked={isSelected}
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                onToggleSelect(mainAlert.machine_id, e.target.checked);
                                            }}
                                        />

                                        {/* Icon Column */}
                                        <Box sx={{ zIndex: 1, mt: 0.5 }}>
                                            {isGroup ? (
                                                <Box sx={{
                                                    width: 32, height: 32, borderRadius: 2,
                                                    bgcolor: 'white', border: 1, borderColor: 'divider',
                                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                    boxShadow: 1
                                                }}>
                                                    <Box sx={{
                                                        width: 24, height: 24, borderRadius: 1.5,
                                                        bgcolor: group.severity === 'critical' ? 'error.lighter' : 'warning.lighter',
                                                        color: group.severity === 'critical' ? 'error.main' : 'warning.main',
                                                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                        fontWeight: 'bold', fontSize: '0.75rem'
                                                    }}>
                                                        {group.items.length}
                                                    </Box>
                                                </Box>
                                            ) : (
                                                <Box sx={{
                                                    width: 16, height: 16, borderRadius: '50%',
                                                    bgcolor: mainAlert.severity === 'critical' ? 'error.main' : 'warning.main',
                                                    boxShadow: `0 0 0 4px ${alpha(mainAlert.severity === 'critical' ? theme.palette.error.main : theme.palette.warning.main, 0.1)}`,
                                                    mt: 0.5, ml: 1
                                                }} />
                                            )}
                                        </Box>

                                        {/* Content Column */}
                                        <Box sx={{ flex: 1, minWidth: 0, ml: 1 }}>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 0.5 }}>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                    {isGroup ? (
                                                        <Typography variant="subtitle2" fontWeight="bold">
                                                            Multiple Events Detected
                                                        </Typography>
                                                    ) : (
                                                        <Typography variant="body2" fontWeight="bold" fontFamily="monospace" sx={{ fontSize: '0.8rem' }}>
                                                            {mainAlert.machine_id}
                                                        </Typography>
                                                    )}

                                                    {/* Relative Time */}
                                                    <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                                        <Clock size={10} />
                                                        {getRelativeTime(mainAlert.timestamp)}
                                                    </Typography>
                                                </Box>

                                                {/* Hover Actions */}
                                                <Box className="actions" sx={{ opacity: 0, transition: 'opacity 0.2s', display: 'flex', gap: 0.5 }}>
                                                    <Tooltip title="Acknowledge">
                                                        <IconButton size="small" sx={{ color: 'success.main', p: 0.5, bgcolor: 'white', border: 1, borderColor: 'divider' }}>
                                                            <Check size={14} />
                                                        </IconButton>
                                                    </Tooltip>
                                                    <Tooltip title="Create Work Order">
                                                        <IconButton size="small" sx={{ color: 'primary.main', p: 0.5, bgcolor: 'white', border: 1, borderColor: 'divider' }}>
                                                            <Wrench size={14} />
                                                        </IconButton>
                                                    </Tooltip>
                                                </Box>
                                            </Box>

                                            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', lineHeight: 1.4 }}>
                                                {isGroup
                                                    ? `${group.items.length} anomalies recorded in this time block. Click to view details.`
                                                    : mainAlert.message}
                                            </Typography>

                                            {isGroup && (
                                                <Box sx={{ mt: 0.5, display: 'flex', alignItems: 'center', gap: 0.5, color: 'primary.main' }}>
                                                    <Typography variant="caption" fontWeight="bold" sx={{ cursor: 'pointer' }}>
                                                        {isExpanded ? 'Collapse' : 'Expand Details'}
                                                    </Typography>
                                                    {isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                                                </Box>
                                            )}
                                        </Box>
                                    </Box>

                                    {/* Expanded Group Items */}
                                    <Collapse in={isExpanded} timeout="auto" unmountOnExit>
                                        <Box sx={{ bgcolor: 'grey.50', pl: 9, pr: 2, pb: 2, borderBottom: 1, borderColor: 'divider' }}>
                                            {group.items.map((item, i) => (
                                                <Box key={i} sx={{
                                                    py: 1, borderTop: i > 0 ? 1 : 0, borderColor: 'divider',
                                                    display: 'flex', alignItems: 'center', gap: 2,
                                                    '&:hover .sub-actions': { opacity: 1 }
                                                }}>
                                                    <Checkbox
                                                        size="small"
                                                        checked={selectedIds.has(item.machine_id)}
                                                        onClick={(e) => onToggleSelect(item.machine_id, e.target.checked)}
                                                        sx={{ p: 0.5 }}
                                                    />
                                                    <Typography variant="caption" fontWeight="bold" fontFamily="monospace" sx={{ minWidth: 60 }}>
                                                        {item.machine_id}
                                                    </Typography>
                                                    <Typography variant="caption" color="text.secondary" sx={{ flex: 1 }}>
                                                        {item.message}
                                                    </Typography>

                                                    {/* Sub-hover actions */}
                                                    <Box className="sub-actions" sx={{ opacity: 0, transition: 'opacity 0.2s', display: 'flex', gap: 0.5 }}>
                                                        <IconButton size="small" sx={{ p: 0.5 }}>
                                                            <Check size={12} />
                                                        </IconButton>
                                                    </Box>
                                                </Box>
                                            ))}
                                        </Box>
                                    </Collapse>
                                </Box>
                            );
                        })}
                    </Box>
                )}
            </CardContent>
        </Card>
    );
}

export default AlertsPanel;
