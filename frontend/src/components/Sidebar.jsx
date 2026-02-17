import React, { useMemo } from 'react';
import {
    Drawer, Box, List, ListItemButton, ListItemIcon, ListItemText,
    Typography, Avatar, Chip, Divider, ListSubheader
} from '@mui/material';
import {
    FactoryOutlined, PsychologyOutlined, KeyboardArrowRight,
    ElectricBoltOutlined, MenuBookOutlined
} from '@mui/icons-material';

/**
 * Sidebar Component (Asset Hierarchy)
 * 
 * Group machines by their manufacturing line.
 * Shows status counters and detailed navigation tree.
 */
function Sidebar({ machines, selectedMachineId, onSelectMachine, view, setView }) {

    // Group machines by Line Name
    const lines = useMemo(() => {
        const groups = {};
        machines.forEach(m => {
            const line = m.line_name || "Unassigned Assets";
            if (!groups[line]) groups[line] = [];
            groups[line].push(m);
        });
        return groups;
    }, [machines]);

    // Calculate aggregated stats
    const criticalCount = machines.filter(m => (m.failure_probability || 0) > 0.8).length;

    return (
        <Drawer
            variant="permanent"
            anchor="left"
            sx={{
                width: 240,
                flexShrink: 0,
                '& .MuiDrawer-paper': {
                    width: 240,
                    boxSizing: 'border-box',
                    bgcolor: 'background.paper',
                    borderRight: '1px solid',
                    borderColor: 'divider',
                },
            }}
        >
            {/* Header */}
            <Box sx={{ p: 3, borderBottom: '1px solid', borderColor: 'divider', bgcolor: 'background.default' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Box sx={{
                        width: 32, height: 32, borderRadius: 2,
                        bgcolor: 'primary.main', color: 'primary.contrastText',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontWeight: 'bold', boxShadow: 2
                    }}>
                        G
                    </Box>
                    <Box>
                        <Typography variant="subtitle1" fontWeight={800} lineHeight={1.2}>
                            PLANT OS
                        </Typography>
                        <Typography variant="caption" color="text.secondary" sx={{ letterSpacing: 1 }}>
                            PREDICTIVE
                        </Typography>
                    </Box>
                </Box>
            </Box>

            {/* Navigation */}
            <Box sx={{ overflow: 'auto', flex: 1, py: 2 }}>
                <List subheader={<ListSubheader sx={{ fontSize: '0.75rem', letterSpacing: 1, lineHeight: '32px' }}>MAIN</ListSubheader>}>
                    <ListItemButton
                        selected={view === 'overview'}
                        onClick={() => setView('overview')}
                        sx={{ borderRadius: 1, mx: 1, mb: 0.5 }}
                    >
                        <ListItemIcon sx={{ minWidth: 40, color: view === 'overview' ? 'primary.main' : 'inherit' }}>
                            <FactoryOutlined fontSize="small" />
                        </ListItemIcon>
                        <ListItemText primary="Plant Overview" primaryTypographyProps={{ fontSize: '0.875rem', fontWeight: 500 }} />
                        {criticalCount > 0 && (
                            <Chip size="small" color="error" label={`${criticalCount} Critical`} sx={{ height: 20, fontSize: '0.65rem', fontWeight: 'bold' }} />
                        )}
                    </ListItemButton>

                    <ListItemButton
                        selected={view === 'pipeline-ops'}
                        onClick={() => setView('pipeline-ops')}
                        sx={{ borderRadius: 1, mx: 1, mb: 0.5 }}
                    >
                        <ListItemIcon sx={{ minWidth: 40, color: view === 'pipeline-ops' ? 'primary.main' : 'inherit' }}>
                            <ElectricBoltOutlined fontSize="small" />
                        </ListItemIcon>
                        <ListItemText primary="Pipeline Ops" primaryTypographyProps={{ fontSize: '0.875rem', fontWeight: 500 }} />
                    </ListItemButton>

                    <ListItemButton
                        selected={view === 'setup-guide'}
                        onClick={() => setView('setup-guide')}
                        sx={{ borderRadius: 1, mx: 1, mb: 0.5 }}
                    >
                        <ListItemIcon sx={{ minWidth: 40, color: view === 'setup-guide' ? 'primary.main' : 'inherit' }}>
                            <MenuBookOutlined fontSize="small" />
                        </ListItemIcon>
                        <ListItemText primary="Setup Guide" primaryTypographyProps={{ fontSize: '0.875rem', fontWeight: 500 }} />
                    </ListItemButton>
                </List>

                <Divider sx={{ my: 1 }} />

                <List subheader={<ListSubheader sx={{ fontSize: '0.75rem', letterSpacing: 1, lineHeight: '32px' }}>ASSETS</ListSubheader>}>
                    {Object.entries(lines).map(([lineName, lineMachines]) => (
                        <Box key={lineName} sx={{ mb: 2 }}>
                            <Typography variant="caption" sx={{ px: 3, mb: 1, display: 'flex', alignItems: 'center', gap: 1, color: 'text.secondary', fontWeight: 600 }}>
                                {lineName}
                            </Typography>
                            {lineMachines.map(machine => {
                                const isCritical = (machine.failure_probability || 0) > 0.8;
                                const isWarning = (machine.failure_probability || 0) > 0.5;
                                const isActive = selectedMachineId === machine.machine_id;

                                return (
                                    <ListItemButton
                                        key={machine.machine_id}
                                        selected={isActive}
                                        onClick={() => {
                                            onSelectMachine(machine.machine_id);
                                            setView('detail');
                                        }}
                                        sx={{
                                            borderRadius: 1, mx: 1, py: 0.5, mb: 0.25,
                                            pl: 4,
                                            borderLeft: isActive ? '3px solid' : '3px solid transparent',
                                            borderColor: 'primary.main'
                                        }}
                                    >
                                        <Box
                                            sx={{
                                                width: 8, height: 8, borderRadius: '50%', mr: 2,
                                                bgcolor: isCritical ? 'error.main' : isWarning ? 'warning.main' : 'success.main',
                                                boxShadow: isCritical ? '0 0 0 2px rgba(220, 38, 38, 0.2)' : 'none'
                                            }}
                                        />
                                        <ListItemText
                                            primary={machine.machine_id}
                                            primaryTypographyProps={{ fontSize: '0.85rem', fontWeight: isActive ? 600 : 400 }}
                                        />
                                        {machine.operational_status === 'RUNNING' && (
                                            <Box sx={{ width: 6, height: 6, borderRadius: '50%', bgcolor: 'success.light', ml: 'auto' }} />
                                        )}
                                    </ListItemButton>
                                );
                            })}
                        </Box>
                    ))}
                </List>
            </Box>

            {/* Footer User Profile */}
            <Box sx={{ p: 2, borderTop: '1px solid', borderColor: 'divider' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Avatar sx={{ width: 32, height: 32, bgcolor: 'grey.300', fontSize: '0.875rem' }}>AG</Avatar>
                    <Box>
                        <Typography variant="body2" fontWeight={600}>Andrew G.</Typography>
                        <Typography variant="caption" color="text.secondary">Admin</Typography>
                    </Box>
                </Box>
            </Box>
        </Drawer>
    );
}

export default Sidebar;
