import React, { useState } from 'react';
import {
    Clock, AlertTriangle, TrendingUp, ArrowRight, Calendar,
    Bell, Flame, AlertOctagon, Zap, Lightbulb, Wrench, Activity, ChevronRight, Check
} from 'lucide-react';
import {
    Card, CardContent, Typography, Box, Chip, Button,
    LinearProgress, Grid, Stack, IconButton, Tooltip, Collapse, Checkbox
} from '@mui/material';
import { alpha } from '@mui/material/styles';

/**
 * CriticalMachineCard (Industrial Alert Style)
 * 
 * High-density alert card for the "Priority Attention" feed.
 */
export function CriticalMachineCard({ machine, onViewDetails, onSchedule, onAlert, isPrimary = false, selected = false, onToggleSelect }) {
    const [isHovered, setIsHovered] = useState(false);

    const prob = (machine.failure_probability || 0) * 100;
    const rulDays = machine.rul_days || 0;

    // Mock "CRITICAL IN" calc (inverse of RUL for demo)
    // < 4h is "Urgent"
    const criticalHours = rulDays * 24;
    const isUrgent = criticalHours < 4;

    // Determine urgency level
    const getUrgency = () => {
        if (prob > 95) return {
            level: 'EXTREME',
            color: 'error',
            gradient: 'linear-gradient(90deg, #ef4444 0%, #dc2626 100%)',
        };
        if (prob > 85) return {
            level: 'HIGH',
            color: 'warning', // Orange-ish in heavy industrial context often maps to warning structure
            gradient: 'linear-gradient(90deg, #f97316 0%, #ef4444 100%)',
        };
        return {
            level: 'ELEVATED',
            color: 'warning',
            gradient: 'linear-gradient(90deg, #f59e0b 0%, #f97316 100%)',
        };
    };

    const urgency = getUrgency();

    return (
        <Card
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            sx={{
                position: 'relative',
                overflow: 'visible', // For hover glow if needed, but here we use box shadow
                borderRadius: 1,
                border: '1px solid',
                borderColor: selected ? 'primary.main' : (isUrgent ? 'error.main' : 'divider'),
                boxShadow: isUrgent ? '0 0 12px rgba(220, 38, 38, 0.15)' : 'none',
                transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                bgcolor: selected ? 'primary.50' : 'background.paper',
                '&:hover': {
                    transform: 'translateY(-1px)',
                    boxShadow: isUrgent ? '0 0 16px rgba(220, 38, 38, 0.3)' : 2,
                    zIndex: 10
                }
            }}
        >
            {isPrimary && (
                <Box sx={{
                    position: 'absolute', top: 0, left: 0, right: 0, height: 2,
                    background: 'linear-gradient(90deg, #ef4444, #f59e0b)'
                }} />
            )}

            {/* Batch Selection Checkbox */}
            <Box sx={{ position: 'absolute', top: 8, left: 8, zIndex: 20 }}>
                <Checkbox
                    size="small"
                    checked={selected}
                    onChange={(e) => onToggleSelect && onToggleSelect(machine.machine_id, e.target.checked)}
                    sx={{ p: 0.5, bgcolor: 'rgba(255,255,255,0.8)', '&:hover': { bgcolor: 'white' } }}
                />
            </Box>

            <CardContent sx={{ p: '12px !important', pl: '36px !important', '&:last-child': { pb: '12px !important' } }}>
                {/* Header: ID & Badges */}
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1.5 }}>
                    <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>
                        {/* Pulsing Status Dot */}
                        <Box sx={{ position: 'relative', display: 'flex' }}>
                            <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: 'error.main', animation: 'pulse 1.5s infinite' }} />
                            {isUrgent && (
                                <Box sx={{ position: 'absolute', inset: -2, borderRadius: '50%', border: '1px solid', borderColor: 'error.main', animation: 'ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite', opacity: 0.5 }} />
                            )}
                        </Box>

                        <Box>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Typography variant="subtitle2" fontWeight="800" fontFamily="monospace" sx={{ fontSize: '0.9rem' }}>
                                    {machine.machine_id}
                                </Typography>
                                {isPrimary && (
                                    <Chip label="PRIMARY ALERT" size="small" color="error" variant="filled" sx={{ height: 16, fontSize: '0.6rem', fontWeight: 'bold' }} />
                                )}
                            </Box>
                            <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
                                {machine.model_number || 'Asset Type N/A'}
                            </Typography>
                        </Box>
                    </Box>

                    {/* Shortcuts & Badge */}
                    <Stack direction="row" spacing={0.5} alignItems="center">
                        <Tooltip title="Quick Acknowledge">
                            <IconButton size="small" onClick={(e) => { e.stopPropagation(); onAlert && onAlert(machine.machine_id); }} sx={{ p: 0.5, color: 'success.main', opacity: 0.8, '&:hover': { opacity: 1, bgcolor: 'success.50' } }}>
                                <Check size={14} />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Create Work Order">
                            <IconButton size="small" onClick={(e) => { e.stopPropagation(); onSchedule && onSchedule(machine.machine_id); }} sx={{ p: 0.5, color: 'primary.main', opacity: 0.8, '&:hover': { opacity: 1, bgcolor: 'primary.50' } }}>
                                <Wrench size={14} />
                            </IconButton>
                        </Tooltip>
                        <Chip
                            label={urgency.level}
                            size="small"
                            sx={{
                                height: 20, fontSize: '0.65rem', fontWeight: '800',
                                bgcolor: alpha(urgency.color === 'error' ? '#ef4444' : '#f59e0b', 0.1),
                                color: urgency.color === 'error' ? 'error.main' : 'warning.main',
                                border: '1px solid', borderColor: urgency.color === 'error' ? 'error.light' : 'warning.light'
                            }}
                        />
                    </Stack>
                </Box>

                {/* Compact Grid Metrics */}
                <Box sx={{
                    display: 'grid',
                    gridTemplateColumns: '1fr 1fr',
                    gap: 1,
                    mb: 1.5,
                    bgcolor: 'grey.50',
                    p: 1,
                    borderRadius: 1,
                    border: '1px solid',
                    borderColor: 'divider'
                }}>
                    <Box>
                        <Typography variant="caption" color="text.secondary" fontWeight="700" sx={{ fontSize: '0.65rem', letterSpacing: 0.5 }}>CRITICAL IN</Typography>
                        <Stack direction="row" alignItems="baseline" spacing={0.5}>
                            <Typography variant="body2" fontWeight="bold" fontFamily="monospace" color={isUrgent ? 'error.main' : 'text.primary'}>
                                {criticalHours < 1 ? '< 1h' : `${criticalHours.toFixed(1)}h`}
                            </Typography>
                        </Stack>
                    </Box>
                    <Box sx={{ pl: 1, borderLeft: '1px solid', borderColor: 'divider' }}>
                        <Typography variant="caption" color="text.secondary" fontWeight="700" sx={{ fontSize: '0.65rem', letterSpacing: 0.5 }}>EST. FAIL</Typography>
                        <Stack direction="row" alignItems="baseline" spacing={0.5}>
                            <Typography variant="body2" fontWeight="bold" fontFamily="monospace" color="text.primary">
                                {new Date(Date.now() + rulDays * 24 * 60 * 60 * 1000).toLocaleDateString(undefined, { month: 'numeric', day: 'numeric' })}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">({rulDays.toFixed(0)}d)</Typography>
                        </Stack>
                    </Box>
                </Box>

                {/* Refined Progress Bar */}
                <Box sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                        <Typography variant="caption" fontWeight="bold" color="text.secondary" fontSize="0.65rem">FAILURE PROBABILITY</Typography>
                        <Typography variant="caption" fontWeight="bold" color={urgency.color + '.main'} fontSize="0.65rem">{prob.toFixed(1)}%</Typography>
                    </Box>
                    <Box sx={{ position: 'relative', height: 4, bgcolor: 'grey.200', borderRadius: 2, overflow: 'hidden' }}>
                        <Box sx={{
                            position: 'absolute', left: 0, top: 0, bottom: 0, width: `${Math.min(prob, 100)}%`,
                            background: urgency.gradient
                        }} />
                        {/* 95% CI Shaded Region (Simulated at the top end) */}
                        <Box sx={{
                            position: 'absolute', right: 0, top: 0, bottom: 0, width: '5%',
                            bgcolor: 'rgba(0,0,0,0.1)', borderLeft: '1px solid rgba(255,255,255,0.5)'
                        }} />
                    </Box>
                </Box>

                {/* Action Chips & Hover Reveal */}
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', height: 24 }}>
                    <Stack direction="row" spacing={1}>
                        <Chip
                            icon={<Calendar size={12} />}
                            label="Schedule"
                            size="small"
                            onClick={() => onSchedule && onSchedule(machine.machine_id)}
                            sx={{
                                height: 24, fontSize: '0.7rem', fontWeight: 'bold', cursor: 'pointer',
                                bgcolor: 'primary.50', color: 'primary.700', border: '1px solid', borderColor: 'primary.100',
                                '&:hover': { bgcolor: 'primary.100' }
                            }}
                        />
                        <Chip
                            icon={<AlertTriangle size={12} />}
                            label="Alert"
                            size="small"
                            onClick={() => onAlert && onAlert(machine.machine_id)}
                            sx={{
                                height: 24, fontSize: '0.7rem', fontWeight: 'bold', cursor: 'pointer',
                                bgcolor: 'transparent', color: 'text.secondary', border: '1px solid', borderColor: 'divider',
                                '&:hover': { bgcolor: 'grey.50', color: 'text.primary' }
                            }}
                        />
                    </Stack>

                    {/* Hover Reveal Shortcut */}
                    <Box sx={{
                        opacity: isHovered ? 1 : 0,
                        transform: isHovered ? 'translateX(0)' : 'translateX(10px)',
                        transition: 'all 0.2s ease',
                        display: 'flex', alignItems: 'center'
                    }}>
                        <Tooltip title="View Sensor Data" arrow placement="left">
                            <IconButton size="small" onClick={() => onViewDetails(machine.machine_id)}>
                                <Activity size={16} className="text-slate-500 hover:text-blue-600" />
                            </IconButton>
                        </Tooltip>
                    </Box>
                </Box>

            </CardContent>
        </Card>
    );
}

export default CriticalMachineCard;
