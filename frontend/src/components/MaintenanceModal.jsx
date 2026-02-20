import React, { useState, useEffect } from 'react';
import {
    Dialog, DialogTitle, DialogContent, DialogActions,
    TextField, Button, Box, Typography, Chip, MenuItem, InputAdornment,
    Alert, Stack, LinearProgress
} from '@mui/material';
import { Calendar, Wrench, AlertTriangle, CheckCircle, Clock } from 'lucide-react';

function MaintenanceModal({ open, onClose, machines = [], onConfirm, initialTitle, initialPriority, initialParts }) {
    const isBatch = machines.length > 1;
    const primaryMachine = machines[0] || {};

    // Form State
    const [title, setTitle] = useState('');
    const [priority, setPriority] = useState('normal');
    const [assignedTo, setAssignedTo] = useState('Team A');
    const [estimatedDuration, setEstimatedDuration] = useState(2); // hours
    const [partsChips, setPartsChips] = useState([]);

    useEffect(() => {
        if (open && machines.length > 0) {
            if (initialTitle != null && initialTitle !== '') {
                setTitle(initialTitle);
            } else if (isBatch) {
                setTitle(`Batch Maintenance: ${machines.length} Assets`);
            } else {
                setTitle(`Predictive Check: ${primaryMachine.machine_id}`);
            }
            if (initialPriority != null && initialPriority !== '') {
                setPriority(String(initialPriority).toLowerCase());
            } else {
                const maxRisk = Math.max(...machines.map(m => m.failure_probability || 0));
                setPriority(maxRisk > 0.8 ? 'critical' : maxRisk > 0.5 ? 'high' : 'normal');
            }
            if (Array.isArray(initialParts) && initialParts.length > 0) {
                setPartsChips(initialParts);
            } else {
                setPartsChips([]);
            }
        }
    }, [open, machines, isBatch, primaryMachine.machine_id, initialTitle, initialPriority, initialParts]);

    const handleSubmit = () => {
        // Validation logic here
        onConfirm({ title, priority, assignedTo, machines });
        onClose();
    };

    return (
        <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
            <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1.5, borderBottom: 1, borderColor: 'divider' }}>
                <Box sx={{ p: 1, bgcolor: 'primary.50', borderRadius: 1, color: 'primary.main' }}>
                    <Calendar size={20} />
                </Box>
                <Box>
                    <Typography variant="h6" fontWeight="bold">
                        {isBatch ? 'Schedule Bulk Maintenance' : 'Schedule Work Order'}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                        {isBatch ? `${machines.length} Assets Selected` : `Asset ID: ${primaryMachine.machine_id}`}
                    </Typography>
                </Box>
            </DialogTitle>

            <DialogContent sx={{ pt: 3 }}>
                {/* Risk Context */}
                {machines.some(m => (m.failure_probability || 0) > 0.8) && (
                    <Alert severity="error" icon={<AlertTriangle size={18} />} sx={{ mb: 3 }}>
                        <strong>Critical Risk Detected:</strong> Immediate action recommended for {machines.filter(m => (m.failure_probability || 0) > 0.8).length} asset(s).
                    </Alert>
                )}

                <Stack spacing={3} sx={{ mt: 1 }}>
                    <TextField
                        label="Work Order Title"
                        fullWidth
                        value={title}
                        onChange={(e) => setTitle(e.target.value)}
                        variant="outlined"
                        InputProps={{
                            startAdornment: <InputAdornment position="start"><Wrench size={16} /></InputAdornment>,
                        }}
                    />

                    <Stack direction="row" spacing={2}>
                        <TextField
                            select
                            label="Priority Level"
                            value={priority}
                            onChange={(e) => setPriority(e.target.value)}
                            fullWidth
                        >
                            <MenuItem value="critical"><Typography color="error" fontWeight="bold">Critical</Typography></MenuItem>
                            <MenuItem value="high"><Typography color="warning.main" fontWeight="bold">High</Typography></MenuItem>
                            <MenuItem value="normal">Normal</MenuItem>
                            <MenuItem value="low">Low</MenuItem>
                        </TextField>

                        <TextField
                            select
                            label="Assigned Team"
                            value={assignedTo}
                            onChange={(e) => setAssignedTo(e.target.value)}
                            fullWidth
                        >
                            <MenuItem value="Team A">Mechanical - Team A</MenuItem>
                            <MenuItem value="Team B">Electrical - Team B</MenuItem>
                            <MenuItem value="Team C">Inspectors</MenuItem>
                        </TextField>
                    </Stack>

                    <Box>
                        <Typography variant="caption" color="text.secondary" gutterBottom>
                            Recommended Parts {partsChips.length > 0 ? '' : '(Auto-Generated)'}
                        </Typography>
                        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                            {isBatch && partsChips.length === 0 ? (
                                <Chip label="Batch Kit A" size="small" variant="outlined" />
                            ) : partsChips.length > 0 ? (
                                partsChips.map((p, i) => <Chip key={i} label={p} size="small" variant="outlined" />)
                            ) : (
                                <>
                                    <Chip label="Bearing Set SKF-22" size="small" variant="outlined" />
                                    <Chip label="Lubricant Pack" size="small" variant="outlined" />
                                    <Chip label="Seal Kit" size="small" variant="outlined" />
                                </>
                            )}
                        </Stack>
                    </Box>

                    <Box sx={{ p: 2, bgcolor: 'grey.50', borderRadius: 2, border: 1, borderColor: 'divider' }}>
                        <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                            Availability Forecast
                        </Typography>
                        <Stack direction="row" alignItems="center" spacing={2}>
                            <Box sx={{ flex: 1 }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                    <Typography variant="caption">Projected Downtime</Typography>
                                    <Typography variant="caption" fontWeight="bold">{estimatedDuration} Hours</Typography>
                                </Box>
                                <LinearProgress variant="determinate" value={30} sx={{ height: 6, borderRadius: 3 }} />
                            </Box>
                            <Chip
                                icon={<CheckCircle size={14} />}
                                label="Window Open"
                                color="success"
                                size="small"
                                variant="filled"
                                sx={{ bgcolor: 'success.lighter', color: 'success.dark', fontWeight: 'bold' }}
                            />
                        </Stack>
                    </Box>

                </Stack>
            </DialogContent>

            <DialogActions sx={{ px: 3, pb: 2 }}>
                <Button onClick={onClose} color="inherit">Cancel</Button>
                <Button
                    onClick={handleSubmit}
                    variant="contained"
                    startIcon={<Calendar size={18} />}
                    sx={{ px: 3 }}
                >
                    Confirm Schedule
                </Button>
            </DialogActions>
        </Dialog>
    );
}

export default MaintenanceModal;
