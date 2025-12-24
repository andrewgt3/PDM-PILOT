import React, { useState, useEffect } from 'react';
import { ClipboardList, Plus, Clock, User, CheckCircle, Send, Calendar, RefreshCw } from 'lucide-react';
import { Card, CardContent, Typography, Box, Button, TextField, MenuItem, Chip, Stack, IconButton, Collapse, Alert, CircularProgress, Divider } from '@mui/material';

/**
 * WorkOrderPanel Component
 * 
 * Real work order management with database persistence.
 */
function WorkOrderPanel({ machine }) {
    const [orders, setOrders] = useState([]);
    const [loading, setLoading] = useState(true);
    const [showForm, setShowForm] = useState(false);
    const [submitting, setSubmitting] = useState(false);
    const [submitted, setSubmitted] = useState(false);

    // Form state
    const [formData, setFormData] = useState({
        title: '',
        description: '',
        priority: 'medium',
        work_type: 'corrective'
    });

    const machineId = machine?.machine_id;

    const fetchOrders = async () => {
        try {
            const url = machineId
                ? `http://localhost:8000/api/enterprise/work-orders?machine_id=${machineId}&limit=10`
                : `http://localhost:8000/api/enterprise/work-orders?limit=10`;

            const response = await fetch(url);
            if (response.ok) {
                const result = await response.json();
                setOrders(result.data || []);
            }
        } catch (err) {
            console.error('[Work Orders API Error]', err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchOrders();
    }, [machineId]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setSubmitting(true);

        try {
            const response = await fetch('http://localhost:8000/api/enterprise/work-orders', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    machine_id: machineId,
                    title: formData.title || 'Maintenance Required',
                    description: formData.description,
                    priority: formData.priority,
                    work_type: formData.work_type
                })
            });

            if (response.ok) {
                setSubmitted(true);
                setShowForm(false);
                setFormData({ title: '', description: '', priority: 'medium', work_type: 'corrective' });
                fetchOrders(); // Refresh list
                setTimeout(() => setSubmitted(false), 3000);
            }
        } catch (err) {
            console.error('[Create Work Order Error]', err);
        } finally {
            setSubmitting(false);
        }
    };

    const updateStatus = async (orderId, newStatus) => {
        try {
            await fetch(`http://localhost:8000/api/enterprise/work-orders/${orderId}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ status: newStatus })
            });
            fetchOrders();
        } catch (err) {
            console.error('[Update Status Error]', err);
        }
    };

    const statusConfig = {
        completed: { color: 'success', label: 'Completed' },
        scheduled: { color: 'info', label: 'Scheduled' },
        in_progress: { color: 'warning', label: 'In Progress' },
        pending: { color: 'default', label: 'Pending' },
        cancelled: { color: 'error', label: 'Cancelled' }
    };

    const priorityConfig = {
        critical: 'error',
        high: 'warning',
        medium: 'warning', // Amber mapped to warning
        low: 'default'
    };

    const formatDate = (dateStr) => {
        if (!dateStr) return '-';
        return new Date(dateStr).toLocaleDateString();
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
                    <ClipboardList size={20} className="text-slate-600" />
                    <Typography variant="subtitle2" fontWeight="bold">Work Orders</Typography>
                </Stack>
                <Stack direction="row" spacing={1}>
                    <IconButton size="small" onClick={fetchOrders} title="Refresh">
                        <RefreshCw size={16} />
                    </IconButton>
                    <Button
                        variant="contained"
                        size="small"
                        startIcon={<Plus size={16} />}
                        onClick={() => setShowForm(!showForm)}
                        sx={{ fontSize: '0.75rem' }}
                    >
                        Create Order
                    </Button>
                </Stack>
            </Box>

            {/* Success Message */}
            <Collapse in={submitted}>
                <Alert severity="success" sx={{ borderRadius: 0 }}>
                    Work order created successfully!
                </Alert>
            </Collapse>

            {/* Quick Create Form */}
            <Collapse in={showForm}>
                <Box component="form" onSubmit={handleSubmit} sx={{ p: 2, bgcolor: 'primary.lighter', borderBottom: 1, borderColor: 'primary.light' }}>
                    <Typography variant="subtitle2" fontWeight="bold" gutterBottom>New Work Order</Typography>
                    <Stack spacing={2}>
                        <TextField
                            label="Title"
                            size="small"
                            fullWidth
                            required
                            value={formData.title}
                            onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                            placeholder="e.g., Replace bearing - predicted failure"
                        />
                        <TextField
                            label="Description"
                            size="small"
                            fullWidth
                            multiline
                            rows={2}
                            value={formData.description}
                            onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                        />
                        <Stack direction="row" spacing={2}>
                            <TextField
                                select
                                label="Priority"
                                size="small"
                                fullWidth
                                value={formData.priority}
                                onChange={(e) => setFormData({ ...formData, priority: e.target.value })}
                            >
                                <MenuItem value="critical">Critical</MenuItem>
                                <MenuItem value="high">High</MenuItem>
                                <MenuItem value="medium">Medium</MenuItem>
                                <MenuItem value="low">Low</MenuItem>
                            </TextField>
                            <TextField
                                select
                                label="Type"
                                size="small"
                                fullWidth
                                value={formData.work_type}
                                onChange={(e) => setFormData({ ...formData, work_type: e.target.value })}
                            >
                                <MenuItem value="corrective">Corrective</MenuItem>
                                <MenuItem value="preventive">Preventive</MenuItem>
                                <MenuItem value="inspection">Inspection</MenuItem>
                            </TextField>
                        </Stack>
                        <Stack direction="row" spacing={1} justifyContent="flex-end">
                            <Button size="small" onClick={() => setShowForm(false)} color="inherit">Cancel</Button>
                            <Button type="submit" variant="contained" size="small" disabled={submitting} startIcon={<Send size={16} />}>
                                {submitting ? 'Creating...' : 'Create Work Order'}
                            </Button>
                        </Stack>
                    </Stack>
                </Box>
            </Collapse>

            {/* Work Orders List */}
            <Box sx={{ flex: 1, overflowY: 'auto', maxHeight: 350 }}>
                {orders.length === 0 ? (
                    <Box sx={{ p: 4, textAlign: 'center', color: 'text.secondary' }}>
                        <ClipboardList size={32} style={{ opacity: 0.3, marginBottom: 8 }} />
                        <Typography variant="body2">No work orders yet</Typography>
                        <Typography variant="caption">Click "Create Order" to add one</Typography>
                    </Box>
                ) : (
                    <Stack divider={<Divider />}>
                        {orders.map(order => {
                            const status = statusConfig[order.status] || statusConfig.pending;
                            return (
                                <Box key={order.work_order_id} sx={{ p: 2, '&:hover': { bgcolor: 'action.hover' } }}>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                                        <Stack direction="row" spacing={1} alignItems="center">
                                            <Typography variant="caption" fontFamily="monospace" color="text.secondary">
                                                {order.work_order_id}
                                            </Typography>
                                            <Chip
                                                label={order.priority?.toUpperCase()}
                                                size="small"
                                                variant="outlined"
                                                color={priorityConfig[order.priority] || 'default'}
                                                sx={{ height: 20, fontSize: '0.65rem' }}
                                            />
                                            <Typography variant="caption" color="text.secondary" sx={{ textTransform: 'capitalize' }}>
                                                {order.work_type}
                                            </Typography>
                                        </Stack>
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                            {order.status === 'pending' && (
                                                <Button size="small" onClick={() => updateStatus(order.work_order_id, 'in_progress')} sx={{ fontSize: '0.7rem', minWidth: 'auto', p: 0.5 }}>
                                                    Start
                                                </Button>
                                            )}
                                            {order.status === 'in_progress' && (
                                                <Button size="small" color="success" onClick={() => updateStatus(order.work_order_id, 'completed')} sx={{ fontSize: '0.7rem', minWidth: 'auto', p: 0.5 }}>
                                                    Complete
                                                </Button>
                                            )}
                                            <Chip label={status.label} color={status.color} size="small" sx={{ height: 20, fontSize: '0.65rem', fontWeight: 'bold' }} />
                                        </Box>
                                    </Box>

                                    <Typography variant="body2" fontWeight="medium" noWrap gutterBottom>
                                        {order.title}
                                    </Typography>

                                    <Stack direction="row" spacing={2} sx={{ mt: 1 }}>
                                        <Stack direction="row" spacing={0.5} alignItems="center">
                                            <Calendar size={12} className="text-slate-400" />
                                            <Typography variant="caption" color="text.secondary">{formatDate(order.created_at)}</Typography>
                                        </Stack>
                                        {order.assigned_to && (
                                            <Stack direction="row" spacing={0.5} alignItems="center">
                                                <User size={12} className="text-slate-400" />
                                                <Typography variant="caption" color="text.secondary">{order.assigned_to}</Typography>
                                            </Stack>
                                        )}
                                    </Stack>
                                </Box>
                            );
                        })}
                    </Stack>
                )}
            </Box>

            {/* Footer */}
            <Box sx={{ px: 2, py: 1, bgcolor: 'grey.50', borderTop: 1, borderColor: 'divider', display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="caption" color="text.secondary">{orders.length} work orders</Typography>
                <Typography variant="caption" color="text.disabled">Source: PDM Database</Typography>
            </Box>
        </Card>
    );
}

export default WorkOrderPanel;
