import React, { useState, useEffect, useCallback } from 'react';
import {
    Box, Card, CardContent, Typography, Button, TextField,
    LinearProgress, Chip, Select, MenuItem, FormControl, InputLabel,
    CircularProgress, Stack, Alert
} from '@mui/material';
import { CheckCircle, XCircle, RefreshCw, Tag } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts';

const API_BASE = 'http://localhost:8000/api/labeling';
const MILESTONES = [50, 100, 200, 500];

function getAuthHeaders() {
    const token = localStorage.getItem('access_token') || localStorage.getItem('token');
    if (token) return { Authorization: `Bearer ${token}` };
    return {};
}

/**
 * Chart thumbnail from feature_snapshot_json: if plottable (numeric arrays), render Recharts line; else key-value summary.
 */
function FeatureThumbnail({ snapshot }) {
    if (!snapshot || typeof snapshot !== 'object') {
        return <Typography variant="caption" color="text.secondary">No features</Typography>;
    }
    const entries = Object.entries(snapshot);
    const numericArrays = entries.filter(([, v]) => Array.isArray(v) && v.length > 0 && typeof v[0] === 'number');
    const scalarPairs = entries.filter(([k, v]) => typeof v === 'number' || typeof v === 'string');

    if (numericArrays.length > 0) {
        const firstArray = numericArrays[0][1];
        const data = firstArray.slice(0, 50).map((y, i) => ({ i, value: y }));
        return (
            <Box sx={{ width: '100%', height: 56 }}>
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
                        <XAxis dataKey="i" hide />
                        <YAxis hide domain={['auto', 'auto']} />
                        <Tooltip contentStyle={{ fontSize: 10 }} />
                        <Line type="monotone" dataKey="value" stroke="hsl(220 70% 50%)" strokeWidth={1} dot={false} />
                    </LineChart>
                </ResponsiveContainer>
            </Box>
        );
    }
    return (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
            {scalarPairs.slice(0, 6).map(([k, v]) => (
                <Chip key={k} size="small" label={`${k}: ${typeof v === 'number' ? v.toFixed(2) : v}`} sx={{ fontSize: '0.65rem', height: 20 }} />
            ))}
        </Box>
    );
}

const MACHINES_API = 'http://localhost:8000/api/machines';

export default function LabelingFeed({ machineIdFilter = null }) {
    const [tasks, setTasks] = useState([]);
    const [coverage, setCoverage] = useState(null);
    const [machines, setMachines] = useState([]);
    const [selectedMachine, setSelectedMachine] = useState(machineIdFilter || '');
    const [loading, setLoading] = useState(true);
    const [submitting, setSubmitting] = useState(null);
    const [notes, setNotes] = useState({});
    const [error, setError] = useState(null);

    useEffect(() => {
        fetch(MACHINES_API)
            .then((res) => res.json())
            .then((data) => setMachines(data.data || []))
            .catch(() => setMachines([]));
    }, []);

    const fetchTasks = useCallback(async () => {
        try {
            const params = new URLSearchParams({ status: 'pending' });
            if (selectedMachine) params.set('machine_id', selectedMachine);
            const res = await fetch(`${API_BASE}/tasks?${params}`, { headers: getAuthHeaders() });
            if (!res.ok) {
                if (res.status === 401) setError('Authentication required');
                else setError(res.statusText);
                setTasks([]);
                return;
            }
            setError(null);
            const data = await res.json();
            setTasks(data.data || []);
        } catch (err) {
            setError(err.message || 'Failed to load tasks');
            setTasks([]);
        }
    }, [selectedMachine]);

    const fetchCoverage = useCallback(async (machineId) => {
        if (!machineId) return;
        try {
            const res = await fetch(`${API_BASE}/coverage/${machineId}`, { headers: getAuthHeaders() });
            if (res.ok) setCoverage(await res.json());
            else setCoverage(null);
        } catch {
            setCoverage(null);
        }
    }, []);

    useEffect(() => {
        setLoading(true);
        Promise.all([fetchTasks(), selectedMachine ? fetchCoverage(selectedMachine) : Promise.resolve()]).finally(() => setLoading(false));
    }, [fetchTasks, fetchCoverage, selectedMachine]);

    const handleLabel = async (taskId, label) => {
        setSubmitting(taskId);
        const body = { label, notes: notes[taskId] || '' };
        try {
            const res = await fetch(`${API_BASE}/tasks/${taskId}/label`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
                body: JSON.stringify(body),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                setError(err.detail || res.statusText);
                setSubmitting(null);
                return;
            }
            setNotes((prev) => ({ ...prev, [taskId]: '' }));
            await fetchTasks();
            if (coverage?.total_anomalies !== undefined && selectedMachine) await fetchCoverage(selectedMachine);
        } catch (err) {
            setError(err.message);
        }
        setSubmitting(null);
    };

    const nextMilestone = MILESTONES.find((m) => (coverage?.labeled ?? 0) < m) ?? MILESTONES[MILESTONES.length - 1];
    const progress = coverage ? Math.min(100, (coverage.labeled / nextMilestone) * 100) : 0;

    return (
        <Box sx={{ p: 2, maxWidth: 900, mx: 'auto' }}>
            <Typography variant="h5" fontWeight={700} sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
                <Tag size={24} />
                Labeling Feed
            </Typography>

            {error && (
                <Alert severity="warning" onClose={() => setError(null)} sx={{ mb: 2 }}>
                    {error}
                </Alert>
            )}

            <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 2 }}>
                <FormControl size="small" sx={{ minWidth: 180 }}>
                    <InputLabel>Machine</InputLabel>
                    <Select
                        value={selectedMachine}
                        label="Machine"
                        onChange={(e) => setSelectedMachine(e.target.value)}
                    >
                        <MenuItem value="">All</MenuItem>
                        {machines.map((m) => (
                            <MenuItem key={m.machine_id} value={m.machine_id}>{m.machine_id}</MenuItem>
                        ))}
                    </Select>
                </FormControl>
                <Button startIcon={<RefreshCw size={16} />} onClick={() => { setLoading(true); fetchTasks(); selectedMachine && fetchCoverage(selectedMachine); setLoading(false); }}>
                    Refresh
                </Button>
            </Stack>

            {coverage && (
                <Card variant="outlined" sx={{ mb: 2 }}>
                    <CardContent>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                            Progress — supervised model trains at 50, 100, 200, 500 labels
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
                            <LinearProgress variant="determinate" value={progress} sx={{ flex: 1, minWidth: 120, height: 8, borderRadius: 1 }} />
                            <Typography variant="body2" fontWeight={600}>
                                {coverage.labeled} / {nextMilestone} labels
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                                (Total: {coverage.total_anomalies} · Confirmed: {coverage.confirmed} · Rejected: {coverage.rejected} · Coverage: {coverage.coverage_pct}%)
                            </Typography>
                        </Box>
                    </CardContent>
                </Card>
            )}

            {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                    <CircularProgress />
                </Box>
            ) : tasks.length === 0 ? (
                <Typography color="text.secondary">No pending labeling tasks. Create tasks from anomaly events to start.</Typography>
            ) : (
                <Stack spacing={2}>
                    {tasks.map((task) => (
                        <Card key={task.task_id} variant="outlined">
                            <CardContent>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 2 }}>
                                    <Box sx={{ flex: 1, minWidth: 200 }}>
                                        <Typography variant="subtitle2" color="text.secondary">
                                            {task.machine_id} · {task.created_at ? new Date(task.created_at).toLocaleString() : task.anomaly_event_id}
                                        </Typography>
                                        <FeatureThumbnail snapshot={task.feature_snapshot_json} />
                                        <TextField
                                            size="small"
                                            placeholder="Notes (optional)"
                                            value={notes[task.task_id] || ''}
                                            onChange={(e) => setNotes((prev) => ({ ...prev, [task.task_id]: e.target.value }))}
                                            sx={{ mt: 1, width: '100%', maxWidth: 320 }}
                                        />
                                    </Box>
                                    <Stack direction="row" spacing={1} alignItems="center">
                                        <Button
                                            variant="contained"
                                            color="error"
                                            size="small"
                                            startIcon={<XCircle size={16} />}
                                            disabled={submitting === task.task_id}
                                            onClick={() => handleLabel(task.task_id, 0)}
                                        >
                                            False Alarm
                                        </Button>
                                        <Button
                                            variant="contained"
                                            color="primary"
                                            size="small"
                                            startIcon={<CheckCircle size={16} />}
                                            disabled={submitting === task.task_id}
                                            onClick={() => handleLabel(task.task_id, 1)}
                                        >
                                            Confirmed Issue
                                        </Button>
                                    </Stack>
                                </Box>
                            </CardContent>
                        </Card>
                    ))}
                </Stack>
            )}
        </Box>
    );
}
