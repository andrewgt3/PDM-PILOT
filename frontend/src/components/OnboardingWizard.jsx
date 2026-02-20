import React, { useState, useCallback, useEffect } from 'react';
import {
    Box,
    Typography,
    Button,
    TextField,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    Slider,
    Stepper,
    Step,
    StepLabel,
    Radio,
    RadioGroup,
    FormControlLabel,
    FormHelperText,
    Stack,
    Paper,
    LinearProgress,
    Alert,
} from '@mui/material';
import { CheckCircle, Error as ErrorIcon, ArrowBack, ArrowForward } from '@mui/icons-material';

const API_BASE = import.meta.env?.VITE_API_BASE ?? 'http://localhost:8000';

const MACHINE_TYPES = ['Robot', 'CNC', 'Press', 'Conveyor', 'Other'];
const SEMANTIC_OPTIONS = ['Torque', 'Current', 'Speed', 'Temperature', 'Vibration', 'Other'];
const NOTIFICATION_METHODS = ['Email', 'Webhook', 'Both'];

const STEP_LABELS = {
    await_minimum_data: 'Collecting Data',
    run_data_quality_check: 'Data quality check',
    compute_healthy_baseline: 'Building Baseline',
    train_bootstrap_model: 'Training Model',
    activate_model: 'Live',
    send_onboarding_complete_notification: 'Live',
};

function slug(str) {
    return String(str || '')
        .trim()
        .replace(/\s+/g, '-')
        .replace(/[^a-zA-Z0-9-_]/g, '')
        .slice(0, 40) || 'machine';
}

function randomAlphanumeric(len = 4) {
    const chars = 'abcdefghijklmnopqrstuvwxyz0123456789';
    let out = '';
    for (let i = 0; i < len; i++) out += chars[Math.floor(Math.random() * chars.length)];
    return out;
}

function generateMachineId(machineName) {
    return `${slug(machineName)}_${randomAlphanumeric(4)}`;
}

export function OnboardingWizard({ onComplete, onCancel, apiBase = API_BASE }) {
    const [step, setStep] = useState(1);
    const [launchSuccess, setLaunchSuccess] = useState(false);
    const [onboardingMachineId, setOnboardingMachineId] = useState(null);
    const [progressStatus, setProgressStatus] = useState(null);
    const [progressError, setProgressError] = useState(null);

    const [form, setForm] = useState({
        machineName: '',
        machineType: 'Robot',
        manufacturer: '',
        model: '',
        location: '',
        machine_id: '',
        assetType: 'abb_robot',
        ip: '',
        port: 4840,
        credentials: { rack: 0, slot: 1, db_number: 1 },
        connectionTest: null,
        signals_detected: [],
        samples: {},
        signal_mapping: {},
        warning_threshold: 0.5,
        critical_threshold: 0.8,
        notification_method: 'Email',
        email: '',
        webhook_url: '',
        downtime_cost_estimate: '',
    });

    const updateForm = useCallback((updates) => {
        setForm((prev) => {
            const next = { ...prev, ...updates };
            if (updates.machineName !== undefined) {
                next.machine_id = generateMachineId(updates.machineName || prev.machineName);
            }
            return next;
        });
    }, []);

    const [verifyLoading, setVerifyLoading] = useState(false);
    const [verifyResult, setVerifyResult] = useState(null);

    const handleTestConnection = useCallback(async () => {
        setVerifyLoading(true);
        setVerifyResult(null);
        try {
            const body = {
                asset_type: form.assetType,
                ip: form.ip,
                port: Number(form.port) || 4840,
                credentials: form.assetType === 'siemens_plc' ? form.credentials : {},
                fetch_samples: 5,
            };
            const res = await fetch(`${apiBase}/api/verify-connection`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) {
                setVerifyResult({ connected: false, error: data.detail || data.error || 'Connection failed' });
                return;
            }
            setVerifyResult(data);
            if (data.connected && data.signals_detected) {
                updateForm({
                    signals_detected: data.signals_detected,
                    samples: data.samples || {},
                    signal_mapping: data.signals_detected.reduce((acc, s) => ({ ...acc, [s]: 'Other' }), {}),
                });
            }
        } catch (e) {
            setVerifyResult({ connected: false, error: e.message || 'Network error' });
        } finally {
            setVerifyLoading(false);
        }
    }, [form.assetType, form.ip, form.port, form.credentials, apiBase, updateForm]);

    const handleStartMonitoring = useCallback(async () => {
        try {
            const payload = {
                machine_id: form.machine_id,
                machine_name: form.machineName,
                machine_type: form.machineType,
                manufacturer: form.manufacturer,
                model: form.model,
                location: form.location,
                connection: {
                    asset_type: form.assetType,
                    ip: form.ip,
                    port: Number(form.port) || 4840,
                    credentials: form.credentials,
                },
                signal_mapping: form.signal_mapping,
                alerts: {
                    warning_threshold: form.warning_threshold,
                    critical_threshold: form.critical_threshold,
                    notification_method: form.notification_method,
                    email: form.email,
                    webhook_url: form.webhook_url,
                    downtime_cost_estimate: form.downtime_cost_estimate ? Number(form.downtime_cost_estimate) : null,
                },
            };
            const res = await fetch(`${apiBase}/api/onboarding/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) throw new Error(data.detail || data.error || 'Start failed');
            setOnboardingMachineId(data.machine_id);
            setLaunchSuccess(true);
        } catch (e) {
            setProgressError(e.message);
        }
    }, [form, apiBase]);

    useEffect(() => {
        if (!launchSuccess || !onboardingMachineId) return;
        const t = setInterval(async () => {
            try {
                const res = await fetch(`${apiBase}/api/onboarding/${onboardingMachineId}`);
                const data = await res.json().catch(() => ({}));
                setProgressStatus(data);
                if (data.status === 'COMPLETE') clearInterval(t);
            } catch {
                // ignore
            }
        }, 4000);
        return () => clearInterval(t);
    }, [launchSuccess, onboardingMachineId, apiBase]);

    const currentStepLabel = progressStatus?.current_step
        ? (STEP_LABELS[progressStatus.current_step] || progressStatus.current_step)
        : 'Connecting';

    if (launchSuccess && onboardingMachineId) {
        const isComplete = progressStatus?.status === 'COMPLETE';
        const isFailed = ['STALLED', 'PAUSED', 'FAILED'].includes(progressStatus?.status);
        return (
            <Box sx={{ p: 3, maxWidth: 600, mx: 'auto' }}>
                <Typography variant="h6" gutterBottom>Onboarding progress</Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                    Machine: {onboardingMachineId}
                </Typography>
                <Stack spacing={1} sx={{ my: 2 }}>
                    {['Connecting', 'Collecting Data', 'Data quality check', 'Building Baseline', 'Training Model', 'Live'].map((label, i) => {
                        const done = isComplete || (label === 'Live' && progressStatus?.current_step === 'send_onboarding_complete_notification') || (i < 5 && progressStatus?.current_step && Object.keys(STEP_LABELS).indexOf(progressStatus.current_step) >= i);
                        const active = currentStepLabel === label;
                        return (
                            <Box key={label} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                {done || (active && isComplete) ? <CheckCircle color="success" fontSize="small" /> : active ? <LinearProgress sx={{ width: 24 }} /> : <Box sx={{ width: 24, height: 24, borderRadius: '50%', bgcolor: 'action.hover' }} />}
                                <Typography variant="body2" fontWeight={active ? 600 : 400}>{label}</Typography>
                            </Box>
                        );
                    })}
                </Stack>
                {progressStatus?.error_message && (
                    <Alert severity="error" sx={{ mt: 2 }}>{progressStatus.error_message}</Alert>
                )}
                {isComplete && (
                    <Button variant="contained" onClick={onComplete} sx={{ mt: 2 }}>Go to fleet</Button>
                )}
                {isFailed && (
                    <Button variant="outlined" onClick={onComplete} sx={{ mt: 2 }}>Back to overview</Button>
                )}
            </Box>
        );
    }

    return (
        <Box sx={{ p: 3, maxWidth: 640, mx: 'auto' }}>
            <Stepper activeStep={step - 1} sx={{ mb: 3 }}>
                {[1, 2, 3, 4, 5].map((s) => (
                    <Step key={s}><StepLabel>Step {s}</StepLabel></Step>
                ))}
            </Stepper>
            {progressError && <Alert severity="error" onClose={() => setProgressError(null)} sx={{ mb: 2 }}>{progressError}</Alert>}

            {/* Step 1 — Machine Identity */}
            {step === 1 && (
                <Paper sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>Machine Identity</Typography>
                    <Stack spacing={2}>
                        <TextField fullWidth label="Machine Name" value={form.machineName} onChange={(e) => updateForm({ machineName: e.target.value })} />
                        <FormControl fullWidth>
                            <InputLabel>Machine Type</InputLabel>
                            <Select value={form.machineType} label="Machine Type" onChange={(e) => updateForm({ machineType: e.target.value })}>
                                {MACHINE_TYPES.map((t) => <MenuItem key={t} value={t}>{t}</MenuItem>)}
                            </Select>
                        </FormControl>
                        <TextField fullWidth label="Manufacturer" value={form.manufacturer} onChange={(e) => updateForm({ manufacturer: e.target.value })} />
                        <TextField fullWidth label="Model" value={form.model} onChange={(e) => updateForm({ model: e.target.value })} />
                        <TextField fullWidth label="Location (e.g. MachineShop/Line_1)" value={form.location} onChange={(e) => updateForm({ location: e.target.value })} placeholder="Area / cell for UNS" />
                        <TextField fullWidth label="Machine ID" value={form.machine_id} InputProps={{ readOnly: true }} helperText="Auto-generated" />
                    </Stack>
                </Paper>
            )}

            {/* Step 2 — Connection Details */}
            {step === 2 && (
                <Paper sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>Connection Details</Typography>
                    <FormControl component="fieldset" sx={{ mb: 2 }}>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom>Asset type</Typography>
                        <RadioGroup row value={form.assetType} onChange={(e) => updateForm({ assetType: e.target.value })}>
                            <FormControlLabel value="abb_robot" control={<Radio />} label="ABB Robot" />
                            <FormControlLabel value="siemens_plc" control={<Radio />} label="Siemens PLC" />
                            <FormControlLabel value="opcua_generic" control={<Radio />} label="OPC-UA Generic" />
                        </RadioGroup>
                    </FormControl>
                    <Stack spacing={2}>
                        <TextField fullWidth label="IP address" value={form.ip} onChange={(e) => updateForm({ ip: e.target.value })} />
                        <TextField type="number" fullWidth label="Port" value={form.port} onChange={(e) => updateForm({ port: e.target.value })} />
                        {form.assetType === 'siemens_plc' && (
                            <>
                                <TextField type="number" fullWidth label="Rack" value={form.credentials.rack} onChange={(e) => updateForm({ credentials: { ...form.credentials, rack: Number(e.target.value) || 0 } })} />
                                <TextField type="number" fullWidth label="Slot" value={form.credentials.slot} onChange={(e) => updateForm({ credentials: { ...form.credentials, slot: Number(e.target.value) || 1 } })} />
                                <TextField type="number" fullWidth label="DB number" value={form.credentials.db_number} onChange={(e) => updateForm({ credentials: { ...form.credentials, db_number: Number(e.target.value) || 1 } })} />
                            </>
                        )}
                        <Button variant="outlined" onClick={handleTestConnection} disabled={verifyLoading}>Test Connection</Button>
                        {verifyResult && (
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                {verifyResult.connected ? <CheckCircle color="success" /> : <ErrorIcon color="error" />}
                                <Typography variant="body2" color={verifyResult.connected ? 'success.main' : 'error.main'}>
                                    {verifyResult.connected ? `Connected. ${(verifyResult.signals_detected || []).length} signals detected.` : verifyResult.error}
                                </Typography>
                            </Box>
                        )}
                    </Stack>
                </Paper>
            )}

            {/* Step 3 — Data Mapping */}
            {step === 3 && (
                <Paper sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>Data Mapping</Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>Map each signal to a semantic meaning. Preview shows sample values.</Typography>
                    <Stack spacing={2} sx={{ mt: 2 }}>
                        {(form.signals_detected || []).map((sig) => (
                            <Box key={sig} sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
                                <Typography variant="body2" sx={{ minWidth: 140 }}>{sig}</Typography>
                                <FormControl size="small" sx={{ minWidth: 160 }}>
                                    <Select value={form.signal_mapping[sig] || 'Other'} onChange={(e) => updateForm({ signal_mapping: { ...form.signal_mapping, [sig]: e.target.value } })}>
                                        {SEMANTIC_OPTIONS.map((o) => <MenuItem key={o} value={o}>{o}</MenuItem>)}
                                    </Select>
                                </FormControl>
                                {form.samples && form.samples[sig] && (
                                    <Typography variant="caption" color="text.secondary">
                                        Sample: {form.samples[sig].slice(0, 10).map((v) => Number(v).toFixed(2)).join(', ')}
                                    </Typography>
                                )}
                            </Box>
                        ))}
                        {(!form.signals_detected || form.signals_detected.length === 0) && (
                            <FormHelperText>Complete Step 2 and run Test Connection to see signals.</FormHelperText>
                        )}
                    </Stack>
                </Paper>
            )}

            {/* Step 4 — Alert Configuration */}
            {step === 4 && (
                <Paper sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>Alert Configuration</Typography>
                    <Stack spacing={2}>
                        <Typography variant="body2">WARNING threshold (failure probability)</Typography>
                        <Slider value={form.warning_threshold} min={0} max={1} step={0.05} valueLabelDisplay="auto" valueLabelFormat={(v) => `${(v * 100).toFixed(0)}%`} onChange={(_, v) => updateForm({ warning_threshold: v })} />
                        <Typography variant="body2">CRITICAL threshold (failure probability)</Typography>
                        <Slider value={form.critical_threshold} min={0} max={1} step={0.05} valueLabelDisplay="auto" valueLabelFormat={(v) => `${(v * 100).toFixed(0)}%`} onChange={(_, v) => updateForm({ critical_threshold: v })} />
                        <FormControl fullWidth>
                            <InputLabel>Notification method</InputLabel>
                            <Select value={form.notification_method} label="Notification method" onChange={(e) => updateForm({ notification_method: e.target.value })}>
                                {NOTIFICATION_METHODS.map((m) => <MenuItem key={m} value={m}>{m}</MenuItem>)}
                            </Select>
                        </FormControl>
                        {(form.notification_method === 'Email' || form.notification_method === 'Both') && (
                            <TextField fullWidth label="Email addresses (comma-separated)" value={form.email} onChange={(e) => updateForm({ email: e.target.value })} />
                        )}
                        {(form.notification_method === 'Webhook' || form.notification_method === 'Both') && (
                            <TextField fullWidth label="Webhook URL" value={form.webhook_url} onChange={(e) => updateForm({ webhook_url: e.target.value })} />
                        )}
                        <TextField fullWidth type="number" label="Estimated cost of unplanned downtime (optional)" value={form.downtime_cost_estimate} onChange={(e) => updateForm({ downtime_cost_estimate: e.target.value })} />
                    </Stack>
                </Paper>
            )}

            {/* Step 5 — Review & Launch */}
            {step === 5 && !launchSuccess && (
                <Paper sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>Review & Launch</Typography>
                    <Stack spacing={0.5} sx={{ mb: 2 }}>
                        <Typography variant="body2"><strong>Machine ID:</strong> {form.machine_id}</Typography>
                        <Typography variant="body2"><strong>Name:</strong> {form.machineName}</Typography>
                        <Typography variant="body2"><strong>Type:</strong> {form.machineType}</Typography>
                        <Typography variant="body2"><strong>Location:</strong> {form.location || '—'}</Typography>
                        <Typography variant="body2"><strong>Connection:</strong> {form.assetType} @ {form.ip}:{form.port}</Typography>
                        <Typography variant="body2"><strong>Thresholds:</strong> WARNING {(form.warning_threshold * 100).toFixed(0)}% / CRITICAL {(form.critical_threshold * 100).toFixed(0)}%</Typography>
                        <Typography variant="body2"><strong>Notification:</strong> {form.notification_method}</Typography>
                    </Stack>
                    <Button variant="contained" onClick={handleStartMonitoring}>Start Monitoring</Button>
                </Paper>
            )}

            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
                <Button startIcon={<ArrowBack />} disabled={step <= 1} onClick={() => setStep((s) => s - 1)}>Back</Button>
                {step < 5 && <Button variant="contained" endIcon={<ArrowForward />} onClick={() => setStep((s) => s + 1)}>Next</Button>}
                {step === 1 && onCancel && <Button onClick={onCancel}>Cancel</Button>}
            </Box>
        </Box>
    );
}

export default OnboardingWizard;
