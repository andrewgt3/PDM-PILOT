import React, { useEffect, useState, useCallback, useRef } from 'react';
import {
    Box, Typography, Button, Chip, IconButton, Tooltip,
    Paper, Fade, Grow, useTheme, LinearProgress,
    Collapse, Alert
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import {
    Refresh as RefreshIcon,
    ArrowForward as ArrowForwardIcon,
    FiberManualRecord as DotIcon,
    Storage as StorageIcon,
    Speed as SpeedIcon,
    Memory as MemoryIcon,
    Sensors as SensorsIcon,
    CleaningServices as CleaningServicesIcon,
    Hub as HubIcon,
    Save as SaveIcon,
    Psychology as PsychologyIcon,
    CloudUpload as CloudUploadIcon,
    AutoFixHigh as AutoFixHighIcon,
    PlayArrow as PlayArrowIcon,
    Dns as DnsIcon,
    Lan as LanIcon,
    Science as ScienceIcon,
    ExpandMore as ExpandMoreIcon,
    Info as InfoIcon
} from '@mui/icons-material';

// Use relative URLs so Vite proxy forwards to API (avoids CORS / connection issues)
const API_BASE = '';

/**
 * PipelineOps — Pipeline Operations Dashboard
 * 
 * Enterprise-grade visual control center for the 6-stage pipeline.
 * Matches the application's light theme (Slate/Indigo).
 */
export default function PipelineOps({ onGoLive }) {
    const theme = useTheme();
    const [pipelineData, setPipelineData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [autoRefresh, setAutoRefresh] = useState(true);
    const [refreshCount, setRefreshCount] = useState(0);
    const [uploading, setUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [uploadStats, setUploadStats] = useState({ current: 0, total: 0 });
    const [bootstrapping, setBootstrapping] = useState(false);
    const [setupGuideOpen, setSetupGuideOpen] = useState(false);
    const fileInputRef = useRef(null);
    const intervalRef = useRef(null);

    const readiness = pipelineData?.readiness || {};
    const archiveReady = readiness.archive_ready ?? false;

    // Stage configuration matched to theme palette
    const STAGE_CONFIG = {
        1: { icon: <SensorsIcon />, color: theme.palette.primary.main, label: 'INGEST', desc: 'OPC UA Poller' },
        2: { icon: <CleaningServicesIcon />, color: theme.palette.secondary.main, label: 'CLEANSE', desc: 'Outlier Removal' },
        3: { icon: <HubIcon />, color: theme.palette.info.main, label: 'CONTEXT', desc: 'Digital Twin Map' },
        4: { icon: <SaveIcon />, color: theme.palette.success.main, label: 'PERSIST', desc: 'TimescaleDB' },
        5: { icon: <PsychologyIcon />, color: theme.palette.warning.main, label: 'INFER', desc: 'XGBoost RUL' },
        6: { icon: <LanIcon />, color: theme.palette.primary.dark, label: 'ORCH', desc: 'API Gateway' },
    };

    // Fetch pipeline status
    const fetchStatus = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/api/pipeline/status`);
            const text = await res.text();
            let data = {};
            try {
                if (text) data = JSON.parse(text);
            } catch (_) {
                throw new Error(text || `HTTP ${res.status}`);
            }
            if (!res.ok) {
                const msg = res.status >= 500
                    ? `API server error (${res.status}). Ensure the backend is running: uvicorn api_server:app --port 8000`
                    : (data.detail || `HTTP ${res.status}`);
                throw new Error(msg);
            }
            setPipelineData(data);
            setError(null);
            setRefreshCount(c => c + 1);
        } catch (err) {
            const isNetworkError = err.name === 'TypeError' || err.message?.includes('fetch') || err.message?.includes('Failed to fetch');
            const msg = isNetworkError
                ? 'API server unreachable. Start the backend: uvicorn api_server:app --port 8000'
                : err.message;
            setError(msg);
            // Fallback state
            setPipelineData({
                stages: [1, 2, 3, 4, 5, 6].map(id => ({
                    id,
                    name: STAGE_CONFIG[id].label,
                    status: 'DOWN',
                    length: 0,
                    stream: null,
                    consumer_groups: [],
                })),
                buffer_size: 0,
                active_machines: 0,
                all_live: false,
            });
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchStatus();
    }, [fetchStatus]);

    useEffect(() => {
        if (autoRefresh) {
            intervalRef.current = setInterval(fetchStatus, 3000);
        }
        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current);
        };
    }, [autoRefresh, fetchStatus]);

    const handleUpload = async (event) => {
        const rawFiles = event.target.files;
        if (!rawFiles || rawFiles.length === 0) return;

        const files = Array.from(rawFiles).filter(f => !f.name.startsWith('.'));
        if (files.length === 0) return;

        setUploading(true);
        setUploadProgress(0);
        setUploadStats({ current: 0, total: files.length });

        const BATCH_SIZE = 50;

        try {
            for (let i = 0; i < files.length; i += BATCH_SIZE) {
                const batch = files.slice(i, i + BATCH_SIZE);
                const formData = new FormData();

                batch.forEach(file => {
                    // Use webkitRelativePath to preserve folder structure
                    const relPath = file.webkitRelativePath || file.name;
                    formData.append('files', file, relPath);
                });

                const res = await fetch(`${API_BASE}/api/ingest/upload`, {
                    method: 'POST',
                    body: formData,
                });

                if (!res.ok) throw new Error(`Batch upload failed: ${res.statusText}`);

                const currentCount = Math.min(i + BATCH_SIZE, files.length);
                setUploadStats({ current: currentCount, total: files.length });
                setUploadProgress(Math.round((currentCount / files.length) * 100));
            }

            // Final success
            fetchStatus();
        } catch (err) {
            console.error('Upload error:', err);
            setError(err.message);
        } finally {
            setUploading(false);
            setUploadProgress(0);
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
        }
    };

    const handleBootstrap = async () => {
        if (!archiveReady) {
            setError(`Archive not found at ${readiness.archive_path || 'NASA_ARCHIVE_PATH'}. Run: python scripts/download_nasa_data.py`);
            return;
        }
        if (!window.confirm("Extract NASA archive to ingest folder and start watcher? This runs in the background.")) return;

        setBootstrapping(true);
        setError(null);
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 60000);
            const res = await fetch(`${API_BASE}/api/ingest/bootstrap`, {
                method: 'POST',
                signal: controller.signal,
            });
            clearTimeout(timeoutId);
            const text = await res.text();
            let data = {};
            try {
                if (text) data = JSON.parse(text);
            } catch (_) {
                throw new Error(text || res.statusText || `HTTP ${res.status}`);
            }

            if (!res.ok) {
                let msg = typeof data.detail === 'string'
                    ? data.detail
                    : Array.isArray(data.detail)
                        ? data.detail.map(d => d.msg || d.message || JSON.stringify(d)).join('; ')
                        : data.message || data.detail?.message || `Bootstrap failed (HTTP ${res.status})`;
                if (res.status >= 500) {
                    msg = `API server error (${res.status}). Ensure the backend is running and NASA_ARCHIVE_PATH points to your archive.`;
                }
                throw new Error(msg);
            }

            // Show success feedback immediately since it's a background task
            window.alert(data.message || "NASA Archive extraction started in the background!");

            fetchStatus();
        } catch (err) {
            console.error('Bootstrap error:', err);
            let msg = err.message;
            if (err.name === 'AbortError') msg = 'Request timed out. The server may be busy copying files. Try again in a minute.';
            else if (err.message?.includes('fetch') || err.message?.includes('Failed to fetch')) msg = 'API server unreachable. Start the backend: uvicorn api_server:app --port 8000';
            setError(msg);
            window.alert(`Error: ${msg}`);
        } finally {
            setBootstrapping(false);
        }
    };

    const stages = pipelineData?.stages || [];
    const allLive = pipelineData?.all_live || false;
    const bufferSize = pipelineData?.buffer_size || 0;
    const activeMachines = pipelineData?.active_machines || 0;
    const totalMessages = stages.reduce((sum, s) => sum + (s.length || 0), 0);

    return (
        <Box sx={{
            minHeight: '100vh',
            bgcolor: theme.palette.background.default,
            color: theme.palette.text.primary,
            position: 'relative',
            overflow: 'hidden', // Contain animations
        }}>
            {/* Subtle Background Pattern */}
            <Box sx={{
                position: 'absolute', inset: 0,
                opacity: 0.4,
                backgroundImage: `radial-gradient(${theme.palette.grey[200]} 1px, transparent 1px)`,
                backgroundSize: '24px 24px',
                pointerEvents: 'none',
            }} />

            {/* Content Container */}
            <Box sx={{ position: 'relative', zIndex: 1, px: 4, py: 4, maxWidth: 1400, mx: 'auto' }}>

                {/* Header */}
                <Fade in timeout={600}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 4 }}>
                        <Box>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
                                <Paper elevation={0} sx={{
                                    width: 40, height: 40, borderRadius: 2,
                                    bgcolor: alpha(theme.palette.primary.main, 0.1),
                                    color: theme.palette.primary.main,
                                    border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                }}>
                                    <LanIcon />
                                </Paper>
                                <Typography variant="h4" fontWeight={800} sx={{ color: theme.palette.text.primary }}>
                                    Pipeline Operations
                                </Typography>
                            </Box>
                            <Typography variant="body1" sx={{ color: theme.palette.text.secondary, ml: 7 }}>
                                Data Setup → Features → Model → 6-Stage Inference Pipeline
                            </Typography>
                        </Box>

                        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
                            {/* Upload - primary data input (always works) */}
                            <Button
                                variant="contained"
                                color="primary"
                                startIcon={uploading ? <DotIcon sx={{ animation: 'pulse 1s infinite' }} /> : <CloudUploadIcon />}
                                onClick={() => fileInputRef.current?.click()}
                                disabled={uploading || bootstrapping}
                                sx={{
                                    height: 32,
                                    textTransform: 'none',
                                    fontWeight: 700,
                                    fontSize: '0.8rem',
                                    boxShadow: 2,
                                    '&:hover': { bgcolor: theme.palette.primary.dark }
                                }}
                            >
                                {uploading ? 'Uploading...' : 'Upload Folder'}
                            </Button>
                            {/* Extract Archive - only when archive exists */}
                            <Tooltip title={archiveReady ? "Extract NASA archive to ingest folder" : `Archive not found. Set NASA_ARCHIVE_PATH or run: python scripts/download_nasa_data.py`} arrow>
                                <span>
                                    <Button
                                        variant="outlined"
                                        startIcon={bootstrapping ? <RefreshIcon sx={{ animation: 'spin 2s linear infinite' }} /> : <AutoFixHighIcon />}
                                        onClick={handleBootstrap}
                                        disabled={!archiveReady || bootstrapping || uploadProgress > 0}
                                        sx={{
                                            height: 32,
                                            textTransform: 'none',
                                            fontWeight: 600,
                                            fontSize: '0.8rem',
                                            borderColor: theme.palette.divider,
                                            color: theme.palette.text.secondary,
                                            '&:hover': {
                                                borderColor: theme.palette.primary.main,
                                                color: theme.palette.primary.main
                                            }
                                        }}
                                    >
                                        {bootstrapping ? 'Extracting...' : 'Extract Archive'}
                                    </Button>
                                </span>
                            </Tooltip>
                            <input
                                type="file"
                                ref={fileInputRef}
                                onChange={handleUpload}
                                style={{ display: 'none' }}
                                webkitdirectory=""
                                directory=""
                                multiple
                            />

                            {/* Simulation Badge */}
                            <Tooltip title="Data source: Replaying 'training_data.pkl' via Ingestion Service" arrow>
                                <Chip
                                    icon={<ScienceIcon fontSize="small" />}
                                    label="SIMULATION MODE"
                                    size="small"
                                    sx={{
                                        bgcolor: alpha(theme.palette.info.main, 0.1),
                                        color: theme.palette.info.dark,
                                        fontWeight: 700,
                                        fontSize: '0.7rem',
                                        border: `1px solid ${alpha(theme.palette.info.main, 0.2)}`,
                                    }}
                                />
                            </Tooltip>

                            <Chip
                                icon={<DotIcon sx={{ fontSize: 10, color: autoRefresh ? theme.palette.success.main : theme.palette.grey[500], animation: autoRefresh ? 'pulse 2s infinite' : 'none' }} />}
                                label={autoRefresh ? 'LIVE POLLING' : 'PAUSED'}
                                onClick={() => setAutoRefresh(!autoRefresh)}
                                sx={{
                                    bgcolor: theme.palette.background.paper,
                                    fontWeight: 600,
                                    fontSize: '0.75rem',
                                    boxShadow: 1,
                                    cursor: 'pointer',
                                    '&:hover': { bgcolor: theme.palette.grey[50] }
                                }}
                            />

                            <IconButton
                                onClick={fetchStatus}
                                size="small"
                                sx={{
                                    bgcolor: theme.palette.background.paper,
                                    boxShadow: 1,
                                    '&:hover': { bgcolor: theme.palette.grey[100] }
                                }}
                            >
                                <RefreshIcon fontSize="small" />
                            </IconButton>
                        </Box>
                    </Box>
                </Fade>

                {/* Data Setup Status */}
                <Fade in timeout={700}>
                    <Box sx={{
                        mb: 3, px: 2, py: 1.5,
                        borderRadius: 2,
                        bgcolor: alpha(theme.palette.background.paper, 0.6),
                        border: `1px solid ${theme.palette.divider}`,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        flexWrap: 'wrap',
                        gap: 2,
                        backdropFilter: 'blur(8px)'
                    }}>
                        <Box sx={{ display: 'flex', gap: 3, alignItems: 'center', flexWrap: 'wrap' }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <StorageIcon sx={{ fontSize: 16, color: theme.palette.text.secondary }} />
                                <Typography variant="caption" fontWeight={700} color="text.secondary">INGEST:</Typography>
                                <Chip label={`${readiness.ingest_file_count ?? 0} files`} size="small" sx={{ height: 20, fontSize: '0.7rem' }} />
                            </Box>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Typography variant="caption" fontWeight={700} color="text.secondary">ARCHIVE:</Typography>
                                <Chip label={archiveReady ? 'Ready' : 'Not found'} size="small" color={archiveReady ? 'success' : 'default'} sx={{ height: 20, fontSize: '0.7rem' }} />
                            </Box>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Typography variant="caption" fontWeight={700} color="text.secondary">FEMTO:</Typography>
                                <Chip label={readiness.femto_ready ? 'Ready' : 'Optional'} size="small" color={readiness.femto_ready ? 'success' : 'default'} sx={{ height: 20, fontSize: '0.7rem' }} />
                            </Box>
                        </Box>
                        <Button
                            size="small"
                            startIcon={<InfoIcon fontSize="small" />}
                            endIcon={<ExpandMoreIcon sx={{ transform: setupGuideOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />}
                            onClick={() => setSetupGuideOpen(!setupGuideOpen)}
                            sx={{ textTransform: 'none', fontSize: '0.75rem', color: theme.palette.text.secondary }}
                        >
                            Setup guide
                        </Button>
                    </Box>
                </Fade>

                {/* Setup Guide (collapsible) */}
                <Collapse in={setupGuideOpen}>
                    <Paper elevation={0} sx={{ mb: 3, p: 2, border: 1, borderColor: 'divider', borderRadius: 2, bgcolor: 'grey.50' }}>
                        <Typography variant="caption" fontWeight={700} color="text.secondary" sx={{ display: 'block', mb: 1.5 }}>CLI STEPS (run in terminal)</Typography>
                        <Box component="pre" sx={{ fontFamily: 'monospace', fontSize: '0.7rem', overflow: 'auto', m: 0, color: 'text.primary' }}>
{`# 1. Get data (choose one or both)
python scripts/download_nasa_data.py     # → archive.zip
python scripts/download_femto_data.py   # → data/downloads/femto_pronostia

# 2. Build features (after Upload or Extract Archive)
python scripts/build_nasa_features_with_physics.py   # NASA only
# OR
python scripts/build_combined_features.py           # NASA + FEMTO

# 3. Train model
python scripts/retrain_rul_with_physics.py

# 4. Validate (optional)
python scripts/validate_rul_on_nasa.py --save`}
                        </Box>
                    </Paper>
                </Collapse>

                {/* System Health / Process Status Bar */}
                <Fade in timeout={800}>
                    <Box sx={{
                        mb: 4, px: 2, py: 1.5,
                        borderRadius: 2,
                        bgcolor: alpha(theme.palette.background.paper, 0.6),
                        border: `1px solid ${theme.palette.divider}`,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        backdropFilter: 'blur(8px)'
                    }}>
                        <Box sx={{ display: 'flex', gap: 4 }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <DnsIcon sx={{ fontSize: 16, color: theme.palette.text.secondary }} />
                                <Typography variant="caption" fontWeight={700} color="text.secondary">
                                    DROP ZONE WATCHER:
                                </Typography>
                                <Chip
                                    label={pipelineData?.watcher_active ? 'ACTIVE' : 'INACTIVE'}
                                    size="small"
                                    color={pipelineData?.watcher_active ? 'success' : 'default'}
                                    sx={{ height: 18, fontSize: '0.65rem', fontWeight: 800 }}
                                />
                            </Box>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <StorageIcon sx={{ fontSize: 16, color: theme.palette.text.secondary }} />
                                <Typography variant="caption" fontWeight={700} color="text.secondary">
                                    REDIS PERSISTENCE:
                                </Typography>
                                <Chip
                                    label="STABLE"
                                    size="small"
                                    color="success"
                                    sx={{ height: 18, fontSize: '0.65rem', fontWeight: 800 }}
                                />
                            </Box>
                        </Box>
                        <Box>
                            <Typography variant="caption" sx={{ color: theme.palette.text.secondary, fontStyle: 'italic' }}>
                                Refreshed: {new Date().toLocaleTimeString()}
                            </Typography>
                        </Box>
                    </Box>
                </Fade>

                {/* Progress Bar for Uploads */}
                {uploading && (
                    <Fade in>
                        <Box sx={{ mb: 4, px: 2 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                                <Typography variant="caption" fontWeight={700} color="primary" sx={{ letterSpacing: 1 }}>
                                    INGESTING DATASET... {uploadProgress}%
                                </Typography>
                                <Typography variant="caption" fontWeight={600} color="text.secondary">
                                    {uploadStats.current} / {uploadStats.total} FILES
                                </Typography>
                            </Box>
                            <LinearProgress
                                variant="determinate"
                                value={uploadProgress}
                                sx={{
                                    height: 10,
                                    borderRadius: 5,
                                    bgcolor: alpha(theme.palette.primary.main, 0.1),
                                    '& .MuiLinearProgress-bar': {
                                        borderRadius: 5,
                                        backgroundImage: `linear-gradient(90deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.light} 100%)`,
                                    }
                                }}
                            />
                        </Box>
                    </Fade>
                )}

                {/* Metrics Row */}
                <Fade in timeout={800}>
                    <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 3, mb: 6 }}>
                        <MetricCard
                            icon={<SpeedIcon />}
                            label="Throughput"
                            value={`${totalMessages.toLocaleString()} msgs`}
                            color={theme.palette.primary.main}
                        />
                        <MetricCard
                            icon={<MemoryIcon />}
                            label="Active Assets"
                            value={activeMachines}
                            color={theme.palette.secondary.main}
                        />
                        <MetricCard
                            icon={<StorageIcon />}
                            label="Buffer Queue"
                            value={bufferSize}
                            color={bufferSize > 0 ? theme.palette.error.main : theme.palette.warning.main}
                        />
                        <MetricCard
                            icon={<DotIcon />}
                            label="System Status"
                            value={allLive ? 'OPERATIONAL' : 'DEGRADED'}
                            valueColor={allLive ? theme.palette.success.main : theme.palette.error.main}
                            color={allLive ? theme.palette.success.main : theme.palette.error.main}
                        />
                        {pipelineData?.training_data && (
                            <MetricCard
                                icon={<ScienceIcon />}
                                label="Training Data"
                                value={pipelineData.training_data.source}
                                color={theme.palette.info.main}
                            />
                        )}
                        {pipelineData?.model_accuracy && (
                            <MetricCard
                                icon={<PsychologyIcon />}
                                label="Model Accuracy"
                                value={pipelineData.model_accuracy.r2 >= 0
                                    ? `R² ${(pipelineData.model_accuracy.r2 * 100).toFixed(1)}% • MAE ${pipelineData.model_accuracy.mae} min`
                                    : `MAE ${pipelineData.model_accuracy.mae} min • Prec ${(pipelineData.model_accuracy.precision * 100).toFixed(0)}%`}
                                color={theme.palette.success.main}
                            />
                        )}
                    </Box>
                </Fade>

                {/* Pipeline Diagram */}
                <Fade in timeout={1000}>
                    <Box sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        mb: 6,
                        overflowX: 'auto',
                        py: 2,
                        gap: 0
                    }}>
                        {stages.map((stage, idx) => {
                            const config = STAGE_CONFIG[stage.id] || {};
                            const isLive = stage.status === 'LIVE';

                            return (
                                <React.Fragment key={stage.id}>
                                    <Grow in timeout={600 + idx * 150}>
                                        <Box sx={{ position: 'relative', minWidth: 160 }}>
                                            <Paper
                                                elevation={0}
                                                sx={{
                                                    p: 2,
                                                    borderRadius: 3,
                                                    border: '1px solid',
                                                    borderColor: isLive ? alpha(config.color, 0.3) : theme.palette.grey[300],
                                                    bgcolor: theme.palette.background.paper,
                                                    textAlign: 'center',
                                                    transition: 'all 0.3s ease',
                                                    position: 'relative',
                                                    overflow: 'hidden',
                                                    boxShadow: isLive ? `0 4px 20px ${alpha(config.color, 0.15)}` : 'none',
                                                    '&:hover': {
                                                        transform: 'translateY(-4px)',
                                                        boxShadow: `0 10px 25px ${alpha(config.color, 0.2)}`,
                                                        borderColor: config.color,
                                                    }
                                                }}
                                            >
                                                {/* Status Stripe */}
                                                <Box sx={{
                                                    position: 'absolute', top: 0, left: 0, right: 0, height: 4,
                                                    bgcolor: isLive ? config.color : theme.palette.grey[300]
                                                }} />

                                                {/* Icon Badge */}
                                                <Box sx={{
                                                    width: 32, height: 32,
                                                    borderRadius: '50%',
                                                    bgcolor: alpha(config.color, 0.1),
                                                    color: config.color,
                                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                    mx: 'auto', mb: 1.5, mt: 1
                                                }}>
                                                    {config.icon}
                                                </Box>

                                                <Typography variant="subtitle2" fontWeight={800} sx={{ color: theme.palette.text.primary, letterSpacing: 0.5 }}>
                                                    {config.label}
                                                </Typography>

                                                <Typography variant="caption" sx={{ display: 'block', color: theme.palette.text.secondary, fontSize: '0.65rem', mb: 1.5, height: 16 }}>
                                                    {config.desc}
                                                </Typography>

                                                {/* Live Counter */}
                                                {stage.length !== null && (
                                                    <Box sx={{ bgcolor: alpha(theme.palette.grey[100], 0.5), py: 0.5, borderRadius: 1 }}>
                                                        <Typography variant="h6" fontWeight={700} sx={{ fontFamily: 'monospace', fontSize: '1rem', color: theme.palette.text.primary }}>
                                                            {stage.length.toLocaleString()}
                                                        </Typography>
                                                    </Box>
                                                )}

                                                {/* Status Label */}
                                                <Chip
                                                    size="small"
                                                    label={stage.status}
                                                    sx={{
                                                        mt: 1.5, height: 20, fontSize: '0.6rem', fontWeight: 700,
                                                        bgcolor: isLive ? alpha(theme.palette.success.main, 0.1) : alpha(theme.palette.error.main, 0.1),
                                                        color: isLive ? theme.palette.success.dark : theme.palette.error.dark
                                                    }}
                                                />
                                            </Paper>
                                        </Box>
                                    </Grow>

                                    {/* Connector Arrow */}
                                    {idx < stages.length - 1 && (
                                        <Box sx={{ display: 'flex', alignItems: 'center', mx: 0, minWidth: 40, justifyContent: 'center' }}>
                                            <Box sx={{
                                                height: 2,
                                                width: 40,
                                                bgcolor: isLive ? theme.palette.grey[300] : theme.palette.grey[200],
                                                position: 'relative'
                                            }}>
                                                {isLive && (
                                                    <Box sx={{
                                                        position: 'absolute',
                                                        width: 8, height: 8,
                                                        bgcolor: config.color,
                                                        borderRadius: '50%',
                                                        top: -3,
                                                        animation: 'flowRight 1.5s linear infinite',
                                                    }} />
                                                )}
                                            </Box>
                                        </Box>
                                    )}
                                </React.Fragment>
                            );
                        })}
                    </Box>
                </Fade>

                {/* Stream Details Table */}
                <Fade in timeout={1200}>
                    <Paper elevation={0} sx={{
                        borderRadius: 3,
                        border: `1px solid ${theme.palette.divider}`,
                        overflow: 'hidden',
                        mb: 4,
                        bgcolor: theme.palette.background.paper,
                        boxShadow: theme.shadows[1]
                    }}>
                        <Box sx={{
                            px: 3, py: 2,
                            borderBottom: `1px solid ${theme.palette.divider}`,
                            bgcolor: theme.palette.grey[50],
                            display: 'flex', justifyContent: 'space-between', alignItems: 'center'
                        }}>
                            <Typography variant="subtitle2" fontWeight={700} color="text.secondary">
                                REDIS STREAM TELEMETRY
                            </Typography>
                            <Typography variant="caption" color="text.disabled" sx={{ fontFamily: 'monospace' }}>
                                REFRESH_ID: {refreshCount}
                            </Typography>
                        </Box>
                        <Box component="table" sx={{ width: '100%', borderCollapse: 'collapse' }}>
                            <thead>
                                <tr style={{ borderBottom: `1px solid ${theme.palette.divider}` }}>
                                    {['Stage', 'Stream Key', 'Messages', 'Status', 'Lag', 'Last Entry ID'].map(h => (
                                        <th key={h} style={{ textAlign: 'left', padding: '12px 24px', fontSize: '0.75rem', color: theme.palette.text.secondary, fontWeight: 600 }}>
                                            {h}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {stages.map((stage, i) => {
                                    const config = STAGE_CONFIG[stage.id] || {};
                                    return (
                                        <tr key={i} style={{ borderBottom: `1px solid ${theme.palette.divider}` }}>
                                            <td style={{ padding: '12px 24px' }}>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                                                    <Box sx={{ color: config.color }}>{config.icon}</Box>
                                                    <Typography variant="body2" fontWeight={600} color="text.primary">{config.label}</Typography>
                                                </Box>
                                            </td>
                                            <td style={{ padding: '12px 24px' }}>
                                                <Typography variant="body2" sx={{ fontFamily: 'monospace', color: 'text.secondary', fontSize: '0.8rem' }}>
                                                    {stage.stream || 'N/A'}
                                                </Typography>
                                            </td>
                                            <td style={{ padding: '12px 24px' }}>
                                                <Typography variant="body2" fontWeight={700} sx={{ fontFamily: 'monospace' }}>
                                                    {stage.length !== null ? stage.length.toLocaleString() : '-'}
                                                </Typography>
                                            </td>
                                            <td style={{ padding: '12px 24px' }}>
                                                <Chip
                                                    label={stage.status}
                                                    size="small"
                                                    sx={{
                                                        height: 20, fontSize: '0.65rem', fontWeight: 700,
                                                        bgcolor: stage.status === 'LIVE' ? alpha(theme.palette.success.main, 0.1) : alpha(theme.palette.error.main, 0.1),
                                                        color: stage.status === 'LIVE' ? theme.palette.success.dark : theme.palette.error.dark
                                                    }}
                                                />
                                            </td>
                                            <td style={{ padding: '12px 24px', color: theme.palette.text.secondary }}>
                                                {stage.consumer_groups?.[0]?.lag || 0}
                                            </td>
                                            <td style={{ padding: '12px 24px' }}>
                                                <Typography variant="caption" sx={{ fontFamily: 'monospace', color: 'text.disabled' }}>
                                                    {stage.last_entry_id || '-'}
                                                </Typography>
                                            </td>
                                        </tr>
                                    )
                                })}
                            </tbody>
                        </Box>
                    </Paper>
                </Fade>

                {/* Go Live CTA */}
                <Fade in timeout={1400}>
                    <Box sx={{ display: 'flex', justifyContent: 'center', pb: 4 }}>
                        <Button
                            variant="contained"
                            size="large"
                            onClick={onGoLive}
                            endIcon={<ArrowForwardIcon />}
                            sx={{
                                px: 6, py: 1.5,
                                borderRadius: 2,
                                fontSize: '1rem',
                                fontWeight: 700,
                                textTransform: 'none',
                                boxShadow: theme.shadows[4],
                                bgcolor: allLive ? theme.palette.success.main : theme.palette.primary.main,
                                '&:hover': {
                                    bgcolor: allLive ? theme.palette.success.dark : theme.palette.primary.dark,
                                    transform: 'translateY(-2px)',
                                }
                            }}
                        >
                            {allLive ? 'All Systems Operational — Open Plant OS' : 'Enter Plant OS'}
                        </Button>
                    </Box>
                </Fade>

                {error && (
                    <Alert severity="error" sx={{ mt: 2 }} onClose={() => setError(null)}>
                        {error}
                    </Alert>
                )}

            </Box>

            {/* Animation Styles */}
            <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        @keyframes flowRight {
          0% { left: 0; opacity: 0; }
          20% { opacity: 1; }
          80% { opacity: 1; }
          100% { left: 100%; opacity: 0; }
        }
      `}</style>
        </Box>
    );
}

function MetricCard({ icon, label, value, color, valueColor }) {
    const theme = useTheme();
    return (
        <Paper elevation={0} sx={{
            p: 2.5,
            borderRadius: 3,
            border: `1px solid ${theme.palette.divider}`,
            display: 'flex', alignItems: 'center', gap: 2,
            bgcolor: theme.palette.background.paper,
            transition: 'all 0.2s',
            '&:hover': {
                borderColor: color,
                transform: 'translateY(-2px)',
                boxShadow: theme.shadows[2]
            }
        }}>
            <Box sx={{
                width: 48, height: 48, borderRadius: 2,
                bgcolor: alpha(color, 0.1),
                color: color,
                display: 'flex', alignItems: 'center', justifyContent: 'center'
            }}>
                {React.cloneElement(icon, { fontSize: 'medium' })}
            </Box>
            <Box>
                <Typography variant="caption" fontWeight={600} color="text.secondary" sx={{ textTransform: 'uppercase', letterSpacing: 0.5 }}>
                    {label}
                </Typography>
                <Typography variant="h5" fontWeight={700} sx={{ color: valueColor || theme.palette.text.primary, mt: 0.5 }}>
                    {value}
                </Typography>
            </Box>
        </Paper>
    );
}
