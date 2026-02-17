import React, { useState, useEffect, useCallback } from 'react';
import {
    Box, Grid, Card, CardContent, Typography, Button, IconButton,
    Tabs, Tab, Chip, Select, MenuItem, FormControl, InputLabel,
    CircularProgress, LinearProgress, Divider, Alert, Stack,
    Tooltip, Paper
} from '@mui/material';
import {
    Psychology, Warning, Hub, TrackChanges, AutoAwesome, Refresh,
    AccessTime, CheckCircle, Cancel, Timeline, TrendingUp,
    ChevronRight, FilterList, Download, Settings, PlayArrow,
    ExpandMore, ExpandLess, Construction, BarChart
} from '@mui/icons-material';
import { DashboardGrid } from '../components/DashboardGrid';

/**
 * AnomalyDiscoveryPage - Dedicated page for AI anomaly detection
 */
function AnomalyDiscoveryPage() {
    const [activeTab, setActiveTab] = useState(0); // 0: anomalies, 1: correlations, 2: insights
    const [anomalies, setAnomalies] = useState([]);
    const [correlations, setCorrelations] = useState([]);
    const [insights, setInsights] = useState([]);
    const [expandedAnomaly, setExpandedAnomaly] = useState(null);
    const [status, setStatus] = useState({ is_trained: false });
    const [loading, setLoading] = useState(true);
    const [detecting, setDetecting] = useState(false);
    const [training, setTraining] = useState(false);
    const [selectedMachine, setSelectedMachine] = useState('all');
    const [machines, setMachines] = useState([]);

    const API_BASE = 'http://localhost:8000/api/discovery';

    // Fetch functions
    const fetchStatus = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/status`);
            if (res.ok) setStatus(await res.json());
        } catch (err) {
            console.error('[Discovery] Status fetch error', err);
        }
    }, []);

    const fetchAnomalies = useCallback(async () => {
        try {
            const url = selectedMachine !== 'all'
                ? `${API_BASE}/anomalies?machine_id=${selectedMachine}&limit=50`
                : `${API_BASE}/anomalies?limit=50`;
            const res = await fetch(url);
            if (res.ok) {
                const data = await res.json();
                setAnomalies(data.data || []);
            }
        } catch (err) {
            console.error('[Discovery] Anomalies fetch error', err);
        }
    }, [selectedMachine]);

    const fetchCorrelations = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/correlations?limit=30`);
            if (res.ok) {
                const data = await res.json();
                setCorrelations(data.data || []);
            }
        } catch (err) {
            console.error('[Discovery] Correlations fetch error', err);
        }
    }, []);

    const fetchInsights = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/insights?limit=20`);
            if (res.ok) {
                const data = await res.json();
                setInsights(data.data || []);
            }
        } catch (err) {
            console.error('[Discovery] Insights fetch error', err);
        }
    }, []);

    const fetchMachines = async () => {
        try {
            const res = await fetch('http://localhost:8000/api/machines');
            if (res.ok) {
                const data = await res.json();
                if (Array.isArray(data)) {
                    setMachines(data);
                } else if (data && Array.isArray(data.data)) {
                    setMachines(data.data);
                } else {
                    setMachines([]);
                }
            }
        } catch (err) {
            console.error('[Discovery] Machines fetch error', err);
            setMachines([]);
        }
    };

    // Actions
    const trainModels = async () => {
        setTraining(true);
        try {
            await fetch(`${API_BASE}/train`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ days_of_data: 7, min_samples: 100 })
            });
            const poll = setInterval(async () => {
                const res = await fetch(`${API_BASE}/status`);
                if (res.ok) {
                    const s = await res.json();
                    if (s.is_trained) {
                        setStatus(s);
                        setTraining(false);
                        clearInterval(poll);
                    }
                }
            }, 3000);
        } catch (err) {
            console.error('[Discovery] Training error', err);
            setTraining(false);
        }
    };

    const runDetection = async () => {
        setDetecting(true);
        try {
            await fetch(`${API_BASE}/detect`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ hours_back: 4, persist: true })
            });
            setTimeout(() => {
                fetchAnomalies();
                setDetecting(false);
            }, 8000);
        } catch (err) {
            console.error('[Discovery] Detection error', err);
            setDetecting(false);
        }
    };

    const runCorrelationAnalysis = async () => {
        try {
            await fetch(`${API_BASE}/correlations/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ days_back: 7 })
            });
            setTimeout(fetchCorrelations, 5000);
        } catch (err) {
            console.error('[Discovery] Correlation analysis error', err);
        }
    };

    const reviewAnomaly = async (detectionId, isValid) => {
        try {
            await fetch(`${API_BASE}/anomalies/${detectionId}/review`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ is_true_positive: isValid, reviewed_by: 'User' })
            });
            fetchAnomalies();
        } catch (err) {
            console.error('[Discovery] Review error', err);
        }
    };

    useEffect(() => {
        const loadAll = async () => {
            setLoading(true);
            await Promise.all([
                fetchStatus(), fetchMachines(), fetchAnomalies(),
                fetchCorrelations(), fetchInsights()
            ]);
            setLoading(false);
        };
        loadAll();

        const interval = setInterval(() => {
            fetchStatus();
            fetchAnomalies();
        }, 60000);
        return () => clearInterval(interval);
    }, [fetchStatus, fetchAnomalies, fetchCorrelations, fetchInsights]);

    useEffect(() => {
        fetchAnomalies();
    }, [selectedMachine, fetchAnomalies]);

    const formatTime = (ts) => {
        if (!ts) return 'Unknown';
        const d = new Date(ts);
        const now = new Date();
        const diffMs = now - d;
        const diffMins = Math.floor(diffMs / 60000);
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
        return d.toLocaleDateString();
    };

    // Stats calculation
    const criticalCount = anomalies.filter(a => a.severity === 'critical').length;
    const highCount = anomalies.filter(a => a.severity === 'high').length;
    const unreviewedCount = anomalies.filter(a => !a.reviewed).length;
    const causalCount = correlations.filter(c => c.granger_causal).length;

    const handleTabChange = (event, newValue) => {
        setActiveTab(newValue);
    };

    // --- Drag and Drop State Management ---
    const defaultPositions = {
        header_panel: { x: 0, y: 0, w: 1350, h: 140 },
        stat_total: { x: 0, y: 160, w: 254, h: 160 },
        stat_critical: { x: 274, y: 160, w: 254, h: 160 },
        stat_high: { x: 548, y: 160, w: 254, h: 160 },
        stat_corr: { x: 822, y: 160, w: 254, h: 160 },
        stat_pending: { x: 1096, y: 160, w: 254, h: 160 },
        main_content: { x: 0, y: 340, w: 1350, h: 800 }
    };

    const [layoutPositions, setLayoutPositions] = useState(() => {
        try {
            const saved = localStorage.getItem('discovery_positions_v4');
            const parsed = saved ? JSON.parse(saved) : null;
            if (parsed && typeof parsed === 'object') {
                return { ...defaultPositions, ...parsed };
            }
            return defaultPositions;
        } catch (e) {
            return defaultPositions;
        }
    });

    const handleUpdate = (id, updates) => {
        const newPositions = {
            ...layoutPositions,
            [id]: { ...layoutPositions[id], ...updates }
        };
        setLayoutPositions(newPositions);
        localStorage.setItem('discovery_positions_v4', JSON.stringify(newPositions));
    };

    const widgetsContent = React.useMemo(() => ({
        header_panel: {
            id: 'header_panel',
            content: (
                <Paper elevation={0} sx={{ height: '100%', p: 3, borderRadius: 3, border: '1px solid', borderColor: 'divider', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                            <Box sx={{
                                p: 1.5, borderRadius: 2, background: 'linear-gradient(135deg, #6366f1 0%, #a855f7 100%)',
                                color: 'white', display: 'flex'
                            }}>
                                <Psychology fontSize="large" />
                            </Box>
                            <Box>
                                <Typography variant="h5" fontWeight="bold">AI Anomaly Discovery</Typography>
                                <Typography variant="body2" color="text.secondary">Unsupervised detection of hidden patterns and correlations</Typography>
                            </Box>
                        </Box>

                        <Stack direction="row" spacing={2} alignItems="center">
                            <Chip
                                icon={<Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: status.is_trained ? 'success.main' : 'warning.main', ml: 1, mr: -0.5 }} />}
                                label={status.is_trained ? 'Models Active' : 'Models Not Trained'}
                                color={status.is_trained ? 'success' : 'warning'}
                                variant="outlined"
                                sx={{ fontWeight: 600 }}
                            />

                            {!status.is_trained && (
                                <Button
                                    variant="contained" color="warning"
                                    startIcon={training ? <CircularProgress size={16} color="inherit" /> : <Settings />}
                                    onClick={trainModels} disabled={training}
                                >
                                    {training ? 'Training...' : 'Train Models'}
                                </Button>
                            )}

                            <Button
                                variant="contained" color="primary"
                                startIcon={detecting ? <CircularProgress size={16} color="inherit" /> : <PlayArrow />}
                                onClick={runDetection} disabled={detecting || !status.is_trained}
                            >
                                {detecting ? 'Analyzing...' : 'Run Detection'}
                            </Button>
                        </Stack>
                    </Box>
                </Paper>
            )
        },
        stat_total: { id: 'stat_total', content: <StatCard label="Total Anomalies" value={anomalies.length} icon={Timeline} color="primary.main" /> },
        stat_critical: { id: 'stat_critical', content: <StatCard label="Critical" value={criticalCount} icon={Warning} color="error.main" /> },
        stat_high: { id: 'stat_high', content: <StatCard label="High Priority" value={highCount} icon={TrendingUp} color="warning.main" /> },
        stat_corr: { id: 'stat_corr', content: <StatCard label="Correlations" value={correlations.length} subValue={`${causalCount} causal`} icon={Hub} color="secondary.main" /> },
        stat_pending: { id: 'stat_pending', content: <StatCard label="Pending Review" value={unreviewedCount} icon={AccessTime} color="info.main" /> },
        main_content: {
            id: 'main_content',
            content: (
                <Card variant="outlined" sx={{ height: '100%', borderRadius: 3, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                    <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                        <Tabs value={activeTab} onChange={handleTabChange} aria-label="discovery tabs">
                            <Tab icon={<Warning fontSize="small" />} iconPosition="start" label="Detected Anomalies" />
                            <Tab icon={<Hub fontSize="small" />} iconPosition="start" label="Cross-Machine Correlations" />
                            <Tab icon={<TrackChanges fontSize="small" />} iconPosition="start" label="AI Insights" />
                        </Tabs>
                    </Box>

                    {/* Toolbar Context */}
                    {activeTab === 0 && (
                        <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center', bgcolor: 'grey.50' }}>
                            <Stack direction="row" spacing={2} alignItems="center">
                                <FilterList fontSize="small" color="action" />
                                <FormControl size="small" sx={{ minWidth: 200 }}>
                                    <Select
                                        value={selectedMachine}
                                        onChange={(e) => setSelectedMachine(e.target.value)}
                                        displayEmpty
                                    >
                                        <MenuItem value="all">All Machines</MenuItem>
                                        {machines.map(m => (
                                            <MenuItem key={m.machine_id} value={m.machine_id}>{m.name || m.machine_id}</MenuItem>
                                        ))}
                                    </Select>
                                </FormControl>
                            </Stack>
                            <Button startIcon={<Refresh />} onClick={fetchAnomalies} size="small">Refresh</Button>
                        </Box>
                    )}

                    {activeTab === 1 && (
                        <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center', bgcolor: 'grey.50' }}>
                            <Typography variant="body2" color="text.secondary">
                                Showing {correlations.length} discovered correlations
                            </Typography>
                            <Button
                                variant="outlined" color="secondary" size="small"
                                startIcon={<AutoAwesome />}
                                onClick={runCorrelationAnalysis}
                            >
                                Run Analysis
                            </Button>
                        </Box>
                    )}

                    <Box sx={{ p: 0, flexGrow: 1, overflow: 'auto' }}>
                        {/* Anomalies List */}
                        {activeTab === 0 && (
                            <Box>
                                {anomalies.length === 0 ? (
                                    <EmptyState
                                        icon={Timeline}
                                        title="No anomalies detected"
                                        description="Click 'Run Detection' to analyze recent sensor data"
                                    />
                                ) : (
                                    anomalies.map(a => {
                                        const isExpanded = expandedAnomaly === a.detection_id;
                                        const severityColor = a.severity === 'critical' ? 'error' :
                                            a.severity === 'high' ? 'warning' :
                                                a.severity === 'medium' ? 'info' : 'success';

                                        return (
                                            <Box key={a.detection_id} sx={{ borderBottom: '1px solid', borderColor: 'divider' }}>
                                                <Box
                                                    onClick={() => setExpandedAnomaly(isExpanded ? null : a.detection_id)}
                                                    sx={{
                                                        p: 3, cursor: 'pointer',
                                                        '&:hover': { bgcolor: 'action.hover' },
                                                        bgcolor: isExpanded ? 'action.selected' : 'inherit',
                                                        opacity: a.reviewed ? 0.6 : 1,
                                                        transition: 'all 0.2s'
                                                    }}
                                                >
                                                    <Grid container alignItems="center" spacing={2}>
                                                        <Grid item xs={12} sm>
                                                            <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mb: 1 }}>
                                                                <Chip size="small" label={a.severity?.toUpperCase()} color={severityColor} sx={{ fontWeight: 'bold', height: 24 }} />
                                                                <Typography variant="subtitle2" fontWeight={600}>{a.machine_id}</Typography>
                                                                <Typography variant="caption" color="text.secondary">•</Typography>
                                                                <Typography variant="caption" sx={{ bgcolor: 'grey.100', px: 1, py: 0.5, rounded: 1 }}>{a.anomaly_type}</Typography>
                                                                <Typography variant="caption" color="text.secondary">{formatTime(a.timestamp)}</Typography>
                                                            </Stack>
                                                            <Typography variant="body2" color="text.secondary" gutterBottom>
                                                                Detection ID: <Box component="span" fontFamily="monospace">{a.detection_id}</Box>
                                                            </Typography>
                                                            <Stack direction="row" spacing={3} alignItems="center">
                                                                <Typography variant="caption" color="text.secondary">
                                                                    Ensemble Score: <Box component="span" fontWeight="bold" color="text.primary">{(a.ensemble_score * 100).toFixed(0)}%</Box>
                                                                </Typography>
                                                                <Typography variant="caption" color="text.secondary">
                                                                    Confidence: <Box component="span" fontWeight="bold" color="text.primary">{(a.confidence * 100).toFixed(0)}%</Box>
                                                                </Typography>
                                                                <Button
                                                                    size="small"
                                                                    endIcon={isExpanded ? <ExpandLess /> : <ExpandMore />}
                                                                    sx={{ fontSize: '0.75rem', p: 0, minWidth: 0 }}
                                                                >
                                                                    {isExpanded ? 'Hide Details' : 'View Details'}
                                                                </Button>
                                                            </Stack>
                                                        </Grid>
                                                        <Grid item>
                                                            {!a.reviewed ? (
                                                                <Stack direction="row" spacing={1} onClick={e => e.stopPropagation()}>
                                                                    <Button
                                                                        variant="outlined" color="success" size="small"
                                                                        startIcon={<CheckCircle />}
                                                                        onClick={() => reviewAnomaly(a.detection_id, true)}
                                                                    >
                                                                        Valid
                                                                    </Button>
                                                                    <Button
                                                                        variant="outlined" color="error" size="small"
                                                                        startIcon={<Cancel />}
                                                                        onClick={() => reviewAnomaly(a.detection_id, false)}
                                                                    >
                                                                        False Positive
                                                                    </Button>
                                                                </Stack>
                                                            ) : (
                                                                <Chip
                                                                    label={a.is_true_positive ? 'Confirmed' : 'False Positive'}
                                                                    color={a.is_true_positive ? 'success' : 'error'}
                                                                    variant="outlined" size="small"
                                                                />
                                                            )}
                                                        </Grid>
                                                    </Grid>
                                                </Box>

                                                {/* Expandable Detail Panel */}
                                                {isExpanded && (
                                                    <Box sx={{ p: 3, bgcolor: 'grey.50', borderTop: '1px solid', borderColor: 'divider' }}>
                                                        <Card variant="outlined" sx={{ borderRadius: 2 }}>
                                                            <CardContent>
                                                                <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                                    <BarChart fontSize="small" /> Top Contributing Factors
                                                                </Typography>
                                                                <Grid container spacing={2}>
                                                                    {Object.entries(a.feature_deviations || {}).map(([feature, importance], idx) => (
                                                                        <Grid item xs={12} sm={6} key={idx}>
                                                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                                                                <Typography variant="caption" fontWeight="medium">{feature}</Typography>
                                                                                <Typography variant="caption">{(importance * 100).toFixed(0)}%</Typography>
                                                                            </Box>
                                                                            <LinearProgress variant="determinate" value={Math.min(importance * 100, 100)} sx={{ height: 6, borderRadius: 3 }} />
                                                                        </Grid>
                                                                    ))}
                                                                    {(!a.feature_deviations || Object.keys(a.feature_deviations).length === 0) && (
                                                                        <Grid item xs={12}>
                                                                            <Typography variant="caption" color="text.secondary">No feature contribution data available.</Typography>
                                                                        </Grid>
                                                                    )}
                                                                </Grid>
                                                                <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid', borderColor: 'divider' }}>
                                                                    <Typography variant="subtitle2" gutterBottom>Analysis Summary:</Typography>
                                                                    <Typography variant="body2" color="text.secondary">
                                                                        Detected via Ensemble Model (Isolation Forest + Temporal Autoencoder). High deviation in vibration patterns correlated with recent operational changes.
                                                                    </Typography>
                                                                </Box>
                                                            </CardContent>
                                                        </Card>
                                                    </Box>
                                                )}
                                            </Box>
                                        );
                                    })
                                )}
                            </Box>
                        )}

                        {/* Correlations Tab */}
                        {activeTab === 1 && (
                            <Box>
                                {correlations.length === 0 ? (
                                    <EmptyState
                                        icon={Hub}
                                        title="No correlations discovered"
                                        description="Click 'Run Analysis' to search for cross-machine patterns"
                                    />
                                ) : (
                                    correlations.map((c, idx) => (
                                        <Box key={idx} sx={{ px: 3, py: 2, borderBottom: '1px solid', borderColor: 'divider', '&:hover': { bgcolor: 'grey.50' } }}>
                                            <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
                                                {c.discovery_type === 'maintenance_impact' ? (
                                                    <Chip
                                                        icon={<Construction fontSize="small" />}
                                                        label="MAINTENANCE"
                                                        size="small" color="warning"
                                                        sx={{ fontWeight: 'bold' }}
                                                    />
                                                ) : (
                                                    <Chip label={c.source_machine_id} size="small" color="primary" />
                                                )}

                                                <ChevronRight color="action" />

                                                <Chip label={c.target_machine_id || c.source_machine_id} size="small" color="secondary" />

                                                {c.granger_causal && (
                                                    <Chip label="CAUSAL" size="small" color="success" variant="outlined" />
                                                )}
                                                <Chip
                                                    label={c.strength} size="small"
                                                    color={c.strength === 'very_strong' ? 'error' : c.strength === 'strong' ? 'warning' : 'default'}
                                                    variant="outlined"
                                                />
                                            </Stack>

                                            <Typography variant="body2" gutterBottom>
                                                <Box component="span" fontWeight="medium">{c.source_feature}</Box>
                                                <Box component="span" mx={1} color="text.secondary">→</Box>
                                                <Box component="span" fontWeight="medium">{c.target_feature}</Box>
                                            </Typography>

                                            {c.explanation && (
                                                <Alert severity="info" sx={{ py: 0, mb: 1 }}>{c.explanation}</Alert>
                                            )}

                                            <Stack direction="row" spacing={3} alignItems="center">
                                                <Typography variant="caption" color="text.secondary">Correlation: <b>{c.correlation_coefficient?.toFixed(3)}</b></Typography>
                                                {c.lag_hours > 0 && <Typography variant="caption" color="text.secondary">Lag: <b>{c.lag_hours}h</b></Typography>}
                                                <Typography variant="caption" color="text.secondary">Confidence: <b>{(c.confidence * 100).toFixed(0)}%</b></Typography>
                                            </Stack>
                                        </Box>
                                    ))
                                )}
                            </Box>
                        )}

                        {/* Insights Tab */}
                        {activeTab === 2 && (
                            <Box>
                                {insights.length === 0 ? (
                                    <EmptyState
                                        icon={TrackChanges}
                                        title="No insights generated"
                                        description="Insights are automatically generated from detected anomalies and correlations"
                                    />
                                ) : (
                                    insights.map((i, idx) => (
                                        <Box key={idx} sx={{ px: 3, py: 2, borderBottom: '1px solid', borderColor: 'divider', '&:hover': { bgcolor: 'grey.50' } }}>
                                            <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
                                                <Chip
                                                    label={i.priority?.toUpperCase()}
                                                    size="small"
                                                    color={i.priority === 'critical' ? 'error' : i.priority === 'high' ? 'warning' : 'default'}
                                                />
                                                <Typography variant="caption" sx={{ bgcolor: 'grey.100', px: 1, py: 0.5, rounded: 1 }}>{i.insight_type}</Typography>
                                            </Stack>
                                            <Typography variant="subtitle2" gutterBottom>{i.title}</Typography>
                                            <Typography variant="body2" color="text.secondary" paragraph>{i.summary}</Typography>

                                            {i.recommended_actions?.length > 0 && (
                                                <Box sx={{ mt: 1 }}>
                                                    <Typography variant="caption" fontWeight="bold">Recommended Actions:</Typography>
                                                    <Box component="ul" sx={{ m: 0, pl: 2 }}>
                                                        {i.recommended_actions.map((action, j) => (
                                                            <Typography component="li" variant="caption" key={j}>{action}</Typography>
                                                        ))}
                                                    </Box>
                                                </Box>
                                            )}
                                        </Box>
                                    ))
                                )}
                            </Box>
                        )}
                    </Box>
                </Card>
            )
        }
    }), [status, anomalies, correlations, insights, activeTab, selectedMachine, training, detecting, expandedAnomaly, machines]); // Add all dependencies

    // Merge content and positions
    const gridItems = React.useMemo(() => {
        const items = {};
        Object.keys(widgetsContent).forEach(key => {
            const pos = layoutPositions[key] || { x: 0, y: 0, w: 400, h: 400 };
            items[key] = {
                ...widgetsContent[key],
                x: pos.x,
                y: pos.y,
                width: pos.w,
                height: pos.h
            };
        });
        return items;
    }, [layoutPositions, widgetsContent]);

    return (
        <Box sx={{ flexGrow: 1 }}>
            <DashboardGrid
                items={gridItems}
                onUpdate={handleUpdate}
            >
                {/* No static overlay needed for this page as everything is draggable, but we could add one if needed */}
            </DashboardGrid>
        </Box>
    );
}

// Helper Components
function StatCard({ label, value, subValue, icon: Icon, color }) {
    return (
        <Card variant="outlined" sx={{ height: '100%', borderRadius: 3 }}>
            <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                    <Box sx={{ color: color }}>
                        <Icon />
                    </Box>
                </Box>
                <Typography variant="h4" fontWeight="bold">{value}</Typography>
                <Typography variant="caption" color="text.secondary">{label}</Typography>
                {subValue && (
                    <Typography variant="caption" display="block" color="text.secondary" sx={{ mt: 0.5, opacity: 0.8 }}>
                        {subValue}
                    </Typography>
                )}
            </CardContent>
        </Card>
    );
}

function EmptyState({ icon: Icon, title, description }) {
    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', py: 8, color: 'text.secondary' }}>
            <Icon sx={{ fontSize: 48, mb: 2, opacity: 0.5 }} />
            <Typography variant="h6" gutterBottom>{title}</Typography>
            <Typography variant="body2">{description}</Typography>
        </Box>
    );
}

export default AnomalyDiscoveryPage;
