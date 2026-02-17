import React, { useState, useEffect, useCallback } from 'react';
import {
    Activity, AlertTriangle, TrendingUp, RefreshCw,
    Zap, Brain, Network, ChevronRight, Clock, Target,
    CheckCircle, XCircle, Eye
} from 'lucide-react';

/**
 * AnomalyDiscoveryPanel Component
 * 
 * Displays AI-discovered anomalies, correlations, and insights.
 */
function AnomalyDiscoveryPanel({ machineId = null }) {
    const [activeTab, setActiveTab] = useState('anomalies');

    // Mock Data for Demonstration
    const mockAnomalies = [
        {
            detection_id: 'det_001',
            severity: 'critical',
            machine_id: 'CNC-004',
            timestamp: new Date(Date.now() - 1000 * 60 * 15).toISOString(), // 15 mins ago
            anomaly_type: 'collective',
            ensemble_score: 0.92,
            confidence: 0.88,
            reviewed: false
        },
        {
            detection_id: 'det_002',
            severity: 'high',
            machine_id: 'ROBOT-ARM-02',
            timestamp: new Date(Date.now() - 1000 * 60 * 45).toISOString(), // 45 mins ago
            anomaly_type: 'point',
            ensemble_score: 0.78,
            confidence: 0.82,
            reviewed: false
        },
        {
            detection_id: 'det_003',
            severity: 'medium',
            machine_id: 'CONVEYOR-01',
            timestamp: new Date(Date.now() - 1000 * 60 * 120).toISOString(), // 2 hours ago
            anomaly_type: 'contextual',
            ensemble_score: 0.65,
            confidence: 0.70,
            reviewed: true
        }
    ];

    const mockCorrelations = [
        {
            source_machine_id: 'CNC-004',
            target_machine_id: 'ROBOT-ARM-02',
            source_feature: 'vibration_y',
            target_feature: 'motor_current',
            correlation_coefficient: 0.85,
            lag_hours: 2,
            strength: 'strong',
            granger_causal: true
        },
        {
            source_machine_id: 'HVAC-01',
            target_machine_id: 'CNC-004',
            source_feature: 'ambient_temp',
            target_feature: 'spindle_temp',
            correlation_coefficient: 0.72,
            lag_hours: 0,
            strength: 'moderate',
            granger_causal: false
        }
    ];

    const mockInsights = [
        {
            priority: 'high',
            insight_type: 'Optimization',
            title: 'Cooling Efficiency Drop',
            summary: 'Correlation detected between ambient temperature rise and spindle overheating events.'
        },
        {
            priority: 'medium',
            insight_type: 'Maintenance',
            title: 'Bearing Wear Pattern',
            summary: 'Vibration signature matches known bearing degradation profile (Type B).'
        }
    ];

    const [anomalies, setAnomalies] = useState(mockAnomalies);
    const [correlations, setCorrelations] = useState(mockCorrelations);
    const [insights, setInsights] = useState(mockInsights);
    const [status, setStatus] = useState({ is_trained: true }); // Mock trained status
    const [loading, setLoading] = useState(false); // Start interactive immediately
    const [detecting, setDetecting] = useState(false);

    const API_BASE = 'http://localhost:8000/api/discovery';

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
            const url = machineId
                ? `${API_BASE}/anomalies?machine_id=${machineId}&limit=20`
                : `${API_BASE}/anomalies?limit=20`;
            const res = await fetch(url);
            if (res.ok) {
                const data = await res.json();
                setAnomalies(data.data || []);
            }
        } catch (err) {
            console.error('[Discovery] Anomalies fetch error', err);
        }
    }, [machineId]);

    const fetchCorrelations = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/correlations?limit=10`);
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
            const res = await fetch(`${API_BASE}/insights?limit=10`);
            if (res.ok) {
                const data = await res.json();
                setInsights(data.data || []);
            }
        } catch (err) {
            console.error('[Discovery] Insights fetch error', err);
        }
    }, []);

    const runDetection = async () => {
        setDetecting(true);
        try {
            await fetch(`${API_BASE}/detect`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ hours_back: 4, persist: true })
            });
            // Wait a bit then refresh
            setTimeout(() => {
                fetchAnomalies();
                setDetecting(false);
            }, 5000);
        } catch (err) {
            console.error('[Discovery] Detection error', err);
            setDetecting(false);
        }
    };

    const reviewAnomaly = async (detectionId, isValid) => {
        try {
            await fetch(`${API_BASE}/anomalies/${detectionId}/review`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    is_true_positive: isValid,
                    reviewed_by: 'User'
                })
            });
            fetchAnomalies();
        } catch (err) {
            console.error('[Discovery] Review error', err);
        }
    };

    useEffect(() => {
        const loadAll = async () => {
            setLoading(true);
            await Promise.all([fetchStatus(), fetchAnomalies(), fetchCorrelations(), fetchInsights()]);
            setLoading(false);
        };
        loadAll();

        // Auto-refresh every 60 seconds
        const interval = setInterval(fetchAnomalies, 60000);
        return () => clearInterval(interval);
    }, [fetchStatus, fetchAnomalies, fetchCorrelations, fetchInsights]);

    const severityConfig = {
        critical: { color: 'text-red-600', bg: 'bg-red-100', border: 'border-red-200' },
        high: { color: 'text-orange-600', bg: 'bg-orange-100', border: 'border-orange-200' },
        medium: { color: 'text-amber-600', bg: 'bg-amber-100', border: 'border-amber-200' },
        low: { color: 'text-blue-600', bg: 'bg-blue-100', border: 'border-blue-200' }
    };

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

    if (loading) {
        return (
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
                <div className="animate-pulse space-y-4">
                    <div className="h-6 bg-slate-200 rounded w-1/3"></div>
                    <div className="h-32 bg-slate-100 rounded"></div>
                </div>
            </div>
        );
    }

    return (
        <div className="h-full flex flex-col bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
            {/* Header */}
            <div className="shrink-0 px-5 py-4 border-b border-slate-100 bg-gradient-to-r from-indigo-50 to-purple-50">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-indigo-100 rounded-lg">
                            <Brain className="w-5 h-5 text-indigo-600" />
                        </div>
                        <div>
                            <h3 className="font-semibold text-slate-900">AI Anomaly Discovery</h3>
                            <p className="text-xs text-slate-500">
                                {status.is_trained ? 'Models trained & active' : 'Models not trained'}
                            </p>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={runDetection}
                            disabled={detecting || !status.is_trained}
                            className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-400 text-white text-xs font-medium rounded-lg transition-colors"
                        >
                            {detecting ? (
                                <>
                                    <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                    Analyzing...
                                </>
                            ) : (
                                <>
                                    <Zap className="w-3.5 h-3.5" />
                                    Run Detection
                                </>
                            )}
                        </button>
                    </div>
                </div>
            </div>

            {/* Tabs */}
            <div className="flex border-b border-slate-100">
                {[
                    { id: 'anomalies', label: 'Anomalies', icon: AlertTriangle, count: anomalies.length },
                    { id: 'correlations', label: 'Correlations', icon: Network, count: correlations.length },
                    { id: 'insights', label: 'Insights', icon: Target, count: insights.length }
                ].map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${activeTab === tab.id
                            ? 'text-indigo-600 border-b-2 border-indigo-600 bg-indigo-50'
                            : 'text-slate-500 hover:text-slate-700 hover:bg-slate-50'
                            }`}
                    >
                        <tab.icon className="w-4 h-4" />
                        {tab.label}
                        {tab.count > 0 && (
                            <span className={`px-1.5 py-0.5 text-[10px] font-bold rounded-full ${activeTab === tab.id ? 'bg-indigo-100 text-indigo-700' : 'bg-slate-100 text-slate-600'
                                }`}>
                                {tab.count}
                            </span>
                        )}
                    </button>
                ))}
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto min-h-0">
                {/* Anomalies Tab */}
                {activeTab === 'anomalies' && (
                    <div className="divide-y divide-slate-100">
                        {anomalies.length === 0 ? (
                            <div className="px-5 py-12 text-center">
                                <Activity className="w-10 h-10 text-slate-300 mx-auto mb-3" />
                                <p className="text-slate-500 font-medium">No anomalies detected</p>
                                <p className="text-xs text-slate-400 mt-1">
                                    Click "Run Detection" to analyze recent data
                                </p>
                            </div>
                        ) : (
                            anomalies.map(a => {
                                const sev = severityConfig[a.severity] || severityConfig.low;
                                return (
                                    <div key={a.detection_id} className={`px-5 py-4 hover:bg-slate-50 ${a.reviewed ? 'opacity-60' : ''}`}>
                                        <div className="flex items-start justify-between">
                                            <div className="flex-1">
                                                <div className="flex items-center gap-2 mb-1">
                                                    <span className={`px-2 py-0.5 text-[10px] font-bold rounded ${sev.bg} ${sev.color}`}>
                                                        {a.severity?.toUpperCase()}
                                                    </span>
                                                    <span className="text-xs font-mono text-slate-500">{a.machine_id}</span>
                                                    <span className="text-xs text-slate-400">•</span>
                                                    <span className="text-xs text-slate-400">{formatTime(a.timestamp)}</span>
                                                </div>
                                                <p className="text-sm text-slate-800 mb-1">
                                                    {a.anomaly_type === 'point' ? 'Unusual point reading' :
                                                        a.anomaly_type === 'contextual' ? 'Contextual anomaly' :
                                                            a.anomaly_type === 'collective' ? 'Temporal pattern anomaly' : 'Anomaly detected'}
                                                </p>
                                                <div className="flex items-center gap-4 text-xs text-slate-500">
                                                    <span>Score: {(a.ensemble_score * 100).toFixed(0)}%</span>
                                                    <span>Confidence: {(a.confidence * 100).toFixed(0)}%</span>
                                                </div>
                                            </div>
                                            {!a.reviewed && (
                                                <div className="flex items-center gap-1">
                                                    <button
                                                        onClick={() => reviewAnomaly(a.detection_id, true)}
                                                        className="p-1.5 text-emerald-600 hover:bg-emerald-50 rounded-lg"
                                                        title="Mark as valid"
                                                    >
                                                        <CheckCircle className="w-4 h-4" />
                                                    </button>
                                                    <button
                                                        onClick={() => reviewAnomaly(a.detection_id, false)}
                                                        className="p-1.5 text-red-600 hover:bg-red-50 rounded-lg"
                                                        title="Mark as false positive"
                                                    >
                                                        <XCircle className="w-4 h-4" />
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                );
                            })
                        )}
                    </div>
                )}

                {/* Correlations Tab */}
                {activeTab === 'correlations' && (
                    <div className="divide-y divide-slate-100">
                        {correlations.length === 0 ? (
                            <div className="px-5 py-12 text-center">
                                <Network className="w-10 h-10 text-slate-300 mx-auto mb-3" />
                                <p className="text-slate-500 font-medium">No correlations discovered</p>
                                <p className="text-xs text-slate-400 mt-1">
                                    Correlations are analyzed automatically
                                </p>
                            </div>
                        ) : (
                            correlations.map((c, idx) => (
                                <div key={idx} className="px-5 py-4 hover:bg-slate-50">
                                    <div className="flex items-center gap-2 mb-2">
                                        <span className="px-2 py-0.5 bg-indigo-100 text-indigo-700 text-[10px] font-bold rounded">
                                            {c.source_machine_id}
                                        </span>
                                        <ChevronRight className="w-4 h-4 text-slate-400" />
                                        <span className="px-2 py-0.5 bg-purple-100 text-purple-700 text-[10px] font-bold rounded">
                                            {c.target_machine_id}
                                        </span>
                                        {c.granger_causal && (
                                            <span className="px-1.5 py-0.5 bg-emerald-100 text-emerald-700 text-[10px] font-bold rounded">
                                                CAUSAL
                                            </span>
                                        )}
                                    </div>
                                    <p className="text-sm text-slate-700 mb-1">
                                        {c.source_feature} → {c.target_feature}
                                    </p>
                                    <div className="flex items-center gap-4 text-xs text-slate-500">
                                        <span>r = {c.correlation_coefficient?.toFixed(2)}</span>
                                        {c.lag_hours > 0 && <span>Lag: {c.lag_hours}h</span>}
                                        <span className="capitalize">{c.strength}</span>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                )}

                {/* Insights Tab */}
                {activeTab === 'insights' && (
                    <div className="divide-y divide-slate-100">
                        {insights.length === 0 ? (
                            <div className="px-5 py-12 text-center">
                                <Target className="w-10 h-10 text-slate-300 mx-auto mb-3" />
                                <p className="text-slate-500 font-medium">No insights generated</p>
                                <p className="text-xs text-slate-400 mt-1">
                                    Insights are generated from detected patterns
                                </p>
                            </div>
                        ) : (
                            insights.map((i, idx) => (
                                <div key={idx} className="px-5 py-4 hover:bg-slate-50">
                                    <div className="flex items-center gap-2 mb-2">
                                        <span className={`px-2 py-0.5 text-[10px] font-bold rounded ${i.priority === 'high' ? 'bg-red-100 text-red-700' :
                                            i.priority === 'medium' ? 'bg-amber-100 text-amber-700' :
                                                'bg-blue-100 text-blue-700'
                                            }`}>
                                            {i.priority?.toUpperCase()}
                                        </span>
                                        <span className="text-xs text-slate-400">{i.insight_type}</span>
                                    </div>
                                    <p className="text-sm font-medium text-slate-800 mb-1">{i.title}</p>
                                    <p className="text-xs text-slate-600">{i.summary}</p>
                                </div>
                            ))
                        )}
                    </div>
                )}
            </div>

            {/* Footer */}
            <div className="px-5 py-3 bg-slate-50 border-t border-slate-100 flex items-center justify-between text-xs">
                <span className="text-slate-500 flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    Auto-refresh: 60s
                </span>
                <span className="text-slate-400">
                    Powered by Isolation Forest + LSTM Autoencoder
                </span>
            </div>
        </div>
    );
}

export default AnomalyDiscoveryPanel;
