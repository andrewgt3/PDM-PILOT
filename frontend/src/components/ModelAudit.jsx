import React, { useState, useEffect } from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, Cell, CartesianGrid } from 'recharts';
import { ShieldCheck, Target, TrendingUp, Activity, CheckCircle, FileText } from 'lucide-react';

const AuditCard = ({ title, value, subtext, icon: Icon, color }) => (
    <div className="card flex flex-col justify-between h-32 relative overflow-hidden group">
        <div className="absolute right-[-10px] top-[-10px] opacity-10 group-hover:opacity-20 transition-opacity">
            <Icon size={80} color={color} />
        </div>
        <div>
            <h3 className="text-slate-400 text-sm font-medium uppercase tracking-wider">{title}</h3>
            <div className="text-3xl font-bold mt-2 text-white">{value}</div>
        </div>
        <div className="text-xs text-slate-500 font-mono flex items-center gap-2">
            <span style={{ color }}>{subtext}</span>
        </div>
    </div>
);

const FoldTable = ({ folds }) => {
    return (
        <div className="card overflow-hidden p-0">
            <div className="p-4 border-b border-[var(--border-subtle)] bg-slate-800/50">
                <h3 className="font-bold text-white flex items-center gap-2">
                    <FileText size={16} className="text-slate-400" />
                    5-Fold Walk-Forward Validation (Real-Time)
                </h3>
            </div>
            <table className="w-full text-left border-collapse">
                <thead>
                    <tr className="bg-slate-800/20 text-xs uppercase tracking-wider text-slate-400">
                        <th className="p-4 font-medium">Validation Fold</th>
                        <th className="p-4 font-medium">Precision</th>
                        <th className="p-4 font-medium">Recall (Sensitivity)</th>
                        <th className="p-4 font-medium">AUC-ROC</th>
                        <th className="p-4 font-medium text-right">Status</th>
                    </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border-subtle)]">
                    {folds.map(fold => (
                        <tr key={fold.id} className="hover:bg-slate-800/30 transition-colors">
                            <td className="p-4 font-mono text-slate-300">Fold #{fold.id}</td>
                            <td className="p-4 text-emerald-400 font-bold">{fold.precision}</td>
                            <td className="p-4 text-emerald-400 font-bold">{fold.recall}</td>
                            <td className="p-4 text-slate-300">{fold.auc}</td>
                            <td className="p-4 text-right">
                                <span className={`badge ${fold.status === 'Pass' ? 'badge-healthy' : 'badge-critical'} text-xs`}>
                                    {fold.status}
                                </span>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

const ModelAudit = () => {
    const [auditData, setAuditData] = useState(null);

    useEffect(() => {
        fetch('/audit_results.json')
            .then(res => res.json())
            .then(data => setAuditData(data))
            .catch(err => console.error("Failed to load audit:", err));
    }, []);

    if (!auditData) return <div className="p-8 text-center">Loading Audit Report...</div>;

    const { summary, folds, roc_curve } = auditData;

    return (
        <div className="flex flex-col gap-6 animate-fade-in pb-10">
            <h1 className="text-2xl font-bold text-white flex items-center gap-2">
                <ShieldCheck className="text-emerald-500" size={28} />
                Model Performance Audit
                <span className="text-sm font-normal text-slate-500 bg-slate-800 px-2 py-1 rounded-full border border-slate-700">
                    Production Candidate v1.0
                </span>
            </h1>

            {/* Top Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <AuditCard
                    title="Avg Precision"
                    value={summary.avg_precision}
                    subtext="Target: > 70%"
                    icon={Target}
                    color="#10b981"
                />
                <AuditCard
                    title="Avg Recall"
                    value={summary.avg_recall}
                    subtext="Target: > 80%"
                    icon={Activity}
                    color="#3b82f6"
                />
                <AuditCard
                    title="F1-Score"
                    value={summary.f1_score}
                    subtext="Harmonic Mean"
                    icon={TrendingUp}
                    color="#8b5cf6"
                />
                <AuditCard
                    title="Model Robustness"
                    value={summary.robustness_score}
                    subtext="5-Fold Validated"
                    icon={CheckCircle}
                    color="#10b981"
                />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Fold Table */}
                <div className="lg:col-span-2">
                    <FoldTable folds={folds} />
                </div>

                {/* ROC Curve Visualization */}
                <div className="card flex flex-col h-full">
                    <h3 className="text-slate-300 font-bold mb-4 flex items-center gap-2">
                        <Activity size={16} className="text-indigo-400" />
                        ROC Curve (Aggregated)
                    </h3>
                    <div className="flex-1 min-h-[250px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={roc_curve}>
                                <defs>
                                    <linearGradient id="colorRoc" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                                <XAxis dataKey="fpr" type="number" domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} stroke="#475569" fontSize={12} label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }} />
                                <YAxis type="number" domain={[0, 1]} stroke="#475569" fontSize={12} label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 10 }} />
                                <Tooltip
                                    formatter={(value) => value.toFixed(3)}
                                    labelFormatter={(label) => `FPR: ${label.toFixed(2)}`}
                                    contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#fff' }}
                                />
                                <Area type="monotone" dataKey="tpr" stroke="#6366f1" strokeWidth={3} fill="url(#colorRoc)" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            <div className="p-4 rounded-lg border border-dashed border-slate-600 bg-slate-800/30 text-slate-400 text-sm font-mono">
                <strong className="text-slate-300">AUDIT CONCLUSION:</strong> The XGBoost Classifier (Cycle 2020) demonstrates
                stable performance across validation folds. {summary.robustness_score === 'PASS' ? 'Recommended for deployment.' : 'Requires retraining.'}
            </div>
        </div>
    );
};

export default ModelAudit;
