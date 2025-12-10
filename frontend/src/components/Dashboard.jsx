import React from 'react';
import { useNavigate } from 'react-router-dom';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from 'recharts';
import { Activity, AlertTriangle, CheckCircle, TrendingUp, ArrowRight } from 'lucide-react';

const KPICard = ({ title, value, subtext, icon: Icon, color }) => (
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

const AlertCard = ({ robot, onClick }) => (
    <div
        onClick={onClick}
        className="card border-l-4 border-l-red-500 cursor-pointer hover:bg-slate-800 transition-colors flex items-center justify-between group"
    >
        <div>
            <div className="flex items-center gap-2 mb-1">
                <span className="font-bold text-white text-lg">{robot.name}</span>
                <span className="badge badge-critical">{robot.status}</span>
            </div>
            <div className="text-sm text-slate-400 font-mono">
                Risk Prov: <span className="text-white">{robot.risk}%</span> | Pred: {robot.prediction}
            </div>
        </div>
        <div className="p-2 rounded-full bg-slate-800 group-hover:bg-slate-700 text-slate-400 group-hover:text-white transition-colors">
            <ArrowRight size={20} />
        </div>
    </div>
);

const Dashboard = ({ robots = [] }) => {
    const navigate = useNavigate();

    // 1. Calculate KPIs
    const criticalCount = robots.filter(r => r.status === 'critical').length;
    const warningCount = robots.filter(r => r.status === 'warning').length;
    const healthyCount = robots.filter(r => r.status === 'healthy').length;
    const avgRisk = robots.reduce((acc, r) => acc + (r.risk || 0), 0) / (robots.length || 1);
    const systemHealth = Math.max(0, 100 - avgRisk);

    // Filter for Active Alerts
    const activeAlerts = robots.filter(r => r.status === 'critical' || r.status === 'warning');

    // 2. Mock History Trend (Simulated 24h ending at current health)
    const trendData = Array.from({ length: 24 }, (_, i) => {
        const time = `${i}:00`;
        const randomVar = (Math.random() - 0.5) * 5;
        // Trend downwards slightly if criticals exist
        const trend = criticalCount > 0 ? -i * 0.5 : 0;
        return {
            time,
            value: Math.min(100, Math.max(0, systemHealth + randomVar + trend)) // Converge to current
        };
    }).map((d, i) => i === 23 ? { time: 'Now', value: systemHealth } : d);

    // 3. Distribution Data
    const pieData = [
        { name: 'Healthy', value: healthyCount, color: '#10b981' },
        { name: 'Warning', value: warningCount, color: '#f59e0b' },
        { name: 'Critical', value: criticalCount, color: '#ef4444' },
    ].filter(d => d.value > 0);

    return (
        <div className="flex flex-col gap-6 animate-fade-in">
            {/* KPI Row */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <KPICard
                    title="System Health"
                    value={`${systemHealth.toFixed(1)}%`}
                    subtext={criticalCount > 0 ? "Degradation Detected" : "Optimal Performance"}
                    icon={Activity}
                    color={systemHealth < 90 ? '#ef4444' : '#10b981'}
                />
                <KPICard
                    title="Active Alerts"
                    value={criticalCount + warningCount}
                    subtext={`${criticalCount} Critical | ${warningCount} Warning`}
                    icon={AlertTriangle}
                    color="#f59e0b"
                />
                <KPICard
                    title="Assets Monitored"
                    value={robots.length}
                    subtext="100% Connectivity"
                    icon={CheckCircle}
                    color="#3b82f6"
                />
                <KPICard
                    title="Avg Failure Prob"
                    value={`${avgRisk.toFixed(1)}%`}
                    subtext=" AI Model Confidence: High"
                    icon={TrendingUp}
                    color="#8b5cf6"
                />
            </div>

            {/* Critical Alerts Section (New) */}
            {activeAlerts.length > 0 && (
                <div>
                    <h3 className="text-slate-300 font-bold mb-3 flex items-center gap-2">
                        <AlertTriangle size={18} className="text-red-500" />
                        Attention Required ({activeAlerts.length})
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {activeAlerts.map(robot => (
                            <AlertCard
                                key={robot.id}
                                robot={robot}
                                onClick={() => navigate(`/assets/${robot.id}`)}
                            />
                        ))}
                    </div>
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[400px]">
                {/* Main Trend Chart */}
                <div className="card col-span-2 flex flex-col">
                    <h3 className="text-slate-300 font-bold mb-4 flex items-center gap-2">
                        <Activity size={16} className="text-indigo-400" />
                        System Health Trend (24h)
                    </h3>
                    <div className="flex-1 min-h-0">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={trendData}>
                                <defs>
                                    <linearGradient id="colorHealth" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <XAxis dataKey="time" stroke="#475569" fontSize={12} tickLine={false} axisLine={false} />
                                <YAxis domain={[0, 100]} stroke="#475569" fontSize={12} tickLine={false} axisLine={false} />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: '#1e293b',
                                        borderColor: '#334155',
                                        borderRadius: '8px',
                                        color: '#f8fafc'
                                    }}
                                />
                                <Area type="monotone" dataKey="value" stroke="#6366f1" strokeWidth={3} fillOpacity={1} fill="url(#colorHealth)" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Status Distribution */}
                <div className="card flex flex-col">
                    <h3 className="text-slate-300 font-bold mb-4 flex items-center gap-2">
                        <AlertTriangle size={16} className="text-amber-400" />
                        Fleet Status Distribution
                    </h3>
                    <div className="flex-1 min-h-0 relative">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={pieData}
                                    innerRadius={60}
                                    outerRadius={80}
                                    paddingAngle={5}
                                    dataKey="value"
                                >
                                    {pieData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} stroke="none" />
                                    ))}
                                </Pie>
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: '#1e293b',
                                        borderColor: '#334155',
                                        borderRadius: '8px',
                                        color: '#f8fafc'
                                    }}
                                />
                                <Legend verticalAlign="bottom" height={36} />
                            </PieChart>
                        </ResponsiveContainer>
                        {/* Centered Text */}
                        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                            <span className="text-2xl font-bold text-white mb-8">{robots.length}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
