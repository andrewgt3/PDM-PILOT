import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ComposedChart, Line, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, BarChart, Bar, Cell } from 'recharts';
import { ArrowLeft, Cpu, Activity, Thermometer, Zap, AlertTriangle } from 'lucide-react';

const MetricCard = ({ label, value, unit, icon: Icon, color }) => (
  <div className="card p-4 flex items-center justify-between">
    <div>
      <div className="text-xs text-slate-500 uppercase font-bold">{label}</div>
      <div className="text-xl font-bold text-white mt-1">
        {value} <span className="text-sm text-slate-500 font-normal">{unit}</span>
      </div>
    </div>
    <div className="p-2 rounded-lg bg-slate-800/50">
      <Icon size={20} color={color} />
    </div>
  </div>
);

const AssetView = ({ robots }) => {
  const { id } = useParams();
  const navigate = useNavigate();
  const robot = robots.find(r => r.id === id);

  if (!robot) return <div className="p-8 text-white">Asset not found</div>;

  // 1. Generate History for Regressions (Simulated based on current sensor values to show drop-off)
  // If critical, we simulate a "ramp up" of torque/temp over time.
  const isCritical = robot.status === 'critical';
  const baseTorque = robot.sensors?.Torque || 40;
  const baseTemp = robot.sensors?.Temperature || 300;

  const historyData = Array.from({ length: 30 }, (_, i) => {
    const t = i;
    const noise = (Math.random() - 0.5) * 2;
    // Simple linear ramp if critical
    const torqueTrend = isCritical ? (t * 0.5) : 0;

    return {
      time: `-${30 - i}m`,
      Torque: Math.max(0, baseTorque - 15 + torqueTrend + noise), // Ramp up to current
      Temperature: Math.max(0, baseTemp - 10 + (torqueTrend * 0.5) + noise),
      Speed: robot.sensors?.['Rotational Speed'] + noise * 5 || 0
    };
  });

  // 2. Linear Regression (Simple approach: y = mx + c)
  // We project 10 points into future
  const lastPoint = historyData[historyData.length - 1];
  const slope = isCritical ? 0.8 : 0.05; // Critical has steep slope
  const futureData = Array.from({ length: 10 }, (_, i) => {
    return {
      time: `+${i + 1}m`,
      PredictedTorque: lastPoint.Torque + (slope * (i + 1))
    }
  });

  const combinedData = [...historyData, ...futureData];

  // 3. Feature Importance (Why is risk high?)
  // Mock logic: If Torque > 50, it's the main driver.
  const featureImportance = [
    { name: 'Torque', value: Math.max(10, (robot.sensors?.Torque || 0)), color: '#8b5cf6' },
    { name: 'Tool Wear', value: Math.max(10, (robot.sensors?.['Tool Wear'] || 0) * 0.3), color: '#ec4899' }, // Scale down for viz
    { name: 'Temperature', value: Math.max(10, ((robot.sensors?.Temperature || 300) - 290)), color: '#f59e0b' },
    { name: 'Rotational Speed', value: 10, color: '#10b981' },
  ].sort((a, b) => b.value - a.value);

  return (
    <div className="flex flex-col gap-6 animate-fade-in pb-10">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button onClick={() => navigate('/assets')} className="p-2 hover:bg-slate-800 rounded-full text-slate-400 hover:text-white transition-colors">
          <ArrowLeft size={20} />
        </button>
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            {robot.name} <span className="text-slate-500 text-lg font-normal">/ {robot.id}</span>
          </h1>
          <div className="flex items-center gap-2 mt-1">
            <span className={`badge ${robot.status === 'critical' ? 'badge-critical' : 'badge-healthy'}`}>
              {robot.status}
            </span>
            <span className="text-xs text-slate-400">Model Confidence: 94%</span>
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Failure Probability"
          value={`${robot.risk}%`}
          unit=""
          icon={Activity}
          color={robot.risk > 50 ? '#ef4444' : '#10b981'}
        />
        <MetricCard
          label="Current Torque"
          value={robot.sensors?.Torque?.toFixed(1) || '--'}
          unit="Nm"
          icon={Zap}
          color="#8b5cf6"
        />
        <MetricCard
          label="Temperature"
          value={robot.sensors?.Temperature?.toFixed(1) || '--'}
          unit="K"
          icon={Thermometer}
          color="#f59e0b"
        />
        <MetricCard
          label="Projected RUL"
          value={robot.prediction || '> 30 Days'}
          unit=""
          icon={Cpu}
          color="#3b82f6"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Analysis Chart */}
        <div className="card col-span-2 h-[450px] flex flex-col">
          <h3 className="text-slate-300 font-bold mb-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity size={16} className="text-indigo-400" />
              Real-time Telemetry & Projection
            </div>
            <div className="flex gap-4 text-xs">
              <span className="flex items-center gap-1 text-slate-400"><div className="w-2 h-2 bg-indigo-500 rounded-full"></div> Torque</span>
              <span className="flex items-center gap-1 text-slate-400"><div className="w-2 h-2 bg-amber-500 rounded-full"></div> Temp</span>
              <span className="flex items-center gap-1 text-slate-400"><div className="w-2 h-[2px] bg-slate-400 border-dashed border-t"></div> Projection</span>
            </div>
          </h3>
          <div className="flex-1 min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={combinedData}>
                <defs>
                  <linearGradient id="colorTorque" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="time" stroke="#475569" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis yAxisId="left" stroke="#475569" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis yAxisId="right" orientation="right" domain={[300, 320]} stroke="#475569" fontSize={12} tickLine={false} axisLine={false} hide />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    borderColor: '#334155',
                    borderRadius: '8px',
                    color: '#f8fafc'
                  }}
                />
                <ReferenceLine x="29m" stroke="#94a3b8" strokeDasharray="3 3" />

                <Area yAxisId="left" type="monotone" dataKey="Torque" stroke="#8b5cf6" strokeWidth={2} fill="url(#colorTorque)" />
                <Line yAxisId="right" type="monotone" dataKey="Temperature" stroke="#f59e0b" strokeWidth={2} dot={false} />
                {/* Regression Line */}
                <Line yAxisId="left" type="monotone" dataKey="PredictedTorque" stroke="#94a3b8" strokeWidth={2} strokeDasharray="5 5" dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Explainable AI / Contributing Factors */}
        <div className="card flex flex-col h-[450px]">
          <h3 className="text-slate-300 font-bold mb-4 flex items-center gap-2">
            <AlertTriangle size={16} className="text-pink-400" />
            Root Cause Analysis (SHAP)
          </h3>
          <p className="text-xs text-slate-500 mb-4">
            Feature contribution to current Failure Probability.
          </p>
          <div className="flex-1 min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart layout="vertical" data={featureImportance} margin={{ left: 0 }}>
                <XAxis type="number" hide />
                <YAxis dataKey="name" type="category" width={100} tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip cursor={{ fill: 'transparent' }} contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '8px', color: '#fff' }} />
                <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20}>
                  {featureImportance.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Recommendations */}
          <div className="mt-4 pt-4 border-t border-[var(--border-subtle)]">
            <h4 className="text-xs font-bold text-slate-400 uppercase mb-2">AI Recommendation</h4>
            <div className="bg-slate-800/50 p-3 rounded text-sm text-slate-300">
              {isCritical
                ? "Immediate inspection of Axis 3 Gearbox required. High torque signature matches bearing failure profile."
                : "System operating within normal parameters. Next scheduled maintenance: 14 days."}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AssetView;
