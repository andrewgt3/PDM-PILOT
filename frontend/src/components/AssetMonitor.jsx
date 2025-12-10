import React from 'react';
import { useNavigate } from 'react-router-dom';
import { AlertCircle, CheckCircle, ArrowRight } from 'lucide-react';

const AssetMonitor = ({ robots }) => {
    const navigate = useNavigate();

    return (
        <div className="flex flex-col gap-6 animate-fade-in">
            <h1 className="text-2xl font-bold text-white flex items-center gap-2">
                Asset Monitor
                <span className="text-sm font-normal text-slate-500 bg-slate-800 px-2 py-1 rounded-full">{robots.length} Units</span>
            </h1>

            <div className="card overflow-hidden p-0">
                <table className="w-full text-left border-collapse">
                    <thead>
                        <tr className="bg-slate-800/50 border-b border-[var(--border-subtle)] text-xs uppercase tracking-wider text-slate-400">
                            <th className="p-4 font-medium">Asset ID</th>
                            <th className="p-4 font-medium">Status</th>
                            <th className="p-4 font-medium">Failure Prob</th>
                            <th className="p-4 font-medium">Prediction (RUL)</th>
                            <th className="p-4 font-medium text-right">Action</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-[var(--border-subtle)]">
                        {robots.map(robot => (
                            <tr
                                key={robot.id}
                                onClick={() => navigate(`/assets/${robot.id}`)}
                                className="hover:bg-slate-800/50 transition-colors cursor-pointer group"
                            >
                                <td className="p-4">
                                    <div className="font-bold text-white">{robot.name}</div>
                                    <div className="text-xs text-slate-500">{robot.id}</div>
                                </td>
                                <td className="p-4">
                                    <span className={`badge ${robot.status === 'critical' ? 'badge-critical' :
                                            robot.status === 'warning' ? 'badge-warning' : 'badge-healthy'
                                        }`}>
                                        {robot.status}
                                    </span>
                                </td>
                                <td className="p-4">
                                    <div className="flex items-center gap-2">
                                        <div className="w-24 h-2 bg-slate-700 rounded-full overflow-hidden">
                                            <div
                                                className={`h-full ${robot.risk > 50 ? 'bg-red-500' :
                                                        robot.risk > 20 ? 'bg-amber-500' : 'bg-emerald-500'
                                                    }`}
                                                style={{ width: `${robot.risk}%` }}
                                            />
                                        </div>
                                        <span className="text-sm font-mono text-slate-300">{robot.risk}%</span>
                                    </div>
                                </td>
                                <td className="p-4 text-sm text-slate-300">
                                    {robot.prediction ? robot.prediction : (
                                        <span className="text-slate-500 italic">--</span>
                                    )}
                                </td>
                                <td className="p-4 text-right">
                                    <button className="text-slate-400 group-hover:text-white transition-colors">
                                        <ArrowRight size={18} />
                                    </button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default AssetMonitor;
