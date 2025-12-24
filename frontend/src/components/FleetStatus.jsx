import React from 'react';

/**
 * FleetStatus Component
 * 
 * Displays a grid of machine status cards with real-time health indicators.
 * Color-coded: Green (Healthy), Yellow (Warning), Red (Critical)
 */
function FleetStatus({ machines }) {
    const getStatusColor = (prob) => {
        if (prob > 0.8) return { bg: 'bg-red-500/20', border: 'border-red-500', text: 'text-red-400' };
        if (prob > 0.5) return { bg: 'bg-yellow-500/20', border: 'border-yellow-500', text: 'text-yellow-400' };
        return { bg: 'bg-emerald-500/20', border: 'border-emerald-500', text: 'text-emerald-400' };
    };

    const getStatusLabel = (prob) => {
        if (prob > 0.8) return 'CRITICAL';
        if (prob > 0.5) return 'WARNING';
        return 'HEALTHY';
    };

    return (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {machines.map((machine) => {
                const colors = getStatusColor(machine.failure_probability);
                return (
                    <div
                        key={machine.machine_id}
                        className={`${colors.bg} ${colors.border} border rounded-xl p-4 backdrop-blur-sm transition-all hover:scale-105`}
                    >
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-gray-300 font-mono text-sm">{machine.machine_id}</span>
                            <span className={`${colors.text} text-xs font-bold px-2 py-1 rounded`}>
                                {getStatusLabel(machine.failure_probability)}
                            </span>
                        </div>

                        {/* Gauge */}
                        <div className="relative h-2 bg-gray-700 rounded-full overflow-hidden mb-3">
                            <div
                                className={`absolute h-full ${machine.failure_probability > 0.8 ? 'bg-red-500' : machine.failure_probability > 0.5 ? 'bg-yellow-500' : 'bg-emerald-500'}`}
                                style={{ width: `${machine.failure_probability * 100}%` }}
                            />
                        </div>

                        <div className="flex justify-between text-xs text-gray-400">
                            <span>AI Confidence</span>
                            <span className={colors.text}>{(machine.failure_probability * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                );
            })}
        </div>
    );
}

export default FleetStatus;
