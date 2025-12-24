import React, { useState } from 'react';
import { Users, User, CheckCircle, Clock, Star, Wrench, Phone, Award, ChevronRight } from 'lucide-react';

/**
 * TechnicianAssignment Component
 * 
 * Shows available technicians with skills matching the required repair.
 * Would integrate with HR/scheduling systems in production.
 */
function TechnicianAssignment({ requiredSkills = ['Bearing Replacement', 'Robot Maintenance'], priority = 'high' }) {
    const [selectedTech, setSelectedTech] = useState(null);

    // Mock technician data
    const technicians = [
        {
            id: 'TECH-001',
            name: 'Mike Thompson',
            photo: null,
            shift: 'Day',
            status: 'available',
            currentLocation: 'Body Shop',
            skills: ['Robot Maintenance', 'Bearing Replacement', 'Servo Motors', 'PLC Programming'],
            certifications: ['FANUC Certified', 'SKF Bearing Specialist'],
            avgResponseTime: 12, // minutes
            completionRate: 98,
            phone: 'x4521'
        },
        {
            id: 'TECH-002',
            name: 'Sarah Chen',
            photo: null,
            shift: 'Day',
            status: 'busy',
            currentTask: 'WO-2024-1847',
            estimatedFree: '15 min',
            currentLocation: 'Stamping',
            skills: ['Robot Maintenance', 'Welding Equipment', 'Hydraulics'],
            certifications: ['ABB Certified'],
            avgResponseTime: 18,
            completionRate: 95,
            phone: 'x4523'
        },
        {
            id: 'TECH-003',
            name: 'Carlos Rodriguez',
            photo: null,
            shift: 'Day',
            status: 'available',
            currentLocation: 'Tool Room',
            skills: ['Bearing Replacement', 'Conveyor Systems', 'Pneumatics'],
            certifications: ['Vibration Analyst Level II'],
            avgResponseTime: 22,
            completionRate: 92,
            phone: 'x4528'
        },
        {
            id: 'TECH-004',
            name: 'Jennifer Park',
            photo: null,
            shift: 'Night',
            status: 'off_shift',
            currentLocation: '-',
            skills: ['Servo Motors', 'PLC Programming', 'HMI Configuration'],
            certifications: ['Siemens Certified'],
            avgResponseTime: 15,
            completionRate: 97,
            phone: 'x4530'
        }
    ];

    // Calculate skill match percentage
    const calculateMatch = (techSkills) => {
        const matches = requiredSkills.filter(skill =>
            techSkills.some(ts => ts.toLowerCase().includes(skill.toLowerCase()))
        );
        return Math.round((matches.length / requiredSkills.length) * 100);
    };

    // Sort by availability and skill match
    const sortedTechnicians = [...technicians].sort((a, b) => {
        // First by status (available first)
        const statusOrder = { available: 0, busy: 1, off_shift: 2 };
        if (statusOrder[a.status] !== statusOrder[b.status]) {
            return statusOrder[a.status] - statusOrder[b.status];
        }
        // Then by skill match
        return calculateMatch(b.skills) - calculateMatch(a.skills);
    });

    const statusConfig = {
        available: { color: 'text-emerald-600', bg: 'bg-emerald-100', dot: 'bg-emerald-500', label: 'Available' },
        busy: { color: 'text-amber-600', bg: 'bg-amber-100', dot: 'bg-amber-500', label: 'Busy' },
        off_shift: { color: 'text-slate-500', bg: 'bg-slate-100', dot: 'bg-slate-400', label: 'Off Shift' }
    };

    const handleAssign = (tech) => {
        setSelectedTech(tech.id);
        // Would trigger API call to scheduling system
        console.log(`Assigning ${tech.name} to work order`);
    };

    return (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
            {/* Header */}
            <div className="px-5 py-4 border-b border-slate-100 bg-slate-50">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Users className="w-5 h-5 text-slate-600" />
                        <h3 className="font-semibold text-slate-900">Technician Assignment</h3>
                    </div>
                    <div className="text-xs text-slate-500">
                        Required: {requiredSkills.join(', ')}
                    </div>
                </div>
            </div>

            {/* Technician List */}
            <div className="divide-y divide-slate-100 max-h-96 overflow-y-auto">
                {sortedTechnicians.map((tech) => {
                    const config = statusConfig[tech.status];
                    const matchPercent = calculateMatch(tech.skills);
                    const isSelected = selectedTech === tech.id;

                    return (
                        <div
                            key={tech.id}
                            className={`px-5 py-4 transition-colors ${isSelected ? 'bg-indigo-50 border-l-4 border-indigo-500' : 'hover:bg-slate-50'
                                } ${tech.status === 'off_shift' ? 'opacity-60' : ''}`}
                        >
                            <div className="flex items-start gap-3">
                                {/* Avatar */}
                                <div className="relative">
                                    <div className="w-10 h-10 bg-slate-200 rounded-full flex items-center justify-center">
                                        <User className="w-5 h-5 text-slate-500" />
                                    </div>
                                    <span className={`absolute -bottom-0.5 -right-0.5 w-3 h-3 ${config.dot} rounded-full border-2 border-white`}></span>
                                </div>

                                {/* Info */}
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2 mb-1">
                                        <span className="font-medium text-slate-900">{tech.name}</span>
                                        <span className={`px-1.5 py-0.5 text-[10px] font-medium rounded ${config.bg} ${config.color}`}>
                                            {config.label}
                                        </span>
                                        {matchPercent === 100 && (
                                            <span className="flex items-center gap-0.5 px-1.5 py-0.5 bg-indigo-100 text-indigo-700 text-[10px] font-bold rounded">
                                                <Star className="w-3 h-3" /> Best Match
                                            </span>
                                        )}
                                    </div>

                                    <div className="flex flex-wrap gap-1 mb-2">
                                        {tech.skills.slice(0, 3).map((skill, idx) => (
                                            <span
                                                key={idx}
                                                className={`px-1.5 py-0.5 text-[10px] rounded ${requiredSkills.some(rs => skill.toLowerCase().includes(rs.toLowerCase()))
                                                        ? 'bg-emerald-100 text-emerald-700 font-medium'
                                                        : 'bg-slate-100 text-slate-600'
                                                    }`}
                                            >
                                                {skill}
                                            </span>
                                        ))}
                                        {tech.skills.length > 3 && (
                                            <span className="text-[10px] text-slate-400">+{tech.skills.length - 3} more</span>
                                        )}
                                    </div>

                                    <div className="flex items-center gap-4 text-xs text-slate-500">
                                        {tech.status === 'busy' && tech.estimatedFree && (
                                            <span className="flex items-center gap-1">
                                                <Clock className="w-3 h-3" />
                                                Free in {tech.estimatedFree}
                                            </span>
                                        )}
                                        {tech.currentLocation !== '-' && (
                                            <span>@ {tech.currentLocation}</span>
                                        )}
                                        <span className="flex items-center gap-1">
                                            <Award className="w-3 h-3" />
                                            {tech.completionRate}% completion
                                        </span>
                                    </div>
                                </div>

                                {/* Actions */}
                                <div className="flex items-center gap-2">
                                    {tech.status === 'available' && (
                                        <button
                                            onClick={() => handleAssign(tech)}
                                            disabled={isSelected}
                                            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${isSelected
                                                    ? 'bg-emerald-600 text-white'
                                                    : 'bg-indigo-600 hover:bg-indigo-700 text-white'
                                                }`}
                                        >
                                            {isSelected ? (
                                                <span className="flex items-center gap-1">
                                                    <CheckCircle className="w-3.5 h-3.5" /> Assigned
                                                </span>
                                            ) : (
                                                'Assign'
                                            )}
                                        </button>
                                    )}
                                    <button className="p-1.5 text-slate-400 hover:text-slate-600 transition-colors">
                                        <Phone className="w-4 h-4" />
                                    </button>
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Footer */}
            <div className="px-5 py-3 bg-slate-50 border-t border-slate-100 flex items-center justify-between text-xs">
                <span className="text-slate-500">
                    {technicians.filter(t => t.status === 'available').length} technicians available
                </span>
                <button className="text-indigo-600 hover:text-indigo-800 font-medium flex items-center gap-1">
                    View Full Schedule <ChevronRight className="w-3 h-3" />
                </button>
            </div>
        </div>
    );
}

export default TechnicianAssignment;
