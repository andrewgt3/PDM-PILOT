import React, { useMemo, useState } from 'react';
import { Plus, ChevronRight, Activity, Zap, Layers } from 'lucide-react';
import { Paper, Slider, Typography, Switch, FormGroup, FormControlLabel, Box, Button, Collapse, IconButton } from '@mui/material';
import { alpha, useTheme } from '@mui/material/styles';

/**
 * FleetTopology - Digital Twin Edition
 * A high-fidelity visual representation of the plant floor with operational forecasting.
 */
export function FleetTopology({ machines, onSelectMachine, width, height }) {
    const theme = useTheme();
    const [showUnassigned, setShowUnassigned] = useState(false);

    // Simulation State
    const [showSimControls, setShowSimControls] = useState(false);
    const [isSimulating, setIsSimulating] = useState(false);
    const [simParams, setSimParams] = useState({
        lineSpeed: 100, // %
        shiftLoad: 80   // % utilization
    });

    const handleSimChange = (param) => (e, newValue) => {
        setSimParams(prev => ({ ...prev, [param]: newValue }));
        if (!isSimulating) setIsSimulating(true);
    };

    // Group machines by shop
    const groupedByShop = useMemo(() => {
        const shops = {};
        machines.forEach(m => {
            const shop = m.shop || "Unassigned";
            if (!shops[shop]) {
                shops[shop] = [];
            }
            shops[shop].push(m);
        });
        return shops;
    }, [machines]);

    const getStatusStyles = (machine) => {
        let prob = (machine?.failure_probability || 0) * 100;

        // Apply Simulation Impact
        if (isSimulating) {
            // Simplified logic: Higher speed/load increases effective probability
            // If lineSpeed > 110%, add 10% risk.
            // If shiftLoad > 90%, add 5% risk.
            const speedPenalty = Math.max(0, simParams.lineSpeed - 100) * 0.5;
            const loadPenalty = Math.max(0, simParams.shiftLoad - 85) * 0.4;
            prob += speedPenalty + loadPenalty;
        }

        if (prob > 80) return {
            color: '#ef4444',
            stroke: '#dc2626',
            pulse: true,
            gradient: 'url(#grad-critical)'
        };
        if (prob > 50) return {
            color: '#fbbf24',
            stroke: '#f59e0b',
            pulse: false,
            gradient: 'url(#grad-warning)'
        };
        return {
            color: '#34d399',
            stroke: '#10b981',
            pulse: false,
            gradient: 'url(#grad-healthy)'
        };
    };

    const TwinNode = ({ machine, x, y, isHub = false }) => {
        const styles = getStatusStyles(machine);
        const size = isHub ? 56 : 48; // Slightly larger for better visibility
        const displayId = machine?.machine_id || '?';

        return (
            <g
                onClick={() => onSelectMachine && onSelectMachine(machine.machine_id)}
                style={{ cursor: 'pointer', pointerEvents: 'all' }}
            >
                {/* Critical Alarm Double Ring Pulse */}
                {styles.pulse && (
                    <>
                        <circle cx={x} cy={y} r={size / 2 + 8} fill="none" stroke={styles.color} strokeWidth="1" opacity="0.3">
                            <animate attributeName="r" from={size / 2 + 8} to={size / 2 + 16} dur="2s" repeatCount="indefinite" />
                            <animate attributeName="opacity" from="0.5" to="0" dur="2s" repeatCount="indefinite" />
                        </circle>
                        <circle cx={x} cy={y} r={size / 2 + 4} fill="none" stroke={styles.color} strokeWidth="1.5" opacity="0.5">
                            <animate attributeName="r" from={size / 2 + 4} to={size / 2 + 10} dur="2s" begin="0.5s" repeatCount="indefinite" />
                            <animate attributeName="opacity" from="0.5" to="0" dur="2s" begin="0.5s" repeatCount="indefinite" />
                        </circle>
                    </>
                )}

                {/* Glassmorphic Base */}
                <circle
                    cx={x}
                    cy={y}
                    r={size / 2}
                    fill="url(#glass-gradient)"
                    stroke={styles.stroke}
                    strokeWidth="2"
                    filter="url(#drop-shadow)"
                    className="transition-all duration-300 hover:stroke-[3px]"
                />

                {/* Status Indicator Dot */}
                <circle cx={x} cy={y - size / 2 + 5} r="3" fill={styles.color} />

                {/* ID Text - Monospaced High Contrast */}
                <text
                    x={x}
                    y={y + 4}
                    textAnchor="middle"
                    fill="#1e293b"
                    fontSize="11"
                    fontWeight="700"
                    fontFamily="monospace"
                    style={{ pointerEvents: 'none', textShadow: '0px 1px 2px rgba(255,255,255,0.8)' }}
                >
                    {displayId}
                </text>
            </g>
        );
    };

    const PipeConnection = ({ x1, y1, x2, y2, statusFn }) => {
        const isStressed = isSimulating && (simParams.lineSpeed > 110 || simParams.shiftLoad > 90);
        const pipeColor = isStressed ? theme.palette.warning.main : "#cbd5e1";
        const flowColor = isStressed ? theme.palette.error.main : "#94a3b8";

        return (
            <g>
                {/* Base Pipe */}
                <path
                    d={`M ${x1} ${y1} L ${x2} ${y2}`}
                    stroke={pipeColor}
                    strokeWidth={isStressed ? 6 : 4} // Thicker if stressed
                    strokeLinecap="round"
                    opacity="0.5"
                    transition="all 0.5s"
                />
                {/* Flow Animation */}
                <path
                    d={`M ${x1} ${y1} L ${x2} ${y2}`}
                    stroke={flowColor}
                    strokeWidth="2"
                    strokeDasharray="4 8"
                    strokeLinecap="round"
                    fill="none"
                    opacity="0.8"
                >
                    <animate
                        attributeName="stroke-dashoffset"
                        from="12"
                        to="0"
                        dur={isSimulating ? `${1.5 * (100 / simParams.lineSpeed)}s` : "1.5s"} // Faster animation
                        repeatCount="indefinite"
                    />
                </path>
            </g>
        );
    };

    // Calculate layout
    const svgWidth = width || 900;
    const svgHeight = height || 320;
    const sectionWidth = svgWidth / 4;

    const shopConfigs = [
        { name: 'Body Shop', x: 10, w: sectionWidth - 20, layout: 'tree', label: 'BODY SHOP' },
        { name: 'Stamping', x: sectionWidth + 10, w: sectionWidth - 20, layout: 'tree', label: 'STAMPING' },
        { name: 'Paint Shop', x: sectionWidth * 2 + 10, w: sectionWidth - 20, layout: 'star', label: 'PAINT SHOP' },
        { name: 'Final Assembly', x: sectionWidth * 3 + 10, w: sectionWidth - 20, layout: 'star', label: 'FINAL ASSEMBLY' },
    ];

    const unassignedCount = (groupedByShop['Unassigned'] || []).length + (groupedByShop['Unassigned Shop'] || []).length;


    const renderShop = (config) => {
        const machines = groupedByShop[config.name] || [];
        const centerX = config.x + config.w / 2;

        // Zone Container
        const zone = (
            <g key={`zone-${config.name}`}>
                <rect
                    x={config.x}
                    y={10}
                    width={config.w}
                    height={svgHeight - 20}
                    rx="12"
                    fill={isSimulating ? alpha(theme.palette.primary.main, 0.05) : "#f8fafc"}
                    stroke={isSimulating ? theme.palette.primary.light : "#e2e8f0"}
                    strokeWidth="1"
                    opacity="0.8"
                    className="transition-colors duration-500"
                />
                {/* Zone Header */}
                <rect
                    x={config.x}
                    y={10}
                    width={config.w}
                    height="32"
                    rx="12"
                    // Clip bottom corners? manually involves path. Simpler: rounded rect + rect to cover
                    fill="#f1f5f9"
                    opacity="0.5"
                    clipPath={`inset(0 0 ${svgHeight - 52}px 0 round 12px 12px 0 0)`}
                />
                <text
                    x={config.x + 16}
                    y={32}
                    textAnchor="start"
                    fill="#64748b"
                    fontSize="10"
                    fontWeight="700"
                    letterSpacing="1.5"
                    style={{ textTransform: 'uppercase' }}
                >
                    {config.label}
                </text>
            </g>
        );

        if (machines.length === 0) return zone;

        // Node Placement
        const positions = [];
        const connections = [];
        const contentStartY = 60;
        const availableHeight = svgHeight - 80;

        if (config.layout === 'tree') {
            // Tree Layout
            const levelStep = Math.min(80, availableHeight / 3);
            const levels = [contentStartY, contentStartY + levelStep, contentStartY + levelStep * 2];

            if (machines[0]) positions.push({ m: machines[0], x: centerX, y: levels[0], isHub: true }); // Root

            // Level 2 (2 nodes max)
            const l2Width = 100;
            if (machines[1]) {
                positions.push({ m: machines[1], x: centerX - l2Width / 2, y: levels[1] });
                connections.push({ from: 0, to: positions.length - 1 });
            }
            if (machines[2]) {
                positions.push({ m: machines[2], x: centerX + l2Width / 2, y: levels[1] });
                connections.push({ from: 0, to: positions.length - 1 });
            }

            // Level 3 (Rest)
            const remaining = machines.slice(3);
            const l3Width = Math.min(config.w - 40, remaining.length * 60);
            const l3Start = centerX - l3Width / 2 + (l3Width / Math.max(1, remaining.length)) / 2;

            remaining.forEach((m, i) => {
                positions.push({
                    m,
                    x: l3Start + i * (l3Width / Math.max(1, remaining.length)),
                    y: levels[2]
                });
                // Connect to nearest parent? Ideally index 1 or 2. 
                // Simple logic: alternate
                connections.push({ from: 1 + (i % 2), to: positions.length - 1 });
            });

        } else {
            // Star Layout
            const hubY = svgHeight / 2 + 10;
            const radius = Math.min(80, availableHeight / 3);

            if (machines[0]) positions.push({ m: machines[0], x: centerX, y: hubY, isHub: true });

            const satellites = machines.slice(1);
            const angleStep = (2 * Math.PI) / Math.max(1, satellites.length);

            satellites.forEach((m, i) => {
                const angle = angleStep * i - Math.PI / 2;
                positions.push({
                    m,
                    x: centerX + radius * Math.cos(angle),
                    y: hubY + radius * Math.sin(angle)
                });
                connections.push({ from: 0, to: positions.length - 1 });
            });
        }

        return (
            <g key={config.name}>
                {zone}
                {connections.map((c, i) => positions[c.from] && positions[c.to] && (
                    <PipeConnection
                        key={`${config.name}-c-${i}`}
                        x1={positions[c.from].x} y1={positions[c.from].y}
                        x2={positions[c.to].x} y2={positions[c.to].y}
                    />
                ))}
                {positions.map((p, i) => (
                    <TwinNode
                        key={`${config.name}-n-${i}`}
                        machine={p.m}
                        x={p.x}
                        y={p.y}
                        isHub={p.isHub}
                    />
                ))}
            </g>
        );
    };

    return (
        <div className="w-full bg-white rounded-lg border border-slate-200 overflow-hidden relative" style={{ height: '100%' }}>
            {/* Simulation Control Panel - Floating Top Right */}
            <Paper
                elevation={3}
                sx={{
                    position: 'absolute',
                    top: 16,
                    right: 16,
                    zIndex: 10,
                    width: showSimControls ? 280 : 48,
                    height: showSimControls ? 'auto' : 48,
                    borderRadius: 4,
                    overflow: 'hidden',
                    transition: 'width 0.3s ease, height 0.3s ease',
                    bgcolor: isSimulating ? 'primary.50' : 'white',
                    border: '1px solid',
                    borderColor: 'divider'
                }}
            >
                {!showSimControls ? (
                    <IconButton onClick={() => setShowSimControls(true)} sx={{ width: '100%', height: '100%' }}>
                        <Zap size={20} className={isSimulating ? "text-amber-500 fill-amber-500" : "text-slate-500"} />
                    </IconButton>
                ) : (
                    <Box sx={{ p: 2 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Zap size={18} className="text-amber-500" />
                                <Typography variant="subtitle2" fontWeight="bold">Operational Forecasting</Typography>
                            </Box>
                            <IconButton size="small" onClick={() => setShowSimControls(false)}>
                                <ChevronRight size={16} />
                            </IconButton>
                        </Box>

                        <FormControlLabel
                            control={<Switch checked={isSimulating} onChange={(e) => setIsSimulating(e.target.checked)} size="small" />}
                            label={<Typography variant="caption" fontWeight="bold">Simulate Impact</Typography>}
                            sx={{ mb: 2 }}
                        />

                        <Collapse in={isSimulating}>
                            <Box sx={{ mb: 2 }}>
                                <Typography variant="caption" color="text.secondary" gutterBottom display="block">
                                    Line Speed ({simParams.lineSpeed}%)
                                </Typography>
                                <Slider
                                    size="small"
                                    value={simParams.lineSpeed}
                                    min={50} max={150}
                                    onChange={handleSimChange('lineSpeed')}
                                    valueLabelDisplay="auto"
                                    color={simParams.lineSpeed > 100 ? 'warning' : 'primary'}
                                />
                            </Box>
                            <Box>
                                <Typography variant="caption" color="text.secondary" gutterBottom display="block">
                                    Shift Load ({simParams.shiftLoad}%)
                                </Typography>
                                <Slider
                                    size="small"
                                    value={simParams.shiftLoad}
                                    min={0} max={120}
                                    onChange={handleSimChange('shiftLoad')}
                                    valueLabelDisplay="auto"
                                    color={simParams.shiftLoad > 90 ? 'error' : 'primary'}
                                />
                            </Box>
                        </Collapse>
                    </Box>
                )}
            </Paper>

            <svg
                width="100%"
                height="100%"
                viewBox={`0 0 ${svgWidth} ${svgHeight}`}
                preserveAspectRatio="xMidYMid meet"
                className="bg-slate-50/20"
            >
                {/* Visual Definitions */}
                <defs>
                    <pattern id="grid-subtle" width="40" height="40" patternUnits="userSpaceOnUse">
                        <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#cbd5e1" strokeWidth="0.5" opacity="0.4" />
                    </pattern>
                    <linearGradient id="glass-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="white" stopOpacity="0.9" />
                        <stop offset="100%" stopColor="white" stopOpacity="0.4" />
                    </linearGradient>
                    <filter id="drop-shadow" x="-50%" y="-50%" width="200%" height="200%">
                        <feGaussianBlur in="SourceAlpha" stdDeviation="2" />
                        <feOffset dx="0" dy="2" result="offsetblur" />
                        <feComponentTransfer>
                            <feFuncA type="linear" slope="0.1" />
                        </feComponentTransfer>
                        <feMerge>
                            <feMergeNode />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>

                {/* Background Grid */}
                <rect width="100%" height="100%" fill="url(#grid-subtle)" />

                {/* Render Shops */}
                {shopConfigs.map(renderShop)}

            </svg>

            {/* Unassigned Assets FAB */}
            {unassignedCount > 0 && (
                <div className="absolute bottom-4 right-4">
                    <button
                        className="flex items-center gap-2 bg-white px-4 py-2 rounded-full shadow-lg border border-slate-200 hover:shadow-xl hover:border-blue-300 transition-all font-semibold text-slate-600 group"
                        onClick={() => setShowUnassigned(!showUnassigned)}
                    >
                        <div className="bg-slate-100 p-1 rounded-full group-hover:bg-blue-50">
                            <Plus size={16} className="text-slate-500 group-hover:text-blue-500" />
                        </div>
                        <span>{unassignedCount} Pending</span>
                        <ChevronRight size={16} className={`text-slate-400 transition-transform ${showUnassigned ? 'rotate-90' : ''}`} />
                    </button>

                    {/* Popover/Sidebar could go here, for now simpler visualization */}
                </div>
            )}
        </div>
    );
}

export default FleetTopology;
