import React, { useMemo, useState, useCallback, useRef, useEffect } from 'react';
import { Plus, ChevronRight, Zap, LayoutGrid, List } from 'lucide-react';
import {
    Paper, Slider, Typography, Switch, FormControlLabel, Box, Collapse, IconButton,
    Drawer, List as MuiList, ListItemButton, ListItemText, ToggleButtonGroup, ToggleButton,
    Table, TableBody, TableCell, TableHead, TableRow, TableContainer,
} from '@mui/material';
import { alpha, useTheme } from '@mui/material/styles';

/**
 * FleetTopology - Shop → Line hierarchy with expand-to-see-machines.
 * Scales to large fleets by showing line-level nodes; individual machines in drawer or list view.
 */
export function FleetTopology({ machines, onSelectMachine, width, height }) {
    const theme = useTheme();
    const [showUnassigned, setShowUnassigned] = useState(false);
    const [viewMode, setViewMode] = useState('topology');
    const [selectedLine, setSelectedLine] = useState(null); // { shop, line }
    const [zoomPan, setZoomPan] = useState({ scale: 1, tx: 0, ty: 0 });
    const [isPanning, setIsPanning] = useState(false);
    const [panStart, setPanStart] = useState({ x: 0, y: 0 });

    // Simulation State
    const [showSimControls, setShowSimControls] = useState(false);
    const [isSimulating, setIsSimulating] = useState(false);
    const [simParams, setSimParams] = useState({
        lineSpeed: 100,
        shiftLoad: 80,
    });

    const handleSimChange = (param) => (e, newValue) => {
        setSimParams(prev => ({ ...prev, [param]: newValue }));
        if (!isSimulating) setIsSimulating(true);
    };

    const zoomPanRef = useRef(null);
    useEffect(() => {
        const el = zoomPanRef.current;
        if (!el) return;
        const onWheel = (e) => {
            e.preventDefault();
            setZoomPan((prev) => {
                const delta = e.deltaY > 0 ? -0.1 : 0.1;
                const next = Math.min(3, Math.max(0.5, prev.scale + delta));
                return { ...prev, scale: next };
            });
        };
        el.addEventListener('wheel', onWheel, { passive: false });
        return () => el.removeEventListener('wheel', onWheel);
    }, []);
    const handleMouseDown = useCallback((e) => {
        if (e.button === 0) setIsPanning(true), setPanStart({ x: e.clientX, y: e.clientY });
    }, []);
    const handleMouseMove = useCallback((e) => {
        if (!isPanning) return;
        setZoomPan((prev) => ({
            ...prev,
            tx: prev.tx + (e.clientX - panStart.x),
            ty: prev.ty + (e.clientY - panStart.y),
        }));
        setPanStart({ x: e.clientX, y: e.clientY });
    }, [isPanning, panStart.x, panStart.y]);
    const handleMouseUp = useCallback(() => setIsPanning(false), []);
    const handleMouseLeave = useCallback(() => setIsPanning(false), []);

    // Group machines by shop, then by line (same as FleetTreemap)
    const groupedByShopAndLine = useMemo(() => {
        const byShop = {};
        machines.forEach((m) => {
            const shop = m.shop || 'Unassigned';
            const line = m.line_name || m.line || 'Unassigned Line';
            if (!byShop[shop]) byShop[shop] = {};
            if (!byShop[shop][line]) byShop[shop][line] = [];
            byShop[shop][line].push(m);
        });
        return byShop;
    }, [machines]);

    const unassignedCount = (groupedByShopAndLine['Unassigned'] ? Object.values(groupedByShopAndLine['Unassigned']).flat() : []).length +
        (groupedByShopAndLine['Unassigned Shop'] ? Object.values(groupedByShopAndLine['Unassigned Shop']).flat() : []).length;

    // Worst status among machines (for line aggregate)
    const getStatusStyles = (machineOrProb) => {
        const prob = typeof machineOrProb === 'number' ? machineOrProb : (machineOrProb?.failure_probability ?? 0) * 100;
        let effective = prob;
        if (isSimulating) {
            const speedPenalty = Math.max(0, simParams.lineSpeed - 100) * 0.5;
            const loadPenalty = Math.max(0, simParams.shiftLoad - 85) * 0.4;
            effective += speedPenalty + loadPenalty;
        }
        if (effective > 80) return { color: '#ef4444', stroke: '#dc2626', pulse: true };
        if (effective > 50) return { color: '#fbbf24', stroke: '#f59e0b', pulse: false };
        return { color: '#34d399', stroke: '#10b981', pulse: false };
    };

    const getLineAggregateStatus = (lineMachines) => {
        if (!lineMachines?.length) return { worstProb: 0, styles: getStatusStyles(0) };
        const worstProb = Math.max(...lineMachines.map((m) => (m.failure_probability ?? 0) * 100));
        return { worstProb, styles: getStatusStyles(worstProb) };
    };

    const LineNode = ({ lineName, lineMachines, x, y, boxWidth, boxHeight, onSelect }) => {
        const count = lineMachines.length;
        const { styles } = getLineAggregateStatus(lineMachines);
        const rx = 8;
        const padding = 6;
        const truncatedName = lineName.length > 14 ? lineName.slice(0, 12) + '…' : lineName;

        return (
            <g
                onClick={() => onSelect?.()}
                style={{ cursor: 'pointer', pointerEvents: 'all' }}
            >
                <rect
                    x={x}
                    y={y}
                    width={boxWidth}
                    height={boxHeight}
                    rx={rx}
                    fill="url(#glass-gradient)"
                    stroke={styles.stroke}
                    strokeWidth="2"
                    filter="url(#drop-shadow)"
                    className="transition-all duration-300 hover:stroke-[3px]"
                />
                <circle cx={x + padding + 5} cy={y + padding + 5} r="4" fill={styles.color} />
                <text
                    x={x + padding + 14}
                    y={y + padding + 9}
                    textAnchor="start"
                    fill="#1e293b"
                    fontSize="10"
                    fontWeight="600"
                    style={{ pointerEvents: 'none' }}
                >
                    {truncatedName}
                </text>
                <text
                    x={x + boxWidth - padding - 4}
                    y={y + padding + 9}
                    textAnchor="end"
                    fill="#64748b"
                    fontSize="10"
                    fontWeight="700"
                    fontFamily="monospace"
                    style={{ pointerEvents: 'none' }}
                >
                    {count}
                </text>
            </g>
        );
    };

    // Calculate layout
    const svgWidth = width || 900;
    const svgHeight = height || 320;
    const sectionWidth = svgWidth / 4;

    const shopConfigs = [
        { name: 'Body Shop', x: 10, w: sectionWidth - 20, label: 'BODY SHOP' },
        { name: 'Stamping', x: sectionWidth + 10, w: sectionWidth - 20, label: 'STAMPING' },
        { name: 'Paint Shop', x: sectionWidth * 2 + 10, w: sectionWidth - 20, label: 'PAINT SHOP' },
        { name: 'Final Assembly', x: sectionWidth * 3 + 10, w: sectionWidth - 20, label: 'FINAL ASSEMBLY' },
    ];

    const contentStartY = 52;

    const renderShop = (config) => {
        const linesMap = groupedByShopAndLine[config.name];
        const lineEntries = linesMap ? Object.entries(linesMap) : [];

        const zone = (
            <g key={`zone-${config.name}`}>
                <rect
                    x={config.x}
                    y={10}
                    width={config.w}
                    height={svgHeight - 20}
                    rx="12"
                    fill={isSimulating ? alpha(theme.palette.primary.main, 0.05) : '#f8fafc'}
                    stroke={isSimulating ? theme.palette.primary.light : '#e2e8f0'}
                    strokeWidth="1"
                    opacity="0.8"
                    className="transition-colors duration-500"
                />
                <rect
                    x={config.x}
                    y={10}
                    width={config.w}
                    height="32"
                    rx="12"
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

        if (lineEntries.length === 0) return zone;

        const boxWidth = config.w - 16;
        const availableHeight = svgHeight - contentStartY - 24;
        const gap = 6;
        const boxHeight = Math.max(24, (availableHeight - (lineEntries.length - 1) * gap) / lineEntries.length);

        return (
            <g key={config.name}>
                {zone}
                {lineEntries.map(([lineName, lineMachines], i) => {
                    const y = contentStartY + i * (boxHeight + gap);
                    const x = config.x + 8;
                    return (
                        <LineNode
                            key={`${config.name}-${lineName}`}
                            lineName={lineName}
                            lineMachines={lineMachines}
                            x={x}
                            y={y}
                            boxWidth={boxWidth}
                            boxHeight={boxHeight}
                            onSelect={() => setSelectedLine({ shop: config.name, line: lineName })}
                        />
                    );
                })}
            </g>
        );
    };

    const drawerMachines = useMemo(() => {
        if (!selectedLine) return [];
        const lines = groupedByShopAndLine[selectedLine.shop];
        return lines?.[selectedLine.line] ?? [];
    }, [selectedLine, groupedByShopAndLine]);

    return (
        <div className="w-full bg-white rounded-lg border border-slate-200 overflow-hidden relative" style={{ height: '100%' }}>
            {/* View mode toggle: Topology | List */}
            <Box sx={{ position: 'absolute', top: 8, left: 8, zIndex: 10 }}>
                <ToggleButtonGroup
                    value={viewMode}
                    exclusive
                    onChange={(e, v) => v != null && setViewMode(v)}
                    size="small"
                    sx={{ bgcolor: 'background.paper', boxShadow: 1 }}
                >
                    <ToggleButton value="topology" aria-label="Topology">
                        <LayoutGrid size={16} style={{ marginRight: 6 }} />
                        Topology
                    </ToggleButton>
                    <ToggleButton value="list" aria-label="List">
                        <List size={16} style={{ marginRight: 6 }} />
                        List
                    </ToggleButton>
                </ToggleButtonGroup>
            </Box>

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

            {viewMode === 'list' ? (
                <TableContainer sx={{ maxHeight: '100%', pt: 5, px: 1 }}>
                    <Table size="small" stickyHeader>
                        <TableHead>
                            <TableRow>
                                <TableCell sx={{ fontWeight: 700 }}>Shop</TableCell>
                                <TableCell sx={{ fontWeight: 700 }}>Line</TableCell>
                                <TableCell sx={{ fontWeight: 700 }}>Machine ID</TableCell>
                                <TableCell sx={{ fontWeight: 700 }}>Name</TableCell>
                                <TableCell sx={{ fontWeight: 700 }}>Status</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {machines.map((m) => {
                                const prob = (m.failure_probability ?? 0) * 100;
                                const status = prob > 80 ? 'Critical' : prob > 50 ? 'Warning' : 'Optimal';
                                return (
                                    <TableRow
                                        key={m.machine_id}
                                        hover
                                        onClick={() => onSelectMachine?.(m.machine_id)}
                                        sx={{ cursor: onSelectMachine ? 'pointer' : 'default' }}
                                    >
                                        <TableCell>{m.shop ?? '—'}</TableCell>
                                        <TableCell>{(m.line_name || m.line) ?? '—'}</TableCell>
                                        <TableCell sx={{ fontFamily: 'monospace' }}>{m.machine_id}</TableCell>
                                        <TableCell>{m.machine_name ?? m.machine_id}</TableCell>
                                        <TableCell sx={{ fontWeight: 600 }}>{status}</TableCell>
                                    </TableRow>
                                );
                            })}
                        </TableBody>
                    </Table>
                </TableContainer>
            ) : (
                <Box
                    ref={zoomPanRef}
                    sx={{
                        width: '100%',
                        height: '100%',
                        overflow: 'hidden',
                        cursor: isPanning ? 'grabbing' : 'grab',
                        userSelect: 'none',
                    }}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseLeave}
                >
                    <Box
                        sx={{
                            transform: `translate(${zoomPan.tx}px, ${zoomPan.ty}px) scale(${zoomPan.scale})`,
                            transformOrigin: '0 0',
                            width: '100%',
                            height: '100%',
                        }}
                    >
                        <svg
                            width="100%"
                            height="100%"
                            viewBox={`0 0 ${svgWidth} ${svgHeight}`}
                            preserveAspectRatio="xMidYMid meet"
                            className="bg-slate-50/20"
                            style={{ pointerEvents: isPanning ? 'none' : 'auto' }}
                        >
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
                            <rect width="100%" height="100%" fill="url(#grid-subtle)" />
                            {shopConfigs.map(renderShop)}
                        </svg>
                    </Box>
                </Box>
            )}

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
                </div>
            )}

            {/* Drawer: machines for selected line */}
            <Drawer
                anchor="right"
                open={Boolean(selectedLine)}
                onClose={() => setSelectedLine(null)}
                PaperProps={{ sx: { width: { xs: '100%', sm: 360 } } }}
            >
                <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
                    <Typography variant="subtitle1" fontWeight="bold">
                        {selectedLine?.shop} → {selectedLine?.line}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                        {drawerMachines.length} machine{drawerMachines.length !== 1 ? 's' : ''}
                    </Typography>
                </Box>
                <MuiList dense sx={{ py: 0 }}>
                    {drawerMachines.map((m) => {
                        const prob = (m.failure_probability ?? 0) * 100;
                        const status = prob > 80 ? 'Critical' : prob > 50 ? 'Warning' : 'Optimal';
                        return (
                            <ListItemButton
                                key={m.machine_id}
                                onClick={() => {
                                    onSelectMachine?.(m.machine_id);
                                    setSelectedLine(null);
                                }}
                            >
                                <ListItemText
                                    primary={m.machine_name || m.machine_id}
                                    secondary={`${m.machine_id} · ${status}`}
                                    primaryTypographyProps={{ fontWeight: 600 }}
                                    secondaryTypographyProps={{ fontFamily: 'monospace', fontSize: '0.75rem' }}
                                />
                            </ListItemButton>
                        );
                    })}
                </MuiList>
            </Drawer>
        </div>
    );
}

export default FleetTopology;
