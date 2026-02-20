import React from 'react';
import { Card, CardContent, CardHeader, Typography, Box, Stack, Button, Divider } from '@mui/material';
import { ArrowDown, ArrowUp, Activity, GitCommit, ArrowRight } from 'lucide-react';
import { useTheme, alpha } from '@mui/material/styles';

/**
 * LineContextWidget
 * Displays the machine's context within the production line (Upstream/Downstream).
 * onCompareTrends: optional callback when "Compare Trends" is clicked (e.g. scroll to trend charts).
 */
function LineContextWidget({ machineId, lineId = 'L-1', onCompareTrends }) {
    const theme = useTheme();

    // Mock Topology Context
    // Ideally this comes from a global topology state or genericized logic
    const context = {
        upstream: { id: 'ST-004', name: 'Stamping Press A', status: 'healthy', load: '98%' },
        current: { id: machineId, status: 'critical' },
        downstream: { id: 'AS-002', name: 'Assembly Bot', status: 'warning', load: '85%' }
    };

    const getStatusColor = (status) => {
        if (status === 'healthy') return theme.palette.success.main;
        if (status === 'warning') return theme.palette.warning.main;
        if (status === 'critical') return theme.palette.error.main;
        return theme.palette.text.disabled;
    };

    return (
        <Card variant="outlined" sx={{ height: '100%', borderRadius: 2, display: 'flex', flexDirection: 'column' }}>
            <CardHeader
                title={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <GitCommit size={18} className="text-slate-500" />
                        <Typography variant="subtitle1" fontWeight="bold">Line Context</Typography>
                    </Box>
                }
                sx={{ bgcolor: 'grey.50', py: 1.5, borderBottom: 1, borderColor: 'divider', flexShrink: 0 }}
            />
            <CardContent sx={{ p: 0, flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                {/* Vertical Flow Visualization */}
                <Box sx={{ p: 3, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0, flexGrow: 1 }}>
                    {/* Upstream */}
                    <AssetNode
                        data={context.upstream}
                        type="upstream"
                        color={getStatusColor(context.upstream.status)}
                    />

                    {/* Connection */}
                    <FlowLine />

                    {/* Current */}
                    <AssetNode
                        data={{ ...context.current, name: 'Current Asset' }} // Override name
                        type="current"
                        isCurrent
                        color={getStatusColor(context.current.status)}
                    />

                    {/* Connection */}
                    <FlowLine />

                    {/* Downstream */}
                    <AssetNode
                        data={context.downstream}
                        type="downstream"
                        color={getStatusColor(context.downstream.status)}
                    />
                </Box>

                <Divider />

                <Box sx={{ p: 2, bgcolor: 'grey.50' }} data-no-drag onClick={() => onCompareTrends?.()}>
                    <Button
                        type="button"
                        variant="outlined"
                        fullWidth
                        startIcon={<Activity size={16} />}
                        size="small"
                        sx={{ bgcolor: 'white', pointerEvents: 'auto' }}
                        onPointerDown={(e) => e.stopPropagation()}
                        onClick={(e) => { e.stopPropagation(); onCompareTrends?.(); }}
                    >
                        Compare Trends
                    </Button>
                </Box>
            </CardContent>
        </Card>
    );
}

function AssetNode({ data, type, isCurrent, color }) {
    return (
        <Box sx={{
            width: '100%',
            p: 1.5,
            borderRadius: 2,
            border: 1,
            borderColor: isCurrent ? 'primary.main' : 'divider',
            bgcolor: isCurrent ? 'primary.50' : 'white',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            boxShadow: isCurrent ? 2 : 1,
            position: 'relative',
            zIndex: 2
        }}>
            <Box>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ textTransform: 'uppercase', fontSize: '0.65rem', fontWeight: 'bold' }}>
                    {type}
                </Typography>
                <Typography variant="body2" fontWeight="bold" fontFamily="monospace">
                    {data.id}
                </Typography>
                <Typography variant="caption" color="text.secondary" noWrap>
                    {data.name}
                </Typography>
            </Box>

            <Box sx={{ textAlign: 'right' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, justifyContent: 'flex-end', mb: 0.5 }}>
                    <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: color }} />
                    <Typography variant="caption" fontWeight="bold" sx={{ color: color, textTransform: 'capitalize' }}>
                        {data.status}
                    </Typography>
                </Box>
                {data.load && (
                    <Typography variant="caption" color="text.secondary">
                        Load: {data.load}
                    </Typography>
                )}
            </Box>
        </Box>
    )
}

function FlowLine() {
    return (
        <Box sx={{ height: 24, width: 2, bgcolor: 'divider', my: 0.5, position: 'relative', zIndex: 1 }}>
            <Box sx={{ position: 'absolute', bottom: -4, left: '50%', transform: 'translateX(-50%)' }}>
                <ArrowDown size={12} className="text-slate-300" />
            </Box>
        </Box>
    )
}

export default LineContextWidget;
