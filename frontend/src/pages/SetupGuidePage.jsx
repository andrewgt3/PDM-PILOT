import React from 'react';
import {
    Box, Typography, Paper, useTheme, alpha,
    Table, TableBody, TableCell, TableRow, TableHead
} from '@mui/material';

/**
 * Setup Guide — Part of Plant OS
 * 
 * Data Setup → Features → Model → 6-Stage Inference Pipeline
 * Accessible from Sidebar for operators and admins.
 */
export default function SetupGuidePage() {
    const theme = useTheme();

    const STAGES = [
        { id: 1, label: 'INGEST', desc: 'OPC UA / File Watcher', color: theme.palette.primary.main },
        { id: 2, label: 'CLEANSE', desc: 'Outlier removal', color: theme.palette.secondary.main },
        { id: 3, label: 'CONTEXT', desc: 'Digital Twin map', color: theme.palette.info.main },
        { id: 4, label: 'PERSIST', desc: 'TimescaleDB', color: theme.palette.success.main },
        { id: 5, label: 'INFER', desc: 'XGBoost RUL', color: theme.palette.warning.main },
        { id: 6, label: 'ORCH', desc: 'API Gateway', color: theme.palette.primary.dark },
    ];

    return (
        <Box sx={{ maxWidth: 900, mx: 'auto', p: 3 }}>
            <Typography variant="h5" fontWeight={800} gutterBottom>
                Pipeline Setup Guide
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Data Setup → Features → Model → 6-Stage Inference Pipeline
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 2 }}>
                Standards: ISO 55001 (Asset Management) | ISO 13372/13374 (Condition Monitoring)
            </Typography>

            {/* 6-Stage Architecture */}
            <Paper elevation={0} sx={{ p: 2, mb: 3, border: 1, borderColor: 'divider', borderRadius: 2 }}>
                <Typography variant="subtitle2" fontWeight={700} color="text.secondary" sx={{ mb: 2, letterSpacing: 1 }}>
                    6-STAGE PIPELINE (Compartmentalized)
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {STAGES.map((s, i) => (
                        <Box
                            key={s.id}
                            sx={{
                                px: 1.5, py: 0.75,
                                borderRadius: 1,
                                bgcolor: alpha(s.color, 0.1),
                                border: `1px solid ${alpha(s.color, 0.3)}`,
                                display: 'flex', alignItems: 'center', gap: 1
                            }}
                        >
                            <Typography variant="caption" fontWeight={800} sx={{ color: s.color }}>{s.id}</Typography>
                            <Typography variant="caption" fontWeight={600}>{s.label}</Typography>
                            <Typography variant="caption" color="text.secondary">{s.desc}</Typography>
                            {i < STAGES.length - 1 && (
                                <Typography variant="caption" color="text.disabled">→</Typography>
                            )}
                        </Box>
                    ))}
                </Box>
            </Paper>

            {/* CLI Steps */}
            <Paper elevation={0} sx={{ p: 2, mb: 3, border: 1, borderColor: 'divider', borderRadius: 2, bgcolor: 'grey.50' }}>
                <Typography variant="subtitle2" fontWeight={700} color="text.secondary" sx={{ mb: 1.5, letterSpacing: 1 }}>
                    CLI STEPS (run in terminal)
                </Typography>
                <Box component="pre" sx={{ fontFamily: 'monospace', fontSize: '0.75rem', overflow: 'auto', m: 0, color: 'text.primary', whiteSpace: 'pre-wrap' }}>
{`# 1. Get data (choose one or both)
python scripts/download_nasa_data.py      # → archive.zip
python scripts/download_femto_data.py    # → data/downloads/femto_pronostia

# 2. Build features (after Upload or Extract Archive in Pipeline Ops)
python scripts/build_nasa_features_with_physics.py   # NASA only
# OR
python scripts/build_combined_features.py            # NASA + FEMTO

# 3. Train model
python scripts/retrain_rul_with_physics.py

# 4. Validate (optional)
python scripts/validate_rul_on_nasa.py --save`}
                </Box>
            </Paper>

            {/* Stage → Component Mapping */}
            <Paper elevation={0} sx={{ p: 2, border: 1, borderColor: 'divider', borderRadius: 2 }}>
                <Typography variant="subtitle2" fontWeight={700} color="text.secondary" sx={{ mb: 2, letterSpacing: 1 }}>
                    STAGE → COMPONENT MAPPING
                </Typography>
                <Table size="small">
                    <TableHead>
                        <TableRow>
                            <TableCell sx={{ fontWeight: 700, fontSize: '0.75rem' }}>Stage</TableCell>
                            <TableCell sx={{ fontWeight: 700, fontSize: '0.75rem' }}>Component</TableCell>
                            <TableCell sx={{ fontWeight: 700, fontSize: '0.75rem' }}>Path</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        <TableRow><TableCell>1 INGEST</TableCell><TableCell>Watcher / OPC-UA</TableCell><TableCell>watcher_service.py, pipeline/ingestion</TableCell></TableRow>
                        <TableRow><TableCell>2 CLEANSE</TableCell><TableCell>Refinery</TableCell><TableCell>refinery.py, pipeline/cleansing</TableCell></TableRow>
                        <TableRow><TableCell>3 CONTEXT</TableCell><TableCell>Digital Twin</TableCell><TableCell>pipeline/contextualization</TableCell></TableRow>
                        <TableRow><TableCell>4 PERSIST</TableCell><TableCell>TimescaleDB Writer</TableCell><TableCell>stream_consumer.py, pipeline/persistence</TableCell></TableRow>
                        <TableRow><TableCell>5 INFER</TableCell><TableCell>XGBoost RUL</TableCell><TableCell>inference_service.py, pipeline/inference</TableCell></TableRow>
                        <TableRow><TableCell>6 ORCH</TableCell><TableCell>API + Frontend</TableCell><TableCell>api_server.py, frontend/</TableCell></TableRow>
                    </TableBody>
                </Table>
            </Paper>
        </Box>
    );
}
