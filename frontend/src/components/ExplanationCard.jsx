import React from 'react';
import { Card, CardContent, Typography, Box, Chip, Stack } from '@mui/material';
import { ArrowUp, ArrowDown, HelpCircle } from 'lucide-react';

/**
 * Plain-language explanation card for why the model produced the current health score.
 * Audience: maintenance technician (clarity over precision).
 *
 * Props:
 *   explanations: array of { feature, display_name, shap_value, direction, magnitude }
 *   healthScore: number 0-100
 *   machineName: string (optional)
 */
function ExplanationCard({ explanations = [], healthScore, machineName }) {
    const hasExplanations = Array.isArray(explanations) && explanations.length > 0;
    const maxAbs = hasExplanations
        ? Math.max(...explanations.map((e) => Math.abs(e.shap_value || 0)), 1e-6)
        : 1;

    const magnitudeColor = (mag) => {
        if (mag === 'HIGH') return '#dc2626';
        if (mag === 'MEDIUM') return '#d97706';
        return '#059669';
    };

    return (
        <Card variant="outlined" sx={{ height: '100%', borderRadius: 2, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <Box sx={{ px: 2, py: 1.5, borderBottom: 1, borderColor: 'divider', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 1 }}>
                <Stack direction="row" spacing={1} alignItems="center">
                    <HelpCircle size={18} color="#64748b" />
                    <Typography variant="subtitle1" fontWeight="bold">Why This Score?</Typography>
                </Stack>
                {healthScore != null && (
                    <Chip
                        label={`Health ${Math.round(Number(healthScore))}%`}
                        size="small"
                        color={healthScore >= 70 ? 'success' : healthScore >= 40 ? 'warning' : 'error'}
                        variant="outlined"
                        sx={{ fontWeight: 'bold' }}
                    />
                )}
            </Box>
            <CardContent sx={{ flex: 1, py: 2, '&:last-child': { pb: 2 } }}>
                {!hasExplanations ? (
                    <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
                        Explanation data not available for this prediction.
                    </Typography>
                ) : (
                    <>
                        {explanations.slice(0, 3).map((item, idx) => {
                            const isRisk = (item.direction || '').toLowerCase() === 'increases_risk';
                            const absVal = Math.abs(item.shap_value || 0);
                            const widthPct = Math.min(100, (absVal / maxAbs) * 100);
                            const mag = (item.magnitude || 'LOW').toUpperCase();
                            const color = magnitudeColor(mag);
                            return (
                                <Box key={idx} sx={{ mb: 2 }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.5 }}>
                                        <Typography variant="body2" fontWeight="600" color="text.primary">
                                            {item.display_name || item.feature || '—'}
                                        </Typography>
                                        <Stack direction="row" spacing={0.5} alignItems="center">
                                            {isRisk ? (
                                                <ArrowUp size={16} color="#dc2626" strokeWidth={2.5} />
                                            ) : (
                                                <ArrowDown size={16} color="#059669" strokeWidth={2.5} />
                                            )}
                                            <Typography
                                                variant="caption"
                                                fontWeight="bold"
                                                sx={{ color: isRisk ? '#dc2626' : '#059669', textTransform: 'uppercase' }}
                                            >
                                                {mag}
                                            </Typography>
                                        </Stack>
                                    </Box>
                                    <Box
                                        sx={{
                                            height: 8,
                                            borderRadius: 1,
                                            bgcolor: '#f1f5f9',
                                            overflow: 'hidden',
                                        }}
                                    >
                                        <Box
                                            sx={{
                                                height: '100%',
                                                width: `${widthPct}%`,
                                                bgcolor: color,
                                                borderRadius: 1,
                                                transition: 'width 0.3s ease',
                                            }}
                                        />
                                    </Box>
                                </Box>
                            );
                        })}
                        {explanations[0] && (
                            <Typography
                                variant="body2"
                                sx={{
                                    mt: 2,
                                    pt: 2,
                                    borderTop: 1,
                                    borderColor: 'divider',
                                    color: 'text.secondary',
                                    fontStyle: 'italic',
                                }}
                            >
                                {explanations[0].direction === 'increases_risk'
                                    ? `The main concern is ${explanations[0].display_name || explanations[0].feature}, which is currently elevated above the normal operating range.`
                                    : `The system is performing better than normal on ${explanations[0].display_name || explanations[0].feature}, which is offsetting other risk factors.`}
                            </Typography>
                        )}
                    </>
                )}
            </CardContent>
        </Card>
    );
}

export default ExplanationCard;
