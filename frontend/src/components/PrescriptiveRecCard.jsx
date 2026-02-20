import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, Box, Button, Stack, CircularProgress } from '@mui/material';
import { Calendar } from 'lucide-react';

export function PrescriptiveRecCard({ machineId, onCreateWorkOrder }) {
    const [rec, setRec] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!machineId) return;
        setLoading(true);
        fetch(`http://localhost:8000/api/recommendations/${machineId}`)
            .then((res) => res.json())
            .then((data) => { setRec(data); })
            .catch(() => setRec(null))
            .finally(() => setLoading(false));
    }, [machineId]);

    if (loading) {
        return (
            <Card variant="outlined" sx={{ height: '100%', borderRadius: 2, display: 'flex', alignItems: 'center', justifyContent: 'center', p: 2 }}>
                <CircularProgress size={24} />
            </Card>
        );
    }

    if (!rec) {
        return (
            <Card variant="outlined" sx={{ height: '100%', borderRadius: 2, p: 2 }}>
                <Typography variant="body2" color="text.secondary">No recommendation available</Typography>
            </Card>
        );
    }

    const component = (rec.parts && rec.parts[0]) ? rec.parts[0] : 'Asset';
    const action = rec.action || '—';
    const timing = rec.timeWindow || '—';
    const evidence = rec.featureEvidence || (rec.reasoning && rec.reasoning.split(/[.!?]/)[0]) || 'Sensor and model analysis';

    return (
        <Card variant="outlined" sx={{ height: '100%', borderRadius: 2, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <Box sx={{ px: 2, py: 1.5, borderBottom: 1, borderColor: 'divider' }}>
                <Typography variant="subtitle2" fontWeight="bold">Prescriptive Recommendation</Typography>
            </Box>
            <CardContent sx={{ flex: 1, p: 2, '&:lastChild': { pb: 2 } }}>
                <Stack spacing={1.5}>
                    <Box>
                        <Typography variant="caption" color="text.secondary">Where</Typography>
                        <Typography variant="body2" fontWeight="500">{component}</Typography>
                    </Box>
                    <Box>
                        <Typography variant="caption" color="text.secondary">What</Typography>
                        <Typography variant="body2" fontWeight="500">{action}</Typography>
                    </Box>
                    <Box>
                        <Typography variant="caption" color="text.secondary">When</Typography>
                        <Typography variant="body2" fontWeight="500">{timing}</Typography>
                    </Box>
                    <Box>
                        <Typography variant="caption" color="text.secondary">Based on</Typography>
                        <Typography variant="caption" display="block" sx={{ fontStyle: 'italic' }}>{evidence}</Typography>
                    </Box>
                    <Button
                        variant="contained"
                        size="small"
                        startIcon={<Calendar size={16} />}
                        onClick={() => onCreateWorkOrder(rec)}
                        sx={{ mt: 1, alignSelf: 'flex-start' }}
                    >
                        Create Work Order
                    </Button>
                </Stack>
            </CardContent>
        </Card>
    );
}
