import React, { useMemo } from 'react';
import { Box, Typography, Grid, Card, CardContent } from '@mui/material';
import { Factory, CheckCircle, AlertTriangle, AlertOctagon, Activity } from 'lucide-react';
import { FleetTreemap } from './FleetTreemap';

/**
 * Reliability engineer view: aggregated KPI dashboard + site-level FleetTreemap only.
 * No machine detail, no raw data, no work orders.
 */
function ReliabilityDashboard({ machines, user }) {
  const { healthy, warning, critical } = useMemo(() => {
    const h = [], w = [], c = [];
    machines.forEach((m) => {
      const prob = m.failure_probability || 0;
      if (prob > 0.8) c.push(m);
      else if (prob > 0.5) w.push(m);
      else h.push(m);
    });
    return { healthy: h, warning: w, critical: c };
  }, [machines]);

  const overallHealth =
    machines.length > 0
      ? Math.round(100 - (machines.reduce((sum, m) => sum + (m.failure_probability || 0), 0) / machines.length) * 100)
      : 100;

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        KPI Dashboard
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Site-level health overview
      </Typography>

      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={6} sm={3}>
          <Card variant="outlined" sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <Factory size={20} />
                <Typography variant="caption" color="text.secondary">Total Assets</Typography>
              </Box>
              <Typography variant="h4" fontWeight="bold">{machines.length}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Card variant="outlined" sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <CheckCircle size={20} color="var(--success)" />
                <Typography variant="caption" color="text.secondary">Optimal</Typography>
              </Box>
              <Typography variant="h4" fontWeight="bold" color="success.main">{healthy.length}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Card variant="outlined" sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <AlertTriangle size={20} />
                <Typography variant="caption" color="text.secondary">Warning</Typography>
              </Box>
              <Typography variant="h4" fontWeight="bold" color="warning.main">{warning.length}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Card variant="outlined" sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <AlertOctagon size={20} />
                <Typography variant="caption" color="text.secondary">Critical</Typography>
              </Box>
              <Typography variant="h4" fontWeight="bold" color="error.main">{critical.length}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Card variant="outlined" sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <Activity size={20} />
                <Typography variant="caption" color="text.secondary">System Health</Typography>
              </Box>
              <Typography
                variant="h4"
                fontWeight="bold"
                sx={{
                  color: overallHealth > 90 ? 'success.main' : overallHealth > 70 ? 'warning.main' : 'error.main',
                }}
              >
                {overallHealth}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Card variant="outlined" sx={{ overflow: 'hidden' }}>
        <Box sx={{ p: 1.5, borderBottom: 1, borderColor: 'divider' }}>
          <Typography variant="subtitle1" fontWeight="bold">Site health</Typography>
          <Typography variant="caption" color="text.secondary">Average health per site</Typography>
        </Box>
        <Box sx={{ height: 320, p: 1 }}>
          <FleetTreemap machines={machines} onSelectMachine={null} />
        </Box>
      </Card>
    </Box>
  );
}

export default ReliabilityDashboard;
