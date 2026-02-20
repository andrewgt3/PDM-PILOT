import React from 'react';
import { Box, Typography, List, ListItemButton, ListItemText, Paper, Grid } from '@mui/material';
import { Wrench } from 'lucide-react';
import WorkOrderPanel from './WorkOrderPanel';

/**
 * Technician view: assigned machines list + work orders panel only.
 */
function TechnicianHome({ machines, assignedMachineIds, onSelectMachine }) {
  const assigned = (assignedMachineIds?.length > 0)
    ? machines.filter((m) => assignedMachineIds.includes(m.machine_id))
    : machines;

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        My machines
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        {assigned.length} assigned asset{assigned.length !== 1 ? 's' : ''}
      </Typography>
      <Grid container spacing={2}>
        <Grid item xs={12} md={4}>
          <Paper variant="outlined" sx={{ p: 1, maxHeight: 400, overflow: 'auto' }}>
            <List dense>
              {assigned.length === 0 ? (
                <Box sx={{ p: 3, textAlign: 'center', color: 'text.secondary' }}>
                  <Wrench size={32} style={{ opacity: 0.4, marginBottom: 8 }} />
                  <Typography variant="body2">No assigned machines</Typography>
                </Box>
              ) : (
                assigned.map((m) => {
                  const prob = m.failure_probability || 0;
                  const isCritical = prob > 0.8;
                  const isWarning = prob > 0.5;
                  return (
                    <ListItemButton
                      key={m.machine_id}
                      onClick={() => onSelectMachine(m.machine_id)}
                      sx={{
                        borderRadius: 1,
                        borderLeft: '3px solid',
                        borderColor: isCritical ? 'error.main' : isWarning ? 'warning.main' : 'success.main',
                      }}
                    >
                      <ListItemText
                        primary={m.machine_id}
                        secondary={m.line_name || null}
                        primaryTypographyProps={{ fontWeight: 600, fontSize: '0.9rem' }}
                      />
                    </ListItemButton>
                  );
                })
              )}
            </List>
          </Paper>
        </Grid>
        <Grid item xs={12} md={8}>
          <WorkOrderPanel machine={null} />
        </Grid>
      </Grid>
    </Box>
  );
}

export default TechnicianHome;
