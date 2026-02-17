import React, { useEffect, useState } from 'react';
import { ThemeProvider, CssBaseline, Box, CircularProgress, Typography } from '@mui/material';
import theme from './theme';
import useWebSocket from './hooks/useWebSocket';
import Sidebar from './components/Sidebar';
import PlantOverview from './components/PlantOverview';
import MachineDetail from './components/MachineDetail';
import AnomalyDiscoveryPage from './pages/AnomalyDiscoveryPage';
import SetupGuidePage from './pages/SetupGuidePage';
import PipelineOps from './components/PipelineOps';
import LoginPage from './components/LoginPage';
import LoadingScreen from './components/LoadingScreen';
import { Factory } from 'lucide-react';
import './index.css';

const API_BASE = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws/stream';

function App() {
  const { messages, latestMessage, isConnected } = useWebSocket(WS_URL);
  const [machines, setMachines] = useState([]);
  const [selectedMachineId, setSelectedMachineId] = useState(null);
  const [view, setView] = useState('overview'); // 'overview' or 'detail'

  // Authentication & Loading State
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Pipeline Ops state — shown after loading, before main app
  const [showPipelineOps, setShowPipelineOps] = useState(true);

  const handleLogin = () => {
    setIsAuthenticated(true);
    setIsLoading(true);
  };

  const handleLoadingComplete = () => {
    setIsLoading(false);
  };

  const handleGoLive = () => {
    setShowPipelineOps(false);
    setView('overview');
  };

  // Fetch initial machine list from REST API
  useEffect(() => {
    fetch(`${API_BASE}/api/machines`)
      .then((res) => res.json())
      .then((data) => {
        if (data.data) {
          setMachines(data.data);
        }
      })
      .catch((err) => console.error('Failed to fetch machines:', err));
  }, []);

  // Update machine status when new WebSocket message arrives
  useEffect(() => {
    if (latestMessage) {
      setMachines((prev) => {
        const updated = [...prev];
        const idx = updated.findIndex(
          (m) => m.machine_id === latestMessage.machine_id
        );

        if (idx >= 0) {
          updated[idx] = {
            ...updated[idx],
            failure_probability: latestMessage.failure_probability,
            last_seen: latestMessage.timestamp,
            rul_days: latestMessage.rul_days,
            operational_status: latestMessage.operational_status,
            recommendation: latestMessage.recommendation,
            anomaly_detected: latestMessage.anomaly_detected,
            line_name: latestMessage.line_name,
            model_number: latestMessage.model_number,
            install_date: latestMessage.install_date
          };
        } else {
          updated.push({
            machine_id: latestMessage.machine_id,
            failure_probability: latestMessage.failure_probability,
            last_seen: latestMessage.timestamp,
            rul_days: latestMessage.rul_days,
            operational_status: latestMessage.operational_status,
            status: latestMessage.failure_probability > 0.8 ? 'CRITICAL' : 'HEALTHY',
            line_name: latestMessage.line_name,
            model_number: latestMessage.model_number,
            install_date: latestMessage.install_date
          });
        }

        return updated;
      });
    }
  }, [latestMessage]);

  // Handle machine selection (switches to detail view)
  const handleSelectMachine = (machineId) => {
    setSelectedMachineId(machineId);
    setView('detail');
  };

  // Handle back to overview
  const handleBackToOverview = () => {
    setSelectedMachineId(null);
    setView('overview');
  };

  // Find the selected machine object
  const selectedMachine = machines.find(m => m.machine_id === selectedMachineId);

  /* New State for Layout Preference */
  const [isFlushLayout, setIsFlushLayout] = useState(() => {
    return localStorage.getItem('app_layout_flush') === 'true';
  });

  const toggleLayout = () => {
    const newVal = !isFlushLayout;
    setIsFlushLayout(newVal);
    localStorage.setItem('app_layout_flush', String(newVal));
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />

      {!isAuthenticated ? (
        <LoginPage onLogin={handleLogin} />
      ) : isLoading ? (
        <LoadingScreen onComplete={handleLoadingComplete} />
      ) : showPipelineOps ? (
        <PipelineOps onGoLive={handleGoLive} />
      ) : (
        <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
          {/* Sidebar Navigation */}
          <Sidebar
            machines={machines}
            selectedMachineId={selectedMachineId}
            view={view}
            setView={setView}
            onSelectMachine={handleSelectMachine}
          />

          {/* Main Content Area */}
          <Box component="main" sx={{
            flexGrow: 1,
            width: '100%', // Ensure it takes available space
            overflow: 'hidden', // Standard behavior
            padding: 0,
            minHeight: '100vh',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <Box sx={{
              p: 0,
              flexGrow: 1,
              overflow: 'auto', // Allow scrolling
              transition: 'padding 0.3s ease'
            }}>
              {/* Top Bar / Status */}
              <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 3, alignItems: 'center', gap: 2 }}>
                {/* Layout Toggle - Temporary Location */}
                <Box
                  onClick={toggleLayout}
                  sx={{
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                    opacity: 0.7,
                    '&:hover': { opacity: 1 },
                    p: 1,
                    borderRadius: 1,
                    bgcolor: 'action.hover'
                  }}
                >
                  <Typography variant="caption" fontWeight="bold">
                    {isFlushLayout ? "COMPACT" : "FULL WIDTH"}
                  </Typography>
                </Box>
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                    px: 1.5,
                    py: 0.75,
                    borderRadius: 4,
                    // Note: MUI palette access in sx is direct e.g. 'success.main'. 
                    // Using custom bg colors for now via alpha or theme colors if defined.
                    // Let's use theme.palette directly or correct aliases.
                    // Reverting to hardcoded or close approximates to ensure no build error if theme structure varies.
                    bgcolor: isConnected ? '#d1fae5' : '#fee2e2',
                    color: isConnected ? '#047857' : '#b91c1c'
                  }}
                >
                  <Box
                    sx={{
                      width: 8,
                      height: 8,
                      borderRadius: '50%',
                      bgcolor: isConnected ? 'success.main' : 'error.main',
                      animation: isConnected ? 'pulse 2s infinite' : 'none',
                      '@keyframes pulse': {
                        '0%': { boxShadow: '0 0 0 0 rgba(5, 150, 105, 0.7)' },
                        '70%': { boxShadow: '0 0 0 10px rgba(5, 150, 105, 0)' },
                        '100%': { boxShadow: '0 0 0 0 rgba(5, 150, 105, 0)' }
                      }
                    }}
                  />
                  <Typography variant="caption" fontWeight="bold">
                    {isConnected ? 'LIVE' : 'OFFLINE'}
                  </Typography>
                </Box>
              </Box>

              {/* View Router */}
              {view === 'pipeline-ops' ? (
                <PipelineOps onGoLive={handleGoLive} />
              ) : view === 'setup-guide' ? (
                <SetupGuidePage />
              ) : view === 'overview' ? (
                <PlantOverview
                  machines={machines}
                  messages={messages}
                  onSelectMachine={handleSelectMachine}
                />
              ) : view === 'anomaly-discovery' ? (
                <AnomalyDiscoveryPage />
              ) : selectedMachine ? (
                <MachineDetail
                  machine={selectedMachine}
                  messages={messages}
                  onBack={handleBackToOverview}
                />
              ) : (
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '60vh', color: 'text.secondary' }}>
                  <Factory style={{ width: 64, height: 64, marginBottom: 16 }} />
                  <Typography>Waiting for fleet data...</Typography>
                </Box>
              )}
            </Box>
          </Box>
        </Box>
      )}
    </ThemeProvider>
  );
}

export default App;
