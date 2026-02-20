import React, { useEffect, useState } from 'react';
import { ThemeProvider, CssBaseline, Box, CircularProgress, Typography } from '@mui/material';
import theme from './theme';
import useWebSocket from './hooks/useWebSocket';
import useCurrentUser from './hooks/useCurrentUser';
import Sidebar from './components/Sidebar';
import PlantOverview from './components/PlantOverview';
import MachineDetail from './components/MachineDetail';
import TechnicianHome from './components/TechnicianHome';
import ReliabilityDashboard from './components/ReliabilityDashboard';
import AnomalyDiscoveryPage from './pages/AnomalyDiscoveryPage';
import SetupGuidePage from './pages/SetupGuidePage';
import PipelineOps from './components/PipelineOps';
import LabelingFeed from './components/LabelingFeed';
import LoginPage from './components/LoginPage';
import LoadingScreen from './components/LoadingScreen';
import OnboardingWizard from './components/OnboardingWizard';
import { Factory } from 'lucide-react';
import './index.css';

// Use relative URLs in dev so Vite proxy forwards to backend (avoids CORS; one place to debug "cannot connect")
const API_BASE = import.meta.env.DEV ? '' : (import.meta.env.VITE_API_BASE || 'http://localhost:8000');
function App() {
  const user = useCurrentUser();
  // Token in state so WebSocket URL updates after login (backend requires ?token= for stream data)
  const [wsToken, setWsToken] = useState(() => typeof localStorage !== 'undefined' && (localStorage.getItem('access_token') || localStorage.getItem('token')));
  const wsBase = import.meta.env.DEV
    ? `${typeof window !== 'undefined' && window.location ? (window.location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + window.location.host : 'ws://localhost:5173'}/ws/stream`
    : (import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/stream');
  const WS_URL = wsToken ? `${wsBase}?token=${encodeURIComponent(wsToken)}` : wsBase;

  const getAuthHeaders = () => {
    const token = localStorage.getItem('access_token') || localStorage.getItem('token');
    if (token) return { Authorization: `Bearer ${token}` };
    return {};
  };

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
    setWsToken(localStorage.getItem('access_token') || localStorage.getItem('token'));
  };

  const handleLoadingComplete = () => {
    setIsLoading(false);
  };

  const handleGoLive = () => {
    setShowPipelineOps(false);
    setView('overview');
  };

  // Fetch initial machine list from REST API (Bearer token for RBAC scoping)
  useEffect(() => {
    fetch(`${API_BASE}/api/machines`, { headers: getAuthHeaders() })
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
            user={user}
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
            <Box
              data-scroll-container="main-content"
              sx={{
                p: 0,
                flexGrow: 1,
                overflow: 'auto', // Allow scrolling
                transition: 'padding 0.3s ease'
              }}
            >
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

              {/* View Router (role-aware) */}
              {view === 'pipeline-ops' ? (
                <PipelineOps onGoLive={handleGoLive} />
              ) : view === 'setup-guide' ? (
                <SetupGuidePage />
              ) : view === 'onboard' ? (
                <OnboardingWizard
                  onComplete={() => {
                    setView('overview');
                    fetch(`${API_BASE}/api/machines`, { headers: getAuthHeaders() }).then((res) => res.json()).then((data) => data.data && setMachines(data.data)).catch(() => {});
                  }}
                  onCancel={() => setView('overview')}
                />
              ) : view === 'overview' ? (
                (user?.role && String(user.role).toLowerCase() === 'technician') ? (
                  <TechnicianHome
                    machines={machines}
                    assignedMachineIds={user.assignedMachineIds}
                    onSelectMachine={handleSelectMachine}
                  />
                ) : (user?.role && String(user.role).toLowerCase() === 'reliability_engineer') ? (
                  <ReliabilityDashboard machines={machines} user={user} />
                ) : (
                  <PlantOverview
                    user={user}
                    machines={machines}
                    messages={messages}
                    onSelectMachine={handleSelectMachine}
                    onStartOnboarding={() => setView('onboard')}
                  />
                )
              ) : view === 'anomaly-discovery' ? (
                <AnomalyDiscoveryPage />
              ) : view === 'labeling-feed' ? (
                <LabelingFeed />
              ) : selectedMachine ? (
                (user?.role && String(user.role).toLowerCase() === 'reliability_engineer') ? (
                  <ReliabilityDashboard machines={machines} user={user} />
                ) : (
                  <MachineDetail
                    machine={selectedMachine}
                    messages={messages}
                    onBack={handleBackToOverview}
                  />
                )
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
