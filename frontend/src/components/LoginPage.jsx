import React, { useState } from 'react';
import { Box, Typography, TextField, Button, InputAdornment, Paper, Stack, useTheme } from '@mui/material';
import { alpha } from '@mui/material/styles';
import { Lock, User, ChevronRight, Hexagon, ShieldCheck } from 'lucide-react';

const LoginPage = ({ onLogin }) => {
    const theme = useTheme();
    const [loading, setLoading] = useState(false);
    const [credentials, setCredentials] = useState({ username: '', password: '' });

    const handleChange = (e) => {
        setCredentials({ ...credentials, [e.target.name]: e.target.value });
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        setLoading(true);
        // Simulate network delay for effect
        setTimeout(() => {
            onLogin();
        }, 800);
    };

    return (
        <Box sx={{
            height: '100vh',
            width: '100vw',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: theme.palette.background.default,
            position: 'relative',
            overflow: 'hidden'
        }}>
            {/* Enterprise Background Pattern */}
            <Box sx={{
                position: 'absolute',
                inset: 0,
                opacity: 0.4,
                backgroundImage: `radial-gradient(${theme.palette.grey[300]} 1px, transparent 1px)`,
                backgroundSize: '24px 24px',
                pointerEvents: 'none',
            }} />

            {/* Abstract Gradient Blobs (Subtle) */}
            <Box sx={{
                position: 'absolute',
                top: '-10%',
                left: '-5%',
                width: '600px',
                height: '600px',
                background: `radial-gradient(circle, ${alpha(theme.palette.primary.main, 0.08)} 0%, transparent 70%)`,
                borderRadius: '50%',
                filter: 'blur(60px)',
                zIndex: 0
            }} />
            <Box sx={{
                position: 'absolute',
                bottom: '-10%',
                right: '-5%',
                width: '500px',
                height: '500px',
                background: `radial-gradient(circle, ${alpha(theme.palette.secondary.main, 0.05)} 0%, transparent 70%)`,
                borderRadius: '50%',
                filter: 'blur(60px)',
                zIndex: 0
            }} />

            <Paper
                elevation={0}
                sx={{
                    position: 'relative',
                    zIndex: 1,
                    width: '100%',
                    maxWidth: 400,
                    p: 5,
                    borderRadius: 3,
                    bgcolor: theme.palette.background.paper,
                    border: `1px solid ${theme.palette.divider}`,
                    boxShadow: theme.shadows[4]
                }}
            >
                <Stack alignItems="center" spacing={1} mb={5}>
                    <Box sx={{
                        p: 1.5,
                        borderRadius: 2,
                        bgcolor: alpha(theme.palette.primary.main, 0.1),
                        color: theme.palette.primary.main,
                        mb: 1
                    }}>
                        <Hexagon size={32} strokeWidth={2} />
                    </Box>
                    <Typography variant="h5" fontWeight="800" letterSpacing={0.5} color="text.primary">
                        PLANT OS
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ letterSpacing: 1.5, textTransform: 'uppercase', fontWeight: 600 }}>
                        Enterprise Fleet Intelligence
                    </Typography>
                </Stack>

                <form onSubmit={handleSubmit}>
                    <Stack spacing={3}>
                        <TextField
                            fullWidth
                            name="username"
                            placeholder="OPERATOR ID"
                            variant="outlined"
                            value={credentials.username}
                            onChange={handleChange}
                            InputProps={{
                                startAdornment: (
                                    <InputAdornment position="start">
                                        <User size={18} color={theme.palette.text.secondary} />
                                    </InputAdornment>
                                ),
                                sx: {
                                    fontFamily: 'monospace',
                                    fontSize: '0.9rem',
                                    bgcolor: theme.palette.grey[50]
                                }
                            }}
                        />

                        <TextField
                            fullWidth
                            name="password"
                            type="password"
                            placeholder="ACCESS KEY"
                            variant="outlined"
                            value={credentials.password}
                            onChange={handleChange}
                            InputProps={{
                                startAdornment: (
                                    <InputAdornment position="start">
                                        <Lock size={18} color={theme.palette.text.secondary} />
                                    </InputAdornment>
                                ),
                                sx: {
                                    fontFamily: 'monospace',
                                    fontSize: '0.9rem',
                                    bgcolor: theme.palette.grey[50]
                                }
                            }}
                        />

                        <Button
                            type="submit"
                            fullWidth
                            variant="contained"
                            size="large"
                            disabled={loading}
                            endIcon={!loading && <ChevronRight size={18} />}
                            sx={{
                                py: 1.5,
                                bgcolor: theme.palette.primary.main,
                                color: '#fff',
                                fontWeight: 'bold',
                                letterSpacing: 1,
                                boxShadow: theme.shadows[2],
                                '&:hover': {
                                    bgcolor: theme.palette.primary.dark,
                                    transform: 'translateY(-1px)',
                                    boxShadow: theme.shadows[4]
                                }
                            }}
                        >
                            {loading ? 'AUTHENTICATING...' : 'INITIATE SESSION'}
                        </Button>
                    </Stack>
                </form>

                <Box sx={{ mt: 4, pt: 3, borderTop: `1px solid ${theme.palette.divider}`, textAlign: 'center' }}>
                    <Stack direction="row" alignItems="center" justifyContent="center" spacing={1} color="text.secondary">
                        <ShieldCheck size={14} />
                        <Typography variant="caption" sx={{ fontFamily: 'monospace', fontSize: '0.7rem' }}>
                            SECURE CONNECTION ESTABLISHED
                        </Typography>
                    </Stack>
                </Box>
            </Paper>
        </Box>
    );
};

export default LoginPage;
