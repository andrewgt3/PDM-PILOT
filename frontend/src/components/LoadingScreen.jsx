import React, { useEffect, useState } from 'react';
import { Box, Typography, LinearProgress, useTheme } from '@mui/material';
import { alpha } from '@mui/material/styles';

const LoadingScreen = ({ onComplete }) => {
    const theme = useTheme();
    const [progress, setProgress] = useState(0);
    const [statusText, setStatusText] = useState('INITIALIZING KERNEL...');

    useEffect(() => {
        const statuses = [
            'ESTABLISHING SECURE HANDSHAKE...',
            'DECRYPTING CONFIGURATION...',
            'CONNECTING TO FLEET MESH...',
            'SYNCING TELEMETRY STREAMS...',
            'LOADING PREDICTIVE MODELS...',
            'SYSTEM READY'
        ];

        // Progress bar and text cycle
        const interval = setInterval(() => {
            setProgress((prev) => {
                const newProgress = prev + 1.5;
                if (newProgress > 100) {
                    clearInterval(interval);
                    setTimeout(onComplete, 500);
                    return 100;
                }

                if (newProgress > 85) setStatusText(statuses[5]);
                else if (newProgress > 70) setStatusText(statuses[4]);
                else if (newProgress > 50) setStatusText(statuses[3]);
                else if (newProgress > 30) setStatusText(statuses[2]);
                else if (newProgress > 15) setStatusText(statuses[1]);

                return newProgress;
            });
        }, 50);

        return () => clearInterval(interval);
    }, [onComplete]);

    return (
        <Box sx={{
            height: '100vh',
            width: '100vw',
            bgcolor: theme.palette.background.default,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            overflow: 'hidden'
        }}>
            {/* Background Pattern */}
            <Box sx={{
                position: 'absolute',
                inset: 0,
                backgroundImage: `radial-gradient(${theme.palette.grey[300]} 1px, transparent 1px)`,
                backgroundSize: '32px 32px',
                opacity: 0.5
            }} />

            {/* Central Spinner Container */}
            <Box sx={{ position: 'relative', width: 160, height: 160, mb: 6 }}>
                {/* Outer Ring - Slow Rotate */}
                <Box sx={{
                    position: 'absolute',
                    inset: 0,
                    borderRadius: '50%',
                    border: '1px solid',
                    borderColor: theme.palette.grey[300],
                    borderTopColor: theme.palette.primary.main,
                    animation: 'spin 3s linear infinite',
                }} />

                {/* Middle Ring - Reverse Fast */}
                <Box sx={{
                    position: 'absolute',
                    inset: 20,
                    borderRadius: '50%',
                    border: '2px solid',
                    borderColor: alpha(theme.palette.background.paper, 0),
                    borderBottomColor: theme.palette.secondary.main,
                    animation: 'spin-reverse 1.5s linear infinite'
                }} />

                {/* Inner Core - Pulse */}
                <Box sx={{
                    position: 'absolute',
                    inset: 48,
                    borderRadius: '50%',
                    bgcolor: alpha(theme.palette.primary.main, 0.05),
                    border: `1px solid ${theme.palette.primary.main}`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    animation: 'pulse 2s ease-in-out infinite',
                    boxShadow: `0 0 20px ${alpha(theme.palette.primary.main, 0.1)}`
                }}>
                    <Typography variant="body1" fontWeight="bold" color="primary.main" fontFamily="monospace">
                        {Math.round(progress)}%
                    </Typography>
                </Box>
            </Box>

            {/* Status Text */}
            <Typography
                variant="body2"
                sx={{
                    fontFamily: 'monospace',
                    letterSpacing: 1.5,
                    color: theme.palette.text.secondary,
                    mb: 2,
                    textTransform: 'uppercase',
                    minHeight: '1.5rem',
                    fontWeight: 600
                }}
            >
                {statusText}
            </Typography>

            {/* Loading Bar */}
            <Box sx={{ width: 240, height: 4, bgcolor: theme.palette.grey[200], borderRadius: 2, overflow: 'hidden' }}>
                <Box sx={{
                    width: `${progress}%`,
                    height: '100%',
                    bgcolor: theme.palette.primary.main,
                    transition: 'width 0.1s linear'
                }} />
            </Box>

            {/* Inline Styles for Keyframes */}
            <style>
                {`
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                    @keyframes spin-reverse {
                        0% { transform: rotate(360deg); }
                        100% { transform: rotate(0deg); }
                    }
                    @keyframes pulse {
                        0%, 100% { opacity: 0.8; transform: scale(1); }
                        50% { opacity: 1; transform: scale(1.05); }
                    }
                `}
            </style>
        </Box>
    );
};

export default LoadingScreen;
