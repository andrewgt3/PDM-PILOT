import React, { useMemo, useState } from 'react';
import { Box, Card, CardContent, Typography, LinearProgress, Button, IconButton, Chip } from '@mui/material';
import { ArrowForward } from '@mui/icons-material';

/**
 * IndustrialTile - High-performance monitoring tile for industrial dashboards.
 * 
 * Features:
 * - Micro-sparklines for 24h trends
 * - Contextual threshold progress bars
 * - "Hero" mode for primary KPIs
 * - Critical state glow effects
 */
export function IndustrialTile({
    title,
    value,
    unit,
    subValue,
    icon: Icon,
    trendData = [],
    dataKey,
    limit,
    limitLabel = 'Max',
    isHero = false,
    riskLevel = 'normal', // normal, warning, critical
    onAction,
    actionLabel = 'View Diagnostics'
}) {
    const [isHovered, setIsHovered] = useState(false);

    // Color Logic
    const colors = {
        normal: {
            main: '#10b981', // emerald-500
            bg: '#0f172a',    // slate-900 (darker)
            border: '#1e293b', // slate-800
            text: '#94a3b8'   // slate-400
        },
        warning: {
            main: '#f59e0b', // amber-500
            bg: '#1c1917',    // stone-900
            border: '#451a03',
            text: '#fdba74'
        },
        critical: {
            main: '#ef4444', // red-500
            bg: '#1a1010',    // dark red tint
            border: '#7f1d1d',
            text: '#fca5a5'
        }
    };

    const currentTheme = colors[riskLevel] || colors.normal;

    // --- Sparkline Logic ---
    const sparklinePath = useMemo(() => {
        if (!trendData || trendData.length < 2) return '';

        // Extract values using dataKey
        const values = trendData.map(d => {
            const val = d[dataKey];
            return typeof val === 'number' ? val : 0;
        });

        const min = Math.min(...values);
        const max = Math.max(...values);
        const range = max - min || 1; // Avoid divide by zero

        // SVG Dimensions: 120w x 30h
        const width = 120;
        const height = 30;

        // Generate Path Points
        // X is equally spaced
        // Y is normalized: 0 at bottom (height), max at top (0)
        const points = values.map((val, i) => {
            const x = (i / (values.length - 1)) * width;
            const normalizedVal = (val - min) / range;
            const y = height - (normalizedVal * height);
            return `${x},${y}`;
        });

        return `M ${points.join(' L ')}`;
    }, [trendData, dataKey]);


    // --- Threshold / Progress Logic ---
    const progressValue = useMemo(() => {
        if (!limit || typeof value !== 'number') return 0;
        const pct = (value / limit) * 100;
        return Math.min(Math.max(pct, 0), 100);
    }, [value, limit]);

    const progressColor = useMemo(() => {
        // Dynamic color based on % of limit
        // > 90% is Red, > 70% is Amber, else Main Color
        if (progressValue > 90) return '#ef4444';
        if (progressValue > 70) return '#f59e0b';
        return '#3b82f6'; // Default blue, or use currentTheme.main
    }, [progressValue]);


    return (
        <Card
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            variant="outlined"
            sx={{
                height: '100%',
                borderRadius: 3,
                // Hero: White with strong border, Normal: Slate-50
                background: isHero ? '#ffffff' : '#f8fafc',
                borderColor: isHero ? (riskLevel === 'critical' ? '#ef4444' : '#3b82f6') : '#e2e8f0', // Blue border for hero
                borderWidth: isHero ? 2 : 1,
                position: 'relative',
                overflow: 'hidden',
                transition: 'all 0.2s ease-in-out',
                boxShadow: isHovered ? '0 10px 15px -3px rgb(0 0 0 / 0.1)' : '0 1px 3px 0 rgb(0 0 0 / 0.1)',
                color: '#1e293b' // Always dark text
            }}
        >
            <CardContent sx={{ p: 2.5, height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                {/* Header */}
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <Box>
                        <Typography
                            variant="caption"
                            fontWeight="700"
                            sx={{
                                color: '#64748b',
                                textTransform: 'uppercase',
                                letterSpacing: '0.05em',
                                fontSize: '0.7rem'
                            }}
                        >
                            {title}
                        </Typography>
                        {isHero && (
                            <Chip
                                label={riskLevel === 'critical' ? 'ATTENTION REQUIRED' : 'SYSTEM HEALTHY'}
                                size="small"
                                sx={{
                                    mt: 1,
                                    height: 20,
                                    fontSize: '0.6rem',
                                    fontWeight: 'bold',
                                    bgcolor: riskLevel === 'critical' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)',
                                    color: riskLevel === 'critical' ? '#fca5a5' : '#6ee7b7',
                                    border: '1px solid',
                                    borderColor: riskLevel === 'critical' ? '#ef4444' : '#10b981'
                                }}
                            />
                        )}
                    </Box>
                    <Box
                        sx={{
                            p: 1,
                            borderRadius: '12px',
                            bgcolor: isHero ? '#eff6ff' : '#ffffff', // Blue tint for Hero
                            border: '1px solid',
                            borderColor: isHero ? '#bfdbfe' : '#e2e8f0',
                            color: riskLevel === 'critical' ? '#ef4444' : (isHero ? '#2563eb' : '#64748b'),
                            boxShadow: riskLevel === 'critical' ? '0 0 15px rgba(239, 68, 68, 0.5)' : 'none',
                            animation: riskLevel === 'critical' ? 'pulse-glow 2s infinite' : 'none',
                            '@keyframes pulse-glow': {
                                '0%': { boxShadow: '0 0 0 0 rgba(239, 68, 68, 0.4)' },
                                '70%': { boxShadow: '0 0 0 10px rgba(239, 68, 68, 0)' },
                                '100%': { boxShadow: '0 0 0 0 rgba(239, 68, 68, 0)' }
                            }
                        }}
                    >
                        {Icon && <Icon size={20} className={riskLevel === 'critical' ? 'animate-pulse' : ''} />}
                    </Box>
                </Box>

                {/* Main Value */}
                <Box sx={{ mt: 'auto', mb: 2 }}>
                    <Typography
                        variant={isHero ? "h3" : "h4"}
                        fontWeight="700"
                        sx={{
                            fontFamily: '"JetBrains Mono", "Roboto Mono", monospace', // Proposed font
                            letterSpacing: '-0.03em',
                            color: '#0f172a' // Always dark
                        }}
                    >
                        {value}
                        <Typography component="span" fontSize="0.5em" sx={{ ml: 0.5, opacity: 0.6 }}>{unit}</Typography>
                    </Typography>
                </Box>

                {/* Visualizations: Progress or Sparkline */}
                <Box>
                    {/* 1. Limit / Progress Bar (If limit exists) */}
                    {limit && (
                        <Box sx={{ mb: 1 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5, fontSize: '0.65rem', color: isHero ? '#64748b' : '#94a3b8' }}>
                                <span>{Math.round(progressValue)}% of Limit</span>
                                <span>{limit} {unit}</span>
                            </Box>
                            <Box sx={{ position: 'relative', height: 4, bgcolor: isHero ? '#1e293b' : '#e2e8f0', borderRadius: 2, overflow: 'hidden' }}>
                                <Box
                                    sx={{
                                        position: 'absolute',
                                        left: 0,
                                        top: 0,
                                        bottom: 0,
                                        width: `${progressValue}%`,
                                        bgcolor: progressColor,
                                        transition: 'width 0.5s ease-out'
                                    }}
                                />
                            </Box>
                        </Box>
                    )}

                    {/* 2. Sparkline (Bottom monochrome trend) */}
                    {trendData.length > 0 && (
                        <Box sx={{ height: 30, display: 'flex', alignItems: 'flex-end', opacity: 0.5 }}>
                            <svg width="100%" height="100%" viewBox="0 0 120 30" preserveAspectRatio="none">
                                <path
                                    d={sparklinePath}
                                    fill="none"
                                    stroke={isHero ? '#94a3b8' : '#64748b'} // Monochrome
                                    strokeWidth="1.5"
                                    vectorEffect="non-scaling-stroke"
                                />
                                {/* Optional Area Fill */}
                                <path
                                    d={`${sparklinePath} V 30 H 0 Z`}
                                    fill={isHero ? '#94a3b8' : '#64748b'}
                                    fillOpacity="0.1"
                                />
                            </svg>
                        </Box>
                    )}
                </Box>

                {/* Hero Action Button (Ghost) */}
                {isHero && (
                    <Box
                        sx={{
                            position: 'absolute',
                            bottom: 16,
                            right: 16,
                            opacity: isHovered ? 1 : 0,
                            transform: isHovered ? 'translateY(0)' : 'translateY(10px)',
                            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
                        }}
                    >
                        <Button
                            variant="outlined"
                            size="small"
                            onClick={onAction}
                            endIcon={<ArrowForward fontSize="small" />}
                            sx={{
                                color: 'white',
                                borderColor: 'rgba(255,255,255,0.3)',
                                backdropFilter: 'blur(4px)',
                                '&:hover': {
                                    borderColor: 'white',
                                    bgcolor: 'rgba(255,255,255,0.1)'
                                }
                            }}
                        >
                            {actionLabel}
                        </Button>
                    </Box>
                )}
            </CardContent>
        </Card>
    );
}
