import { createTheme } from '@mui/material/styles';

const theme = createTheme({
    palette: {
        mode: 'light',
        primary: {
            main: '#4f46e5', // Indigo-600
            light: '#818cf8',
            dark: '#4338ca',
            contrastText: '#ffffff',
        },
        secondary: {
            main: '#7c3aed', // Violet-600
            light: '#a78bfa',
            dark: '#5b21b6',
            contrastText: '#ffffff',
        },
        error: {
            main: '#dc2626', // Red-600
        },
        warning: {
            main: '#d97706', // Amber-600
        },
        info: {
            main: '#2563eb', // Blue-600
        },
        success: {
            main: '#059669', // Emerald-600
        },
        background: {
            default: '#f8fafc', // Slate-50
            paper: '#ffffff',
        },
        text: {
            primary: '#0f172a', // Slate-900
            secondary: '#475569', // Slate-600
        },
    },
    typography: {
        fontFamily: [
            '-apple-system',
            'BlinkMacSystemFont',
            '"Segoe UI"',
            'Roboto',
            '"Helvetica Neue"',
            'Arial',
            'sans-serif',
        ].join(','),
        h1: { fontSize: '2.5rem', fontWeight: 700 },
        h2: { fontSize: '2rem', fontWeight: 600 },
        h3: { fontSize: '1.75rem', fontWeight: 600 },
        h4: { fontSize: '1.5rem', fontWeight: 600 },
        h5: { fontSize: '1.25rem', fontWeight: 600 },
        h6: { fontSize: '1rem', fontWeight: 600 },
    },
    components: {
        MuiCard: {
            styleOverrides: {
                root: {
                    borderRadius: 12,
                    boxShadow: '0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)',
                },
            },
        },
        MuiButton: {
            styleOverrides: {
                root: {
                    borderRadius: 8,
                    textTransform: 'none',
                    fontWeight: 600,
                },
            },
        },
    },
});

export default theme;
