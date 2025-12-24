import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Custom React hook for WebSocket connection to FastAPI backend.
 * 
 * Automatically reconnects on disconnect and buffers messages.
 * 
 * @param {string} url - WebSocket URL (e.g., ws://localhost:8000/ws/stream)
 * @returns {Object} - { messages, latestMessage, isConnected, error }
 */
export function useWebSocket(url) {
    const [messages, setMessages] = useState([]);
    const [latestMessage, setLatestMessage] = useState(null);
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState(null);
    const wsRef = useRef(null);
    const reconnectTimeoutRef = useRef(null);

    const connect = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) return;

        try {
            const ws = new WebSocket(url);
            wsRef.current = ws;

            ws.onopen = () => {
                console.log('[WS] Connected to', url);
                setIsConnected(true);
                setError(null);
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    setLatestMessage(data);
                    setMessages((prev) => {
                        // Keep last 100 messages for charting
                        const updated = [data, ...prev];
                        return updated.slice(0, 100);
                    });
                } catch (e) {
                    console.error('[WS] Parse error:', e);
                }
            };

            ws.onerror = (event) => {
                console.error('[WS] Error:', event);
                setError('WebSocket connection error');
            };

            ws.onclose = () => {
                console.log('[WS] Disconnected, reconnecting in 3s...');
                setIsConnected(false);
                // Auto-reconnect after 3 seconds
                reconnectTimeoutRef.current = setTimeout(connect, 3000);
            };
        } catch (e) {
            console.error('[WS] Connection failed:', e);
            setError(e.message);
        }
    }, [url]);

    useEffect(() => {
        connect();

        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
            }
        };
    }, [connect]);

    return { messages, latestMessage, isConnected, error };
}

export default useWebSocket;
