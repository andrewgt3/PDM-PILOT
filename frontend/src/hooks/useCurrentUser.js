import { useMemo, useState, useEffect } from 'react';

const TOKEN_KEY = 'access_token';
const FALLBACK_KEY = 'token';

function getToken() {
  return localStorage.getItem(TOKEN_KEY) || localStorage.getItem(FALLBACK_KEY) || null;
}

function base64UrlDecode(str) {
  const base64 = str.replace(/-/g, '+').replace(/_/g, '/');
  const padded = base64.padEnd(base64.length + (4 - base64.length % 4) % 4, '=');
  try {
    return decodeURIComponent(
      atob(padded)
        .split('')
        .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    );
  } catch {
    return null;
  }
}

function decodePayload(token) {
  if (!token || typeof token !== 'string') return null;
  const parts = token.trim().split('.');
  if (parts.length !== 3) return null;
  const raw = base64UrlDecode(parts[1]);
  if (!raw) return null;
  try {
    const payload = JSON.parse(raw);
    return {
      userId: payload.sub ?? null,
      role: payload.role ?? null,
      siteId: payload.site_id ?? null,
      assignedMachineIds: Array.isArray(payload.assigned_machine_ids) ? payload.assigned_machine_ids : [],
      username: payload.username ?? null,
    };
  } catch {
    return null;
  }
}

const EMPTY_USER = {
  userId: null,
  role: null,
  siteId: null,
  assignedMachineIds: [],
  username: null,
};

/**
 * Decodes JWT from localStorage and returns current user claims.
 * Re-decodes only when the token string changes (memoized).
 * @returns {{ userId, role, siteId, assignedMachineIds, username }}
 */
export function useCurrentUser() {
  const [token, setToken] = useState(() => getToken());

  useEffect(() => {
    const handler = () => setToken(getToken());
    window.addEventListener('storage', handler);
    return () => window.removeEventListener('storage', handler);
  }, []);

  return useMemo(() => {
    const decoded = decodePayload(token);
    if (!decoded) return EMPTY_USER;
    return {
      userId: decoded.userId,
      role: decoded.role,
      siteId: decoded.siteId,
      assignedMachineIds: decoded.assignedMachineIds,
      username: decoded.username,
    };
  }, [token]);
}

export default useCurrentUser;
