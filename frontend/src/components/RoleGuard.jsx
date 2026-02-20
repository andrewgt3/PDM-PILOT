import React from 'react';
import { useCurrentUser } from '../hooks/useCurrentUser';

/**
 * Renders children only when the current user's role meets the requirement.
 * Otherwise renders null.
 *
 * @param {Object} props
 * @param {React.ReactNode} props.children
 * @param {string[]} [props.allowedRoles] - Roles that can see the content (e.g. ['admin', 'engineer'])
 * @param {function(string): boolean} [props.require] - Permission function (e.g. canViewFleet)
 */
function RoleGuard({ children, allowedRoles, require }) {
  const { role } = useCurrentUser();

  if (role == null) return null;

  const roleLower = String(role).toLowerCase();

  if (allowedRoles != null) {
    const allowed = allowedRoles.map((r) => String(r).toLowerCase());
    if (!allowed.includes(roleLower)) return null;
  }

  if (typeof require === 'function' && !require(role)) return null;

  return <>{children}</>;
}

export default RoleGuard;
