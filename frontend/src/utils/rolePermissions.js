/**
 * Role-based permission checks. All functions accept role as string;
 * null/undefined is treated as no permission.
 */

export function canViewFleet(role) {
  if (role == null) return false;
  const r = String(role).toLowerCase();
  return ['admin', 'engineer', 'plant_manager', 'operator'].includes(r);
}

export function canViewRawData(role) {
  if (role == null) return false;
  return String(role).toLowerCase() !== 'reliability_engineer';
}

export function canCreateWorkOrder(role) {
  if (role == null) return false;
  const r = String(role).toLowerCase();
  return !['reliability_engineer', 'technician'].includes(r);
}

export function canViewAllSites(role) {
  if (role == null) return false;
  const r = String(role).toLowerCase();
  return ['admin', 'reliability_engineer'].includes(r);
}

export function canManageUsers(role) {
  if (role == null) return false;
  return String(role).toLowerCase() === 'admin';
}
