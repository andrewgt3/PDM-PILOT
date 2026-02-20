"""
RBAC service: visible machines and raw-data access by role.

Uses token-derived user fields (role, site_id, assigned_machine_ids).
Plant_manager scope: machines for user.site_id derived from station_config (shop).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from schemas.security import UserRole

if TYPE_CHECKING:
    pass

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STATION_CONFIG_PATH = _PROJECT_ROOT / "pipeline" / "station_config.json"


class UserInToken(Protocol):
    """Protocol for user-like objects with token-derived RBAC fields."""

    @property
    def role(self) -> UserRole: ...
    @property
    def site_id(self) -> str | None: ...
    @property
    def assigned_machine_ids(self) -> list[str]: ...


def _machine_ids_for_site_from_config(site_id: str) -> list[str]:
    """Return machine IDs that belong to the given site_id using station_config (shop = site_id)."""
    if not site_id:
        return []
    try:
        with open(_STATION_CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        node_mappings = data.get("node_mappings") or {}
        return [
            mid
            for mid, info in node_mappings.items()
            if isinstance(info, dict) and (info.get("shop") or "") == site_id
        ]
    except Exception:
        return []


def get_visible_machine_ids(user: UserInToken) -> list[str] | None:
    """
    Return the list of machine IDs the user is allowed to see, or None for no restriction.

    - admin / engineer / operator: None (all machines visible).
    - plant_manager: list of machine IDs in user.site_id (from config shop); [] if no site_id.
    - reliability_engineer: [] (no machine detail; aggregates only).
    - technician: user.assigned_machine_ids.
    """
    if user.role in (UserRole.ADMIN, UserRole.ENGINEER, UserRole.OPERATOR):
        return None
    if user.role == UserRole.PLANT_MANAGER:
        if not user.site_id:
            return []
        return _machine_ids_for_site_from_config(user.site_id)
    if user.role == UserRole.RELIABILITY_ENGINEER:
        return []
    if user.role == UserRole.TECHNICIAN:
        return list(user.assigned_machine_ids) if user.assigned_machine_ids else []
    # viewer and any unknown role: no restriction by default (or return [] if viewer should be restricted)
    return None


def can_access_machine(user: UserInToken, machine_id: str) -> bool:
    """
    Return True if the user is allowed to access the given machine.

    admin / engineer / operator: always True.
    Otherwise: True iff machine_id is in get_visible_machine_ids(user).
    """
    if user.role in (UserRole.ADMIN, UserRole.ENGINEER, UserRole.OPERATOR):
        return True
    visible = get_visible_machine_ids(user)
    if visible is None:
        return True
    return machine_id in visible


def can_access_raw_data(user: UserInToken) -> bool:
    """
    Return True if the user is allowed to access raw sensor readings.

    reliability_engineer: False (aggregates only).
    All other roles: True.
    """
    return user.role != UserRole.RELIABILITY_ENGINEER
