"""
RBAC tests: role-based machine and raw-data access.

- Technician: only assigned machines; 403 for unassigned.
- Reliability engineer: 403 on raw data (e.g. /api/features).
- Plant manager: only their site's machines.
- Admin: sees all.
"""

import pytest
import pytest_asyncio

from services.auth_service import create_access_token
from schemas.security import UserRole


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_token():
    """Token for admin (all machines)."""
    return create_access_token(
        user_id="admin-test",
        username="admin_test",
        role=UserRole.ADMIN,
    )


@pytest.fixture
def technician_token_assigned_wb001():
    """Token for technician assigned only to WB-001."""
    return create_access_token(
        user_id="tech-wb001",
        username="tech_wb001",
        role=UserRole.TECHNICIAN,
        assigned_machine_ids=["WB-001"],
    )


@pytest.fixture
def plant_manager_body_shop_token():
    """Token for plant_manager with site_id = Body Shop (from station_config)."""
    return create_access_token(
        user_id="pm-body",
        username="pm_body",
        role=UserRole.PLANT_MANAGER,
        site_id="Body Shop",
    )


@pytest.fixture
def reliability_engineer_token():
    """Token for reliability_engineer (no raw data)."""
    return create_access_token(
        user_id="re-engineer",
        username="re_engineer",
        role=UserRole.RELIABILITY_ENGINEER,
    )


@pytest.mark.asyncio
async def test_technician_cannot_access_unassigned_machine(client, technician_token_assigned_wb001):
    """Technician assigned only to WB-001 gets 403 for WB-002 (features)."""
    resp = await client.get(
        "/api/features",
        params={"machine_id": "WB-002", "limit": 10},
        headers=_headers(technician_token_assigned_wb001),
    )
    assert resp.status_code == 403
    assert "Not authorized" in resp.text or "not permitted" in resp.text.lower()


@pytest.mark.asyncio
async def test_reliability_engineer_403_on_features(client, reliability_engineer_token):
    """Reliability engineer gets 403 on /api/features (raw data not permitted)."""
    resp = await client.get(
        "/api/features",
        params={"limit": 10},
        headers=_headers(reliability_engineer_token),
    )
    assert resp.status_code == 403
    assert "Raw data" in resp.text or "not permitted" in resp.text.lower()


@pytest.mark.asyncio
async def test_plant_manager_sees_only_site_machines(client, plant_manager_body_shop_token):
    """Plant manager with site Body Shop sees only Body Shop machines (e.g. WB-001, WB-002, WB-003)."""
    resp = await client.get("/api/machines", headers=_headers(plant_manager_body_shop_token))
    assert resp.status_code == 200, resp.text
    data = resp.json() or {}
    machines = (data.get("data") if isinstance(data, dict) else None) or []
    machine_ids = [m.get("machine_id") for m in machines if m and m.get("machine_id")]
    # Body Shop in station_config has WB-001, WB-002, WB-003
    for mid in machine_ids:
        assert mid in ("WB-001", "WB-002", "WB-003"), f"Plant manager should only see Body Shop machines, got {mid}"
    assert "HP-200" not in machine_ids


@pytest.mark.asyncio
async def test_admin_sees_all_machines(client, admin_token):
    """Admin sees all machines (no filter)."""
    resp = await client.get("/api/machines", headers=_headers(admin_token))
    assert resp.status_code == 200, resp.text
    data = resp.json() or {}
    machines = (data.get("data") if isinstance(data, dict) else None) or []
    assert isinstance(machines, list)


@pytest.mark.asyncio
async def test_technician_sees_only_assigned_machines(client, technician_token_assigned_wb001):
    """Technician with only WB-001 assigned gets only WB-001 in /api/machines."""
    resp = await client.get("/api/machines", headers=_headers(technician_token_assigned_wb001))
    assert resp.status_code == 200, resp.text
    data = resp.json() or {}
    machines = (data.get("data") if isinstance(data, dict) else None) or []
    for m in machines:
        assert m and m.get("machine_id") == "WB-001"
    assert len(machines) <= 1
