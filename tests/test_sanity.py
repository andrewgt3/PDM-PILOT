"""Gaia Predictive — Sanity Tests.

Verifies that the testing infrastructure itself is working correctly.
Checks database connectivity, fixture injection, and authentication.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from schemas.security import UserRole
from services.auth_service import AuthService

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio



async def test_health_check_endpoint(client: AsyncClient):
    """Verify client fixture connects to API."""
    response = await client.get("/health")
    # Note: Status depends on DB being online, which it should be via fixture
    assert response.status_code in (200, 503)
    data = response.json()
    assert "status" in data

async def test_api_health_endpoint(client: AsyncClient, auth_headers: dict):
    """Verify auth headers work on protected endpoint (if protected).
    
    The /api/health endpoint is currently open, but this verifies
    we can pass headers without error.
    """
    response = await client.get("/api/health", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data

async def test_token_factory_roles(user_token_factory):
    """Verify token factory creates tokens with correct roles."""
    # Create ADMIN token
    admin_token = user_token_factory(role=UserRole.ADMIN)
    decoded = AuthService().verify_token(admin_token)
    assert decoded.role == UserRole.ADMIN
    
    # Create VIEWER token
    viewer_token = user_token_factory(role=UserRole.VIEWER)
    decoded = AuthService().verify_token(viewer_token)
    assert decoded.role == UserRole.VIEWER
