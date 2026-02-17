"""Tests for error handling and clean error responses.

Verifies that:
1. Invalid JSON returns 422 with INVALID_INPUT error
2. Non-existent resources return 404 with ResourceNotFound
3. Auth errors return appropriate status codes
"""

import pytest
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    """Create async test client."""
    from api_server import app
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def auth_headers():
    """Get valid auth token for protected endpoints."""
    from auth_utils import create_access_token
    token = create_access_token(data={"sub": "test_user", "role": "admin"})
    return {"Authorization": f"Bearer {token}"}


class TestBadJSONHandling:
    """Test that malformed JSON returns clean errors."""
    
    @pytest.mark.anyio
    async def test_invalid_json_body(self, client, auth_headers):
        """Send completely invalid JSON and verify clean error response."""
        # Use the enterprise token endpoint which accepts POST with body
        response = await client.post(
            "/api/enterprise/token",
            content="{ not valid json }",
            headers={"Content-Type": "application/json"}
        )
        
        # Should return 422 for malformed JSON
        assert response.status_code == 422
        data = response.json()
        
        # Should have clean error structure, not Python traceback
        assert "error" in data or "detail" in data
        assert "traceback" not in str(data).lower()


class TestResourceNotFound:
    """Test 404 responses for non-existent resources."""
    
    @pytest.mark.anyio
    async def test_nonexistent_machine(self, client, auth_headers):
        """Request a machine ID that doesn't exist."""
        response = await client.get(
            "/api/machines/NONEXISTENT_MACHINE_12345",
            headers=auth_headers
        )
        
        # Should return 404, not 500
        assert response.status_code == 404
        data = response.json()
        
        # Should have clean error message
        assert "error" in data or "detail" in data
        assert "RESOURCE_NOT_FOUND" in str(data) or "not found" in str(data).lower()


class TestAuthErrors:
    """Test authentication error responses."""
    
    @pytest.mark.anyio
    async def test_invalid_token_format(self, client):
        """Malformed token should return 401."""
        # Use a properly formatted but invalid JWT
        fake_jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        
        response = await client.get(
            "/api/machines",
            headers={"Authorization": f"Bearer {fake_jwt}"}
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data


class TestHealthEndpoint:
    """Test health check doesn't require auth."""
    
    @pytest.mark.anyio
    async def test_health_no_auth_required(self, client):
        """Health endpoint should not require authentication."""
        response = await client.get("/health")
        
        # Should return 200 or 503, not 401/403
        assert response.status_code in (200, 503)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
