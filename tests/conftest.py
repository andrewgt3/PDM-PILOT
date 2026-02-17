"""Gaia Predictive — Pytest Configuration & Fixtures.

Provides a rigorous testing environment with:
1. Database rollback strategy (tests run in transactions that are rolled back).
2. AsyncClient for testing FastAPI endpoints.
3. Factory fixtures for creating authentication tokens.

Usage:
    async def test_my_endpoint(client, db_session, auth_headers):
        response = await client.get("/api/machines", headers=auth_headers)
        assert response.status_code == 200
"""

import asyncio
from typing import AsyncGenerator, Callable
import uuid

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Import the FastAPI app
try:
    # Try the new structure first if it exists
    from app.main import app
except ImportError:
    # Fallback to the current flat structure
    from api_server import app

from config import get_settings
from database import get_db
from services.auth_service import create_access_token
from schemas.security import UserRole

# Config
settings = get_settings()


# =============================================================================
# Event Loop & Database Engine
# =============================================================================

# event_loop fixture removed to allow pytest-asyncio to handle it automatically
# with session scope defined in pytest.ini


@pytest_asyncio.fixture(scope="session")
async def db_engine():
    """Create a single database engine for the entire test session.
    
    This engine is shared across all tests for performance.
    
    IMPORTANT: Override DB_NAME env var to use test database if not set.
    This prevents tests from accidentally wiping production data.
    """
    import os
    
    # Override to test database if not explicitly set
    if "DB_NAME" not in os.environ:
        os.environ["DB_NAME"] = "gaia_predictive_test"
    
    # Clear settings cache to pick up the override
    from config import get_settings
    get_settings.cache_clear()
    settings = get_settings()
    
    # Create engine independent of the application's engine
    engine = create_async_engine(
        settings.database.async_dsn,
        pool_pre_ping=True,
        poolclass=None,
        connect_args={"server_settings": {"jit": "off"}, "ssl": False}, # Disable SSL for local tests
    )
    
    yield engine
    
    await engine.dispose()


# =============================================================================
# Database Session (Transaction Rollback)
# =============================================================================

@pytest_asyncio.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Provide a database session wrapped in a transaction that rolls back.
    
    CRITICAL: This ensures tests are isolated and don't pollute the database.
    Each test gets a fresh transaction on the connection.
    """
    # 1. Connect to the database
    connection = await db_engine.connect()
    
    # 2. Begin a transaction
    transaction = await connection.begin()
    
    # 3. Create a session bound to this connection
    session_factory = async_sessionmaker(
        bind=connection,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    session = session_factory()
    
    # 4. Override the application's get_db dependency
    # This forces the app to use OUR session (which will be rolled back)
    app.dependency_overrides[get_db] = lambda: session
    
    yield session
    
    # 5. Cleanup
    # Restore the dependency
    app.dependency_overrides.pop(get_db, None)
    
    # Close session and rollback transaction
    await session.close()
    if transaction.is_active:
        await transaction.rollback()
    await connection.close()


# =============================================================================
# API Client
# =============================================================================

@pytest_asyncio.fixture
async def client(db_session) -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client for testing API endpoints.
    
    The client is bound to the test app and uses the overridden db_session
    via dependency injection.
    """
    # ASGITransport allows calling the app directly without a network socket
    transport = ASGITransport(app=app)
    
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# =============================================================================
# Authentication Fixtures
# =============================================================================

@pytest.fixture
def user_token_factory() -> Callable[[str, UserRole], str]:
    """Factory fixture to create valid JWT tokens for testing."""
    
    def _create_token(
        user_id: str | None = None,
        role: UserRole = UserRole.VIEWER
    ) -> str:
        if user_id is None:
            user_id = str(uuid.uuid4())
            
        return create_access_token(
            user_id=user_id,
            username="testuser",
            role=role,
        )
        
    return _create_token


@pytest.fixture
def auth_headers(user_token_factory) -> dict[str, str]:
    """Headers for an authenticated ADMIN user.
    
    Use this for tests that require full privileges.
    """
    token = user_token_factory(role=UserRole.ADMIN)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def operator_headers(user_token_factory) -> dict[str, str]:
    """Headers for an authenticated OPERATOR user."""
    token = user_token_factory(role=UserRole.OPERATOR)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def viewer_headers(user_token_factory) -> dict[str, str]:
    """Headers for an authenticated VIEWER user."""
    token = user_token_factory(role=UserRole.VIEWER)
    return {"Authorization": f"Bearer {token}"}
