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


# =============================================================================
# Integration test fixtures (require full Docker stack; use pytest -m integration)
# =============================================================================

# Redis channel for ingestion (only path to TimescaleDB per architecture)
REDIS_CHANNEL = "sensor_stream"
# Vibration length per spec: 128 floats (match N_SAMPLES)
INTEGRATION_N_SAMPLES = 128


def _check_docker_stack() -> bool:
    """Return True if TimescaleDB (5432), Redis (6379), and Mosquitto (1883) are reachable."""
    import socket
    from config import get_settings
    get_settings.cache_clear()
    s = get_settings()
    checks = [
        (s.database.host, s.database.port),
        (s.redis.host, s.redis.port),
    ]
    mqtt_host = getattr(s.mqtt, "broker_host", "localhost")
    mqtt_port = getattr(s.mqtt, "broker_port", 1883)
    checks.append((mqtt_host, mqtt_port))
    for host, port in checks:
        try:
            with socket.create_connection((host, port), timeout=2):
                pass
        except (OSError, TypeError):
            return False
    return True


@pytest.fixture(scope="session")
def docker_stack():
    """Ensure TimescaleDB, Redis, and Mosquitto are running; skip integration tests if not. Do NOT auto-start Docker."""
    if not _check_docker_stack():
        pytest.skip("Docker stack not running.")
    return True


@pytest.fixture
def test_machine_id() -> str:
    """Default test machine id for integration tests."""
    return "TEST-001"


@pytest.fixture
def test_new_machine_id() -> str:
    """Machine id for full onboarding flow (test_07)."""
    return "TEST-NEW-001"


@pytest.fixture(autouse=False)
def clean_test_data(test_machine_id, test_new_machine_id):
    """Teardown: remove integration test data for test machines. Uses synchronous psycopg2."""
    yield
    from config import get_settings
    import psycopg2
    get_settings.cache_clear()
    s = get_settings().database
    try:
        conn = psycopg2.connect(
            host=s.host,
            port=s.port,
            dbname=s.name,
            user=s.user,
            password=s.password.get_secret_value(),
            connect_timeout=5,
        )
        cur = conn.cursor()
        for table in [
            "cwru_features",
            "sensor_readings",
            "data_profiles",
            "alert_current_state",
            "work_orders",
            "onboarding_status",
        ]:
            try:
                cur.execute(
                    "DELETE FROM " + table + " WHERE machine_id IN (%s, %s)",
                    (test_machine_id, test_new_machine_id),
                )
            except Exception:
                pass
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass


def publish_n_readings(
    machine_id: str,
    n: int,
    anomaly: bool = False,
    redis_host: str | None = None,
    redis_port: int | None = None,
    uns_topic_override: str | None = None,
) -> int:
    """
    Publish n synthetic sensor readings to Redis sensor_stream only (no MQTT).
    Payload: machine_id, timestamp, rotational_speed, temperature, torque, tool_wear,
    vibration_raw (128 floats), uns_topic. Returns count of messages published.
    """
    import json
    import time
    import redis
    from config import get_settings
    get_settings.cache_clear()
    s = get_settings()
    r = redis.Redis(
        host=redis_host or s.redis.host,
        port=redis_port or s.redis.port,
        password=s.redis.password.get_secret_value() if s.redis.password else None,
        decode_responses=True,
        socket_timeout=5,
    )
    base_ts = time.time()
    if anomaly:
        torque, rotational_speed, tool_wear, temperature = 95.0, 1800.0, 250.0, 85.0
    else:
        torque, rotational_speed, tool_wear, temperature = 42.0, 1400.0, 50.0, 45.0
    uns_topic = uns_topic_override or f"PlantAGI/TestFactory/TestCell/{machine_id}/ABB_Robot/telemetry/main"
    from datetime import datetime, timezone
    published = 0
    for i in range(n):
        ts = base_ts - (n - i) * 0.1
        # Use subsecond precision so each message has a unique (machine_id, timestamp) for cwru_features.
        ts_iso = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        payload = {
            "machine_id": machine_id,
            "timestamp": ts_iso,
            "rotational_speed": rotational_speed,
            "temperature": temperature,
            "torque": torque,
            "tool_wear": tool_wear,
            "vibration_raw": [0.01 * (1 + (i + j) % 10) for j in range(INTEGRATION_N_SAMPLES)],
            "uns_topic": uns_topic,
        }
        r.publish(REDIS_CHANNEL, json.dumps(payload))
        published += 1
    return published
