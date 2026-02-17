"""Gaia Predictive — Database Connection Manager.

Fault-tolerant async database connection management for TimescaleDB
using SQLAlchemy 2.0 (Async) and asyncpg with connection pooling.

TISAX/SOC2 Compliance:
    - Connection strings use SecretStr (never logged)
    - Pre-ping validates connections before use
    - Proper shutdown disposes all connections
    - All operations logged with correlation IDs

Usage:
    from database import get_db, init_database, shutdown_database
    
    # In FastAPI lifespan
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await init_database()
        yield
        await shutdown_database()
    
    # In routes
    @app.get("/machines")
    async def get_machines(db: AsyncSession = Depends(get_db)):
        result = await db.execute(select(Machine))
        return result.scalars().all()
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import AsyncAdaptedQueuePool

from config import get_settings
from logger import get_logger

# =============================================================================
# Module State
# =============================================================================
# Engine and session maker are created separately for proper lifecycle management

_engine: AsyncEngine | None = None
_session_maker: async_sessionmaker[AsyncSession] | None = None

logger = get_logger(__name__)


# =============================================================================
# Engine Factory
# =============================================================================

def _create_engine() -> AsyncEngine:
    """Create the async database engine with connection pooling.
    
    Returns:
        Configured AsyncEngine with pre-ping and optimized pool settings.
    
    Raises:
        RuntimeError: If database URL is not configured.
    """
    settings = get_settings()
    db_settings = settings.database
    
    logger.info(
        "Creating database engine",
        host=db_settings.host,
        port=db_settings.port,
        database=db_settings.name,
        pool_size=db_settings.pool_size,
        max_overflow=db_settings.max_overflow,
    )
    
    engine = create_async_engine(
        db_settings.async_dsn,
        # Connection pool configuration
        poolclass=AsyncAdaptedQueuePool,
        pool_size=db_settings.pool_size,
        max_overflow=db_settings.max_overflow,
        pool_timeout=30,  # Seconds to wait for a connection
        pool_recycle=1800,  # Recycle connections after 30 minutes
        pool_pre_ping=True,  # CRITICAL: Validate connections before use
        
        # Connection arguments for asyncpg
        connect_args={
            "server_settings": {
                "application_name": "gaia-predictive",
                "jit": "off",  # Disable JIT for more predictable latency
            },
            "command_timeout": 60,  # Query timeout in seconds
        },
        
        # Echo SQL for debugging (disabled in production)
        echo=settings.debug,
        echo_pool=settings.debug,
        
        # Hide password from URL in logs
        hide_parameters=True,
    )
    
    # Register event listeners for connection lifecycle
    _register_engine_events(engine)
    
    return engine


def _create_session_maker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create the async session maker bound to the engine.
    
    Args:
        engine: The async database engine.
    
    Returns:
        Configured session maker for creating database sessions.
    """
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,  # Avoid lazy loading issues with async
        autocommit=False,
        autoflush=False,
    )


def _register_engine_events(engine: AsyncEngine) -> None:
    """Register event listeners for connection monitoring and slow query logging."""
    import time
    
    # Store query start times in connection info
    QUERY_START_KEY = "_query_start_time"
    SLOW_QUERY_THRESHOLD_MS = 500
    
    @event.listens_for(engine.sync_engine, "connect")
    def on_connect(dbapi_connection: Any, connection_record: Any) -> None:
        """Log new database connections."""
        logger.debug("Database connection established")
    
    @event.listens_for(engine.sync_engine, "checkout")
    def on_checkout(
        dbapi_connection: Any,
        connection_record: Any,
        connection_proxy: Any,
    ) -> None:
        """Log connection checkout from pool."""
        logger.debug("Connection checked out from pool")
    
    @event.listens_for(engine.sync_engine, "checkin")
    def on_checkin(dbapi_connection: Any, connection_record: Any) -> None:
        """Log connection return to pool."""
        logger.debug("Connection returned to pool")
    
    @event.listens_for(engine.sync_engine, "invalidate")
    def on_invalidate(
        dbapi_connection: Any,
        connection_record: Any,
        exception: BaseException | None,
    ) -> None:
        """Log connection invalidation."""
        logger.warning(
            "Connection invalidated",
            error=str(exception) if exception else None,
        )
    
    # =============================================================================
    # Slow Query Logging
    # =============================================================================
    
    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def before_cursor_execute(
        conn: Any,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        """Record query start time for slow query detection."""
        conn.info[QUERY_START_KEY] = time.perf_counter()
    
    @event.listens_for(engine.sync_engine, "after_cursor_execute")
    def after_cursor_execute(
        conn: Any,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        """Log slow queries (>500ms)."""
        start_time = conn.info.pop(QUERY_START_KEY, None)
        if start_time is None:
            return
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        if elapsed_ms > SLOW_QUERY_THRESHOLD_MS:
            # Truncate long queries for readability
            truncated_statement = statement[:500] + "..." if len(statement) > 500 else statement
            
            logger.warning(
                "slow_query",
                query=truncated_statement,
                latency_ms=round(elapsed_ms, 2),
                threshold_ms=SLOW_QUERY_THRESHOLD_MS,
            )


# =============================================================================
# Lifecycle Management
# =============================================================================

async def init_database() -> None:
    """Initialize the database engine and session maker.
    
    Call this during application startup (e.g., in FastAPI lifespan).
    Validates connectivity by executing a test query.
    
    Raises:
        RuntimeError: If database connection fails.
    
    Example:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await init_database()
            yield
            await shutdown_database()
    """
    global _engine, _session_maker
    
    if _engine is not None:
        logger.warning("Database already initialized, skipping")
        return
    
    try:
        _engine = _create_engine()
        _session_maker = _create_session_maker(_engine)
        
        # Validate connection
        async with _engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            result.fetchone()
        
        logger.info("Database initialized successfully")
        
    except Exception as exc:
        logger.error(
            "Database initialization failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        # Clean up partial initialization
        if _engine is not None:
            await _engine.dispose()
            _engine = None
        _session_maker = None
        raise RuntimeError(f"Failed to initialize database: {exc}") from exc


async def shutdown_database() -> None:
    """Shutdown the database engine and dispose all connections.
    
    Call this during application shutdown to properly release resources.
    
    Example:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await init_database()
            yield
            await shutdown_database()
    """
    global _engine, _session_maker
    
    if _engine is None:
        logger.warning("Database not initialized, nothing to shutdown")
        return
    
    logger.info("Shutting down database connections")
    
    try:
        await _engine.dispose()
        logger.info("Database connections disposed successfully")
    except Exception as exc:
        logger.error(
            "Error disposing database connections",
            error_type=type(exc).__name__,
            error=str(exc),
        )
    finally:
        _engine = None
        _session_maker = None


def get_engine() -> AsyncEngine:
    """Get the current database engine.
    
    Returns:
        The initialized async engine.
    
    Raises:
        RuntimeError: If database is not initialized.
    """
    if _engine is None:
        raise RuntimeError(
            "Database not initialized. Call init_database() first."
        )
    return _engine


# =============================================================================
# FastAPI Dependency
# =============================================================================

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database session injection.
    
    Provides a database session with automatic commit/rollback handling.
    The session is committed if no exception occurs, rolled back otherwise.
    
    Yields:
        AsyncSession: Database session for the request.
    
    Raises:
        RuntimeError: If database is not initialized.
        DBAPIError: If database operation fails.
    
    Example:
        @app.get("/machines/{machine_id}")
        async def get_machine(
            machine_id: uuid.UUID,
            db: AsyncSession = Depends(get_db),
        ) -> Machine:
            result = await db.execute(
                select(Machine).where(Machine.id == machine_id)
            )
            return result.scalar_one_or_none()
    """
    if _session_maker is None:
        raise RuntimeError(
            "Database not initialized. Call init_database() first."
        )
    
    session = _session_maker()
    
    try:
        yield session
        await session.commit()
    except (OperationalError, InterfaceError) as exc:
        # Connection-level errors - the connection may be stale
        await session.rollback()
        logger.error(
            "Database connection error",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    except DBAPIError as exc:
        # Other database errors (constraint violations, etc.)
        await session.rollback()
        logger.error(
            "Database operation failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    except Exception as exc:
        # Application-level exceptions - rollback and re-raise
        await session.rollback()
        raise
    finally:
        await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager version of get_db for non-FastAPI usage.
    
    Use this when you need a database session outside of a request context,
    such as in background tasks or CLI commands.
    
    Yields:
        AsyncSession: Database session.
    
    Example:
        async def background_task():
            async with get_db_context() as db:
                result = await db.execute(select(Machine))
                machines = result.scalars().all()
    """
    async for session in get_db():
        yield session
        return


# =============================================================================
# Health Check
# =============================================================================

async def check_database_health() -> dict[str, Any]:
    """Check database connectivity for health endpoints.
    
    Returns:
        Dictionary with health status and connection pool info.
    
    Example:
        @app.get("/health")
        async def health_check():
            db_health = await check_database_health()
            return {"database": db_health}
    """
    if _engine is None:
        return {
            "status": "unhealthy",
            "error": "Database not initialized",
        }
    
    try:
        async with _engine.begin() as conn:
            # Execute a simple query to verify connectivity
            result = await conn.execute(text("SELECT 1"))
            result.fetchone()
        
        # Get pool statistics
        pool = _engine.pool
        
        return {
            "status": "healthy",
            "pool": {
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
            },
        }
    
    except Exception as exc:
        logger.error(
            "Database health check failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return {
            "status": "unhealthy",
            "error": str(exc),
        }


# =============================================================================
# Transaction Helpers
# =============================================================================

@asynccontextmanager
async def transaction(session: AsyncSession) -> AsyncGenerator[AsyncSession, None]:
    """Explicit transaction context manager with savepoint support.
    
    Use this when you need explicit transaction control within a request,
    such as when you want to rollback part of an operation.
    
    Args:
        session: The database session.
    
    Yields:
        The same session, within a savepoint.
    
    Example:
        async def transfer_funds(db: AsyncSession, from_id: str, to_id: str):
            async with transaction(db):
                await db.execute(update(...))  # Debit
                await db.execute(update(...))  # Credit
                # Both succeed or both rollback
    """
    async with session.begin_nested():
        yield session


async def execute_raw(
    session: AsyncSession,
    query: str,
    params: dict[str, Any] | None = None,
) -> Any:
    """Execute a raw SQL query with proper parameterization.
    
    SECURITY: Always use parameterized queries, never string interpolation.
    
    Args:
        session: The database session.
        query: SQL query with :param placeholders.
        params: Query parameters.
    
    Returns:
        Query result.
    
    Example:
        result = await execute_raw(
            db,
            "SELECT * FROM machines WHERE plant_id = :plant_id",
            {"plant_id": "ford-dearborn-01"},
        )
    """
    result = await session.execute(text(query), params or {})
    return result
