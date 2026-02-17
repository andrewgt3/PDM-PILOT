# Gaia Predictive — Coding Standards & Contribution Guidelines

> **Project:** Gaia Predictive — Industrial Predictive Maintenance Platform  
> **Clients:** Ford / General Motors  
> **Compliance:** TISAX / SOC2  
> **Stack:** Python 3.11+, FastAPI (Async), TimescaleDB (Asyncpg), Redis, Docker

---

## Table of Contents

1. [Overview](#overview)
2. [Type Hinting (Strict Mode)](#type-hinting-strict-mode)
3. [Docstrings (Google Style)](#docstrings-google-style)
4. [Error Handling](#error-handling)
5. [Async Best Practices](#async-best-practices)
6. [Security Requirements](#security-requirements)
7. [Code Review Checklist](#code-review-checklist)

---

## Overview

This document defines the **mandatory coding standards** for all contributors to the Gaia Predictive platform. These standards are non-negotiable and exist to ensure:

- **Reliability:** Zero tolerance for runtime surprises in production environments
- **Auditability:** Full traceability for TISAX/SOC2 compliance
- **Maintainability:** Clear, self-documenting code that any team member can understand

> [!CAUTION]
> **Pull requests that violate these standards will be rejected.** No exceptions.

---

## Type Hinting (Strict Mode)

### Enforcement

All Python files **must** pass `mypy` in strict mode:

```bash
mypy --strict src/
```

### Rules

| Rule | Requirement |
|------|-------------|
| Function arguments | **All** arguments must have explicit type annotations |
| Return types | **All** functions must declare return types (including `-> None`) |
| Variables | Use explicit types for non-obvious assignments |
| `Any` usage | **Prohibited** unless approved by Principal Architect |
| Optional types | Use `Optional[T]` or `T | None` (Python 3.10+) explicitly |

### Examples

```python
# ✅ CORRECT: Fully typed function signature
async def get_machine_health(
    machine_id: uuid.UUID,
    db: AsyncSession,
    include_history: bool = False,
) -> MachineHealthResponse:
    """Retrieve health metrics for a specific machine."""
    ...

# ❌ INCORRECT: Missing type hints
async def get_machine_health(machine_id, db, include_history=False):
    ...
```

```python
# ✅ CORRECT: Typed collections and generics
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T", bound=BaseSensorReading)

def aggregate_readings(readings: Sequence[T]) -> dict[str, float]:
    """Aggregate sensor readings into statistical summary."""
    ...

# ❌ INCORRECT: Untyped collections
def aggregate_readings(readings):
    ...
```

### Pydantic Models

All API request/response models **must** use Pydantic with strict validation:

```python
from pydantic import BaseModel, Field, ConfigDict

class SensorDataRequest(BaseModel):
    """Incoming sensor telemetry payload."""
    
    model_config = ConfigDict(strict=True)
    
    machine_id: uuid.UUID
    timestamp: datetime
    readings: list[SensorReading] = Field(..., min_length=1)
    correlation_id: str = Field(..., pattern=r"^[a-f0-9\-]{36}$")
```

---

## Docstrings (Google Style)

### Enforcement

All public functions, classes, and modules **must** have Google-style docstrings. Private functions (`_prefixed`) should have docstrings if logic is non-trivial.

### Format

```python
def calculate_rul(
    sensor_data: SensorDataFrame,
    model: PredictiveModel,
    confidence_threshold: float = 0.85,
) -> RULPrediction:
    """Calculate Remaining Useful Life for a machine component.
    
    Uses the trained XGBoost model to predict when a component will
    require maintenance based on current sensor telemetry patterns.
    
    Args:
        sensor_data: Normalized sensor readings from the past 24 hours.
            Must contain at least vibration, temperature, and pressure.
        model: Pre-loaded predictive model instance.
        confidence_threshold: Minimum confidence score to return a
            prediction. Predictions below this threshold raise an error.
    
    Returns:
        RULPrediction containing estimated days until failure and
        confidence score.
    
    Raises:
        InsufficientDataError: If sensor_data contains fewer than 100
            data points.
        LowConfidenceError: If prediction confidence is below the
            specified threshold.
        ModelNotLoadedError: If the model has not been properly
            initialized.
    
    Example:
        >>> prediction = calculate_rul(sensor_data, model)
        >>> print(f"Days remaining: {prediction.days_remaining}")
        Days remaining: 45
    """
    ...
```

### Required Sections

| Section | When Required |
|---------|---------------|
| **Summary** | Always (first line, imperative mood) |
| **Extended Description** | When behavior is non-obvious |
| **Args** | When function takes parameters |
| **Returns** | When function returns a value |
| **Raises** | When function can raise exceptions |
| **Example** | For public API functions |

### Class Docstrings

```python
class MaintenanceScheduler:
    """Orchestrates predictive maintenance scheduling across plant assets.
    
    This class coordinates between the ML prediction engine and the
    plant's maintenance management system (MMS) to schedule proactive
    maintenance windows.
    
    Attributes:
        plant_id: Unique identifier for the manufacturing plant.
        prediction_horizon_days: How far ahead to generate predictions.
        mms_client: Client for communicating with the MMS API.
    
    Example:
        >>> scheduler = MaintenanceScheduler(plant_id="ford-dearborn-01")
        >>> await scheduler.generate_weekly_schedule()
    """
    
    plant_id: str
    prediction_horizon_days: int
    mms_client: MMSClient
```

---

## Error Handling

### Golden Rules

> [!IMPORTANT]
> **Rule #1:** Never use bare `except:` or `except Exception:`  
> **Rule #2:** Always catch the most specific exception possible  
> **Rule #3:** Always log exceptions with full context before re-raising

### Prohibited Patterns

```python
# ❌ FORBIDDEN: Bare except clause (hides all errors including KeyboardInterrupt)
try:
    await db.execute(query)
except:
    pass

# ❌ FORBIDDEN: Catching Exception without specificity
try:
    result = await external_api.fetch()
except Exception as e:
    logger.error(f"Something went wrong: {e}")
    return None

# ❌ FORBIDDEN: Silently swallowing errors
try:
    process_data(payload)
except ValueError:
    pass  # Never silently ignore errors
```

### Required Patterns

```python
# ✅ CORRECT: Specific exception handling with context
from asyncpg import PostgresError, InterfaceError
from app.exceptions import (
    DatabaseConnectionError,
    DataValidationError,
    ExternalServiceError,
)

async def store_sensor_reading(reading: SensorReading) -> StoredReading:
    """Persist sensor reading to TimescaleDB.
    
    Raises:
        DatabaseConnectionError: If database connection fails.
        DataValidationError: If reading fails constraint checks.
    """
    try:
        async with db_pool.acquire() as conn:
            result = await conn.fetchrow(INSERT_QUERY, reading.dict())
            return StoredReading.model_validate(result)
    
    except InterfaceError as e:
        logger.error(
            "Database connection failed",
            extra={
                "machine_id": str(reading.machine_id),
                "error_type": type(e).__name__,
                "error_detail": str(e),
            },
        )
        raise DatabaseConnectionError(
            f"Failed to connect to TimescaleDB: {e}"
        ) from e
    
    except PostgresError as e:
        logger.error(
            "Database operation failed",
            extra={
                "machine_id": str(reading.machine_id),
                "pg_error_code": getattr(e, 'sqlstate', 'UNKNOWN'),
                "error_detail": str(e),
            },
        )
        raise DataValidationError(
            f"Failed to store reading: {e}"
        ) from e
```

### Custom Exception Hierarchy

All custom exceptions **must** inherit from the project's base exception:

```python
# app/exceptions.py

class GaiaPredictiveError(Exception):
    """Base exception for all Gaia Predictive errors."""
    
    def __init__(self, message: str, correlation_id: str | None = None) -> None:
        self.correlation_id = correlation_id
        super().__init__(message)


class DatabaseConnectionError(GaiaPredictiveError):
    """Raised when database operations fail due to connectivity."""
    pass


class DataValidationError(GaiaPredictiveError):
    """Raised when data fails validation constraints."""
    pass


class ExternalServiceError(GaiaPredictiveError):
    """Raised when external API calls fail."""
    pass


class PredictionError(GaiaPredictiveError):
    """Raised when ML prediction pipeline fails."""
    pass
```

### FastAPI Exception Handlers

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(GaiaPredictiveError)
async def gaia_exception_handler(
    request: Request,
    exc: GaiaPredictiveError,
) -> JSONResponse:
    """Handle all Gaia-specific exceptions with structured response."""
    logger.error(
        f"Request failed: {exc}",
        extra={
            "correlation_id": exc.correlation_id,
            "path": request.url.path,
            "method": request.method,
        },
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "message": str(exc),
            "correlation_id": exc.correlation_id,
        },
    )
```

---

## Async Best Practices

### Golden Rules

> [!WARNING]
> **Blocking I/O in async routes will halt the entire event loop.**  
> This is a critical performance and reliability violation.

### Prohibited Patterns

```python
# ❌ FORBIDDEN: Blocking I/O in async context
@app.get("/machines/{machine_id}")
async def get_machine(machine_id: uuid.UUID) -> Machine:
    # This blocks the event loop!
    result = requests.get(f"{EXTERNAL_API}/machines/{machine_id}")
    return Machine(**result.json())

# ❌ FORBIDDEN: Synchronous file I/O
@app.post("/upload")
async def upload_file(file: UploadFile) -> dict:
    # open() is blocking!
    with open(f"/data/{file.filename}", "wb") as f:
        f.write(await file.read())
    return {"status": "ok"}

# ❌ FORBIDDEN: time.sleep in async context
@app.get("/delayed")
async def delayed_response() -> dict:
    time.sleep(5)  # Blocks the entire event loop for 5 seconds!
    return {"status": "done"}

# ❌ FORBIDDEN: Synchronous database calls
@app.get("/stats")
async def get_stats() -> Stats:
    # psycopg2 is synchronous!
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM machines")
    ...
```

### Required Patterns

```python
# ✅ CORRECT: Async HTTP client (httpx or aiohttp)
import httpx

@app.get("/machines/{machine_id}")
async def get_machine(machine_id: uuid.UUID) -> Machine:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{EXTERNAL_API}/machines/{machine_id}",
            timeout=10.0,
        )
        response.raise_for_status()
        return Machine(**response.json())
```

```python
# ✅ CORRECT: Async file I/O with aiofiles
import aiofiles

@app.post("/upload")
async def upload_file(file: UploadFile) -> dict[str, str]:
    async with aiofiles.open(f"/data/{file.filename}", "wb") as f:
        content = await file.read()
        await f.write(content)
    return {"status": "ok"}
```

```python
# ✅ CORRECT: Async sleep
import asyncio

@app.get("/delayed")
async def delayed_response() -> dict[str, str]:
    await asyncio.sleep(5)  # Non-blocking!
    return {"status": "done"}
```

```python
# ✅ CORRECT: Async database access with asyncpg
import asyncpg

@app.get("/stats")
async def get_stats(db: AsyncSession = Depends(get_db)) -> Stats:
    result = await db.execute(
        text("SELECT COUNT(*) as count FROM machines")
    )
    row = result.fetchone()
    return Stats(machine_count=row.count)
```

### Running Blocking Code Safely

When you **must** use blocking code (e.g., CPU-intensive ML inference):

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Create a dedicated thread pool for blocking operations
ml_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ml-")

@app.post("/predict")
async def predict_failure(request: PredictionRequest) -> PredictionResponse:
    """Run ML prediction in thread pool to avoid blocking."""
    loop = asyncio.get_event_loop()
    
    # Run blocking ML inference in thread pool
    prediction = await loop.run_in_executor(
        ml_executor,
        model.predict,  # Blocking function
        request.features,
    )
    
    return PredictionResponse(
        machine_id=request.machine_id,
        failure_probability=prediction.probability,
        confidence=prediction.confidence,
    )
```

### Async Context Managers

Always use async context managers for resources:

```python
# ✅ CORRECT: Async context manager for database sessions
from contextlib import asynccontextmanager
from typing import AsyncGenerator

@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional scope around a series of operations."""
    session = AsyncSession(engine)
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
```

---

## Security Requirements

### Input Validation

- All user inputs **must** be validated via Pydantic models
- SQL queries **must** use parameterized statements (never string interpolation)
- File paths **must** be validated against path traversal attacks

### Secrets Management

```python
# ❌ FORBIDDEN: Hardcoded secrets
DATABASE_URL = "postgresql://admin:supersecret123@localhost/gaia"

# ✅ CORRECT: Environment-based secrets
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    database_url: str
    redis_url: str
    jwt_secret: str
    
    model_config = ConfigDict(env_file=".env")
```

### Logging

- **Never** log sensitive data (passwords, tokens, PII)
- Always include `correlation_id` for request tracing
- Use structured logging (JSON format in production)

---

## Code Review Checklist

Before submitting a PR, verify:

- [ ] All functions have complete type annotations
- [ ] `mypy --strict` passes with zero errors
- [ ] All public functions have Google-style docstrings
- [ ] No bare `except:` or `except Exception:` clauses
- [ ] All async routes use async I/O (no blocking calls)
- [ ] Custom exceptions inherit from `GaiaPredictiveError`
- [ ] No hardcoded secrets or credentials
- [ ] Unit tests cover all new code paths
- [ ] No `# type: ignore` comments without justification

---

## Tooling Configuration

### pyproject.toml

```toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true

[tool.ruff]
line-length = 88
target-version = "py311"
select = ["E", "F", "I", "N", "W", "UP", "ANN", "ASYNC", "S", "B"]

[tool.ruff.per-file-ignores]
"tests/*" = ["S101", "ANN"]
```

---

**Last Updated:** December 2024  
**Approved By:** Principal Software Architect  
**Review Cycle:** Quarterly
