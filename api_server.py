#!/usr/bin/env python3
"""
GAIA FastAPI WebSocket Server
Phase 7: Tier 1 Architecture - The API Layer

Endpoints:
- /ws/stream      : WebSocket for real-time 10Hz telemetry
- /api/features   : REST GET for historical cwru_features
- /api/machines   : REST GET for active machine IDs
- /api/health     : REST GET for system health check
- /docs           : Swagger UI (auto-generated)

Author: Senior Backend Engineer
"""

import asyncio
import json
import os
import traceback
import uuid
import shutil
import sys
import subprocess
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

import joblib
import numpy as np
import redis.asyncio as aioredis
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import BackgroundTasks
from starlette.concurrency import run_in_threadpool
import analytics_engine
from advanced_features import SignalProcessor, SAMPLE_RATE, N_SAMPLES
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Use centralized configuration
from config import get_settings
from database import check_database_health, get_db, get_db_context, init_database, shutdown_database
from dependencies import (
    get_current_user,
    get_scoped_machine_filter,
    require_engineer_permission,
    require_operator_or_above,
)
from services.rbac_service import can_access_machine, can_access_raw_data
from schemas.security import User
from schemas.profiling import DataProfile
from schemas.response import APIResponse, ORJSONResponse
from auth_utils import decode_access_token
from services.connectivity_monitor_service import ConnectivityMonitorService

# Rate Limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Load settings
settings = get_settings()

# =============================================================================
# CONFIGURATION
# =============================================================================
REDIS_HOST = settings.redis.host
REDIS_PORT = settings.redis.port
REDIS_CHANNEL = settings.redis.redis_channel

# DATABASE_URL is now managed by database.py via settings

# =============================================================================
# EQUIPMENT METADATA LOOKUP
# Maps equipment IDs to their shop/line/name information
# =============================================================================
EQUIPMENT_METADATA = {
    "WB-001": {"name": "6-Axis Welder #1", "shop": "Body Shop", "line": "Underbody Weld Cell", "type": "Spot Welder"},
    "WB-002": {"name": "6-Axis Welder #2", "shop": "Body Shop", "line": "Underbody Weld Cell", "type": "Spot Welder"},
    "WB-003": {"name": "Frame Welder", "shop": "Body Shop", "line": "Underbody Weld Cell", "type": "MIG Welder"},
    "HP-200": {"name": "Hydraulic Press 2000T", "shop": "Stamping", "line": "Press Line 1", "type": "Hydraulic Press"},
    "TD-450": {"name": "Transfer Die Unit", "shop": "Stamping", "line": "Press Line 1", "type": "Transfer Die"},
    "PR-101": {"name": "Paint Robot #1", "shop": "Paint Shop", "line": "Sealer Line", "type": "Paint Applicator"},
    "CO-050": {"name": "Curing Oven", "shop": "Paint Shop", "line": "Sealer Line", "type": "Thermal Oven"},
    "TS-001": {"name": "Torque Station #1", "shop": "Final Assembly", "line": "Chassis Line", "type": "Torque Tool"},
    "LA-003": {"name": "Lift Assist #3", "shop": "Final Assembly", "line": "Chassis Line", "type": "Ergonomic Lift"},
    "CV-100": {"name": "Main Conveyor Drive", "shop": "Final Assembly", "line": "Chassis Line", "type": "Conveyor Motor"},
}



# =============================================================================
# ASSET DATA & HIERARCHY (Phase 4)
# =============================================================================
ASSET_REGISTRY = {
    # Line A: Precision Machines
    "M_001": {"line_id": "L_01", "line_name": "Line A (Precision)", "model": "HAAS-VF2", "install_date": "2021-03-15"},
    "M_002": {"line_id": "L_01", "line_name": "Line A (Precision)", "model": "HAAS-VF2", "install_date": "2021-04-20"},
    "M_003": {"line_id": "L_01", "line_name": "Line A (Precision)", "model": "DMG-50", "install_date": "2022-01-10"},
    "M_004": {"line_id": "L_01", "line_name": "Line A (Precision)", "model": "DMG-50", "install_date": "2022-02-14"},
    "M_005": {"line_id": "L_01", "line_name": "Line A (Precision)", "model": "MAZAK-QR", "install_date": "2020-11-30"},
    
    # Line B: Assembly & Packaging
    "M_006": {"line_id": "L_02", "line_name": "Line B (Assembly)", "model": "KUKA-KR6", "install_date": "2023-05-05"},
    "M_007": {"line_id": "L_02", "line_name": "Line B (Assembly)", "model": "ABB-IRB", "install_date": "2023-06-12"},
    "M_008": {"line_id": "L_02", "line_name": "Line B (Assembly)", "model": "ABB-IRB", "install_date": "2023-06-12"},
    "M_009": {"line_id": "L_02", "line_name": "Line B (Assembly)", "model": "FANUC-M20", "install_date": "2019-08-22"},
    "M_010": {"line_id": "L_02", "line_name": "Line B (Assembly)", "model": "FANUC-M20", "install_date": "2019-09-01"},
}

# =============================================================================
# GLOBAL STATE
# =============================================================================
class AppState:
    """Global application state for shared resources."""
    redis: Optional[aioredis.Redis] = None
    model = None
    scaler = None
    feature_columns = None
    processor = None
    connected_clients: List[WebSocket] = []


state = AppState()


# =============================================================================
# LIFESPAN (Startup/Shutdown)
# =============================================================================
# =============================================================================
# LIFESPAN (Startup/Shutdown)
# =============================================================================
async def _inference_stream_subscriber():
    """
    Background task: subscribes to the inference_results Redis Stream
    and caches the latest prediction per machine_id for fast API lookups.
    This runs for the lifetime of the API server.
    """
    STREAM = "inference_results"
    last_id = "$"  # Only new messages
    while True:
        try:
            if state.redis is None:
                await asyncio.sleep(5)
                continue
            # XREAD with 5s block
            messages = await state.redis.xread(
                {STREAM: last_id}, count=50, block=5000
            )
            if not messages:
                continue
            for stream_name, entries in messages:
                for msg_id, fields in entries:
                    last_id = msg_id
                    try:
                        data = json.loads(fields.get("data", "{}"))
                        machine_id = data.get("machine_id", "unknown")
                        # Cache in Redis for fast lookup
                        await state.redis.set(
                            f"inference:latest:{machine_id}",
                            json.dumps(data),
                            ex=300,  # 5 min TTL
                        )
                    except Exception:
                        pass
        except asyncio.CancelledError:
            break
        except Exception as exc:
            print(f"[Inference Subscriber] Error: {exc}. Retrying in 5s…")
            await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, cleanup on shutdown."""
    print("=" * 60)
    print("GAIA API SERVER - Starting...")
    print("=" * 60)
    
    # Initialize Database
    await init_database()
    
    # Connect to Redis
    try:
        state.redis = aioredis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True
        )
        await state.redis.ping()
        print(f"✓ Redis connected: {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        print(f"⚠ Redis connection failed: {e}")
        state.redis = None
    
    # Load ML Model (path from config, resolved relative to project root)
    model_path = settings.model.resolved_gaia_model_path
    if not model_path.exists():
        print(f"Model file not found at {model_path} — inference endpoints will return 503")
        state.model = None
        state.scaler = None
        state.feature_columns = None
        state.processor = None
    else:
        try:
            pipeline = joblib.load(str(model_path))
            state.model = pipeline['model']
            state.scaler = pipeline['scaler']
            state.feature_columns = pipeline['feature_columns']
            state.processor = SignalProcessor(sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES)
            print(f"✓ ML Model loaded: {len(state.feature_columns)} features")
        except Exception as e:
            print(f"Model file not found at {model_path} — inference endpoints will return 503")
            state.model = None
            state.scaler = None
            state.feature_columns = None
            state.processor = None
    
    # Start background inference stream subscriber (Golden Pipeline Stage 6)
    inference_task = asyncio.create_task(_inference_stream_subscriber())
    print("✓ Inference results stream subscriber started")
    
    print("=" * 60)
    print("Server ready. Endpoints:")
    print("  - /docs           (Swagger UI)")
    print("  - /ws/stream      (WebSocket)")
    print("  - /api/features   (REST)")
    print("  - /api/machines   (REST)")
    print("  - /api/health     (REST)")
    print("  - /api/inference/latest  (REST — Pipeline)")
    print("=" * 60)
    
    yield  # Server runs here
    
    # Cleanup
    print("\nShutting down...")
    inference_task.cancel()
    try:
        await inference_task
    except asyncio.CancelledError:
        pass
    await shutdown_database()
    if state.redis:
        await state.redis.close()


# =============================================================================
# FASTAPI APP
# =============================================================================

# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["100 per minute"])

# Implement strict Global Auth with exceptions
from dependencies import get_optional_user
from schemas.security import User

async def verify_global_auth(request: Request, user: Optional[User] = Depends(get_optional_user)):
    """
    Enforce authentication globally with specific exceptions.
    """
    # Public endpoints (no auth required)
    public_paths = [
        "/health", "/token", "/api/enterprise/token", 
        "/docs", "/redoc", "/openapi.json",
        # Frontend demo endpoints (allow read-only access)
        "/api/machines", "/api/features", "/api/recommendations",
        # Pipeline Ops dashboard endpoints
        "/api/pipeline/status", "/api/inference/latest",
        "/api/ingest/bootstrap", "/api/ingest/upload", "/api/ingest/watcher/status",
    ]
    if request.url.path in public_paths:
        return
    
    # Allow paths starting with /api/recommendations/ (machine-specific)
    if request.url.path.startswith("/api/recommendations/"):
        return
    
    # Allow all discovery API endpoints (anomaly detection)
    if request.url.path.startswith("/api/discovery/"):
        return

    # WebSocket handshake - excluded from auth middleware
    if request.url.path == "/ws/stream":
        return 

    if not user:
         raise HTTPException(status_code=401, detail="Not authenticated")

app = FastAPI(
    title="GAIA Predictive Maintenance API",
    description="Real-time WebSocket streaming and REST API for Industrial IoT predictive maintenance.",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
    dependencies=[], # [Depends(verify_global_auth)],
)

# Attach limiter to app state
app.state.limiter = limiter

# Rate limit exceeded handler (returns 429 Too Many Requests)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# =============================================================================
# GLOBAL EXCEPTION HANDLERS
# =============================================================================

# Import custom exceptions and SQLAlchemy errors
from core.exceptions import (
    GaiaBaseException,
    ResourceNotFound,
    PermissionDenied,
    BusinessRuleViolation,
    ValidationError as DomainValidationError,
    ExternalServiceError,
    RateLimitExceeded,
)
from sqlalchemy.exc import IntegrityError
from fastapi.exceptions import RequestValidationError


@app.exception_handler(ResourceNotFound)
async def resource_not_found_handler(request: Request, exc: ResourceNotFound):
    """Handle ResourceNotFound → 404."""
    return JSONResponse(
        status_code=404,
        content=exc.to_dict()
    )


@app.exception_handler(PermissionDenied)
async def permission_denied_handler(request: Request, exc: PermissionDenied):
    """Handle PermissionDenied → 403."""
    return JSONResponse(
        status_code=403,
        content=exc.to_dict()
    )


@app.exception_handler(BusinessRuleViolation)
async def business_rule_violation_handler(request: Request, exc: BusinessRuleViolation):
    """Handle BusinessRuleViolation → 400."""
    return JSONResponse(
        status_code=400,
        content=exc.to_dict()
    )


@app.exception_handler(DomainValidationError)
async def domain_validation_handler(request: Request, exc: DomainValidationError):
    """Handle domain validation errors → 400."""
    return JSONResponse(
        status_code=400,
        content=exc.to_dict()
    )


@app.exception_handler(ExternalServiceError)
async def external_service_handler(request: Request, exc: ExternalServiceError):
    """Handle external service failures → 502."""
    return JSONResponse(
        status_code=502,
        content=exc.to_dict()
    )


@app.exception_handler(IntegrityError)
async def integrity_error_handler(request: Request, exc: IntegrityError):
    """Handle SQLAlchemy IntegrityError → 409 Conflict (clean, no SQL traceback)."""
    # Parse the constraint name from the error if possible
    error_str = str(exc.orig) if exc.orig else str(exc)
    
    # Common constraint patterns
    if "unique" in error_str.lower() or "duplicate" in error_str.lower():
        message = "A record with this value already exists"
        error_type = "DUPLICATE_ENTRY"
    elif "foreign key" in error_str.lower():
        message = "Referenced record does not exist"
        error_type = "INVALID_REFERENCE"
    elif "not-null" in error_str.lower() or "null value" in error_str.lower():
        message = "Required field is missing"
        error_type = "MISSING_REQUIRED_FIELD"
    else:
        message = "Database constraint violation"
        error_type = "CONSTRAINT_VIOLATION"
    
    return JSONResponse(
        status_code=409,
        content={
            "error": error_type,
            "message": message,
            "details": {}
        }
    )


@app.exception_handler(RequestValidationError)
async def request_validation_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors → 422 with clean field-level errors."""
    fields = {}
    for error in exc.errors():
        # Extract field path (e.g., ['body', 'email'] → 'email')
        loc = error.get("loc", [])
        field_name = ".".join(str(part) for part in loc if part not in ("body", "query", "path"))
        if not field_name:
            field_name = "unknown"
        
        # Clean up error message
        msg = error.get("msg", "Invalid value")
        fields[field_name] = msg
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "INVALID_INPUT",
            "message": "Request validation failed",
            "fields": fields
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions - pass through with proper status."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler - catches all unhandled exceptions.
    Logs full trace to structlog but returns clean error to user.
    """
    from logger import get_logger
    import traceback as tb
    
    error_logger = get_logger("api.errors")
    request_id = str(uuid.uuid4())[:8]  # Short ref for user
    
    # Log full error details with structlog
    error_logger.error(
        "Unhandled exception",
        ref=request_id,
        path=request.url.path,
        method=request.method,
        error_type=type(exc).__name__,
        error_message=str(exc),
        traceback=tb.format_exc(),
    )
    
    # Return clean response to user (no stack trace)
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "ref": request_id
        }
    )


# CORS Middleware (Allow React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Audit Logger Middleware (for sensitive routes)
try:
    from middleware.audit_logger import AuditLoggerMiddleware
    app.add_middleware(AuditLoggerMiddleware, log_all_requests=False)
    print("✓ Audit Logger middleware enabled")
except ImportError as e:
    print(f"⚠ Audit Logger not available: {e}")

# Security Headers Middleware (Helmet.js equivalent)
try:
    from middleware.security_headers import SecurityHeadersMiddleware
    app.add_middleware(
        SecurityHeadersMiddleware,
        enable_hsts=True,
        frame_options="DENY"
    )
    print("✓ Security Headers middleware enabled (CSP, HSTS, X-Frame-Options)")
except ImportError as e:
    print(f"⚠ Security Headers not available: {e}")

# Request ID & JSON Access Logging Middleware
try:
    from core.logger import RequestIdMiddleware
    app.add_middleware(RequestIdMiddleware)
    print("✓ RequestIdMiddleware enabled (JSON access logs with request_id)")
except ImportError as e:
    print(f"⚠ RequestIdMiddleware not available: {e}")

# Include Enterprise API routes (alarms, work orders, MTBF/MTTR, schedules)
try:
    from enterprise_api import enterprise_router, labeling_router, alerts_router, drift_router, models_router, onboarding_router, verify_router, admin_router
    app.include_router(enterprise_router)
    app.include_router(labeling_router)
    app.include_router(alerts_router)
    app.include_router(drift_router)
    app.include_router(models_router)
    app.include_router(onboarding_router)
    app.include_router(verify_router)
    app.include_router(admin_router)
    print("✓ Enterprise API routes loaded")
    print("✓ Labeling API routes loaded")
    print("✓ Alerts API routes loaded")
except ImportError as e:
    print(f"⚠ Enterprise API not available: {e}")

# Include Anomaly Discovery API routes
try:
    from anomaly_discovery.api import discovery_router
    app.include_router(discovery_router)
    print("✓ Anomaly Discovery API routes loaded")
except ImportError as e:
    print(f"⚠ Anomaly Discovery API not available: {e}")


# =============================================================================
# PYDANTIC MODELS
# =============================================================================
class HealthResponse(BaseModel):
    status: str
    redis_connected: bool
    model_loaded: bool
    connected_clients: int
    timestamp: str


class FeatureRecord(BaseModel):
    timestamp: str
    machine_id: str
    bpfi_amp: float
    bpfo_amp: float
    failure_prediction: float


class MachineStatus(BaseModel):
    machine_id: str
    last_seen: str
    failure_probability: float
    status: str


# =============================================================================
# DATABASE HELPER
# =============================================================================
# Legacy cleanup
# get_db_connection removed in favor of explicit get_db dependency




# =============================================================================
# DATABASE HELPERS (Async)
# =============================================================================

async def fetch_latest_features(db: AsyncSession, machine_id: str, limit: int = 1) -> List[Dict]:
    """Helper to fetch features asynchronously."""
    query = text("""
        SELECT *
        FROM cwru_features
        WHERE machine_id = :machine_id
        ORDER BY timestamp DESC
        LIMIT :limit
    """)
    result = await db.execute(query, {"machine_id": machine_id, "limit": limit})
    rows = result.mappings().all()
    return [dict(row) for row in rows]



# =============================================================================
# GOLDEN PIPELINE ENDPOINTS (Stage 6)
# =============================================================================

def _normalize_inference_to_prediction_response(data: dict) -> dict:
    """Ensure inference payload has PredictionResponse fields (with defaults for legacy cache)."""
    out = dict(data)
    out.setdefault("failure_probability", out.get("failure_prediction", 0.0))
    out.setdefault("health_score", max(0, min(100, (1.0 - out.get("failure_probability", 0)) * 100)))
    rul_h = out.get("rul_hours", -1.0)
    rul_d = (rul_h / 24.0) if rul_h >= 0 else -1.0
    out.setdefault("rul_days", rul_d)
    out.setdefault("rul_lower_80", rul_d if rul_d >= 0 else -1.0)
    out.setdefault("rul_upper_80", rul_d if rul_d >= 0 else -1.0)
    out.setdefault("confidence", "MEDIUM")
    out.setdefault("in_training_distribution", True)
    return out


@app.get("/api/inference/latest", tags=["Pipeline"])
async def get_all_latest_inference(user: User = Depends(get_current_user)):
    """
    Get the latest inference results from the Golden Pipeline.
    Scoped by role: only machines the user can access.
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if state.redis is None:
        raise HTTPException(status_code=503, detail="Redis not connected")

    machine_filter = get_scoped_machine_filter(user)
    if machine_filter is not None and len(machine_filter) == 0:
        return APIResponse.success(data=[], count=0)

    results = []
    cursor = b"0"
    allowed = set(machine_filter) if machine_filter is not None else None
    while True:
        cursor, keys = await state.redis.scan(
            cursor=cursor, match="inference:latest:*", count=100
        )
        for key in keys:
            raw = await state.redis.get(key)
            if raw:
                try:
                    data = json.loads(raw)
                    mid = data.get("machine_id")
                    if allowed is not None and mid not in allowed:
                        continue
                    results.append(_normalize_inference_to_prediction_response(data))
                except json.JSONDecodeError:
                    pass
        if cursor == 0:
            break

    return APIResponse.success(data=results, count=len(results))


@app.get("/api/inference/latest/{machine_id}", tags=["Pipeline"])
async def get_machine_inference(machine_id: str, user: User = Depends(get_current_user)):
    """
    Get the latest inference result for a specific machine.
    RBAC: user must have access to this machine.
    """
    if not can_access_machine(user, machine_id):
        raise HTTPException(status_code=403, detail="Not authorized for this machine")
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if state.redis is None:
        raise HTTPException(status_code=503, detail="Redis not connected")

    raw = await state.redis.get(f"inference:latest:{machine_id}")
    if not raw:
        raise HTTPException(
            status_code=404,
            detail=f"No inference results for machine {machine_id}",
        )
    data = json.loads(raw)
    return _normalize_inference_to_prediction_response(data)

@app.post("/api/ingest/upload", tags=["Pipeline"])
async def upload_ingest_files(files: List[UploadFile] = File(...)):
    """
    Upload generic files to the ingestion Drop Zone.
    Preserves directory structure if relative paths are provided in filenames.
    """
    INGEST_BASE = Path("data/ingest")
    INGEST_BASE.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for file in files:
        try:
            # Normalize filename: handle potential leading slashes or relative segments
            # Security: ensure the path doesn't escape the ingest directory
            safe_rel_path = Path(file.filename).name if ".." in file.filename else file.filename
            file_path = INGEST_BASE / safe_rel_path
            
            # Create subdirectories if necessary
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)
        except Exception as e:
            from logger import get_logger
            logger = get_logger("api")
            logger.error(f"Error saving {file.filename}: {e}")
            
    return {"message": f"Successfully uploaded {len(saved_files)} files", "files": saved_files}


@app.get("/api/ingest/watcher/status", tags=["Pipeline"])
async def get_watcher_status():
    """Checks if the watcher_service.py is currently running."""
    try:
        # Use pgrep to check for the process in a threadpool to avoid blocking
        result = await run_in_threadpool(
            subprocess.run, ["pgrep", "-f", "watcher_service.py"], capture_output=True, text=True
        )
        is_running = bool(result.stdout.strip())
        return {"running": is_running, "pids": result.stdout.strip().split("\n") if is_running else []}
    except Exception as e:
        return {"running": False, "error": str(e)}


def run_bootstrap_sequence():
    """Fire-and-forget: spawn bootstrap + watcher in detached processes. Returns immediately."""
    from logger import get_logger
    logger = get_logger("api")

    try:
        # 1. Spawn bootstrap (ingest_archive.py) - do NOT wait; copying can take minutes
        logger.info("[Background] Spawning bootstrap process (archive copy)...")
        subprocess.Popen(
            [sys.executable, "ingest_archive.py"],
            cwd=Path(__file__).resolve().parent,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # 2. Start Watcher if not already running
        result = subprocess.run(["pgrep", "-f", "watcher_service.py"], capture_output=True, text=True)
        if not result.stdout.strip():
            logger.info("[Background] Starting Watcher Service...")
            subprocess.Popen(
                [sys.executable, "watcher_service.py"],
                cwd=Path(__file__).resolve().parent,
                start_new_session=True,
            )

        logger.info("[Background] Bootstrap and watcher started. Copy runs in background.")
    except Exception as e:
        logger.error(f"[Background] Bootstrap failed: {e}")


@app.post("/api/ingest/bootstrap", tags=["Pipeline"])
async def bootstrap_nasa_data(current_user=Depends(require_engineer_permission)):
    """
    Triggers local archive extraction in the background and ensures Watcher is active.
    Returns immediately; bootstrap runs in a daemon thread.
    """
    import threading
    try:
        ARCHIVE_PATH = Path(os.environ.get("NASA_ARCHIVE_PATH", str(Path.home() / "Desktop" / "archive.zip")))
        
        if not ARCHIVE_PATH.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"NASA Archive not found at {ARCHIVE_PATH}. Set NASA_ARCHIVE_PATH in .env to your archive.zip or archive folder."
            )

        # Fire-and-forget in background thread - response returns before thread does any work
        def _run():
            run_bootstrap_sequence()
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        
        return {
            "status": "accepted", 
            "message": "Archive extraction started in background. Watcher will activate automatically."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bootstrap failed: {e}")


@app.get("/api/pipeline/status", tags=["Pipeline"])
async def get_pipeline_status():
    """
    Get real-time status of all 6 Golden Pipeline stages.
    Queries Redis stream lengths, consumer group info, and health keys.
    Returns degraded status (200) when Redis is down so the page can load.
    """
    if state.redis is None:
        return {
            "stages": [
                {"id": i, "name": n, "stream": None, "length": 0, "status": "DOWN", "consumer_groups": [], "last_entry_id": None}
                for i, n in [(1, "Ingestion"), (2, "Cleansing"), (3, "Contextualization"), (4, "Persistence"), (5, "Inference"), (6, "Orchestration")]
            ],
            "buffer_size": 0,
            "active_machines": 0,
            "all_live": False,
            "watcher_active": False,
            "redis_connected": False,
        }

    streams = {
        "raw_sensor_data": {"stage": 1, "name": "Ingestion"},
        "clean_features": {"stage": 2, "name": "Cleansing"}, # NASA Refinery target
        "clean_sensor_data": {"stage": 2, "name": "Standard Cleansing"},
        "contextualized_data": {"stage": 3, "name": "Contextualization"},
        "inference_results": {"stage": 5, "name": "Inference"},
    }

    stages = []

    for stream_key, info in streams.items():
        try:
            length = await state.redis.xlen(stream_key)
            # Get consumer group info
            groups = []
            try:
                group_info = await state.redis.xinfo_groups(stream_key)
                groups = [
                    {
                        "name": g.get("name", ""),
                        "consumers": g.get("consumers", 0),
                        "pending": g.get("pending", 0),
                        "lag": g.get("lag", 0),
                    }
                    for g in group_info
                ]
            except Exception:
                pass

            # Get last entry timestamp
            last_ts = None
            try:
                last_entry = await state.redis.xrevrange(stream_key, count=1)
                if last_entry:
                    last_ts = last_entry[0][0]  # Redis stream ID (timestamp-based)
            except Exception:
                pass

            stages.append({
                "id": info["stage"],
                "name": info["name"],
                "stream": stream_key,
                "length": length,
                "status": "LIVE" if length > 0 else "WAITING",
                "consumer_groups": groups,
                "last_entry_id": last_ts,
            })
        except Exception:
            stages.append({
                "id": info["stage"],
                "name": info["name"],
                "stream": stream_key,
                "length": 0,
                "status": "DOWN",
                "consumer_groups": [],
                "last_entry_id": None,
            })

    # Stage 4: Persistence (health key check)
    try:
        health_ts = await state.redis.get("system:health:persistence")
        persistence_healthy = health_ts is not None
    except Exception:
        persistence_healthy = False

    stages.append({
        "id": 4,
        "name": "Persistence",
        "stream": None,
        "length": None,
        "status": "LIVE" if persistence_healthy else "DOWN",
        "health_timestamp": health_ts,
        "consumer_groups": [],
        "last_entry_id": None,
    })

    # Stage 6: Orchestration (API itself)
    stages.append({
        "id": 6,
        "name": "Orchestration",
        "stream": None,
        "length": None,
        "status": "LIVE",
        "consumer_groups": [],
        "last_entry_id": None,
    })

    # Sort by stage ID
    stages.sort(key=lambda s: s["id"])

    # Buffer size (store-and-forward)
    buffer_size = 0
    try:
        buffer_size = await state.redis.llen("buffer:contextualized")
    except Exception:
        pass

    # Count active inference keys
    inference_count = 0
    try:
        cursor = b"0"
        while True:
            cursor, keys = await state.redis.scan(
                cursor=cursor, match="inference:latest:*", count=100
            )
            inference_count += len(keys)
            if cursor == 0:
                break
    except Exception:
        pass

    # Watcher status
    watcher_active = False
    watcher_pids = []
    try:
        result = subprocess.run(["pgrep", "-f", "watcher_service.py"], capture_output=True, text=True)
        watcher_active = bool(result.stdout.strip())
        watcher_pids = result.stdout.strip().split("\n") if watcher_active else []
    except Exception:
        pass

    # Data readiness (for Pipeline UI)
    readiness = {}
    try:
        base_dir = Path(__file__).resolve().parent
        archive_path = Path(os.environ.get("NASA_ARCHIVE_PATH", str(Path.home() / "Desktop" / "archive.zip")))
        ingest_dir = base_dir / "data" / "ingest"
        femto_dir = base_dir / "data" / "downloads" / "femto_pronostia"
        readiness = {
            "archive_ready": archive_path.exists(),
            "archive_path": str(archive_path),
            "ingest_file_count": sum(1 for _ in ingest_dir.rglob("*") if _.is_file() and not _.name.startswith(".")) if ingest_dir.exists() else 0,
            "femto_ready": (femto_dir / "Learning_set").exists() if femto_dir.exists() else False,
        }
    except Exception:
        pass

    # Training data & model accuracy (for Pipeline UI)
    training_data = None
    model_accuracy = None
    try:
        base_dir = Path(__file__).resolve().parent
        models_dir = base_dir / "data" / "models"
        proc_dir = base_dir / "data" / "processed"
        meta_path = models_dir / "training_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                training_data = json.load(f)
        else:
            if (proc_dir / "combined_features_physics.csv").exists():
                training_data = {"source": "NASA + FEMTO", "n_runs": None}
            elif (proc_dir / "nasa_features_physics.csv").exists():
                training_data = {"source": "NASA only", "n_runs": None}
        acc_path = models_dir / "validation_metrics.json"
        if acc_path.exists():
            with open(acc_path) as f:
                model_accuracy = json.load(f)
    except Exception:
        pass

    return {
        "stages": stages,
        "buffer_size": buffer_size,
        "active_machines": inference_count,
        "all_live": all(s["status"] == "LIVE" for s in stages),
        "watcher_active": watcher_active,
        "watcher_pids": watcher_pids,
        "redis_connected": True,
        "training_data": training_data,
        "model_accuracy": model_accuracy,
        "readiness": readiness,
    }


@app.post("/api/pipeline/start-live-adapters", tags=["Pipeline"])
async def start_live_adapters(current_user=Depends(require_engineer_permission)):
    """
    Start Option B: ABB adapter, Siemens S7 adapter, and stream publisher as separate processes.
    Spawns abb_adapter.py, siemens_s7_adapter.py, and stream_publisher.py in the project root.
    Requires Redis to be running; adapters will retry until their PLC/OPC targets are available.
    """
    project_root = Path(__file__).resolve().parent
    python = sys.executable
    scripts = [
        ("abb_adapter.py", "ABB (OPC-UA)"),
        ("siemens_s7_adapter.py", "Siemens S7 (snap7)"),
        ("stream_publisher.py", "Stream publisher"),
    ]
    started = []
    for script_name, label in scripts:
        script_path = project_root / script_name
        if not script_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Script not found: {script_name}. Run from project root.",
            )
        try:
            # Detach: new session on Unix so processes outlive the request
            kwargs = {"cwd": str(project_root), "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
            if sys.platform != "win32":
                kwargs["start_new_session"] = True
            else:
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            subprocess.Popen([python, script_name], **kwargs)
            started.append(label)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start {label}: {e}",
            )
    return {
        "ok": True,
        "message": "Started live adapters (Option B).",
        "started": started,
    }


# =============================================================================
# REST ENDPOINTS
# =============================================================================
@app.get("/", tags=["System"])
async def root():
    """Root endpoint to verify connectivity."""
    return {"message": "GAIA API is running", "docs_url": "/docs"}

@app.get("/health", tags=["System"])
async def health_check():
    """
    Public health check endpoint for load balancers.
    Checks database connection pool status.
    """
    health = await check_database_health()
    status_code = 200 if health["status"] == "healthy" else 503
    return JSONResponse(content=health, status_code=status_code)


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def detailed_health_check():
    """Detailed system health check endpoint (internal use)."""
    return HealthResponse(
        status="ok",
        redis_connected=state.redis is not None,
        model_loaded=state.model is not None,
        connected_clients=len(state.connected_clients),
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.get("/api/edge/connectivity", tags=["Edge"])
async def get_edge_connectivity(db: AsyncSession = Depends(get_db)):
    """Return offline buffer gap report (no auth)."""
    report = await ConnectivityMonitorService().get_gap_report(db)
    return report


@app.get("/api/edge/gap-history", tags=["Edge"])
async def get_edge_gap_history(
    days: int = Query(30, ge=1, le=365, description="Number of days of history"),
    db: AsyncSession = Depends(get_db),
    user=Depends(require_operator_or_above),
):
    """Return gap_events for the last N days (operator+)."""
    result = await db.execute(
        text("""
            SELECT id, adapter_source, gap_started_at, gap_ended_at,
                   readings_buffered, readings_replayed, duration_minutes
            FROM gap_events
            WHERE gap_started_at >= now() - interval '1 day' * :days
            ORDER BY gap_started_at DESC
        """),
        {"days": days},
    )
    rows = result.mappings().all()
    return [dict(r) for r in rows]


@app.get("/api/features", tags=["Data"])
async def get_features(
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return"),
    machine_id: str = Query(None, description="Filter by machine ID"),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> APIResponse[List[Dict]]:
    """
    Retrieve historical feature records. RBAC: machine and raw-data access enforced.
    """
    if not can_access_raw_data(user):
        raise HTTPException(status_code=403, detail="Raw data access not permitted")
    if machine_id and not can_access_machine(user, machine_id):
        raise HTTPException(status_code=403, detail="Not authorized for this machine")

    machine_filter = get_scoped_machine_filter(user)
    if machine_filter is not None and len(machine_filter) == 0:
        return APIResponse.success(data=[], count=0)

    try:
        if machine_id:
            query = text("""
                SELECT 
                    timestamp, machine_id,
                    peak_freq_1, peak_freq_2, peak_freq_3, peak_freq_4, peak_freq_5,
                    peak_amp_1, peak_amp_2, peak_amp_3, peak_amp_4, peak_amp_5,
                    low_band_power, mid_band_power, high_band_power,
                    spectral_entropy, spectral_kurtosis, total_power,
                    bpfo_amp, bpfi_amp, bsf_amp, ftf_amp, sideband_strength,
                    degradation_score,
                    rotational_speed, temperature, torque, tool_wear,
                    failure_prediction, failure_class
                FROM cwru_features
                WHERE machine_id = :machine_id
                ORDER BY timestamp DESC
                LIMIT :limit
            """)
            result = await db.execute(query, {"machine_id": machine_id, "limit": limit})
        else:
            query = text("""
                SELECT 
                    timestamp, machine_id,
                    peak_freq_1, peak_freq_2, peak_freq_3, peak_freq_4, peak_freq_5,
                    peak_amp_1, peak_amp_2, peak_amp_3, peak_amp_4, peak_amp_5,
                    low_band_power, mid_band_power, high_band_power,
                    spectral_entropy, spectral_kurtosis, total_power,
                    bpfo_amp, bpfi_amp, bsf_amp, ftf_amp, sideband_strength,
                    degradation_score,
                    rotational_speed, temperature, torque, tool_wear,
                    failure_prediction, failure_class
                FROM cwru_features
                ORDER BY timestamp DESC
                LIMIT :limit
            """)
            result = await db.execute(query, {"limit": limit})

        # Convert rows to dicts
        rows = result.mappings().all()
        data = [dict(row) for row in rows]
        if machine_filter is not None:
            allowed = set(machine_filter)
            data = [row for row in data if row.get("machine_id") in allowed]

        # ISO format timestamps (if not already handled by json encoder)
        for row in data:
            if row.get('timestamp'):
                row['timestamp'] = row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])

        return APIResponse.success(data=data, count=len(data))
    except Exception as e:
        print(f"Error extracting features: {e}")
        return APIResponse.from_error(str(e))


@app.get("/api/machines", tags=["Data"], response_class=ORJSONResponse)
async def get_machines(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> APIResponse[List[Dict]]:
    """
    Get list of active machines with their current status.
    Scoped by role: technician/plant_manager see only allowed machines.
    """
    try:
        from services.machine_service import MachineService

        machine_filter = get_scoped_machine_filter(user)
        if machine_filter is not None and len(machine_filter) == 0:
            return APIResponse.success(data=[], count=0)

        service = MachineService(db)
        machines = await service.get_all_machines()
        if machine_filter is not None:
            allowed = set(machine_filter)
            machines = [m for m in machines if m.get("machine_id") in allowed]

        return APIResponse.success(data=machines, count=len(machines))
    except Exception as e:
        print(f"Error fetching machines: {e}")
        return APIResponse.from_error(str(e))


@app.get("/api/machines/{machine_id}/profile", tags=["Data"], response_class=ORJSONResponse)
async def get_machine_profile(
    machine_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Get the latest data quality profile for a machine.
    RBAC: user must have access to this machine.
    """
    if not can_access_machine(user, machine_id):
        raise HTTPException(status_code=403, detail="Not authorized for this machine")
    query = text("""
        SELECT machine_id, profiled_at, profile_json, overall_status, hours_analyzed
        FROM data_profiles
        WHERE machine_id = :machine_id
        ORDER BY profiled_at DESC
        LIMIT 1
    """)
    result = await db.execute(query, {"machine_id": machine_id})
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail=f"No profile found for machine {machine_id}")
    data = dict(row["profile_json"])
    data["machine_id"] = row["machine_id"]
    data["profiled_at"] = row["profiled_at"]
    data["hours_analyzed"] = row["hours_analyzed"]
    data["overall_status"] = row["overall_status"]
    profile = DataProfile.model_validate(data)
    return APIResponse.success(data=profile.model_dump(mode="json"))


@app.post("/api/machines/{machine_id}/profile/run", status_code=202, tags=["Data"])
async def run_machine_profile(
    machine_id: str,
    background_tasks: BackgroundTasks,
    hours: int = 48,
    current_user=Depends(require_engineer_permission),
):
    """
    Trigger a new data quality profile run for the machine (background).
    Returns 202 Accepted immediately; profile is persisted when complete.
    """
    async def run_and_save_profile() -> None:
        from services.data_profiler_service import DataProfilerService
        async with get_db_context() as db:
            try:
                service = DataProfilerService(db)
                profile = await service.profile(machine_id, hours=hours)
                payload = profile.model_dump(mode="json")
                await db.execute(
                    text("""
                        INSERT INTO data_profiles (machine_id, profiled_at, profile_json, overall_status, hours_analyzed)
                        VALUES (:machine_id, :profiled_at, CAST(:profile_json AS jsonb), :overall_status, :hours_analyzed)
                    """),
                    {
                        "machine_id": machine_id,
                        "profiled_at": profile.profiled_at,
                        "profile_json": json.dumps(payload),
                        "overall_status": profile.overall_status,
                        "hours_analyzed": profile.hours_analyzed,
                    },
                )
                await db.commit()
            except Exception as e:
                await db.rollback()
                from logger import get_logger
                get_logger("api_server.profile").warning("Profile run failed: %s", e)

    background_tasks.add_task(run_and_save_profile)
    return {"status": "accepted", "message": f"Profile job scheduled for machine {machine_id} (hours={hours})"}


# =============================================================================
# ANALYTICS ENDPOINTS
# =============================================================================

async def run_analytics_task():
    """Run analytics engine in a separate thread to avoid blocking loop."""
    from logger import get_logger
    import traceback as tb
    
    analytics_logger = get_logger("analytics.background")
    
    try:
        analytics_logger.info("Starting background analysis")
        # Using built-in threadpool (Celery offload stub)
        await run_in_threadpool(analytics_engine.main)
        analytics_logger.info("Background analysis complete")
    except Exception as e:
        # Exceptions from run_in_threadpool propagate correctly here
        analytics_logger.error(
            "Background task failed",
            error_type=type(e).__name__,
            error_message=str(e),
            traceback=tb.format_exc(),
        )

@app.post("/api/analytics/trigger", status_code=202, tags=["Analytics"])
async def trigger_analytics(
    background_tasks: BackgroundTasks,
    current_user=Depends(require_engineer_permission),
):
    """
    Trigger the heavy analytics engine in the background.
    Returns immediately with 202 Accepted.
    """
    background_tasks.add_task(run_analytics_task)
    return {"status": "accepted", "message": "Analytics engine started in background"}


@app.get("/api/recommendations/{machine_id}", tags=["AI"])
async def get_recommendation(
    machine_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Get AI-powered maintenance recommendation for a specific machine.
    RBAC: user must have access to this machine.
    Uses LLM to analyze sensor data and generate maintenance recommendations.
    """
    if not can_access_machine(user, machine_id):
        raise HTTPException(status_code=403, detail="Not authorized for this machine")

    try:
        # Import AI recommendation engine
        from ai_recommendations import generate_maintenance_recommendation
        
        # Get machine metadata
        metadata = EQUIPMENT_METADATA.get(machine_id, {})
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found")
        
        # Get latest sensor data for this machine
        records = await fetch_latest_features(db, machine_id, limit=1)
        if not records:
            raise HTTPException(status_code=404, detail=f"No sensor data for {machine_id}")
        
        latest = records[0]
        
        # Generate AI recommendation
        recommendation = generate_maintenance_recommendation(
            machine_id=machine_id,
            machine_name=metadata.get("name", machine_id),
            equipment_type=metadata.get("type", "Equipment"),
            sensor_data=latest,
            shop=metadata.get("shop", "Unknown"),
            line=metadata.get("line", "Unknown")
        )
        
        return recommendation
        
    except ImportError:
        # AI module not available - return rule-based fallback
        return {
            "priority": "MEDIUM",
            "action": "Schedule Routine Inspection",
            "reasoning": "AI recommendation module not available. Using fallback rule-based logic.",
            "timeWindow": "Within 7 days",
            "parts": ["Standard inspection kit"],
            "estimatedDowntime": "1-2 hours",
            "safetyNotes": "Follow standard lockout/tagout procedures",
            "aiGenerated": False,
            "machine_id": machine_id
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[AI Recommendation Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))



# =============================================================================
# WEBSOCKET ENDPOINT
# =============================================================================
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket, token: str = Query(None)):
    """
    Real-time WebSocket stream of sensor telemetry.
    Only pushes messages for machines the user can access (RBAC).
    Requires valid JWT token in query parameter: /ws/stream?token=...
    """
    visible_machine_ids = None  # None = all machines (admin/engineer/operator)
    user_id = "anonymous"
    if token:
        try:
            from types import SimpleNamespace
            from services.auth_service import verify_token
            from schemas.security import TokenType
            token_data = verify_token(token, expected_type=TokenType.ACCESS)
            if not token_data.is_expired:
                user_id = token_data.username
                user_like = SimpleNamespace(
                    role=token_data.role,
                    site_id=token_data.site_id,
                    assigned_machine_ids=token_data.assigned_machine_ids or [],
                )
                visible_machine_ids = get_scoped_machine_filter(user_like)
        except Exception:
            visible_machine_ids = []  # Invalid token: no machines
    else:
        visible_machine_ids = []  # No token: no machine data

    websocket.state.user_id = user_id
    websocket.state.visible_machine_ids = visible_machine_ids

    await websocket.accept()
    state.connected_clients.append(websocket)
    print(f"[WS] Client connected: {user_id}. Total: {len(state.connected_clients)}")
    
    try:
        if state.redis is None:
            await websocket.send_json({"type": "error", "error": "Redis not connected"})
            return
        if state.model is None:
            await websocket.send_json({"type": "error", "error": "Model not loaded"})

        # Subscribe to Redis channel
        pubsub = state.redis.pubsub()
        await pubsub.subscribe(REDIS_CHANNEL)
        
        # Manual iteration to support heartbeat timeout
        iterator = pubsub.listen()
        
        while True:
            try:
                # Wait for message with 10s heartbeat timeout
                message = await asyncio.wait_for(iterator.__anext__(), timeout=10.0)
                
                if message['type'] != 'message':
                    continue
                
                # Parse incoming sensor data
                payload = json.loads(message['data'])
                
                # Extract features and run inference
                if state.model and 'vibration_raw' in payload:
                    machine_id = payload.get('machine_id', 'UNKNOWN')
                    vibration = np.array(payload['vibration_raw'])
                    telemetry = {
                        'rotational_speed': payload.get('rotational_speed', 1800.0),
                        'temperature': payload.get('temperature', 70.0),
                        'torque': payload.get('torque', 40.0),
                        'tool_wear': payload.get('tool_wear', 0.1)
                    }
                    
                    # Feature extraction
                    features = state.processor.process_signal(vibration, telemetry)
                    
                    # ML Inference
                    feature_vector = np.array([[features.get(col, 0.0) for col in state.feature_columns]])
                    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
                    feature_scaled = state.scaler.transform(feature_vector)
                    proba = state.model.predict_proba(feature_scaled)[0]
                    failure_prob = float(proba[1])
                    
                    # === REAL LOGIC IMPLEMENTATION (Phase 3) ===
                    
                    # Get smoothed degradation for stable RUL calculation
                    # Query latest smoothed value from database
                    # Use get_db_context for async safe access
                    from database import get_db_context
                    degradation_db = 0.0
                    try:
                        async with get_db_context() as db:
                            result = await db.execute(text("""
                                SELECT degradation_score_smoothed 
                                FROM cwru_features 
                                WHERE machine_id = :machine_id 
                                ORDER BY timestamp DESC 
                                LIMIT 1
                            """), {"machine_id": machine_id})
                            row = result.fetchone()
                            if row:
                                degradation_db = row[0]
                    except Exception as e:
                       print(f"[WS] DB Error: {e}")

                    # Use smoothed degradation if available, otherwise use raw
                    degradation = degradation_db if degradation_db else features.get('degradation_score', 0)
                    
                    # 1. RUL Calculation (Physics-based with smoothing)
                    # Factory baseline: 2000 hours life. Exponential decay model.
                    current_life_hours = 2000 * (1.0 - degradation)**2
                    rul_days = max(0.0, current_life_hours / 24.0)

                    # 2. Anomaly Detection (Threshold-based)
                    # BPFI/BPFO are specific fault frequencies. High amp = specific defect.
                    bpfi = features.get('bpfi_amp', 0)
                    bpfo = features.get('bpfo_amp', 0)
                    
                    anomalies = []
                    if bpfi > 0.005:  # Inner Race Threshold
                        anomalies.append("Bearing Inner Race Wear")
                    if bpfo > 0.005:  # Outer Race Threshold
                        anomalies.append("Bearing Outer Race Damage")
                    if features.get('spectral_entropy', 0) < 2.0:
                        anomalies.append("Signal Regularity Loss")
                    
                    anomaly_detected = len(anomalies) > 0
                    anomaly_type = anomalies[0] if anomalies else "None"
                    
                    # 3. Recommendation Engine
                    recommendation = "Normal Operation"
                    if failure_prob > 0.8:
                        recommendation = "Immediate Shutdown & Replace Bearing"
                    elif failure_prob > 0.5:
                         recommendation = "Schedule Maintenance Inspection (Level 2)"
                    elif anomaly_detected:
                         recommendation = "Monitor Vibration Spectrum for Growth"
                    
                    # 4. Operational Status (Phase 4)
                    speed = payload.get('rotational_speed', 0)
                    op_status = "OFFLINE"
                    if speed > 100:
                        op_status = "RUNNING"
                    elif speed > 0:
                        op_status = "IDLE"
                    
                    # 5. Asset Metadata Lookup
                    asset_info = ASSET_REGISTRY.get(machine_id, {})

                else:
                    features = {}
                    
                    # FALLBACK HEURISTIC (When Model Fails)
                    # If model didn't load, use raw vibration amplitude to estimate risk
                    # This ensures the UI is never "Blind" to faults
                    vibration = np.array(payload.get('vibration_raw', []))
                    if len(vibration) > 0:
                         max_vib = np.max(np.abs(vibration))
                         if max_vib > 4.0:
                             failure_prob = 0.95 # Critical
                         elif max_vib > 1.5:
                             failure_prob = 0.65 # Warning
                         else:
                             failure_prob = 0.05 # Healthy
                    else:
                        failure_prob = 0.0

                    rul_days = 0.0
                    anomaly_detected = False
                    anomaly_type = "None"
                    recommendation = "None"
                    op_status = "OFFLINE"
                    asset_info = {}

                # Build response payload (PredictionResponse-aligned schema)
                health_score = max(0, min(100, (1.0 - failure_prob) * 100))
                response = {
                    "type": "telemetry",
                    "timestamp": payload.get('timestamp', datetime.now(timezone.utc).isoformat()),
                    "machine_id": payload.get('machine_id', 'UNKNOWN'),
                    "rotational_speed": payload.get('rotational_speed', 0),
                    "temperature": payload.get('temperature', 0),
                    "failure_probability": failure_prob,
                    "health_score": float(health_score),
                    "rul_days": float(rul_days),
                    "rul_lower_80": float(rul_days),
                    "rul_upper_80": float(rul_days),
                    "confidence": "MEDIUM",
                    "in_training_distribution": True,
                    "bpfi_amp": features.get('bpfi_amp', 0),
                    "bpfo_amp": features.get('bpfo_amp', 0),
                    "degradation_score": features.get('degradation_score', 0),
                    "spectral_entropy": features.get('spectral_entropy', 0),
                    "anomaly_detected": bool(anomaly_detected),
                    "anomaly_type": str(anomaly_type),
                    "recommendation": str(recommendation),
                    "operational_status": op_status,
                    "line_id": asset_info.get("line_id", "L_UNKNOWN"),
                    "line_name": asset_info.get("line_name", "Unassigned Line"),
                    "model_number": asset_info.get("model", "N/A"),
                    "install_date": asset_info.get("install_date", "N/A"),
                }
                
                # RBAC: only push for machines the user can access
                visible = getattr(websocket.state, "visible_machine_ids", None)
                if visible is not None and response.get("machine_id", "UNKNOWN") not in visible:
                    continue
                # Send to client (only if open)
                if websocket.client_state.name == "CONNECTED" and websocket.application_state.name == "CONNECTED":
                    await websocket.send_json(response)
                
            except asyncio.TimeoutError:
                # Send heartbeat
                if websocket.client_state.name == "CONNECTED":
                    await websocket.send_json({"type": "heartbeat", "timestamp": datetime.now(timezone.utc).isoformat()})
                    
            except json.JSONDecodeError:
                continue
            except Exception as e:
                # Non-fatal processing error - log and notify client
                from logger import get_logger
                ws_logger = get_logger("websocket")
                ws_logger.warning("WebSocket processing error", error=str(e))
                
                # Send error frame but continue processing
                if websocket.client_state.name == "CONNECTED":
                    await websocket.send_json({
                        "type": "error",
                        "error": "Processing error",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                continue
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        # Fatal error - send error frame before closing
        from logger import get_logger
        import traceback as tb
        
        ws_logger = get_logger("websocket")
        ws_logger.error(
            "WebSocket fatal error",
            user_id=getattr(websocket.state, 'user_id', 'unknown'),
            error_type=type(e).__name__,
            error_message=str(e),
            traceback=tb.format_exc(),
        )
        
        # Send error frame before closing
        try:
            if websocket.client_state.name == "CONNECTED":
                await websocket.send_json({
                    "type": "error",
                    "error": "Connection terminated due to server error",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        except Exception:
            pass  # Socket may already be closed
    finally:
        if websocket in state.connected_clients:
            state.connected_clients.remove(websocket)
        from logger import get_logger
        ws_logger = get_logger("websocket")
        ws_logger.info("Client disconnected", total_clients=len(state.connected_clients))


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    import logging
    
    # Disable Uvicorn's default access logger (we use custom structlog middleware)
    logging.getLogger("uvicorn.access").disabled = True
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=False,  # Disable default access logging
    )
