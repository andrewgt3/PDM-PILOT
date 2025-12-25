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
import traceback
import uuid
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
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import BackgroundTasks
from starlette.concurrency import run_in_threadpool
import analytics_engine
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Use centralized configuration
from config import get_settings
from database import check_database_health, get_db, init_database, shutdown_database
from dependencies import get_current_user
from schemas.response import APIResponse, ORJSONResponse
from auth_utils import decode_access_token

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
REDIS_CHANNEL = 'sensor_stream'

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
    
    # Load ML Model
    try:
        pipeline = joblib.load('gaia_model.pkl')
        state.model = pipeline['model']
        state.scaler = pipeline['scaler']
        state.feature_columns = pipeline['feature_columns']
        state.processor = SignalProcessor(sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES)
        print(f"✓ ML Model loaded: {len(state.feature_columns)} features")
    except Exception as e:
        print(f"⚠ ML Model not loaded: {e}")
    
    print("=" * 60)
    print("Server ready. Endpoints:")
    print("  - /docs           (Swagger UI)")
    print("  - /ws/stream      (WebSocket)")
    print("  - /api/features   (REST)")
    print("  - /api/machines   (REST)")
    print("  - /api/health     (REST)")
    print("=" * 60)
    
    yield  # Server runs here
    
    # Cleanup
    print("\nShutting down...")
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
    # Public endpoints
    if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
        return

    # WebSocket handshake usually doesn't have headers, handled separately or needs query param auth
    # For now, excluding it or letting it fail if using standard headers
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
    dependencies=[Depends(verify_global_auth)],
)

# Attach limiter to app state
app.state.limiter = limiter

# Rate limit exceeded handler (returns 429 Too Many Requests)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# =============================================================================
# GLOBAL EXCEPTION HANDLERS
# =============================================================================
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
    Logs full trace to console but returns clean error to user.
    """
    error_id = str(uuid.uuid4())
    
    # Log full error trace to console
    print(f"\n{'='*60}")
    print(f"[ERROR] Reference ID: {error_id}")
    print(f"[ERROR] Path: {request.url.path}")
    print(f"[ERROR] Method: {request.method}")
    print(f"{'='*60}")
    traceback.print_exc()
    print(f"{'='*60}\n")
    
    # Return clean response to user (no stack trace)
    return JSONResponse(
        status_code=500,
        content={"detail": f"An internal error occurred. Reference ID: {error_id}"}
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

# Include Enterprise API routes (alarms, work orders, MTBF/MTTR, schedules)
try:
    from enterprise_api import enterprise_router
    app.include_router(enterprise_router)
    print("✓ Enterprise API routes loaded")
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


@app.get("/api/features", tags=["Data"])
async def get_features(
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return"),
    machine_id: str = Query(None, description="Filter by machine ID"),
    db: AsyncSession = Depends(get_db)
) -> APIResponse[List[Dict]]:
    """
    Retrieve historical feature records.
    """
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
                    degradation_score, degradation_score_smoothed,
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
                    degradation_score, degradation_score_smoothed,
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
        
        # ISO format timestamps (if not already handled by json encoder)
        for row in data:
            if row.get('timestamp'):
                row['timestamp'] = row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])

        return APIResponse.success(data=data, count=len(data))
    except Exception as e:
        print(f"Error extracting features: {e}")
        return APIResponse.error(str(e))


@app.get("/api/machines", tags=["Data"], response_class=ORJSONResponse)
async def get_machines(db: AsyncSession = Depends(get_db)) -> APIResponse[List[Dict]]:
    """
    Get list of active machines with their current status.
    Calculations offloaded to SQL for performance.
    """
    try:
        # DISTINCT ON is Postgres specific
        # Calculate Status and RUL directly in DB
        query = text("""
            SELECT DISTINCT ON (machine_id)
                machine_id,
                timestamp as last_seen,
                failure_prediction,
                COALESCE(degradation_score_smoothed, degradation_score, 0.0) as degradation_score,
                
                -- RUL Calculation: 2000 * (1 - degradation)^2 / 24
                GREATEST(0, (2000 * POWER(1.0 - COALESCE(degradation_score_smoothed, degradation_score, 0.0), 2)) / 24.0) as rul_days,
                
                -- Status Logic
                CASE 
                    WHEN failure_prediction > 0.8 THEN 'CRITICAL'
                    WHEN failure_prediction > 0.5 THEN 'WARNING'
                    ELSE 'HEALTHY'
                END as status
                
            FROM cwru_features
            ORDER BY machine_id, timestamp DESC
            LIMIT 100
        """)
        
        result = await db.execute(query)
        rows = result.mappings().all()
        
        machines = []
        for row in rows:
            # Metadata lookup (in-memory)
            machine_id = row['machine_id']
            metadata = EQUIPMENT_METADATA.get(machine_id, {})
            
            machines.append({
                "machine_id": machine_id,
                "machine_name": metadata.get("name", machine_id),
                "shop": metadata.get("shop", "Unassigned Shop"),
                "line_name": metadata.get("line", "Unassigned Line"),
                "equipment_type": metadata.get("type", "Equipment"),
                "last_seen": row['last_seen'],
                "failure_probability": row['failure_prediction'],
                "rul_days": float(row['rul_days']),
                "status": row['status']
            })
            
        return APIResponse.success(data=machines, count=len(machines))
    except Exception as e:
        print(f"Error fetching machines: {e}")
        return APIResponse.error(str(e))



# =============================================================================
# ANALYTICS ENDPOINTS
# =============================================================================

async def run_analytics_task():
    """Run analytics engine in a separate thread to avoid blocking loop."""
    try:
        print("[Analytics] Starting background analysis...")
        # Stub for Celery worker offload
        # await run_in_threadpool(analytics_engine.main)
        # Using built-in threadpool for now as requested
        await run_in_threadpool(analytics_engine.main)
        print("[Analytics] Background analysis complete.")
    except Exception as e:
        print(f"[Analytics] Error in background task: {e}")

@app.post("/api/analytics/trigger", status_code=202, tags=["Analytics"])
async def trigger_analytics(background_tasks: BackgroundTasks):
    """
    Trigger the heavy analytics engine in the background.
    Returns immediately with 202 Accepted.
    """
    background_tasks.add_task(run_analytics_task)
    return {"status": "accepted", "message": "Analytics engine started in background"}


@app.get("/api/recommendations/{machine_id}", tags=["AI"])
async def get_recommendation(machine_id: str, db: AsyncSession = Depends(get_db)):
    """
    Get AI-powered maintenance recommendation for a specific machine.
    
    Uses LLM (OpenAI/Azure/Ollama) to analyze current sensor data and
    generate intelligent, contextual maintenance recommendations.
    
    The response includes:
    - priority: CRITICAL, HIGH, MEDIUM, or LOW
    - action: Recommended maintenance action
    - reasoning: AI explanation based on sensor data
    - timeWindow: When to perform maintenance
    - parts: List of required parts
    - estimatedDowntime: Expected duration
    - safetyNotes: Safety considerations
    """
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
    
    Subscribes to Redis sensor_stream channel and broadcasts
    processed data to connected clients at ~10Hz.
    
    Requires valid JWT token in query parameter: /ws/stream?token=...
    """
    # 1. Validate Token
    if not token:
        print("[WS] Connection rejected: No token provided")
        await websocket.close(code=1008, reason="Missing authentication token")
        return
        
    payload = decode_access_token(token)
    if not payload:
        print("[WS] Connection rejected: Invalid token")
        await websocket.close(code=1008, reason="Invalid authentication token")
        return
        
    # 2. Accept & Store User
    user_id = payload.get("sub")
    websocket.state.user_id = user_id
    
    await websocket.accept()
    state.connected_clients.append(websocket)
    print(f"[WS] Client connected: {user_id}. Total: {len(state.connected_clients)}")
    
    try:
        if state.redis is None:
            await websocket.send_json({"error": "Redis not connected"})
            return
        
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
                
                # Build response payload
                response = {
                    "timestamp": payload.get('timestamp', datetime.now(timezone.utc).isoformat()),
                    "machine_id": payload.get('machine_id', 'UNKNOWN'),
                    "rotational_speed": payload.get('rotational_speed', 0),
                    "temperature": payload.get('temperature', 0),
                    "failure_probability": failure_prob,
                    "bpfi_amp": features.get('bpfi_amp', 0),
                    "bpfo_amp": features.get('bpfo_amp', 0),
                    "degradation_score": features.get('degradation_score', 0),
                    "spectral_entropy": features.get('spectral_entropy', 0),
                    
                    # New Real-Time Intelligence (Phase 3)
                    "rul_days": float(rul_days),
                    "anomaly_detected": bool(anomaly_detected),
                    "anomaly_type": str(anomaly_type),
                    "recommendation": str(recommendation),
                    
                    # Asset Core (Phase 4)
                    "operational_status": op_status,
                    "line_id": asset_info.get("line_id", "L_UNKNOWN"),
                    "line_name": asset_info.get("line_name", "Unassigned Line"),
                    "model_number": asset_info.get("model", "N/A"),
                    "install_date": asset_info.get("install_date", "N/A"),
                }
                
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
                print(f"[WS] Processing error: {e}")
                # Send error to client for debugging
                await websocket.send_json({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "machine_id": "ERROR",
                    "error": str(e)
                })
                continue
                
    except WebSocketDisconnect:
        pass
    finally:
        state.connected_clients.remove(websocket)
        print(f"[WS] Client disconnected. Total: {len(state.connected_clients)}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
