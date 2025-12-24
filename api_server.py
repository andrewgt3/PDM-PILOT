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

import os
import json
import asyncio
import uuid
import traceback
from datetime import datetime, timezone
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
import redis.asyncio as aioredis
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Rate Limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Load environment variables
load_dotenv()

# Import custom modules
from advanced_features import SignalProcessor, SAMPLE_RATE, N_SAMPLES


# =============================================================================
# CONFIGURATION
# =============================================================================
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_CHANNEL = 'sensor_stream'

DATABASE_URL = os.getenv('DATABASE_URL', None)

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
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, cleanup on shutdown."""
    print("=" * 60)
    print("GAIA API SERVER - Starting...")
    print("=" * 60)
    
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
    if state.redis:
        await state.redis.close()


# =============================================================================
# FASTAPI APP
# =============================================================================

# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["100 per minute"])

app = FastAPI(
    title="GAIA Predictive Maintenance API",
    description="Real-time WebSocket streaming and REST API for Industrial IoT predictive maintenance.",
    version="1.0.0",
    lifespan=lifespan
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
def get_db_connection():
    """Create PostgreSQL connection."""
    if not DATABASE_URL:
        raise Exception("Database not configured")
    return psycopg2.connect(DATABASE_URL)


def query_features(limit: int = 100, machine_id: str = None) -> List[Dict]:
    """Query cwru_features table."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        if machine_id:
            query = """
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
                WHERE machine_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """
            cursor.execute(query, (machine_id, limit))
        else:
            query = """
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
                LIMIT %s
            """
            cursor.execute(query, (limit,))
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Convert timestamps to ISO format
        for row in rows:
            if row['timestamp']:
                row['timestamp'] = row['timestamp'].isoformat()
        
        return rows
    except Exception as e:
        print(f"Database error: {e}")
        return []


def get_active_machines() -> List[Dict]:
    """Get list of active machines with latest status."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT DISTINCT ON (machine_id)
                machine_id,
                timestamp as last_seen,
                failure_prediction,
                COALESCE(degradation_score_smoothed, degradation_score, 0.0) as degradation_score
            FROM cwru_features
            ORDER BY machine_id, timestamp DESC
            LIMIT 10
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        machines = []
        for row in rows:
            prob = row['failure_prediction'] or 0.0
            degradation = row.get('degradation_score', 0.0) or 0.0
            
            # Simple RUL Calculation (matches WebSocket logic)
            # Using smoothed degradation for stability
            current_life_hours = 2000 * (1.0 - degradation)**2
            rul_days = max(0.0, current_life_hours / 24.0)
            
            status = "CRITICAL" if prob > 0.8 else "WARNING" if prob > 0.5 else "HEALTHY"
            
            # Get equipment metadata from lookup
            machine_id = row['machine_id']
            metadata = EQUIPMENT_METADATA.get(machine_id, {})
            
            machines.append({
                "machine_id": machine_id,
                "machine_name": metadata.get("name", machine_id),
                "shop": metadata.get("shop", "Unassigned Shop"),
                "line_name": metadata.get("line", "Unassigned Line"),
                "equipment_type": metadata.get("type", "Equipment"),
                "last_seen": row['last_seen'].isoformat() if row['last_seen'] else None,
                "failure_probability": prob,
                "rul_days": float(rul_days),
                "status": status
            })
        
        return machines
    except Exception as e:
        print(f"Database error: {e}")
        return []


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
    
    Tests database connectivity and returns appropriate status.
    No authentication required.
    """
    db_status = "offline"
    
    try:
        if DATABASE_URL:
            conn = psycopg2.connect(DATABASE_URL)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            conn.close()
            db_status = "online"
    except Exception as e:
        print(f"[Health Check] Database error: {e}")
        db_status = "offline"
    
    if db_status == "online":
        return JSONResponse(
            status_code=200,
            content={"status": "healthy", "database": "online"}
        )
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "database": "offline"}
        )


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
    machine_id: str = Query(None, description="Filter by machine ID")
):
    """
    Retrieve historical feature records from the database.
    
    Returns the latest N records from cwru_features table,
    optionally filtered by machine_id.
    """
    records = query_features(limit=limit, machine_id=machine_id)
    return {"count": len(records), "data": records}


@app.get("/api/machines", tags=["Data"])
async def get_machines():
    """
    Get list of active machines with their current status.
    
    Returns each machine's last seen timestamp and failure probability.
    """
    machines = get_active_machines()
    return {"count": len(machines), "data": machines}


@app.get("/api/recommendations/{machine_id}", tags=["AI"])
async def get_recommendation(machine_id: str):
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
        records = query_features(limit=1, machine_id=machine_id)
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
async def websocket_stream(websocket: WebSocket):
    """
    Real-time WebSocket stream of sensor telemetry.
    
    Subscribes to Redis sensor_stream channel and broadcasts
    processed data to connected clients at ~10Hz.
    """
    await websocket.accept()
    state.connected_clients.append(websocket)
    print(f"[WS] Client connected. Total: {len(state.connected_clients)}")
    
    try:
        if state.redis is None:
            await websocket.send_json({"error": "Redis not connected"})
            return
        
        # Subscribe to Redis channel
        pubsub = state.redis.pubsub()
        await pubsub.subscribe(REDIS_CHANNEL)
        
        async for message in pubsub.listen():
            if message['type'] != 'message':
                continue
            
            try:
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
                    cursor = state.conn.cursor()
                    cursor.execute("""
                        SELECT degradation_score_smoothed 
                        FROM cwru_features 
                        WHERE machine_id = %s 
                        ORDER BY timestamp DESC 
                        LIMIT 1
                    """, (machine_id,))
                    result = cursor.fetchone()
                    cursor.close()
                    
                    # Use smoothed degradation if available, otherwise use raw
                    degradation = result[0] if result else features.get('degradation_score', 0)
                    
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
