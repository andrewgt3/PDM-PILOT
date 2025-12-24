#!/usr/bin/env python3
"""
Enterprise Features API Module
Provides endpoints for alarms, work orders, MTBF/MTTR, and shift schedules.

Import this into api_server.py to add the routes.
"""

import os
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Query, Depends, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# Rate Limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

# Import validated schemas with strict security rules
from schemas import (
    AlarmCreateValidated,
    WorkOrderCreateValidated,
    WorkOrderUpdateValidated,
    TelemetryDataValidated,
    BulkTelemetryValidated,
)

# Import authentication utilities
from auth_utils import (
    get_password_hash,
    verify_password,
    create_access_token,
    decode_access_token,
)

# Create router for enterprise features
enterprise_router = APIRouter(prefix="/api/enterprise", tags=["Enterprise"])

# Initialize rate limiter for this module
limiter = Limiter(key_func=get_remote_address)

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/enterprise/token")

DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")


def get_db():
    """Get database connection."""
    return psycopg2.connect(DATABASE_URL)


# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================

# Temporary hardcoded credentials (TODO: Replace with database lookup)
TEMP_USERS = {
    "admin": {
        "username": "admin",
        "hashed_password": get_password_hash("secret123"),
        "role": "admin",
        "email": "admin@plantagi.com"
    }
}


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Token payload data."""
    username: Optional[str] = None
    role: Optional[str] = None


@enterprise_router.post("/token", response_model=Token, tags=["Authentication"])
@limiter.limit("5 per minute")  # Strict rate limit to prevent password guessing
async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2-compatible token login endpoint.
    
    Accepts username and password, returns JWT access token.
    
    **Rate Limited:** 5 requests per minute to prevent brute-force attacks.
    
    **Temporary Credentials (for testing):**
    - Username: `admin`
    - Password: `secret123`
    
    **Usage:**
    ```
    curl -X POST "http://localhost:8000/api/enterprise/token" \\
         -H "Content-Type: application/x-www-form-urlencoded" \\
         -d "username=admin&password=secret123"
    ```
    """
    # Look up user (temporary hardcoded, will be replaced with DB lookup)
    user = TEMP_USERS.get(form_data.username)
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Verify password
    if not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Create access token
    access_token = create_access_token(
        data={
            "sub": user["username"],
            "role": user["role"],
            "email": user["email"]
        }
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Dependency to get current user from JWT token.
    
    Usage in protected endpoints:
        @router.get("/protected")
        async def protected_route(current_user: dict = Depends(get_current_user)):
            return {"user": current_user["sub"]}
    """
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"}
    )
    
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    
    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception
    
    return payload


async def get_current_admin_user(current_user: dict = Depends(get_current_user)) -> dict:
    """
    Dependency to require admin role.
    
    Usage in admin-only endpoints:
        @router.delete("/dangerous")
        async def admin_only(user: dict = Depends(get_current_admin_user)):
            ...
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    return current_user


# =============================================================================
# PYDANTIC MODELS (Legacy aliases - use validated versions)
# =============================================================================
# These are aliased to the validated versions for backward compatibility
AlarmCreate = AlarmCreateValidated
WorkOrderCreate = WorkOrderCreateValidated
WorkOrderUpdate = WorkOrderUpdateValidated


# =============================================================================
# TELEMETRY INGESTION ENDPOINT (NEW - with strict validation)
# =============================================================================
@enterprise_router.post("/telemetry", tags=["Telemetry"])
async def ingest_telemetry(
    data: TelemetryDataValidated,
    current_user: dict = Depends(get_current_user)  # Requires authentication
):
    """
    Ingest validated telemetry data from sensors.
    
    Validation Rules (422 error if violated):
    - machine_id: 3-50 chars, alphanumeric with underscores/hyphens
    - temperature: -50°C to 200°C
    - vibration_x: >= 0 (non-negative)
    - rotational_speed: >= 0 RPM
    """
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            INSERT INTO sensor_readings 
            (machine_id, timestamp, rotational_speed, temperature, torque, tool_wear)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id, timestamp
        """, (
            data.machine_id,
            data.timestamp or datetime.now(timezone.utc),
            data.rotational_speed,
            data.temperature,
            data.torque,
            data.tool_wear
        ))
        
        result = cursor.fetchone()
        conn.commit()
        cursor.close()
        conn.close()
        
        return {
            "status": "accepted",
            "reading_id": result['id'],
            "timestamp": result['timestamp'].isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@enterprise_router.post("/telemetry/bulk", tags=["Telemetry"])
async def ingest_bulk_telemetry(
    data: BulkTelemetryValidated,
    current_user: dict = Depends(get_current_user)  # Requires authentication
):
    """
    Bulk ingest validated telemetry data (up to 1000 readings).
    
    Each reading is validated with the same rules as single ingestion.
    """
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Prepare bulk insert
        values = []
        for reading in data.readings:
            values.append((
                reading.machine_id,
                reading.timestamp or datetime.now(timezone.utc),
                reading.rotational_speed,
                reading.temperature,
                reading.torque,
                reading.tool_wear
            ))
        
        # Execute bulk insert
        cursor.executemany("""
            INSERT INTO sensor_readings 
            (machine_id, timestamp, rotational_speed, temperature, torque, tool_wear)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, values)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return {
            "status": "accepted",
            "readings_ingested": len(data.readings)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ALARM ENDPOINTS
# =============================================================================
@enterprise_router.get("/alarms")
async def get_alarms(
    machine_id: str = Query(None),
    active_only: bool = Query(True),
    limit: int = Query(50, ge=1, le=200)
):
    """Get alarms, optionally filtered by machine and status."""
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = "SELECT * FROM pdm_alarms WHERE 1=1"
        params = []
        
        if machine_id:
            query += " AND machine_id = %s"
            params.append(machine_id)
        
        if active_only:
            query += " AND active = TRUE"
        
        query += " ORDER BY timestamp DESC LIMIT %s"
        params.append(limit)
        
        cursor.execute(query, params)
        alarms = cursor.fetchall()
        
        # Convert timestamps
        for alarm in alarms:
            if alarm['timestamp']:
                alarm['timestamp'] = alarm['timestamp'].isoformat()
            if alarm['acknowledged_at']:
                alarm['acknowledged_at'] = alarm['acknowledged_at'].isoformat()
        
        cursor.close()
        conn.close()
        
        return {"count": len(alarms), "data": alarms}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@enterprise_router.post("/alarms")
async def create_alarm(
    alarm: AlarmCreate,
    current_user: dict = Depends(get_current_user)  # Requires authentication
):
    """Create a new alarm (typically called by monitoring system)."""
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Generate alarm ID
        cursor.execute("SELECT COUNT(*) FROM pdm_alarms WHERE DATE(timestamp) = CURRENT_DATE")
        count = cursor.fetchone()['count']
        alarm_id = f"ALM-{datetime.now().strftime('%Y%m%d')}-{count + 1:04d}"
        
        cursor.execute("""
            INSERT INTO pdm_alarms 
            (alarm_id, machine_id, severity, code, message, trigger_type, trigger_value, threshold_value)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
        """, (alarm_id, alarm.machine_id, alarm.severity, alarm.code, alarm.message,
              alarm.trigger_type, alarm.trigger_value, alarm.threshold_value))
        
        new_alarm = cursor.fetchone()
        conn.commit()
        cursor.close()
        conn.close()
        
        return new_alarm
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@enterprise_router.post("/alarms/{alarm_id}/acknowledge")
async def acknowledge_alarm(
    alarm_id: str,
    acknowledged_by: str = "System",
    current_user: dict = Depends(get_current_user)  # Requires authentication
):
    """Acknowledge an alarm."""
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            UPDATE pdm_alarms 
            SET acknowledged = TRUE, acknowledged_by = %s, acknowledged_at = NOW()
            WHERE alarm_id = %s
            RETURNING *
        """, (acknowledged_by, alarm_id))
        
        alarm = cursor.fetchone()
        if not alarm:
            raise HTTPException(status_code=404, detail="Alarm not found")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return alarm
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@enterprise_router.post("/alarms/{alarm_id}/resolve")
async def resolve_alarm(
    alarm_id: str,
    current_user: dict = Depends(get_current_user)  # Requires authentication
):
    """Mark an alarm as resolved/inactive."""
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            UPDATE pdm_alarms 
            SET active = FALSE, resolved_at = NOW()
            WHERE alarm_id = %s
            RETURNING *
        """, (alarm_id,))
        
        alarm = cursor.fetchone()
        if not alarm:
            raise HTTPException(status_code=404, detail="Alarm not found")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return alarm
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# WORK ORDER ENDPOINTS
# =============================================================================
@enterprise_router.get("/work-orders")
async def get_work_orders(
    machine_id: str = Query(None),
    status: str = Query(None),
    limit: int = Query(50, ge=1, le=200)
):
    """Get work orders with optional filters."""
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = "SELECT * FROM work_orders WHERE 1=1"
        params = []
        
        if machine_id:
            query += " AND machine_id = %s"
            params.append(machine_id)
        
        if status:
            query += " AND status = %s"
            params.append(status)
        
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        
        cursor.execute(query, params)
        orders = cursor.fetchall()
        
        # Convert timestamps
        for order in orders:
            for field in ['created_at', 'started_at', 'completed_at']:
                if order.get(field):
                    order[field] = order[field].isoformat()
            if order.get('scheduled_date'):
                order['scheduled_date'] = order['scheduled_date'].isoformat()
        
        cursor.close()
        conn.close()
        
        return {"count": len(orders), "data": orders}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@enterprise_router.post("/work-orders")
async def create_work_order(
    wo: WorkOrderCreate,
    current_user: dict = Depends(get_current_user)  # Requires authentication
):
    """Create a new work order."""
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Generate work order ID
        year = datetime.now().year
        cursor.execute("SELECT COUNT(*) FROM work_orders WHERE EXTRACT(YEAR FROM created_at) = %s", (year,))
        count = cursor.fetchone()['count']
        wo_id = f"WO-{year}-{count + 1:04d}"
        
        cursor.execute("""
            INSERT INTO work_orders 
            (work_order_id, machine_id, title, description, priority, work_type, 
             scheduled_date, estimated_duration_hours)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
        """, (wo_id, wo.machine_id, wo.title, wo.description, wo.priority, wo.work_type,
              wo.scheduled_date, wo.estimated_duration_hours))
        
        new_order = cursor.fetchone()
        conn.commit()
        
        # Convert timestamps for response
        for field in ['created_at', 'started_at', 'completed_at']:
            if new_order.get(field):
                new_order[field] = new_order[field].isoformat()
        
        cursor.close()
        conn.close()
        
        return new_order
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@enterprise_router.patch("/work-orders/{work_order_id}")
async def update_work_order(
    work_order_id: str,
    update: WorkOrderUpdate,
    current_user: dict = Depends(get_current_user)  # Requires authentication
):
    """Update a work order's status, assignment, or notes."""
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Build dynamic update
        updates = []
        params = []
        
        if update.status:
            updates.append("status = %s")
            params.append(update.status)
            
            # Auto-set timestamps based on status
            if update.status == 'in_progress':
                updates.append("started_at = NOW()")
            elif update.status == 'completed':
                updates.append("completed_at = NOW()")
        
        if update.assigned_to:
            updates.append("assigned_to = %s")
            params.append(update.assigned_to)
        
        if update.notes:
            updates.append("notes = %s")
            params.append(update.notes)
        
        if update.actual_duration_hours:
            updates.append("actual_duration_hours = %s")
            params.append(update.actual_duration_hours)
        
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        params.append(work_order_id)
        query = f"UPDATE work_orders SET {', '.join(updates)} WHERE work_order_id = %s RETURNING *"
        
        cursor.execute(query, params)
        order = cursor.fetchone()
        
        if not order:
            raise HTTPException(status_code=404, detail="Work order not found")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return order
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MTBF/MTTR ENDPOINTS
# =============================================================================
@enterprise_router.get("/reliability/{machine_id}")
async def get_reliability_metrics(machine_id: str):
    """
    Calculate MTBF and MTTR for a specific machine.
    Uses failure_events table and work_orders for calculations.
    """
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get failure event count in last 365 days
        cursor.execute("""
            SELECT COUNT(*) as failure_count
            FROM failure_events 
            WHERE machine_id = %s 
            AND timestamp > NOW() - INTERVAL '365 days'
        """, (machine_id,))
        failure_data = cursor.fetchone()
        failure_count = failure_data['failure_count'] or 0
        
        # Calculate total operating hours (assume 24/7 minus downtime)
        # For now, estimate based on time range
        days_tracked = 365
        hours_per_day = 20  # Assume 20 hours/day average
        total_uptime = days_tracked * hours_per_day
        
        # Get average repair time from completed work orders
        cursor.execute("""
            SELECT AVG(actual_duration_hours) as avg_mttr
            FROM work_orders 
            WHERE machine_id = %s 
            AND status = 'completed'
            AND actual_duration_hours IS NOT NULL
        """, (machine_id,))
        mttr_data = cursor.fetchone()
        mttr = mttr_data['avg_mttr'] or 3.0  # Default 3 hours if no data
        
        # Calculate MTBF
        if failure_count > 0:
            mtbf = total_uptime / failure_count
        else:
            mtbf = total_uptime  # No failures = full uptime
        
        # Calculate availability
        availability = (mtbf / (mtbf + mttr)) * 100 if (mtbf + mttr) > 0 else 99.0
        
        cursor.close()
        conn.close()
        
        return {
            "machine_id": machine_id,
            "mtbf_hours": round(mtbf, 1),
            "mttr_hours": round(mttr, 1),
            "availability_percent": round(availability, 2),
            "failure_count_ytd": failure_count,
            "total_uptime_hours": total_uptime,
            "calculation_period": "Last 365 days"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SHIFT SCHEDULE ENDPOINTS
# =============================================================================
@enterprise_router.get("/schedule")
async def get_schedule():
    """Get current shift schedule and production mode."""
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get shifts
        cursor.execute("SELECT * FROM shift_schedule WHERE is_active = TRUE ORDER BY start_time")
        shifts = cursor.fetchall()
        
        # Convert time objects to strings
        for shift in shifts:
            shift['start_time'] = str(shift['start_time'])
            shift['end_time'] = str(shift['end_time'])
        
        # Get current production mode
        cursor.execute("""
            SELECT * FROM production_mode 
            WHERE effective_until IS NULL OR effective_until > NOW()
            ORDER BY effective_from DESC 
            LIMIT 1
        """)
        mode = cursor.fetchone()
        
        # Determine current shift
        current_time = datetime.now().time()
        current_shift = None
        for shift in shifts:
            start = datetime.strptime(shift['start_time'], '%H:%M:%S').time()
            end = datetime.strptime(shift['end_time'], '%H:%M:%S').time()
            
            if start < end:
                if start <= current_time <= end:
                    current_shift = shift['shift_name']
                    break
            else:  # Overnight shift
                if current_time >= start or current_time <= end:
                    current_shift = shift['shift_name']
                    break
        
        cursor.close()
        conn.close()
        
        return {
            "shifts": shifts,
            "current_shift": current_shift,
            "production_mode": mode['mode'] if mode else 'normal',
            "weekly_hours": mode['weekly_hours'] if mode else 120,
            "wear_factor": mode['wear_factor'] if mode else 1.0
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@enterprise_router.post("/schedule/mode")
async def set_production_mode(
    mode: str,
    weekly_hours: float = 120,
    current_user: dict = Depends(get_current_admin_user)  # Admin only!
):
    """Set the current production mode (normal, overtime, reduced)."""
    if mode not in ['normal', 'overtime', 'reduced']:
        raise HTTPException(status_code=400, detail="Mode must be: normal, overtime, or reduced")
    
    wear_factors = {'normal': 1.0, 'overtime': 1.15, 'reduced': 0.85}
    
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Expire current mode
        cursor.execute("""
            UPDATE production_mode 
            SET effective_until = NOW() 
            WHERE effective_until IS NULL
        """)
        
        # Insert new mode
        cursor.execute("""
            INSERT INTO production_mode (mode, weekly_hours, wear_factor)
            VALUES (%s, %s, %s)
            RETURNING *
        """, (mode, weekly_hours, wear_factors[mode]))
        
        new_mode = cursor.fetchone()
        conn.commit()
        cursor.close()
        conn.close()
        
        return new_mode
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ALARM GENERATOR - Call this to check thresholds and create alarms
# =============================================================================
def check_and_create_alarms(machine_id: str, failure_probability: float, 
                            degradation_score: float, bpfi_amp: float, bpfo_amp: float):
    """
    Check sensor values against thresholds and create alarms if needed.
    Call this from the stream processing pipeline.
    """
    alarms_created = []
    
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check for existing active alarms to avoid duplicates
        cursor.execute("""
            SELECT code FROM pdm_alarms 
            WHERE machine_id = %s AND active = TRUE
        """, (machine_id,))
        existing_codes = [row['code'] for row in cursor.fetchall()]
        
        # Threshold checks
        checks = [
            {
                'condition': failure_probability > 0.8,
                'code': 'PDM-CRIT-001',
                'severity': 'critical',
                'message': f'Critical failure probability: {failure_probability*100:.1f}%',
                'trigger_type': 'failure_probability',
                'trigger_value': failure_probability,
                'threshold': 0.8
            },
            {
                'condition': failure_probability > 0.5 and failure_probability <= 0.8,
                'code': 'PDM-WARN-001',
                'severity': 'warning',
                'message': f'Elevated failure probability: {failure_probability*100:.1f}%',
                'trigger_type': 'failure_probability',
                'trigger_value': failure_probability,
                'threshold': 0.5
            },
            {
                'condition': degradation_score > 0.7,
                'code': 'PDM-WARN-002',
                'severity': 'warning',
                'message': f'High degradation score: {degradation_score*100:.1f}%',
                'trigger_type': 'degradation',
                'trigger_value': degradation_score,
                'threshold': 0.7
            },
            {
                'condition': bpfi_amp > 0.5,
                'code': 'PDM-INFO-001',
                'severity': 'info',
                'message': f'Inner race bearing fault amplitude elevated: {bpfi_amp:.4f}g',
                'trigger_type': 'bpfi_amplitude',
                'trigger_value': bpfi_amp,
                'threshold': 0.5
            }
        ]
        
        for check in checks:
            if check['condition'] and check['code'] not in existing_codes:
                # Generate alarm ID
                cursor.execute("SELECT COUNT(*) FROM pdm_alarms WHERE DATE(timestamp) = CURRENT_DATE")
                count = cursor.fetchone()['count']
                alarm_id = f"ALM-{datetime.now().strftime('%Y%m%d')}-{count + 1:04d}"
                
                cursor.execute("""
                    INSERT INTO pdm_alarms 
                    (alarm_id, machine_id, severity, code, message, trigger_type, trigger_value, threshold_value)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (alarm_id, machine_id, check['severity'], check['code'], check['message'],
                      check['trigger_type'], check['trigger_value'], check['threshold']))
                
                alarms_created.append(alarm_id)
        
        # Also check if we should record a failure event
        if failure_probability > 0.8:
            cursor.execute("""
                SELECT id FROM failure_events 
                WHERE machine_id = %s 
                AND timestamp > NOW() - INTERVAL '1 hour'
            """, (machine_id,))
            if not cursor.fetchone():
                cursor.execute("""
                    INSERT INTO failure_events 
                    (machine_id, event_type, failure_probability, degradation_score, bpfi_amp, bpfo_amp)
                    VALUES (%s, 'predicted_failure', %s, %s, %s, %s)
                """, (machine_id, failure_probability, degradation_score, bpfi_amp, bpfo_amp))
        
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"[Alarm Generator Error] {e}")
    
    return alarms_created


# Export for use in api_server.py
__all__ = ['enterprise_router', 'check_and_create_alarms']
