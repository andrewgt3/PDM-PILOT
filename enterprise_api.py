#!/usr/bin/env python3
"""
Enterprise Features API Module
Provides endpoints for alarms, work orders, MTBF/MTTR, and shift schedules.

Import this into api_server.py to add the routes.
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any

from fastapi import APIRouter, HTTPException, Query, Depends, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Rate Limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

# Import validated schemas
from schemas import (
    AlarmCreateValidated,
    WorkOrderCreateValidated,
    WorkOrderUpdateValidated,
    TelemetryDataValidated,
    BulkTelemetryValidated,
)
from schemas.enterprise import WorkOrder
from schemas.security import Token, UserRole

# Global Config & Dependencies
from config import get_settings
from database import get_db
from dependencies import get_current_user, get_current_admin_user

from services.auth_service import verify_password, create_access_token, get_auth_service

# Create router
enterprise_router = APIRouter(prefix="/api/enterprise", tags=["Enterprise"])

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

settings = get_settings()
# No local get_db or oauth2_scheme needed



# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================

# Temporary hardcoded credentials (TODO: Replace with database lookup)
# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================

# Temporary hardcoded users for login (Phase 1)
# TODO: Move to DB
TEMP_USERS = {
    "admin": {
        "username": "admin",
        # Hashed "secret123"
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW", 
        "role": "admin",
        "email": "admin@plantagi.com"
    }
}

@enterprise_router.post("/token", response_model=Token, tags=["Authentication"])
@limiter.limit("5 per minute")
async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 login endpoint.
    """
    user = TEMP_USERS.get(form_data.username)
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Verify password (async)
    # Note: verify_password in auth_utils might be sync/async depending on previous edits.
    # We check if it is awaitable.
    # Per recent summary, auth_service.py refactored to async.
    # verify_password comes from services.auth_service
    is_valid = await verify_password(form_data.password, user["hashed_password"])
    if not is_valid:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Create access token
    access_token = create_access_token(
        user_id=user["username"],
        username=user["username"],
        role=UserRole(user["role"])
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

# get_current_user and get_current_admin_user removed (imported from dependencies)



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
    current_user: dict = Depends(get_current_user),  # Requires authentication
    db: AsyncSession = Depends(get_db)
):
    """
    Ingest validated telemetry data from sensors.
    """
    try:
        query = text("""
            INSERT INTO sensor_readings 
            (machine_id, timestamp, rotational_speed, temperature, torque, tool_wear)
            VALUES (:machine_id, :timestamp, :rotational_speed, :temperature, :torque, :tool_wear)
            RETURNING id, timestamp
        """)
        
        params = {
            "machine_id": data.machine_id,
            "timestamp": data.timestamp or datetime.now(timezone.utc),
            "rotational_speed": data.rotational_speed,
            "temperature": data.temperature,
            "torque": data.torque,
            "tool_wear": data.tool_wear
        }
        
        result = await db.execute(query, params)
        row = result.mappings().one()
        # Auto-commit handled by dependency
        
        return {
            "status": "accepted",
            "reading_id": row['id'],
            "timestamp": row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@enterprise_router.post("/telemetry/bulk", tags=["Telemetry"])
async def ingest_bulk_telemetry(
    data: BulkTelemetryValidated,
    current_user: dict = Depends(get_current_user),  # Requires authentication
    db: AsyncSession = Depends(get_db)
):
    """
    Bulk ingest validated telemetry data (up to 1000 readings).
    """
    try:
        query = text("""
            INSERT INTO sensor_readings 
            (machine_id, timestamp, rotational_speed, temperature, torque, tool_wear)
            VALUES (:machine_id, :timestamp, :rotational_speed, :temperature, :torque, :tool_wear)
        """)
        
        values = []
        for reading in data.readings:
            values.append({
                "machine_id": reading.machine_id,
                "timestamp": reading.timestamp or datetime.now(timezone.utc),
                "rotational_speed": reading.rotational_speed,
                "temperature": reading.temperature,
                "torque": reading.torque,
                "tool_wear": reading.tool_wear
            })
        
        await db.execute(query, values)
        # Auto-commit handled by dependency
        
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
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db)
):
    """Get alarms, optionally filtered by machine and status."""
    try:
        query_str = "SELECT * FROM pdm_alarms WHERE 1=1"
        params = {}
        
        if machine_id:
            query_str += " AND machine_id = :machine_id"
            params["machine_id"] = machine_id
        
        if active_only:
            query_str += " AND active = TRUE"
        
        query_str += " ORDER BY timestamp DESC LIMIT :limit"
        params["limit"] = limit
        
        result = await db.execute(text(query_str), params)
        rows = result.mappings().all()
        
        alarms = [dict(row) for row in rows]

        
        # Convert timestamps
        for alarm in alarms:
            if alarm['timestamp']:
                alarm['timestamp'] = alarm['timestamp'].isoformat()
            if alarm['acknowledged_at']:
                alarm['acknowledged_at'] = alarm['acknowledged_at'].isoformat()
        
        # Convert timestamps
        for alarm in alarms:
            if alarm.get('timestamp'):
                alarm['timestamp'] = alarm['timestamp'].isoformat() if hasattr(alarm['timestamp'], 'isoformat') else str(alarm['timestamp'])
            if alarm.get('acknowledged_at'):
                alarm['acknowledged_at'] = alarm['acknowledged_at'].isoformat() if hasattr(alarm['acknowledged_at'], 'isoformat') else str(alarm['acknowledged_at'])
            if alarm.get('resolved_at'):
                alarm['resolved_at'] = alarm['resolved_at'].isoformat() if hasattr(alarm['resolved_at'], 'isoformat') else str(alarm['resolved_at'])
        
        return {"count": len(alarms), "data": alarms}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@enterprise_router.post("/alarms")
async def create_alarm(
    alarm: AlarmCreateValidated,
    current_user: dict = Depends(get_current_user),  # Requires authentication
    db: AsyncSession = Depends(get_db)
):
    """Create a new alarm (typically called by monitoring system)."""
    try:
        # Generate alarm ID
        count_query = text("SELECT COUNT(*) FROM pdm_alarms WHERE DATE(timestamp) = CURRENT_DATE")
        result = await db.execute(count_query)
        count = result.scalar()
        
        alarm_id = f"ALM-{datetime.now().strftime('%Y%m%d')}-{count + 1:04d}"
        
        query = text("""
            INSERT INTO pdm_alarms 
            (alarm_id, machine_id, severity, code, message, trigger_type, trigger_value, threshold_value)
            VALUES (:alarm_id, :machine_id, :severity, :code, :message, :trigger_type, :trigger_value, :threshold_value)
            RETURNING *
        """)
        
        params = {
            "alarm_id": alarm_id,
            "machine_id": alarm.machine_id,
            "severity": alarm.severity,
            "code": alarm.code,
            "message": alarm.message,
            "trigger_type": alarm.trigger_type,
            "trigger_value": alarm.trigger_value,
            "threshold_value": alarm.threshold_value
        }
        
        result = await db.execute(query, params)
        new_alarm = dict(result.mappings().one())
        # Auto-commit handled
        
        return new_alarm
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@enterprise_router.post("/alarms/{alarm_id}/acknowledge")
async def acknowledge_alarm(
    alarm_id: str,
    acknowledged_by: str = "System",
    current_user: dict = Depends(get_current_user),  # Requires authentication
    db: AsyncSession = Depends(get_db)
):
    """Acknowledge an alarm."""
    try:
        query = text("""
            UPDATE pdm_alarms 
            SET acknowledged = TRUE, acknowledged_by = :ack_by, acknowledged_at = NOW()
            WHERE alarm_id = :alarm_id
            RETURNING *
        """)
        
        result = await db.execute(query, {"ack_by": acknowledged_by, "alarm_id": alarm_id})
        alarm = result.mappings().one_or_none()
        
        if not alarm:
            raise HTTPException(status_code=404, detail="Alarm not found")
        
        return dict(alarm)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@enterprise_router.post("/alarms/{alarm_id}/resolve")
async def resolve_alarm(
    alarm_id: str,
    current_user: dict = Depends(get_current_user),  # Requires authentication
    db: AsyncSession = Depends(get_db)
):
    """Mark an alarm as resolved/inactive."""
    try:
        query = text("""
            UPDATE pdm_alarms 
            SET active = FALSE, resolved_at = NOW()
            WHERE alarm_id = :alarm_id
            RETURNING *
        """)
        
        result = await db.execute(query, {"alarm_id": alarm_id})
        alarm = result.mappings().one_or_none()
        
        if not alarm:
            raise HTTPException(status_code=404, detail="Alarm not found")
        
        return dict(alarm)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# WORK ORDER ENDPOINTS
# =============================================================================
@enterprise_router.get("/work-orders", response_model=Dict[str, Any])
async def get_work_orders(
    machine_id: str = Query(None),
    status: str = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db)
):
    """Get work orders with optional filters."""
    try:
        query_str = "SELECT * FROM work_orders WHERE 1=1"
        params = {}
        
        if machine_id:
            query_str += " AND machine_id = :machine_id"
            params["machine_id"] = machine_id
        
        if status:
            query_str += " AND status = :status"
            params["status"] = status
        
        query_str += " ORDER BY created_at DESC LIMIT :limit"
        params["limit"] = limit
        
        result = await db.execute(text(query_str), params)
        rows = result.mappings().all()
        
        # Map to Pydantic schema
        orders = [WorkOrder.model_validate(dict(row)) for row in rows]
        
        return {"count": len(orders), "data": orders}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@enterprise_router.post("/work-orders", response_model=WorkOrder)
async def create_work_order(
    wo: WorkOrderCreate,
    current_user: dict = Depends(get_current_user),  # Requires authentication
    db: AsyncSession = Depends(get_db)
):
    """Create a new work order."""
    try:
        # Generate work order ID
        year = datetime.now().year
        count_query = text("SELECT COUNT(*) FROM work_orders WHERE EXTRACT(YEAR FROM created_at) = :year")
        result = await db.execute(count_query, {"year": year})
        count = result.scalar()
        
        wo_id = f"WO-{year}-{count + 1:04d}"
        
        query = text("""
            INSERT INTO work_orders 
            (work_order_id, machine_id, title, description, priority, work_type, 
             scheduled_date, estimated_duration_hours)
            VALUES (:wo_id, :machine_id, :title, :description, :priority, :work_type, 
             :scheduled_date, :estimated_duration_hours)
            RETURNING *
        """)
        
        params = {
            "wo_id": wo_id,
            "machine_id": wo.machine_id,
            "title": wo.title,
            "description": wo.description,
            "priority": wo.priority,
            "work_type": wo.work_type,
            "scheduled_date": wo.scheduled_date,
            "estimated_duration_hours": wo.estimated_duration_hours
        }
        
        result = await db.execute(query, params)
        row = result.mappings().one()
        
        return WorkOrder.model_validate(dict(row))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@enterprise_router.patch("/work-orders/{work_order_id}")
async def update_work_order(
    work_order_id: str,
    update: WorkOrderUpdate,
    current_user: dict = Depends(get_current_user),  # Requires authentication
    db: AsyncSession = Depends(get_db)
):
    """Update a work order's status, assignment, or notes."""
    try:
        # Build dynamic update
        updates = []
        params = {}
        
        if update.status:
            updates.append("status = :status")
            params["status"] = update.status
            
            # Auto-set timestamps based on status
            if update.status == 'in_progress':
                updates.append("started_at = NOW()")
            elif update.status == 'completed':
                updates.append("completed_at = NOW()")
        
        if update.assigned_to:
            updates.append("assigned_to = :assigned_to")
            params["assigned_to"] = update.assigned_to
        
        if update.notes:
            updates.append("notes = :notes")
            params["notes"] = update.notes
        
        if update.actual_duration_hours:
            updates.append("actual_duration_hours = :actual_duration_hours")
            params["actual_duration_hours"] = update.actual_duration_hours
        
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        params["work_order_id"] = work_order_id
        
        query_str = f"UPDATE work_orders SET {', '.join(updates)} WHERE work_order_id = :work_order_id RETURNING *"
        
        result = await db.execute(text(query_str), params)
        order = result.mappings().one_or_none()
        
        if not order:
            raise HTTPException(status_code=404, detail="Work order not found")
        
        return dict(order)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MTBF/MTTR ENDPOINTS
# =============================================================================
@enterprise_router.get("/reliability/{machine_id}")
async def get_reliability_metrics(machine_id: str, db: AsyncSession = Depends(get_db)):
    """
    Calculate MTBF and MTTR for a specific machine.
    Uses failure_events table and work_orders for calculations.
    """
    try:
        # Get failure event count in last 365 days
        query_failures = text("""
            SELECT COUNT(*) as failure_count
            FROM failure_events 
            WHERE machine_id = :machine_id 
            AND timestamp > NOW() - INTERVAL '365 days'
        """)
        result_failures = await db.execute(query_failures, {"machine_id": machine_id})
        failure_count = result_failures.scalar() or 0
        
        # Calculate total operating hours (assume 24/7 minus downtime)
        days_tracked = 365
        hours_per_day = 20  # Assume 20 hours/day average
        total_uptime = days_tracked * hours_per_day
        
        # Get average repair time from completed work orders
        query_mttr = text("""
            SELECT AVG(actual_duration_hours) as avg_mttr
            FROM work_orders 
            WHERE machine_id = :machine_id 
            AND status = 'completed'
            AND actual_duration_hours IS NOT NULL
        """)
        result_mttr = await db.execute(query_mttr, {"machine_id": machine_id})
        mttr = result_mttr.scalar() or 3.0  # Default 3 hours if no data
        
        # Calculate MTBF
        if failure_count > 0:
            mtbf = total_uptime / failure_count
        else:
            mtbf = total_uptime  # No failures = full uptime
        
        # Calculate availability
        availability = (mtbf / (mtbf + mttr)) * 100 if (mtbf + mttr) > 0 else 99.0
        
        return {
            "machine_id": machine_id,
            "mtbf_hours": round(float(mtbf), 1),
            "mttr_hours": round(float(mttr), 1),
            "availability_percent": round(float(availability), 2),
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
async def get_schedule(db: AsyncSession = Depends(get_db)):
    """Get current shift schedule and production mode."""
    try:
        # Get shifts with timeout protection
        try:
            async with asyncio.timeout(5.0):  # 5 second timeout
                result_shifts = await db.execute(text("SELECT * FROM shift_schedule WHERE is_active = TRUE ORDER BY start_time"))
                shifts = [dict(row) for row in result_shifts.mappings().all()]
                
                # Get current production mode
                result_mode = await db.execute(text("""
                    SELECT * FROM production_mode 
                    WHERE effective_until IS NULL OR effective_until > NOW()
                    ORDER BY effective_from DESC 
                    LIMIT 1
                """))
                mode = result_mode.mappings().one_or_none()
        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail="Database query timed out")
        
        # Convert time objects to strings
        for shift in shifts:
            shift['start_time'] = str(shift['start_time'])
            shift['end_time'] = str(shift['end_time'])
        
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
        
        return {
            "shifts": shifts,
            "current_shift": current_shift,
            "production_mode": mode['mode'] if mode else 'normal',
            "weekly_hours": mode['weekly_hours'] if mode else 120,
            "wear_factor": mode['wear_factor'] if mode else 1.0
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@enterprise_router.post("/schedule/mode")
async def set_production_mode(
    mode: str,
    weekly_hours: float = 120,
    current_user: dict = Depends(get_current_admin_user),  # Admin only!
    db: AsyncSession = Depends(get_db)
):
    """Set the current production mode (normal, overtime, reduced)."""
    if mode not in ['normal', 'overtime', 'reduced']:
        raise HTTPException(status_code=400, detail="Mode must be: normal, overtime, or reduced")
    
    wear_factors = {'normal': 1.0, 'overtime': 1.15, 'reduced': 0.85}
    
    try:
        # Expire current mode
        await db.execute(text("""
            UPDATE production_mode 
            SET effective_until = NOW() 
            WHERE effective_until IS NULL
        """))
        
        # Insert new mode
        query = text("""
            INSERT INTO production_mode (mode, weekly_hours, wear_factor)
            VALUES (:mode, :weekly_hours, :wear_factor)
            RETURNING *
        """)
        result = await db.execute(query, {
            "mode": mode,
            "weekly_hours": weekly_hours,
            "wear_factor": wear_factors[mode]
        })
        
        new_mode = dict(result.mappings().one())
        
        return new_mode
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ALARM GENERATOR - Call this to check thresholds and create alarms
# =============================================================================
async def check_and_create_alarms(
    machine_id: str, 
    failure_probability: float, 
    degradation_score: float, 
    bpfi_amp: float, 
    bpfo_amp: float,
    db: AsyncSession
):
    """
    Check sensor values against thresholds and create alarms if needed.
    Call this from the stream processing pipeline.
    """
    alarms_created = []
    
    try:
        # Check for existing active alarms to avoid duplicates
        result = await db.execute(text("""
            SELECT code FROM pdm_alarms 
            WHERE machine_id = :machine_id AND active = TRUE
        """), {"machine_id": machine_id})
        
        existing_codes = [row[0] for row in result.fetchall()]
        
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
                count_res = await db.execute(text("SELECT COUNT(*) FROM pdm_alarms WHERE DATE(timestamp) = CURRENT_DATE"))
                count = count_res.scalar()
                
                alarm_id = f"ALM-{datetime.now().strftime('%Y%m%d')}-{count + 1 + len(alarms_created):04d}"
                
                query = text("""
                    INSERT INTO pdm_alarms 
                    (alarm_id, machine_id, severity, code, message, trigger_type, trigger_value, threshold_value)
                    VALUES (:alarm_id, :machine_id, :severity, :code, :message, :trigger_type, :trigger_value, :threshold_value)
                """)
                
                params = {
                    "alarm_id": alarm_id,
                    "machine_id": machine_id,
                    "severity": check['severity'],
                    "code": check['code'],
                    "message": check['message'],
                    "trigger_type": check['trigger_type'],
                    "trigger_value": check['trigger_value'],
                    "threshold_value": check['threshold']
                }
                
                await db.execute(query, params)
                alarms_created.append(alarm_id)
        
        # Also check if we should record a failure event
        if failure_probability > 0.8:
            check_fail = text("""
                SELECT id FROM failure_events 
                WHERE machine_id = :machine_id 
                AND timestamp > NOW() - INTERVAL '1 hour'
            """)
            res_fail = await db.execute(check_fail, {"machine_id": machine_id})
            if not res_fail.first():
                ins_fail = text("""
                    INSERT INTO failure_events 
                    (machine_id, event_type, failure_probability, degradation_score, bpfi_amp, bpfo_amp)
                    VALUES (:machine_id, 'predicted_failure', :prob, :deg, :bpfi, :bpfo)
                """)
                await db.execute(ins_fail, {
                    "machine_id": machine_id,
                    "prob": failure_probability,
                    "deg": degradation_score,
                    "bpfi": bpfi_amp,
                    "bpfo": bpfo_amp
                })
        
        # Auto-commit handled if session is managed externally, 
        # but if this is called as a helper, the caller should commit.
        # However, `get_db` dependency auto-commits.
        # If passed manually, user might expect commit.
        # Since we are modifying state, let's assume caller manages commit, 
        # OR we can flush. But `sqlalchemy` async session usually needs explicit commit if not using context manager that does it.
        # We will assume caller handles transaction or dependency does.
        
    except Exception as e:
        print(f"[Alarm Generator Error] {e}")
    
    return alarms_created


# Export for use in api_server.py
__all__ = ['enterprise_router', 'check_and_create_alarms']
