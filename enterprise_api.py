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

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
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
from dependencies import (
    get_current_user,
    get_current_admin_user,
    get_scoped_machine_filter,
    require_operator_permission,
    require_engineer_permission,
    require_admin_permission,
    require_role,
)
from schemas.security import User, UserRole
from services.rbac_service import can_access_machine

from services.auth_service import verify_password, create_access_token, get_auth_service
from middleware.audit_logger import get_client_ip, write_audit_entry

# Create router
enterprise_router = APIRouter(prefix="/api/enterprise", tags=["Enterprise"])
labeling_router = APIRouter(prefix="/api/labeling", tags=["Labeling"])
alerts_router = APIRouter(prefix="/api/alerts", tags=["Alerts"])
drift_router = APIRouter(prefix="/api/drift", tags=["Drift"])
models_router = APIRouter(prefix="/api/models", tags=["Models"])
onboarding_router = APIRouter(prefix="/api/onboarding", tags=["Onboarding"])
verify_router = APIRouter(prefix="/api", tags=["Onboarding"])
admin_router = APIRouter(prefix="/api/admin", tags=["Admin"])

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

# Demo user for development only (not used in production/staging).
# TODO: Move to DB for production.
TEMP_USERS = {
    "admin": {
        "username": "admin",
        # Hashed "secret123" - regenerated for bcrypt compatibility
        "hashed_password": "$2b$12$LIHnanXRW.tNQMk9IXfYE.G1sET8AO32Thl/M8poeZULPvkI0FJP.",
        "role": "admin",
        "email": "admin@plantagi.com"
    }
}

@enterprise_router.post("/token", response_model=Token, tags=["Authentication"])
@limiter.limit("5 per minute")
async def login_for_access_token(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """
    OAuth2 login endpoint.
    """
    ip_address = get_client_ip(request)
    # Demo user exists only in development; in production/staging do not seed TEMP_USERS.
    user = TEMP_USERS.get(form_data.username) if settings.environment == "development" else None

    if not user:
        await write_audit_entry(
            db,
            ip_address=ip_address,
            action="login_failed",
            username=form_data.username,
            details={"reason": "user_not_found"},
        )
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"}
        )

    is_valid = await verify_password(form_data.password, user["hashed_password"])
    if not is_valid:
        await write_audit_entry(
            db,
            ip_address=ip_address,
            action="login_failed",
            username=form_data.username,
            user_id=user["username"],
            details={"reason": "invalid_password"},
        )
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"}
        )

    site_id: Optional[str] = user.get("site_id")
    assigned_machine_ids: List[str] = list(user.get("assigned_machine_ids") or [])
    # If DB has users and machine_assignments, load site_id and assignments for JWT
    try:
        result = await db.execute(
            text("SELECT id, site_id FROM users WHERE username = :username"),
            {"username": form_data.username},
        )
        row = result.mappings().first()
        if row:
            site_id = site_id or (row["site_id"] if row.get("site_id") else None)
            uid = row["id"]
            assgn = await db.execute(
                text("SELECT machine_id FROM machine_assignments WHERE user_id = :uid"),
                {"uid": uid},
            )
            assigned_machine_ids = [r["machine_id"] for r in assgn.mappings().fetchall()]
    except Exception:
        pass

    access_token = create_access_token(
        user_id=user["username"],
        username=user["username"],
        role=UserRole(user["role"]),
        site_id=site_id,
        assigned_machine_ids=assigned_machine_ids or None,
    )

    await write_audit_entry(
        db,
        ip_address=ip_address,
        action="login",
        user_id=user["username"],
        username=user["username"],
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 1800,
    }

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
    current_user: dict = Depends(require_operator_permission),
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
    current_user: dict = Depends(require_operator_permission),
    db: AsyncSession = Depends(get_db)
):
    """
    Bulk ingest validated telemetry data (up to 1000 readings).
    Uses TelemetryService for business logic encapsulation.
    """
    try:
        from services.telemetry_service import TelemetryService
        
        service = TelemetryService(db)
        count = await service.process_batch(data)
        
        return {
            "status": "accepted",
            "readings_ingested": count
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
    current_user: User = Depends(require_operator_permission),
    db: AsyncSession = Depends(get_db)
):
    """Get alarms, optionally filtered by machine and status. RBAC: only visible machines."""
    try:
        machine_filter = get_scoped_machine_filter(current_user)
        if machine_filter is not None and len(machine_filter) == 0:
            return {"count": 0, "data": []}
        if machine_id and machine_filter is not None and machine_id not in machine_filter:
            return {"count": 0, "data": []}

        query_str = "SELECT * FROM pdm_alarms WHERE 1=1"
        params = {}
        if machine_id:
            query_str += " AND machine_id = :machine_id"
            params["machine_id"] = machine_id
        if machine_filter is not None and not machine_id:
            if not machine_filter:
                return {"count": 0, "data": []}
            placeholders = ", ".join([f":m{i}" for i in range(len(machine_filter))])
            query_str += f" AND machine_id IN ({placeholders})"
            for i, mid in enumerate(machine_filter):
                params[f"m{i}"] = mid
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
    current_user: dict = Depends(require_engineer_permission),
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
    current_user: dict = Depends(require_operator_permission),
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
    current_user: dict = Depends(require_operator_permission),
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
    current_user: User = Depends(require_operator_permission),
    db: AsyncSession = Depends(get_db)
):
    """Get work orders. technician: only assigned to them; plant_manager: their site; admin/engineer: all."""
    try:
        query_str = "SELECT * FROM work_orders WHERE 1=1"
        params = {}
        if current_user.role == UserRole.TECHNICIAN:
            query_str += " AND assigned_to = :assigned_to"
            params["assigned_to"] = current_user.username
        elif current_user.role == UserRole.PLANT_MANAGER:
            machine_filter = get_scoped_machine_filter(current_user)
            if machine_filter is None or len(machine_filter) == 0:
                return {"count": 0, "data": []}
            placeholders = ", ".join([f":wo_m{i}" for i in range(len(machine_filter))])
            query_str += f" AND machine_id IN ({placeholders})"
            for i, mid in enumerate(machine_filter):
                params[f"wo_m{i}"] = mid
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
        orders = [WorkOrder.model_validate(dict(row)) for row in rows]
        return {"count": len(orders), "data": orders}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@enterprise_router.post("/work-orders", response_model=WorkOrder)
async def create_work_order(
    wo: WorkOrderCreate,
    current_user: User = Depends(require_role(["admin", "engineer", "technician"])),
    db: AsyncSession = Depends(get_db)
):
    """Create a new work order. Technician: only for their assigned machines."""
    if current_user.role == UserRole.TECHNICIAN:
        if not current_user.assigned_machine_ids or wo.machine_id not in current_user.assigned_machine_ids:
            raise HTTPException(status_code=403, detail="Not authorized to create work order for this machine")
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
    current_user: dict = Depends(require_engineer_permission),
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
async def get_reliability_metrics(
    machine_id: str,
    current_user: User = Depends(require_operator_permission),
    db: AsyncSession = Depends(get_db)
):
    """
    Calculate MTBF and MTTR for a specific machine.
    RBAC: user must have access to this machine.
    """
    if not can_access_machine(current_user, machine_id):
        raise HTTPException(status_code=403, detail="Not authorized for this machine")
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
async def get_schedule(
    current_user: dict = Depends(require_operator_permission),
    db: AsyncSession = Depends(get_db)
):
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
    current_user: dict = Depends(require_engineer_permission),
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


# =============================================================================
# LABELING (HUMAN-IN-THE-LOOP)
# =============================================================================

from fastapi import BackgroundTasks

# Pydantic body for submit label
class SubmitLabelBody(BaseModel):
    label: int  # 0 or 1
    notes: Optional[str] = None


class FeedbackFromAlarmBody(BaseModel):
    alarm_id: str
    label: int  # 0 = False Alarm, 1 = Confirmed Issue


@labeling_router.post("/feedback-from-alarm")
async def feedback_from_alarm(
    body: FeedbackFromAlarmBody,
    current_user: dict = Depends(require_operator_permission),
    db: AsyncSession = Depends(get_db),
):
    """Create a labeling task from an alarm and submit the label (0=False Alarm, 1=Confirmed Issue)."""
    if body.label not in (0, 1):
        raise HTTPException(status_code=400, detail="label must be 0 or 1")
    try:
        from labeling_engine import create_labeling_task_from_alarm, submit_label
        task_id = await create_labeling_task_from_alarm(db, body.alarm_id)
        # If task was already completed, do not submit again (labels has unique task_id)
        check = await db.execute(
            text("SELECT status FROM labeling_tasks WHERE task_id = :tid"),
            {"tid": task_id},
        )
        row = check.mappings().first()
        if row and row["status"] == "completed":
            return {"status": "ok", "already_labeled": True}
        submitted_by = getattr(current_user, "username", None) or (current_user.get("username") if isinstance(current_user, dict) else None) or "unknown"
        await submit_label(db, task_id, body.label, submitted_by, notes="feedback-from-alarm")
        return {"status": "ok"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@labeling_router.get("/tasks")
async def get_labeling_tasks(
    machine_id: Optional[str] = Query(None),
    status: str = Query("pending"),
    current_user: dict = Depends(require_operator_permission),
    db: AsyncSession = Depends(get_db),
):
    """List labeling tasks with optional machine_id and status filters."""
    try:
        query_str = "SELECT task_id, anomaly_event_id, machine_id, feature_snapshot_json, created_at, status FROM labeling_tasks WHERE 1=1"
        params = {}
        if machine_id:
            query_str += " AND machine_id = :machine_id"
            params["machine_id"] = machine_id
        if status:
            query_str += " AND status = :status"
            params["status"] = status
        query_str += " ORDER BY created_at DESC"
        result = await db.execute(text(query_str), params)
        rows = result.mappings().all()
        tasks = []
        for row in rows:
            d = dict(row)
            if d.get("created_at"):
                d["created_at"] = d["created_at"].isoformat() if hasattr(d["created_at"], "isoformat") else str(d["created_at"])
            tasks.append(d)
        return {"count": len(tasks), "data": tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@labeling_router.post("/tasks/from-anomaly/{anomaly_event_id}")
async def create_task_from_anomaly(
    anomaly_event_id: str,
    current_user: dict = Depends(require_operator_permission),
    db: AsyncSession = Depends(get_db),
):
    """Create a labeling task from an anomaly detection event if not already present."""
    try:
        from labeling_engine import create_labeling_task
        # Check if task already exists for this anomaly
        check = await db.execute(
            text("SELECT task_id FROM labeling_tasks WHERE anomaly_event_id = :aid"),
            {"aid": anomaly_event_id},
        )
        existing = check.mappings().first()
        if existing:
            return {"task_id": existing["task_id"], "created": False}
        task_id = await create_labeling_task(db, anomaly_event_id)
        return {"task_id": task_id, "created": True}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@labeling_router.post("/tasks/{task_id}/label")
async def submit_task_label(
    task_id: str,
    body: SubmitLabelBody,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(require_operator_permission),
    db: AsyncSession = Depends(get_db),
):
    """Submit a label (0=False Alarm, 1=Confirmed Issue) for a task. Triggers retrain at milestones."""
    if body.label not in (0, 1):
        raise HTTPException(status_code=400, detail="label must be 0 or 1")
    try:
        from labeling_engine import submit_label, get_labeled_dataset, label_coverage_report
        submitted_by = getattr(current_user, "username", None) or (current_user.get("username") if isinstance(current_user, dict) else None) or "unknown"
        result = await submit_label(db, task_id, body.label, submitted_by, body.notes)
        machine_id = result["machine_id"]

        # Count labeled rows for this machine (join labels + labeling_tasks)
        count_result = await db.execute(
            text("""
                SELECT COUNT(*) AS n FROM labels l
                JOIN labeling_tasks lt ON l.task_id = lt.task_id
                WHERE lt.machine_id = :mid
            """),
            {"mid": machine_id},
        )
        count = count_result.scalar() or 0

        # Trigger retrain only when count equals a milestone (shadow deployment via retraining_service)
        if count in (50, 100, 200, 500):
            async def _schedule_retrain(mid: str, cnt: int):
                from pathlib import Path
                from database import get_db_context
                from labeling_engine import get_labeled_dataset as get_ds
                from services.retraining_service import trigger_retraining

                async with get_db_context() as session:
                    df = await get_ds(session, mid, min_labels=50)
                if df is None or len(df) < 50:
                    return
                if "label" in df.columns and "anomaly" not in df.columns:
                    df = df.rename(columns={"label": "anomaly"})
                out_dir = Path("data")
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                csv_path = out_dir / f"labeled_export_{mid}_{ts}.csv"
                df.to_csv(csv_path, index=False)
                reason = f"label_milestone_{cnt}"
                trigger_retraining(mid, reason, data_path=str(csv_path))

            background_tasks.add_task(_schedule_retrain, machine_id, count)

        return {"status": "ok", "machine_id": machine_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@labeling_router.get("/coverage/{machine_id}")
async def get_labeling_coverage(
    machine_id: str,
    current_user: dict = Depends(require_operator_permission),
    db: AsyncSession = Depends(get_db),
):
    """Return label coverage report for the machine."""
    try:
        from labeling_engine import label_coverage_report
        report = await label_coverage_report(db, machine_id)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ALERTS (tiered alert engine)
# =============================================================================

class AcknowledgeAlertBody(BaseModel):
    notes: Optional[str] = None


@alerts_router.get("/active")
async def get_active_alerts(
    current_user: dict = Depends(require_operator_permission),
):
    """Return all currently active alerts (state != HEALTHY) with state and duration."""
    try:
        from services.alert_engine import get_active_alerts as get_active
        from datetime import datetime, timezone
        rows = get_active()
        now = datetime.now(timezone.utc)
        for r in rows:
            entered = r.get("entered_at")
            if entered:
                if isinstance(entered, str):
                    try:
                        entered = datetime.fromisoformat(entered.replace("Z", "+00:00"))
                    except Exception:
                        entered = None
                if entered:
                    r["duration_seconds"] = int((now - entered).total_seconds())
        return {"count": len(rows), "data": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@alerts_router.post("/{machine_id}/acknowledge")
async def acknowledge_alert(
    machine_id: str,
    body: AcknowledgeAlertBody,
    current_user: dict = Depends(require_operator_permission),
    db: AsyncSession = Depends(get_db),
):
    """Acknowledge the current alert for a machine with user and notes."""
    try:
        username = getattr(current_user, "username", None) or (current_user.get("username") if isinstance(current_user, dict) else None) or "unknown"
        await db.execute(
            text("""
                UPDATE alert_current_state
                SET acknowledged_at = NOW(), acknowledged_by = :by, notes = COALESCE(:notes, notes)
                WHERE machine_id = :mid
            """),
            {"mid": machine_id, "by": username, "notes": body.notes or ""},
        )
        await db.commit()
        from services.alert_engine import get_state
        state = get_state(machine_id)
        if not state:
            raise HTTPException(status_code=404, detail="No alert state found for this machine")
        return state
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@alerts_router.get("/{machine_id}/history")
async def get_alert_history(
    machine_id: str,
    limit: int = Query(100, ge=1, le=500),
    current_user: dict = Depends(require_operator_permission),
    db: AsyncSession = Depends(get_db),
):
    """Return state transition history for the machine."""
    try:
        result = await db.execute(
            text("""
                SELECT from_state, to_state, at_timestamp, prediction_snapshot
                FROM alert_state_history
                WHERE machine_id = :mid
                ORDER BY at_timestamp DESC
                LIMIT :lim
            """),
            {"mid": machine_id, "lim": limit},
        )
        rows = result.mappings().all()
        data = []
        for r in rows:
            d = dict(r)
            if d.get("at_timestamp"):
                d["at_timestamp"] = d["at_timestamp"].isoformat() if hasattr(d["at_timestamp"], "isoformat") else str(d["at_timestamp"])
            data.append(d)
        return {"count": len(data), "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Drift (model drift reports)
# =============================================================================


@drift_router.get("/{machine_id}/latest")
async def get_drift_latest(
    machine_id: str,
    current_user: dict = Depends(require_engineer_permission),
    db: AsyncSession = Depends(get_db)
):
    """Return the latest drift report for the given machine from drift_reports. 404 if none."""
    result = await db.execute(
        text("""
            SELECT machine_id, checked_at, worst_psi, drifted_features_json,
                   prediction_drift_pct, overall_status, recommended_action
            FROM drift_reports
            WHERE machine_id = :mid
            ORDER BY checked_at DESC
            LIMIT 1
        """),
        {"mid": machine_id},
    )
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="No drift report found for this machine")
    r = dict(row)
    if r.get("checked_at") and hasattr(r["checked_at"], "isoformat"):
        r["checked_at"] = r["checked_at"].isoformat()
    dfj = r.pop("drifted_features_json", None)
    r["drifted_features"] = dfj if isinstance(dfj, list) else (json.loads(dfj) if isinstance(dfj, str) else [])
    return r


@drift_router.get("/{machine_id}/history")
async def get_drift_history(
    machine_id: str,
    days: int = Query(30, ge=1, le=365),
    current_user: dict = Depends(require_engineer_permission),
    db: AsyncSession = Depends(get_db)
):
    """Return drift reports for the machine in the given date range (from drift_reports)."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    result = await db.execute(
        text("""
            SELECT machine_id, checked_at, worst_psi, drifted_features_json,
                   prediction_drift_pct, overall_status, recommended_action
            FROM drift_reports
            WHERE machine_id = :mid AND checked_at >= :since
            ORDER BY checked_at DESC
        """),
        {"mid": machine_id, "since": since},
    )
    rows = result.mappings().fetchall()
    out = []
    for r in rows:
        row = dict(r)
        if row.get("checked_at") and hasattr(row["checked_at"], "isoformat"):
            row["checked_at"] = row["checked_at"].isoformat()
        dfj = row.pop("drifted_features_json", None)
        row["drifted_features"] = dfj if isinstance(dfj, list) else (json.loads(dfj) if isinstance(dfj, str) else [])
        out.append(row)
    return out


@drift_router.post("/{machine_id}/run", status_code=202)
async def post_drift_run(
    machine_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(require_engineer_permission),
):
    """Enqueue a drift check for the machine; returns 202 Accepted."""
    from services.drift_monitor_service import DriftMonitorService
    def _run():
        svc = DriftMonitorService()
        svc.run_drift_check(machine_id)
    background_tasks.add_task(_run)
    return {"message": f"Drift check enqueued for {machine_id}"}


# =============================================================================
# Models (shadow report, promote, rollback)
# =============================================================================


@models_router.get("/shadow-report/{machine_id}")
async def get_shadow_report(
    machine_id: str,
    current_user: dict = Depends(require_engineer_permission),
    db: AsyncSession = Depends(get_db)
):
    """Return 24-hour shadow comparison for the machine: agreement rate and divergence counts."""
    since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    result = await db.execute(
        text("""
            SELECT production_prediction, staging_prediction, agree
            FROM shadow_predictions
            WHERE machine_id = :mid AND timestamp >= :since
            ORDER BY timestamp DESC
        """),
        {"mid": machine_id, "since": since},
    )
    rows = result.mappings().all()
    if not rows:
        return {
            "machine_id": machine_id,
            "window_hours": 24,
            "total_count": 0,
            "agreement_rate": None,
            "divergence_staging_critical_production_healthy": 0,
            "divergence_production_critical_staging_healthy": 0,
        }
    total = len(rows)
    agreed = sum(1 for r in rows if r.get("agree") is True)
    staging_crit_prod_healthy = sum(
        1 for r in rows
        if (r.get("staging_prediction") or 0) > 0.5 and (r.get("production_prediction") or 0) <= 0.5
    )
    prod_crit_staging_healthy = sum(
        1 for r in rows
        if (r.get("production_prediction") or 0) > 0.5 and (r.get("staging_prediction") or 0) <= 0.5
    )
    return {
        "machine_id": machine_id,
        "window_hours": 24,
        "total_count": total,
        "agreement_rate": round(agreed / total, 4) if total else None,
        "divergence_staging_critical_production_healthy": staging_crit_prod_healthy,
        "divergence_production_critical_staging_healthy": prod_crit_staging_healthy,
    }


@models_router.post("/promote/{run_id}")
async def promote_staging(
    run_id: str,
    request: Request,
    current_user: dict = Depends(require_admin_permission),
    db: AsyncSession = Depends(get_db),
):
    """Promote the current Staging model to Production. Admin only."""
    from services.retraining_service import promote_staging_to_production, _load_registry
    ok = promote_staging_to_production(run_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Promotion failed: staging model not found or run_id mismatch")
    reg = _load_registry()
    await write_audit_entry(
        db,
        ip_address=get_client_ip(request),
        action="model_promotion",
        user_id=getattr(current_user, "id", None) and str(current_user.id) or None,
        username=getattr(current_user, "username", None),
        resource_type="model",
        resource_id=run_id,
        after_value={"run_id": run_id, "production_run_id": reg.get("production_run_id")},
        details={"run_id": run_id, "promoted_by": getattr(current_user, "username", None)},
    )
    return {"status": "ok", "message": f"Staging run_id={run_id} promoted to production"}


@models_router.post("/rollback")
async def rollback_production(
    request: Request,
    current_user: dict = Depends(require_admin_permission),
    db: AsyncSession = Depends(get_db),
):
    """Revert Production to the previous version. Admin only."""
    from services.retraining_service import rollback_production as do_rollback, _load_registry
    reg_before = _load_registry()
    ok = do_rollback()
    if not ok:
        raise HTTPException(status_code=400, detail="Rollback failed: no previous production backup")
    reg_after = _load_registry()
    await write_audit_entry(
        db,
        ip_address=get_client_ip(request),
        action="model_rollback",
        user_id=getattr(current_user, "id", None) and str(current_user.id) or None,
        username=getattr(current_user, "username", None),
        resource_type="model",
        resource_id=reg_after.get("production_run_id"),
        before_value={"production_run_id": reg_before.get("production_run_id")},
        after_value={"production_run_id": reg_after.get("production_run_id")},
        details={"rolled_back_by": getattr(current_user, "username", None)},
    )
    return {"status": "ok", "message": "Production reverted to previous version"}


# =============================================================================
# ADMIN — Users and assignments (for admin UI)
# =============================================================================

@admin_router.get("/users")
async def get_admin_users(
    current_user: User = Depends(require_admin_permission),
    db: AsyncSession = Depends(get_db),
):
    """
    List all users with roles and machine assignments. Admin only.
    Used for admin UI to manage technician assignments.
    """
    try:
        result = await db.execute(text("""
            SELECT id, username, email, full_name, role, site_id, is_active, created_at, updated_at
            FROM users
            ORDER BY username
        """))
        rows = result.mappings().all()
        users_out = []
        for row in rows:
            uid = row["id"]
            assgn_result = await db.execute(
                text("SELECT machine_id FROM machine_assignments WHERE user_id = :uid"),
                {"uid": uid},
            )
            assigned = [r["machine_id"] for r in assgn_result.mappings().fetchall()]
            users_out.append({
                "id": uid,
                "username": row["username"],
                "email": row.get("email"),
                "full_name": row.get("full_name"),
                "role": row["role"],
                "site_id": row.get("site_id"),
                "is_active": row.get("is_active", True),
                "assigned_machine_ids": assigned,
                "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
                "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
            })
        return {"count": len(users_out), "data": users_out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ADMIN — Audit log (append-only table)
# =============================================================================

@admin_router.get("/audit-log")
async def get_audit_log(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    username: Optional[str] = Query(None, description="Filter by username"),
    action: Optional[str] = Query(None, description="Filter by action or resource type"),
    from_date: Optional[datetime] = Query(None, description="Start of date range (UTC)"),
    to_date: Optional[datetime] = Query(None, description="End of date range (UTC)"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(require_admin_permission),
    db: AsyncSession = Depends(get_db),
):
    """
    Return paginated audit log entries with optional filters.
    Admin only. Data is read from the append-only audit_log table.
    """
    conditions = ["1=1"]
    params: Dict[str, Any] = {"limit": limit, "offset": offset}
    if user_id:
        conditions.append("user_id = :user_id")
        params["user_id"] = user_id
    if username:
        conditions.append("username = :username")
        params["username"] = username
    if action:
        conditions.append("(action = :action OR resource_type = :action)")
        params["action"] = action
    if from_date:
        conditions.append("timestamp_utc >= :from_date")
        params["from_date"] = from_date
    if to_date:
        conditions.append("timestamp_utc <= :to_date")
        params["to_date"] = to_date
    where = " AND ".join(conditions)
    result = await db.execute(
        text(f"""
            SELECT id, timestamp_utc, user_id, username, ip_address, action, method, path,
                   status_code, resource_type, resource_id, before_value, after_value, details
            FROM audit_log
            WHERE {where}
            ORDER BY timestamp_utc DESC
            LIMIT :limit OFFSET :offset
        """),
        params,
    )
    rows = result.mappings().all()
    data = []
    for r in rows:
        d = dict(r)
        if d.get("timestamp_utc"):
            d["timestamp_utc"] = d["timestamp_utc"].isoformat() if hasattr(d["timestamp_utc"], "isoformat") else str(d["timestamp_utc"])
        data.append(d)
    return {"count": len(data), "data": data}


# =============================================================================
# VERIFY CONNECTION (onboarding wizard)
# =============================================================================

@verify_router.post("/verify-connection")
async def verify_connection(request: Request):
    """
    Test connection to ABB Robot, Siemens PLC, or OPC-UA Generic.
    Returns connected, signals_detected, optional samples for data mapping step.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    asset_type = (body.get("asset_type") or "abb_robot").lower()
    ip = body.get("ip") or "127.0.0.1"
    port = int(body.get("port") or 4840)
    credentials = body.get("credentials") or {}
    fetch_samples = min(10, max(0, int(body.get("fetch_samples") or 0)))

    if asset_type == "siemens_plc":
        try:
            from siemens_s7_adapter import SiemensS7Adapter
            adapter = SiemensS7Adapter(
                plc_ip=ip,
                rack=int(credentials.get("rack", 0)),
                slot=int(credentials.get("slot", 1)),
                db_number=int(credentials.get("db_number", 1)),
            )
            result = await asyncio.to_thread(adapter.test_read)
            if result is None:
                return {"connected": False, "signals_detected": [], "error": "Connection or read failed", "samples": None}
            telemetry = result.get("telemetry") or {}
            signals_detected = list(telemetry.keys())
            samples = {k: [float(telemetry.get(k, 0))] for k in signals_detected}
            if fetch_samples > 1 and signals_detected:
                for _ in range(fetch_samples - 1):
                    r2 = await asyncio.to_thread(adapter.test_read)
                    if r2 and r2.get("telemetry"):
                        for k in signals_detected:
                            v = r2["telemetry"].get(k, 0)
                            try:
                                samples.setdefault(k, []).append(float(v))
                            except (TypeError, ValueError):
                                samples.setdefault(k, []).append(0.0)
                    await asyncio.sleep(0.2)
            return {"connected": True, "signals_detected": signals_detected, "error": None, "samples": samples}
        except Exception as e:
            return {"connected": False, "signals_detected": [], "error": str(e), "samples": None}

    if asset_type in ("abb_robot", "opcua_generic"):
        url = f"opc.tcp://{ip}:{port}"
        try:
            from asyncua import Client
            from config import get_settings
            abb = get_settings().abb

            async def _connect_and_read():
                async with Client(url=url) as client:
                    node_speed = client.get_node(abb.node_speed)
                    node_torque = client.get_node(abb.node_torque)
                    node_joints = client.get_node(abb.node_joints)
                    await node_speed.read_value()
                    await node_torque.read_value()
                    await node_joints.read_value()
                return ["rotational_speed", "torque", "joints"]

            signals_detected = await _connect_and_read()
            samples = {}
            if fetch_samples > 1:
                async with Client(url=url) as client:
                    ns = client.get_node(abb.node_speed)
                    nt = client.get_node(abb.node_torque)
                    for _ in range(fetch_samples):
                        s = await ns.read_value()
                        t = await nt.read_value()
                        samples.setdefault("rotational_speed", []).append(float(s) if s is not None else 0.0)
                        samples.setdefault("torque", []).append(float(t) if t is not None else 0.0)
                        samples.setdefault("joints", []).append(0.0)
                        await asyncio.sleep(0.2)
            else:
                samples = {"rotational_speed": [0.0], "torque": [0.0], "joints": [0.0]}
            return {"connected": True, "signals_detected": signals_detected, "error": None, "samples": samples}
        except Exception as e:
            return {"connected": False, "signals_detected": [], "error": str(e), "samples": None}

    return {"connected": False, "signals_detected": [], "error": f"Unknown asset_type: {asset_type}", "samples": None}


# =============================================================================
# ONBOARDING ENDPOINTS
# =============================================================================

@onboarding_router.post("/start")
async def onboarding_start(request: Request):
    """
    Start onboarding from wizard: write station_config, set PENDING, trigger Prefect flow.
    Body: machine_id, machine_name, machine_type, manufacturer, model, location, connection, signal_mapping, alerts.
    """
    import os
    from pathlib import Path
    from services.onboarding_helpers import set_onboarding_pending

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    machine_id = body.get("machine_id")
    if not machine_id:
        raise HTTPException(status_code=400, detail="machine_id required")
    machine_name = body.get("machine_name") or machine_id
    machine_type = body.get("machine_type") or "Other"
    location = (body.get("location") or "").strip()
    parts = location.split("/") if location else []
    shop = parts[0] if len(parts) > 0 else "Unassigned"
    line = parts[1] if len(parts) > 1 else (parts[0] if parts else "Unassigned")
    connection = body.get("connection") or {}

    base = Path(__file__).resolve().parent.parent
    config_path = os.getenv("STATION_CONFIG_PATH", str(base / "pipeline" / "station_config.json"))
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {"node_mappings": {}, "metadata": {"version": "1.0.0"}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load station_config: {e}")

    node_mappings = config.get("node_mappings") or {}
    uns_path = f"PlantAGI/{shop}/{line}/{machine_id}"
    node_mappings[machine_id] = {
        "asset_name": machine_name,
        "shop": shop,
        "line": line,
        "equipment_type": machine_type,
        "opc_node_id": f"ns=2;s={machine_id}",
        "criticality": "medium",
        "uns_path": uns_path,
        "connection": connection,
    }
    config["node_mappings"] = node_mappings
    try:
        tmp_path = config_path + ".tmp." + str(os.getpid())
        with open(tmp_path, "w") as f:
            json.dump(config, f, indent=2)
        os.replace(tmp_path, config_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write station_config: {e}")

    set_onboarding_pending(machine_id)
    onboarding_id = machine_id
    PREFECT_API_URL = os.getenv("PREFECT_API_URL", "")
    if PREFECT_API_URL:
        try:
            from prefect import get_client
            with get_client(sync_client=True) as client:
                deployment = client.read_deployment_by_name("new-machine-onboarding", "new-machine-onboarding")
                if deployment:
                    run = client.create_flow_run_from_deployment(
                        deployment_id=deployment.id,
                        parameters={"machine_id": machine_id},
                    )
                    onboarding_id = str(run.id) if getattr(run, "id", None) else machine_id
        except Exception:
            pass
    return {"machine_id": machine_id, "onboarding_id": onboarding_id}


@onboarding_router.get("")
async def list_onboarding_statuses(
    current_user: dict = Depends(require_operator_permission),
    db: AsyncSession = Depends(get_db)
):
    """Return all machines and their onboarding status (latest row per machine)."""
    result = await db.execute(
        text("""
            SELECT DISTINCT ON (machine_id)
                machine_id, started_at, completed_at, current_step, status, error_message, model_id
            FROM onboarding_status
            ORDER BY machine_id, started_at DESC
        """),
    )
    rows = result.mappings().all()
    return [dict(r) for r in rows]


@onboarding_router.get("/{machine_id}")
async def get_onboarding_status(
    machine_id: str,
    current_user: dict = Depends(require_operator_permission),
    db: AsyncSession = Depends(get_db)
):
    """Return current onboarding status for the given machine_id. 404 if no row."""
    result = await db.execute(
        text("""
            SELECT machine_id, started_at, completed_at, current_step, status, error_message, model_id
            FROM onboarding_status
            WHERE machine_id = :machine_id
            ORDER BY started_at DESC
            LIMIT 1
        """),
        {"machine_id": machine_id},
    )
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail=f"No onboarding status for machine_id={machine_id}")
    return dict(row)


@onboarding_router.post("/{machine_id}/restart")
async def restart_onboarding(machine_id: str):
    """Restart failed/stalled onboarding: set status PENDING, clear error, trigger Prefect flow. Returns 202 Accepted."""
    from services.onboarding_helpers import update_onboarding_status
    update_onboarding_status(
        machine_id=machine_id,
        status="PENDING",
        current_step=None,
        error_message=None,
    )
    import os
    PREFECT_API_URL = os.getenv("PREFECT_API_URL", "")
    if PREFECT_API_URL:
        try:
            from prefect import get_client
            with get_client(sync_client=True) as client:
                deployment = client.read_deployment_by_name("new-machine-onboarding", "new-machine-onboarding")
                if deployment:
                    client.create_flow_run_from_deployment(
                        deployment_id=deployment.id,
                        parameters={"machine_id": machine_id},
                    )
                else:
                    raise HTTPException(status_code=503, detail="Prefect deployment not found")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed to trigger flow: {e}")
    return JSONResponse(status_code=202, content={"status": "accepted", "message": f"Onboarding restarted for {machine_id}"})


# Export for use in api_server.py
__all__ = ['enterprise_router', 'labeling_router', 'alerts_router', 'drift_router', 'models_router', 'onboarding_router', 'verify_router', 'admin_router', 'check_and_create_alarms']
