"""Gaia Predictive — Alarm Service.

Encapsulates alarm creation, threshold checking, and failure event tracking.
Replaces the check_and_create_alarms function from enterprise_api.py.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from logger import get_logger

logger = get_logger(__name__)


class AlarmThresholdResult(BaseModel):
    """Result of checking a threshold."""
    triggered: bool
    code: str
    severity: str
    message: str
    trigger_type: str
    trigger_value: float
    threshold: float


class AlarmService:
    """Service for alarm management and threshold monitoring.
    
    This service encapsulates:
    - Threshold-based alarm creation
    - Duplicate alarm prevention
    - Failure event recording
    """
    
    def __init__(self, db: AsyncSession):
        """Initialize service with database session."""
        self.db = db
        self.logger = logger.bind(service="AlarmService")

    async def check_and_create_alarms(
        self,
        machine_id: str,
        failure_probability: float,
        degradation_score: float,
        bpfi_amp: float,
        bpfo_amp: float
    ) -> List[str]:
        """Check sensor values against thresholds and create alarms if needed.
        
        Args:
            machine_id: Machine identifier.
            failure_probability: ML model failure probability (0-1).
            degradation_score: Calculated degradation score (0-1).
            bpfi_amp: Inner race bearing fault amplitude.
            bpfo_amp: Outer race bearing fault amplitude.
            
        Returns:
            List of created alarm IDs.
        """
        alarms_created = []
        
        try:
            # Check for existing active alarms to avoid duplicates
            result = await self.db.execute(text("""
                SELECT code FROM pdm_alarms 
                WHERE machine_id = :machine_id AND active = TRUE
            """), {"machine_id": machine_id})
            
            existing_codes = [row[0] for row in result.fetchall()]
            
            # Define threshold checks
            checks = self._get_threshold_checks(
                failure_probability, degradation_score, bpfi_amp, bpfo_amp
            )
            
            for check in checks:
                if check.triggered and check.code not in existing_codes:
                    alarm_id = await self._create_alarm(machine_id, check)
                    alarms_created.append(alarm_id)
            
            # Record failure event if critical
            if failure_probability > 0.8:
                await self._record_failure_event(
                    machine_id, failure_probability, degradation_score, bpfi_amp, bpfo_amp
                )
            
        except Exception as e:
            self.logger.error("Alarm check failed", error=str(e), machine_id=machine_id)
        
        return alarms_created

    def _get_threshold_checks(
        self,
        failure_probability: float,
        degradation_score: float,
        bpfi_amp: float,
        bpfo_amp: float
    ) -> List[AlarmThresholdResult]:
        """Generate list of threshold checks based on current values."""
        return [
            AlarmThresholdResult(
                triggered=failure_probability > 0.8,
                code='PDM-CRIT-001',
                severity='critical',
                message=f'Critical failure probability: {failure_probability*100:.1f}%',
                trigger_type='failure_probability',
                trigger_value=failure_probability,
                threshold=0.8
            ),
            AlarmThresholdResult(
                triggered=0.5 < failure_probability <= 0.8,
                code='PDM-WARN-001',
                severity='warning',
                message=f'Elevated failure probability: {failure_probability*100:.1f}%',
                trigger_type='failure_probability',
                trigger_value=failure_probability,
                threshold=0.5
            ),
            AlarmThresholdResult(
                triggered=degradation_score > 0.7,
                code='PDM-WARN-002',
                severity='warning',
                message=f'High degradation score: {degradation_score*100:.1f}%',
                trigger_type='degradation',
                trigger_value=degradation_score,
                threshold=0.7
            ),
            AlarmThresholdResult(
                triggered=bpfi_amp > 0.5,
                code='PDM-INFO-001',
                severity='info',
                message=f'Inner race bearing fault amplitude elevated: {bpfi_amp:.4f}g',
                trigger_type='bpfi_amplitude',
                trigger_value=bpfi_amp,
                threshold=0.5
            ),
        ]

    async def _create_alarm(self, machine_id: str, check: AlarmThresholdResult) -> str:
        """Create a new alarm record."""
        # Generate alarm ID
        count_res = await self.db.execute(
            text("SELECT COUNT(*) FROM pdm_alarms WHERE DATE(timestamp) = CURRENT_DATE")
        )
        count = count_res.scalar()
        
        alarm_id = f"ALM-{datetime.now().strftime('%Y%m%d')}-{count + 1:04d}"
        
        query = text("""
            INSERT INTO pdm_alarms 
            (alarm_id, machine_id, severity, code, message, trigger_type, trigger_value, threshold_value)
            VALUES (:alarm_id, :machine_id, :severity, :code, :message, :trigger_type, :trigger_value, :threshold_value)
        """)
        
        await self.db.execute(query, {
            "alarm_id": alarm_id,
            "machine_id": machine_id,
            "severity": check.severity,
            "code": check.code,
            "message": check.message,
            "trigger_type": check.trigger_type,
            "trigger_value": check.trigger_value,
            "threshold_value": check.threshold,
        })
        
        self.logger.info(
            "Alarm created",
            alarm_id=alarm_id,
            machine_id=machine_id,
            severity=check.severity,
            code=check.code
        )
        
        return alarm_id

    async def _record_failure_event(
        self,
        machine_id: str,
        failure_probability: float,
        degradation_score: float,
        bpfi_amp: float,
        bpfo_amp: float
    ) -> None:
        """Record a failure event if not already recorded recently."""
        # Check for recent failure event
        check_query = text("""
            SELECT id FROM failure_events 
            WHERE machine_id = :machine_id 
            AND timestamp > NOW() - INTERVAL '1 hour'
        """)
        result = await self.db.execute(check_query, {"machine_id": machine_id})
        
        if not result.first():
            insert_query = text("""
                INSERT INTO failure_events 
                (machine_id, event_type, failure_probability, degradation_score, bpfi_amp, bpfo_amp)
                VALUES (:machine_id, 'predicted_failure', :prob, :deg, :bpfi, :bpfo)
            """)
            await self.db.execute(insert_query, {
                "machine_id": machine_id,
                "prob": failure_probability,
                "deg": degradation_score,
                "bpfi": bpfi_amp,
                "bpfo": bpfo_amp
            })
            
            self.logger.warning(
                "Failure event recorded",
                machine_id=machine_id,
                failure_probability=failure_probability
            )

    async def acknowledge_alarm(self, alarm_id: str, acknowledged_by: str) -> Optional[dict]:
        """Acknowledge an alarm."""
        query = text("""
            UPDATE pdm_alarms 
            SET acknowledged = TRUE, acknowledged_by = :ack_by, acknowledged_at = NOW()
            WHERE alarm_id = :alarm_id
            RETURNING *
        """)
        
        result = await self.db.execute(query, {"ack_by": acknowledged_by, "alarm_id": alarm_id})
        row = result.mappings().one_or_none()
        
        if row:
            self.logger.info("Alarm acknowledged", alarm_id=alarm_id, by=acknowledged_by)
            return dict(row)
        return None

    async def resolve_alarm(self, alarm_id: str) -> Optional[dict]:
        """Resolve an alarm (mark as inactive)."""
        query = text("""
            UPDATE pdm_alarms 
            SET active = FALSE, resolved_at = NOW()
            WHERE alarm_id = :alarm_id
            RETURNING *
        """)
        
        result = await self.db.execute(query, {"alarm_id": alarm_id})
        row = result.mappings().one_or_none()
        
        if row:
            self.logger.info("Alarm resolved", alarm_id=alarm_id)
            return dict(row)
        return None

    async def commit_or_rollback(self) -> bool:
        """Commit transaction or rollback on error."""
        try:
            await self.db.commit()
            return True
        except Exception as e:
            await self.db.rollback()
            self.logger.error("Transaction rolled back", error=str(e))
            return False
