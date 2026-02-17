"""Gaia Predictive — Telemetry Service.

Encapsulates all business logic for telemetry ingestion and processing.
Handles single readings and bulk batch inserts with proper transaction management.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from schemas import TelemetryDataValidated, BulkTelemetryValidated
from logger import get_logger

logger = get_logger(__name__)


class TelemetryService:
    """Service for telemetry data ingestion and processing.
    
    This service encapsulates:
    - Single telemetry reading insertion
    - Bulk batch processing with transaction safety
    - Data validation and normalization
    """
    
    def __init__(self, db: AsyncSession):
        """Initialize service with database session."""
        self.db = db
        self.logger = logger.bind(service="TelemetryService")

    async def insert_reading(self, data: TelemetryDataValidated) -> dict:
        """Insert a single telemetry reading.
        
        Args:
            data: Validated telemetry data.
            
        Returns:
            Dict with reading_id and timestamp.
        """
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
        
        result = await self.db.execute(query, params)
        row = result.mappings().one()
        
        self.logger.debug("Inserted telemetry reading", machine_id=data.machine_id)
        
        return {
            "reading_id": row['id'],
            "timestamp": row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])
        }

    async def process_batch(self, data: BulkTelemetryValidated) -> int:
        """Process a batch of telemetry readings.
        
        Uses executemany for efficient bulk insertion.
        
        Args:
            data: Validated bulk telemetry data (up to 1000 readings).
            
        Returns:
            Number of readings ingested.
        """
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
        
        await self.db.execute(query, values)
        
        self.logger.info("Processed telemetry batch", count=len(values))
        
        return len(values)

    async def insert_cwru_features(
        self,
        machine_id: str,
        timestamp: datetime,
        failure_prediction: float,
        degradation_score: float,
        bpfi_amp: float,
        bpfo_amp: float,
        spectral_entropy: float,
        **extra_features
    ) -> None:
        """Insert processed CWRU features (from stream consumer).
        
        Args:
            machine_id: Machine identifier.
            timestamp: Reading timestamp.
            failure_prediction: ML model failure probability.
            degradation_score: Calculated degradation score.
            bpfi_amp: Inner race bearing fault amplitude.
            bpfo_amp: Outer race bearing fault amplitude.
            spectral_entropy: Spectral entropy value.
            **extra_features: Additional feature columns.
        """
        query = text("""
            INSERT INTO cwru_features 
            (machine_id, timestamp, failure_prediction, degradation_score, 
             bpfi_amp, bpfo_amp, spectral_entropy)
            VALUES (:machine_id, :timestamp, :failure_prediction, :degradation_score,
                    :bpfi_amp, :bpfo_amp, :spectral_entropy)
        """)
        
        params = {
            "machine_id": machine_id,
            "timestamp": timestamp,
            "failure_prediction": failure_prediction,
            "degradation_score": degradation_score,
            "bpfi_amp": bpfi_amp,
            "bpfo_amp": bpfo_amp,
            "spectral_entropy": spectral_entropy,
        }
        
        await self.db.execute(query, params)
        
        self.logger.debug(
            "Inserted CWRU features",
            machine_id=machine_id,
            failure_prediction=failure_prediction
        )

    async def commit_or_rollback(self) -> bool:
        """Commit transaction or rollback on error.
        
        Returns:
            True if committed successfully, False if rolled back.
        """
        try:
            await self.db.commit()
            return True
        except Exception as e:
            await self.db.rollback()
            self.logger.error("Transaction rolled back", error=str(e))
            return False
