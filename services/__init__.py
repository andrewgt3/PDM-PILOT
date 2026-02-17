"""Gaia Predictive — Service Layer.

This package contains business logic services that encapsulate
domain operations and keep API routes thin.

Services:
    - AuthService: Password hashing, JWT token management
    - MachineService: Fleet status, health calculations
    - TelemetryService: Sensor data ingestion, batch processing
    - AlarmService: Threshold monitoring, alarm management
    - BaseService: Generic CRUD operations (for ORM models)

Usage:
    from services import MachineService, TelemetryService, AlarmService
    
    # In FastAPI route
    async def get_machines(db: AsyncSession = Depends(get_db)):
        service = MachineService(db)
        return await service.get_all_machines()
"""

from services.auth_service import AuthService, get_auth_service
from services.machine_service import MachineService
from services.telemetry_service import TelemetryService
from services.alarm_service import AlarmService
from services.base import BaseService

__all__ = [
    "AuthService",
    "get_auth_service",
    "MachineService", 
    "TelemetryService",
    "AlarmService",
    "BaseService",
]
