"""Gaia Predictive — Machine Service.

Encapsulates all business logic for Machine assets including 
fleet status, health calculations, and maintenance workflows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import uuid

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from services.base import BaseService
from logger import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STATION_CONFIG_PATH = _PROJECT_ROOT / "pipeline" / "station_config.json"


def _metadata_for_machine(machine_id: str) -> Dict[str, str]:
    """Return name, shop, line, type from EQUIPMENT_METADATA or station_config node_mappings."""
    # Hardcoded first (legacy fleet)
    if machine_id in EQUIPMENT_METADATA:
        m = EQUIPMENT_METADATA[machine_id]
        return {
            "name": m.get("name", machine_id),
            "shop": m.get("shop", "Unassigned Shop"),
            "line": m.get("line", "Unassigned Line"),
            "type": m.get("type", "Equipment"),
        }
    # Azure PM and others: from station_config (ingestor writes asset_name, shop, line, equipment_type)
    try:
        if _STATION_CONFIG_PATH.exists():
            with open(_STATION_CONFIG_PATH, encoding="utf-8") as f:
                data = json.load(f)
            node_mappings = data.get("node_mappings") or {}
            info = node_mappings.get(str(machine_id)) if isinstance(node_mappings.get(str(machine_id)), dict) else None
            if info:
                return {
                    "name": info.get("asset_name", machine_id),
                    "shop": info.get("shop", "Unassigned Shop"),
                    "line": info.get("line", "Unassigned Line"),
                    "type": info.get("equipment_type", "Equipment"),
                }
    except Exception as e:
        logger.debug("station_config_read_failed", machine_id=machine_id, error=str(e))
    return {
        "name": machine_id,
        "shop": "Unassigned Shop",
        "line": "Unassigned Line",
        "type": "Equipment",
    }


# Equipment metadata lookup (moved from api_server.py)
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


class MachineService:
    """Service for managing Machine lifecycle and fleet intelligence.
    
    This service encapsulates:
    - Fleet status queries with RUL calculations
    - Individual machine status lookups
    - Health score calculations
    - Maintenance workflow triggers
    """
    
    def __init__(self, db: AsyncSession):
        """Initialize service with database session."""
        self.db = db
        self.logger = logger.bind(service="MachineService")

    async def get_all_machines(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get list of active machines with their current status.
        
        Calculations (RUL, Status) are offloaded to SQL for performance.
        
        Returns:
            List of machine dictionaries with status, RUL, and metadata.
        """
        query = text("""
            SELECT DISTINCT ON (machine_id)
                machine_id,
                timestamp as last_seen,
                failure_prediction,
                COALESCE(degradation_score, 0.0) as degradation_score,
                
                -- RUL Calculation: 2000 * (1 - degradation)^2 / 24
                GREATEST(0, (2000 * POWER(1.0 - COALESCE(degradation_score, 0.0), 2)) / 24.0) as rul_days,
                
                -- Status Logic
                CASE 
                    WHEN failure_prediction > 0.8 THEN 'CRITICAL'
                    WHEN failure_prediction > 0.5 THEN 'WARNING'
                    ELSE 'HEALTHY'
                END as status
                
            FROM cwru_features
            ORDER BY machine_id, timestamp DESC
            LIMIT :limit
        """)
        
        result = await self.db.execute(query, {"limit": limit})
        rows = result.mappings().all()
        
        machines = []
        for row in rows:
            machine_id = row['machine_id']
            metadata = _metadata_for_machine(str(machine_id))
            machines.append({
                "machine_id": machine_id,
                "machine_name": metadata["name"],
                "shop": metadata["shop"],
                "line_name": metadata["line"],
                "equipment_type": metadata["type"],
                "last_seen": row['last_seen'],
                "failure_probability": row['failure_prediction'],
                "rul_days": float(row['rul_days']),
                "status": row['status']
            })
        
        self.logger.debug("Fetched machines", count=len(machines))
        return machines

    async def get_machine_status(self, machine_id: str) -> Dict[str, Any]:
        """Get detailed status for a specific machine.
        
        Args:
            machine_id: The machine identifier.
            
        Returns:
            Machine status dict.
            
        Raises:
            ResourceNotFound: If machine_id has no data.
        """
        from core.exceptions import ResourceNotFound
        
        query = text("""
            SELECT 
                machine_id,
                timestamp as last_seen,
                failure_prediction,
                COALESCE(degradation_score, 0.0) as degradation_score,
                GREATEST(0, (2000 * POWER(1.0 - COALESCE(degradation_score, 0.0), 2)) / 24.0) as rul_days,
                CASE 
                    WHEN failure_prediction > 0.8 THEN 'CRITICAL'
                    WHEN failure_prediction > 0.5 THEN 'WARNING'
                    ELSE 'HEALTHY'
                END as status,
                bpfi_amp, bpfo_amp, spectral_entropy
            FROM cwru_features
            WHERE machine_id = :machine_id
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        
        result = await self.db.execute(query, {"machine_id": machine_id})
        row = result.mappings().one_or_none()
        
        if not row:
            raise ResourceNotFound("Machine", machine_id)
        
        metadata = _metadata_for_machine(str(machine_id))
        return {
            "machine_id": machine_id,
            "machine_name": metadata["name"],
            "shop": metadata["shop"],
            "line_name": metadata["line"],
            "equipment_type": metadata["type"],
            "last_seen": row['last_seen'],
            "failure_probability": row['failure_prediction'],
            "degradation_score": float(row['degradation_score']),
            "rul_days": float(row['rul_days']),
            "status": row['status'],
            "bpfi_amp": row['bpfi_amp'],
            "bpfo_amp": row['bpfo_amp'],
            "spectral_entropy": row['spectral_entropy'],
        }

    async def get_critical_machines(self) -> List[Dict[str, Any]]:
        """Get machines in CRITICAL or WARNING status.
        
        Returns:
            List of machines requiring attention.
        """
        all_machines = await self.get_all_machines()
        return [m for m in all_machines if m['status'] in ('CRITICAL', 'WARNING')]

