"""Gaia Predictive — Machine Schemas.

Pydantic models for Machine API requests and responses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field

class MachineBase(BaseModel):
    """Shared properties for Machine models."""
    name: str = Field(..., min_length=1, max_length=255)
    type: str = Field(..., min_length=1, max_length=50)
    location: Optional[str] = Field(None, max_length=100)
    is_active: bool = True

class MachineCreate(MachineBase):
    """Payload for creating a new machine."""
    pass

class MachineUpdate(BaseModel):
    """Payload for updating a machine."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    type: Optional[str] = None
    location: Optional[str] = None
    is_active: Optional[bool] = None

class Machine(MachineBase):
    """Full Machine resource response."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    health_score: float
    predicted_rul_days: Optional[float]
    created_at: datetime
    updated_at: datetime
