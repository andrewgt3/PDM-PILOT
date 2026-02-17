"""Gaia Predictive — Machine ORM Model.

Defines the Machine entity for TimescaleDB/PostgreSQL.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, String, Float
from sqlalchemy.dialects.postgresql import UUID

from db.base import Base  # Assuming this exists or will exist

class Machine(Base):
    """Machine entity representing a physical asset."""
    __tablename__ = "machines"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    type = Column(String(50), nullable=False)  # e.g., 'CNC', 'Press'
    location = Column(String(100), nullable=True)
    
    # RUL (Remaining Useful Life) metrics
    health_score = Column(Float, default=100.0)
    predicted_rul_days = Column(Float, nullable=True)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
