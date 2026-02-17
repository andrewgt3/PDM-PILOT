from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

class WorkOrder(BaseModel):
    """Work Order response schema."""
    model_config = ConfigDict(from_attributes=True)

    work_order_id: str
    machine_id: str
    title: str
    description: Optional[str] = None
    priority: str
    work_type: str
    status: str
    scheduled_date: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_to: Optional[str] = None
    notes: Optional[str] = None
    actual_duration_hours: Optional[float] = None
    estimated_duration_hours: Optional[float] = None
    created_at: datetime
