"""Add cmms_sync_status to work_orders for CMMS/SAP integration.

Revision ID: 008_work_orders_cmms_sync
Revises: 007_audit_log_append_only
Create Date: 2026-02-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "008_work_orders_cmms_sync"
down_revision: Union[str, None] = "007_audit_log_append_only"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "work_orders",
        sa.Column("cmms_sync_status", sa.String(20), server_default="pending", nullable=True),
    )


def downgrade() -> None:
    op.drop_column("work_orders", "cmms_sync_status")
