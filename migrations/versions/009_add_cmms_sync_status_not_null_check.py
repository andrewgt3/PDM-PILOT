"""Add NOT NULL and CHECK constraint to work_orders.cmms_sync_status.

Revision ID: 009_cmms_sync_not_null_check
Revises: 008_work_orders_cmms_sync
Create Date: 2026-02-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "009_cmms_sync_not_null_check"
down_revision: Union[str, None] = "008_work_orders_cmms_sync"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Ensure existing NULLs become 'pending' before adding NOT NULL
    op.execute(
        "UPDATE work_orders SET cmms_sync_status = 'pending' WHERE cmms_sync_status IS NULL"
    )
    op.alter_column(
        "work_orders",
        "cmms_sync_status",
        existing_type=sa.String(20),
        server_default="pending",
        nullable=False,
    )
    op.create_check_constraint(
        "ck_work_orders_cmms_sync_status",
        "work_orders",
        "cmms_sync_status IN ('pending', 'synced', 'failed')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_work_orders_cmms_sync_status", "work_orders", type_="check")
    op.alter_column(
        "work_orders",
        "cmms_sync_status",
        existing_type=sa.String(20),
        server_default=None,
        nullable=True,
    )
