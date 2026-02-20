"""Add gap_events table for edge connectivity / offline buffer visibility.

Revision ID: 012_gap_events
Revises: 011_baselines_drift
Create Date: 2026-02-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "012_gap_events"
down_revision: Union[str, None] = "011_baselines_drift"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "gap_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("adapter_source", sa.String(32), nullable=False),
        sa.Column("gap_started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gap_ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("readings_buffered", sa.Integer(), nullable=True),
        sa.Column("readings_replayed", sa.Integer(), nullable=True),
        sa.Column("duration_minutes", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_gap_events_adapter_started",
        "gap_events",
        ["adapter_source", "gap_started_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_gap_events_adapter_started", table_name="gap_events")
    op.drop_table("gap_events")
