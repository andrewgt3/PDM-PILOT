"""Add uns_topic to sensor_readings for UNS message metadata (E2E test_01).

Revision ID: 010_sensor_readings_uns_topic
Revises: 009_cmms_sync_not_null_check
Create Date: 2026-02-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "010_sensor_readings_uns_topic"
down_revision: Union[str, None] = "009_cmms_sync_not_null_check"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "sensor_readings",
        sa.Column("uns_topic", sa.String(512), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("sensor_readings", "uns_topic")
