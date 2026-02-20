"""Add alert_current_state and alert_state_history for tiered alert engine.

Revision ID: 003_alert_state_tables
Revises: 002_labeling_tables
Create Date: 2026-02-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "003_alert_state_tables"
down_revision: Union[str, None] = "002_labeling_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "alert_current_state",
        sa.Column("machine_id", sa.String(50), nullable=False),
        sa.Column("state", sa.String(20), nullable=False, server_default="HEALTHY"),
        sa.Column("cycle_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("entered_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("acknowledged_by", sa.String(100), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("machine_id"),
    )
    op.create_index("idx_alert_current_state_state", "alert_current_state", ["state"], unique=False)

    op.create_table(
        "alert_state_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("machine_id", sa.String(50), nullable=False),
        sa.Column("from_state", sa.String(20), nullable=False),
        sa.Column("to_state", sa.String(20), nullable=False),
        sa.Column("at_timestamp", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("prediction_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_alert_state_history_machine_time", "alert_state_history", ["machine_id", "at_timestamp"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_alert_state_history_machine_time", table_name="alert_state_history")
    op.drop_table("alert_state_history")
    op.drop_index("idx_alert_current_state_state", table_name="alert_current_state")
    op.drop_table("alert_current_state")
