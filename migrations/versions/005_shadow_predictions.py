"""Add shadow_predictions table for staging vs production comparison.

Revision ID: 005_shadow_predictions
Revises: 004_model_drift_reports
Create Date: 2026-02-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "005_shadow_predictions"
down_revision: Union[str, None] = "004_model_drift_reports"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "shadow_predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("machine_id", sa.String(50), nullable=False),
        sa.Column("production_prediction", sa.Float(), nullable=True),
        sa.Column("staging_prediction", sa.Float(), nullable=True),
        sa.Column("agree", sa.Boolean(), nullable=True),
        sa.Column("features_snapshot_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_shadow_predictions_machine_timestamp",
        "shadow_predictions",
        ["machine_id", "timestamp"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_shadow_predictions_machine_timestamp", table_name="shadow_predictions")
    op.drop_table("shadow_predictions")
