"""Add model_baselines and drift_reports for DriftMonitorService.

Revision ID: 011_model_baselines_drift_reports
Revises: 010_sensor_readings_uns_topic
Create Date: 2026-02-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "011_baselines_drift"
down_revision: Union[str, None] = "010_sensor_readings_uns_topic"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_baselines",
        sa.Column("machine_id", sa.String(255), nullable=False),
        sa.Column("model_version", sa.String(255), nullable=False),
        sa.Column("feature_stats_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("baseline_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.PrimaryKeyConstraint("machine_id", "model_version"),
        if_not_exists=True,
    )
    op.create_table(
        "drift_reports",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("machine_id", sa.String(255), nullable=False),
        sa.Column("checked_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("worst_psi", sa.Float(), nullable=True),
        sa.Column("drifted_features_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("prediction_drift_pct", sa.Float(), nullable=True),
        sa.Column("overall_status", sa.String(20), nullable=False),
        sa.Column("recommended_action", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_index(
        "idx_drift_reports_machine_checked",
        "drift_reports",
        ["machine_id", "checked_at"],
        unique=False,
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("idx_drift_reports_machine_checked", table_name="drift_reports")
    op.drop_table("drift_reports")
    op.drop_table("model_baselines")
