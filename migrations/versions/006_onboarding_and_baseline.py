"""Add onboarding_status and machine_baseline_stats tables.

Revision ID: 006_onboarding_and_baseline
Revises: 005_shadow_predictions
Create Date: 2026-02-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "006_onboarding_and_baseline"
down_revision: Union[str, None] = "005_shadow_predictions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "onboarding_status",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("machine_id", sa.String(50), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("current_step", sa.String(100), nullable=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("model_id", sa.String(255), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_onboarding_status_machine_id", "onboarding_status", ["machine_id"], unique=False)
    op.create_index("idx_onboarding_status_status", "onboarding_status", ["status"], unique=False)

    op.create_table(
        "machine_baseline_stats",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("machine_id", sa.String(50), nullable=False),
        sa.Column("metric_name", sa.String(80), nullable=False),
        sa.Column("mean", sa.Float(), nullable=False),
        sa.Column("std", sa.Float(), nullable=False),
        sa.Column("p5", sa.Float(), nullable=False),
        sa.Column("p95", sa.Float(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("machine_id", "metric_name", name="uq_machine_baseline_stats_machine_metric"),
    )
    op.create_index("idx_machine_baseline_stats_machine_id", "machine_baseline_stats", ["machine_id"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_machine_baseline_stats_machine_id", table_name="machine_baseline_stats")
    op.drop_table("machine_baseline_stats")
    op.drop_index("idx_onboarding_status_status", table_name="onboarding_status")
    op.drop_index("idx_onboarding_status_machine_id", table_name="onboarding_status")
    op.drop_table("onboarding_status")
