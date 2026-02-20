"""Add model_drift_reports table for drift monitoring.

Revision ID: 004_model_drift_reports
Revises: 003_alert_state_tables
Create Date: 2026-02-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "004_model_drift_reports"
down_revision: Union[str, None] = "003_alert_state_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_drift_reports",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("machine_id", sa.String(50), nullable=False),
        sa.Column("checked_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("data_drift_psi", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("drifted_features", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("concept_drift_f1_delta", sa.Float(), nullable=True),
        sa.Column("overall_status", sa.String(20), nullable=False),
        sa.Column("recommended_action", sa.Text(), nullable=True),
        sa.Column("report_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_model_drift_reports_machine_checked",
        "model_drift_reports",
        ["machine_id", "checked_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_model_drift_reports_machine_checked", table_name="model_drift_reports")
    op.drop_table("model_drift_reports")
