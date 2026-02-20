"""Add labeling_tasks and labels tables for human-in-the-loop labeling.

Revision ID: 002_labeling_tables
Revises: 001_data_profiles
Create Date: 2026-02-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "002_labeling_tables"
down_revision: Union[str, None] = "001_data_profiles"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "labeling_tasks",
        sa.Column("task_id", sa.String(36), nullable=False),
        sa.Column("anomaly_event_id", sa.String(30), nullable=False),
        sa.Column("machine_id", sa.String(20), nullable=False),
        sa.Column("feature_snapshot_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.PrimaryKeyConstraint("task_id"),
    )
    op.create_index("idx_labeling_tasks_machine_status", "labeling_tasks", ["machine_id", "status"], unique=False)

    op.create_table(
        "labels",
        sa.Column("label_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("task_id", sa.String(36), nullable=False),
        sa.Column("label", sa.SmallInteger(), nullable=False),
        sa.Column("submitted_by", sa.String(100), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.PrimaryKeyConstraint("label_id"),
        sa.ForeignKeyConstraint(["task_id"], ["labeling_tasks.task_id"], ondelete="CASCADE"),
        sa.UniqueConstraint("task_id", name="uq_labels_task_id"),
    )
    op.create_index("idx_labels_task_id", "labels", ["task_id"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_labels_task_id", table_name="labels")
    op.drop_table("labels")
    op.drop_index("idx_labeling_tasks_machine_status", table_name="labeling_tasks")
    op.drop_table("labeling_tasks")
