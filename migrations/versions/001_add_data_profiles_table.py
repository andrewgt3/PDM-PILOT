"""Add data_profiles table for data quality reports.

Revision ID: 001_data_profiles
Revises:
Create Date: 2026-02-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_data_profiles"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "data_profiles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("machine_id", sa.String(50), nullable=False),
        sa.Column("profiled_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("profile_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("overall_status", sa.String(20), nullable=False),
        sa.Column("hours_analyzed", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_data_profiles_machine_profiled",
        "data_profiles",
        ["machine_id", "profiled_at"],
        unique=False,
        postgresql_ops={"profiled_at": "DESC"},
    )


def downgrade() -> None:
    op.drop_index("idx_data_profiles_machine_profiled", table_name="data_profiles")
    op.drop_table("data_profiles")
