"""RBAC: sites, users, machine_assignments for role-based machine and site scoping.

Revision ID: 013_rbac_sites_assignments
Revises: 012_gap_events
Create Date: 2026-02-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "013_rbac_sites_assignments"
down_revision: Union[str, None] = "012_gap_events"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Sites: tenant/site dimension (machines derived from station_config or app logic)
    op.create_table(
        "sites",
        sa.Column("site_id", sa.String(64), nullable=False),
        sa.Column("site_name", sa.String(255), nullable=True),
        sa.Column("tenant_id", sa.String(64), nullable=True),
        sa.PrimaryKeyConstraint("site_id"),
    )

    # Users: required for machine_assignments FKs (no existing users table in migrations)
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("username", sa.String(50), nullable=False),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column("full_name", sa.String(255), nullable=True),
        sa.Column("role", sa.String(32), nullable=False, server_default="viewer"),
        sa.Column("site_id", sa.String(64), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("hashed_password", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("last_login", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_users_username", "users", ["username"], unique=True)
    op.create_index("idx_users_site_id", "users", ["site_id"], unique=False)

    # Machine assignments: technician scope and plant_manager site linkage
    op.create_table(
        "machine_assignments",
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("machine_id", sa.String(64), nullable=False),
        sa.Column("site_id", sa.String(64), nullable=False),
        sa.Column("assigned_by", sa.Integer(), nullable=False),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.PrimaryKeyConstraint("user_id", "machine_id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["assigned_by"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index("idx_machine_assignments_site_id", "machine_assignments", ["site_id"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_machine_assignments_site_id", table_name="machine_assignments")
    op.drop_table("machine_assignments")
    op.drop_index("idx_users_site_id", table_name="users")
    op.drop_index("idx_users_username", table_name="users")
    op.drop_table("users")
    op.drop_table("sites")
