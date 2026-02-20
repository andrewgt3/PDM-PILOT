"""Add tenant_id to all data tables for row-level multi-tenancy.

Revision ID: 014_add_tenant_id
Revises: 013_rbac_sites_assignments
Create Date: 2026-02-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "014_add_tenant_id"
down_revision: Union[str, None] = "013_rbac_sites_assignments"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TENANT_TABLES = [
    "sensor_readings",
    "cwru_features",
    "pdm_alarms",
    "work_orders",
    "data_profiles",
    "alert_current_state",
    "alert_state_history",
    "gap_events",
    "model_baselines",
    "drift_reports",
    "onboarding_status",
    "labeling_tasks",
    "labels",
    "machine_assignments",
    "shadow_predictions",
    "failure_events",
]


def upgrade() -> None:
    # 1. Create tenants table and insert default
    op.create_table(
        "tenants",
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("tenant_name", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.PrimaryKeyConstraint("tenant_id"),
    )
    op.execute(
        sa.text("INSERT INTO tenants (tenant_id, tenant_name, created_at, active) VALUES ('default', 'Default Tenant', NOW(), true)")
    )

    # 2. Add tenant_id column to all data tables
    for table in TENANT_TABLES:
        op.add_column(table, sa.Column("tenant_id", sa.String(64), nullable=False, server_default=sa.text("'default'")))

    # 3. Backfill (no-op when column is NOT NULL DEFAULT; safe to run)
    for table in TENANT_TABLES:
        op.execute(sa.text(f"UPDATE {table} SET tenant_id = 'default' WHERE tenant_id IS NULL"))

    # 4. Single-column index on tenant_id for every table
    for table in TENANT_TABLES:
        op.create_index(
            f"idx_{table}_tenant",
            table,
            ["tenant_id"],
            unique=False,
        )

    # 5. Composite indexes for heavy query tables
    op.create_index(
        "idx_cwru_tenant_machine",
        "cwru_features",
        ["tenant_id", "machine_id"],
        unique=False,
    )
    op.create_index(
        "idx_readings_tenant_machine",
        "sensor_readings",
        ["tenant_id", "machine_id"],
        unique=False,
    )

    # 6. FK: machine_assignments.site_id -> sites.site_id
    op.create_foreign_key(
        "fk_machine_assignments_site",
        "machine_assignments",
        "sites",
        ["site_id"],
        ["site_id"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_machine_assignments_site", "machine_assignments", type_="foreignkey")

    op.drop_index("idx_readings_tenant_machine", table_name="sensor_readings")
    op.drop_index("idx_cwru_tenant_machine", table_name="cwru_features")

    for table in TENANT_TABLES:
        op.drop_index(f"idx_{table}_tenant", table_name=table)
        op.drop_column(table, "tenant_id")

    op.execute(sa.text("DELETE FROM tenants WHERE tenant_id = 'default'"))
    op.drop_table("tenants")
