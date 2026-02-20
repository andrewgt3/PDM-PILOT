"""Add append-only audit_log table for comprehensive audit trail.

Revision ID: 007_audit_log_append_only
Revises: 006_onboarding_and_baseline
Create Date: 2026-02-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "007_audit_log_append_only"
down_revision: Union[str, None] = "006_onboarding_and_baseline"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("timestamp_utc", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("user_id", sa.String(64), nullable=True),
        sa.Column("username", sa.String(255), nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=False),
        sa.Column("action", sa.String(255), nullable=False),
        sa.Column("method", sa.String(10), nullable=True),
        sa.Column("path", sa.String(1024), nullable=True),
        sa.Column("status_code", sa.SmallInteger(), nullable=True),
        sa.Column("resource_type", sa.String(64), nullable=True),
        sa.Column("resource_id", sa.String(255), nullable=True),
        sa.Column("before_value", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("after_value", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_audit_log_timestamp", "audit_log", ["timestamp_utc"], unique=False)
    op.create_index("idx_audit_log_user_id", "audit_log", ["user_id"], unique=False)
    op.create_index("idx_audit_log_action", "audit_log", ["action"], unique=False)
    op.create_index("idx_audit_log_resource", "audit_log", ["resource_type", "resource_id"], unique=False)

    # Append-only: forbid UPDATE and DELETE
    op.execute("""
        CREATE OR REPLACE FUNCTION audit_log_deny_update_delete()
        RETURNS TRIGGER AS $$
        BEGIN
          RAISE EXCEPTION 'audit_log is append-only: UPDATE and DELETE are not allowed';
        END;
        $$ LANGUAGE plpgsql
    """)
    op.execute("""
        CREATE TRIGGER audit_log_append_only
        BEFORE UPDATE OR DELETE ON audit_log
        FOR EACH ROW EXECUTE PROCEDURE audit_log_deny_update_delete()
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS audit_log_append_only ON audit_log")
    op.execute("DROP FUNCTION IF EXISTS audit_log_deny_update_delete()")
    op.drop_index("idx_audit_log_resource", table_name="audit_log")
    op.drop_index("idx_audit_log_action", table_name="audit_log")
    op.drop_index("idx_audit_log_user_id", table_name="audit_log")
    op.drop_index("idx_audit_log_timestamp", table_name="audit_log")
    op.drop_table("audit_log")
