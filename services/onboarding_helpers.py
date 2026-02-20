"""Shared helpers for onboarding_status (sync DB updates). Used by flows and stream_consumer."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def get_connection():
    """Sync psycopg2 connection using config."""
    from config import get_settings
    s = get_settings().database
    import psycopg2
    return psycopg2.connect(
        host=s.host,
        port=s.port,
        dbname=s.name,
        user=s.user,
        password=s.password.get_secret_value(),
        connect_timeout=5,
    )


def update_onboarding_status(
    machine_id: str,
    current_step: Optional[str] = None,
    status: Optional[str] = None,
    error_message: Optional[str] = None,
    model_id: Optional[str] = None,
    completed_at: Optional[datetime] = None,
) -> None:
    """Insert or update onboarding_status for the machine. Pass only fields to update."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id FROM onboarding_status WHERE machine_id = %s ORDER BY started_at DESC LIMIT 1
            """,
            (machine_id,),
        )
        row = cur.fetchone()
        if row:
            updates = []
            params = []
            if current_step is not None:
                updates.append("current_step = %s")
                params.append(current_step)
            if status is not None:
                updates.append("status = %s")
                params.append(status)
            if error_message is not None:
                updates.append("error_message = %s")
                params.append(error_message)
            if model_id is not None:
                updates.append("model_id = %s")
                params.append(model_id)
            if completed_at is not None:
                updates.append("completed_at = %s")
                params.append(completed_at)
            if updates:
                params.append(machine_id)
                cur.execute(
                    "UPDATE onboarding_status SET "
                    + ", ".join(updates)
                    + " WHERE id = (SELECT id FROM onboarding_status WHERE machine_id = %s ORDER BY started_at DESC LIMIT 1)",
                    params,
                )
        else:
            cur.execute(
                """
                INSERT INTO onboarding_status (machine_id, started_at, current_step, status, error_message, model_id, completed_at)
                VALUES (%s, NOW(), %s, %s, %s, %s, %s)
                """,
                (
                    machine_id,
                    current_step,
                    status or "PENDING",
                    error_message,
                    model_id,
                    completed_at,
                ),
            )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def set_onboarding_pending(machine_id: str) -> None:
    """Set or insert onboarding_status to PENDING with started_at=NOW(). Used by stream_consumer."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE onboarding_status SET status = 'PENDING', started_at = NOW(), current_step = NULL, error_message = NULL
            WHERE machine_id = %s
            """,
            (machine_id,),
        )
        if cur.rowcount == 0:
            cur.execute(
                """
                INSERT INTO onboarding_status (machine_id, started_at, status)
                VALUES (%s, NOW(), 'PENDING')
                """,
                (machine_id,),
            )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def set_onboarding_complete(machine_id: str) -> None:
    """Mark a machine as onboarding COMPLETE (e.g. for Azure PM demo so it appears as live)."""
    update_onboarding_status(
        machine_id,
        current_step="send_onboarding_complete_notification",
        status="COMPLETE",
        error_message="",
        completed_at=datetime.now(timezone.utc),
    )
