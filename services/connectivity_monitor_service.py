"""
Connectivity monitor for edge offline buffers.
Aggregates buffer stats from adapter SQLite DBs and records gap_events in TimescaleDB.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

ADAPTER_KEYS = ("opc_ua", "s7", "abb")


def _resolve_path(db_path: str) -> Path:
    p = Path(db_path)
    if p.is_absolute():
        return p
    return (_PROJECT_ROOT / db_path).resolve()


def _query_sqlite_pending(path: Path) -> tuple[int, Optional[str]]:
    """Return (pending_count, oldest_buffered_at_iso) for the given SQLite path. Thread-safe, opens and closes."""
    if not path.exists() or path == Path(":memory:"):
        return 0, None
    try:
        uri = f"file:{path.resolve()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=1.0)
        try:
            cur = conn.execute(
                """SELECT COUNT(*), MIN(buffered_at) FROM buffered_readings WHERE replayed = 0"""
            )
            row = cur.fetchone()
            count = row[0] or 0
            oldest = row[1] if row and row[1] else None
            return count, oldest
        finally:
            conn.close()
    except Exception:
        return 0, None


class ConnectivityMonitorService:
    """
    Aggregates buffer stats from adapter SQLite files and writes gap_events
    when status transitions ONLINE -> BUFFERING or BUFFERING -> ONLINE.
    """

    def __init__(self) -> None:
        self._last_pending: dict[str, int] = {k: 0 for k in ADAPTER_KEYS}
        self._lock = threading.Lock()

    def _get_buffer_paths(self) -> dict[str, Path]:
        from config import get_settings
        s = get_settings()
        base = _resolve_path(s.edge.offline_buffer_path)
        return {
            "opc_ua": base,
            "s7": _resolve_path("data/offline_buffer_s7.db"),
            "abb": _resolve_path("data/offline_buffer_abb.db"),
        }

    def _read_all_adapters_sync(self) -> dict[str, tuple[int, Optional[str]]]:
        """Run SQLite reads (blocking). Call from thread pool from async context."""
        paths = self._get_buffer_paths()
        result = {}
        for key in ADAPTER_KEYS:
            result[key] = _query_sqlite_pending(paths[key])
        return result

    async def get_gap_report(self, db: AsyncSession) -> dict[str, Any]:
        """
        Aggregate buffer stats from adapter DBs, update gap_events on transitions,
        return report dict.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, self._read_all_adapters_sync)

        adapters_report: dict[str, dict[str, Any]] = {}
        total_pending = 0
        oldest_ts: Optional[datetime] = None
        oldest_iso: Optional[str] = None

        for key in ADAPTER_KEYS:
            count, oldest_at = data[key]
            total_pending += count
            adapters_report[key] = {
                "pending": count,
                "oldest_at": oldest_at or "",
            }
            if oldest_at:
                try:
                    dt = datetime.fromisoformat(oldest_at.replace("Z", "+00:00"))
                    if oldest_ts is None or dt < oldest_ts:
                        oldest_ts = dt
                        oldest_iso = oldest_at
                except (ValueError, TypeError):
                    pass

        now = datetime.now(timezone.utc)
        gap_duration_minutes: Optional[float] = None
        if oldest_ts is not None:
            gap_duration_minutes = (now - oldest_ts).total_seconds() / 60.0

        if total_pending == 0:
            status = "ONLINE"
        else:
            status = "BUFFERING"

        with self._lock:
            for key in ADAPTER_KEYS:
                count = data[key][0]
                prev = self._last_pending[key]
                if prev == 0 and count > 0:
                    await db.execute(
                        text("""
                            INSERT INTO gap_events
                            (adapter_source, gap_started_at, gap_ended_at, readings_buffered, readings_replayed, duration_minutes)
                            VALUES (:src, :started, NULL, NULL, NULL, NULL)
                        """),
                        {"src": key, "started": now},
                    )
                elif prev > 0 and count == 0:
                    duration_minutes = None
                    if oldest_ts:
                        duration_minutes = (now - oldest_ts).total_seconds() / 60.0
                    subq = text("""
                        UPDATE gap_events SET
                            gap_ended_at = :ended,
                            duration_minutes = EXTRACT(EPOCH FROM (:ended - gap_started_at)) / 60.0
                        WHERE id = (
                            SELECT id FROM gap_events
                            WHERE adapter_source = :src AND gap_ended_at IS NULL
                            ORDER BY gap_started_at DESC LIMIT 1
                        )
                    """)
                    await db.execute(subq, {"ended": now, "src": key})
                self._last_pending[key] = count

        return {
            "total_pending": total_pending,
            "oldest_gap_started_at": oldest_iso,
            "gap_duration_minutes": gap_duration_minutes,
            "adapters": adapters_report,
            "status": status,
        }
