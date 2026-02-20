"""
Offline buffer for edge adapters.
SQLite-backed, thread-safe; persists readings when Redis/upstream is unreachable.
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(db_path: str) -> Path:
    """Resolve db_path relative to project root if not absolute."""
    p = Path(db_path)
    if p.is_absolute():
        return p
    return (_PROJECT_ROOT / db_path).resolve()


class OfflineBuffer:
    """
    SQLite-backed buffer for telemetry readings. Thread-safe via a single lock.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            from config import get_settings
            db_path = get_settings().edge.offline_buffer_path
        if db_path == ":memory:":
            self._path = Path(":memory:")
            self._lock = threading.Lock()
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            path = _resolve_path(db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._path = path
            self._lock = threading.Lock()
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS buffered_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    machine_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    buffered_at TEXT NOT NULL,
                    replayed INTEGER DEFAULT 0
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_replayed ON buffered_readings(replayed, timestamp)"
            )
            self._conn.commit()

    def write(self, machine_id: str, timestamp: str, payload_dict: dict[str, Any], source: str) -> int:
        """Insert one reading. Thread-safe. Returns row id."""
        payload_json = json.dumps(payload_dict)
        buffered_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO buffered_readings
                   (machine_id, timestamp, payload_json, source, buffered_at, replayed)
                   VALUES (?, ?, ?, ?, ?, 0)""",
                (machine_id, timestamp, payload_json, source, buffered_at),
            )
            self._conn.commit()
            return cur.lastrowid

    def read_pending(self, limit: int = 500) -> list[dict]:
        """Return up to limit unreplayed rows ordered by timestamp ASC."""
        with self._lock:
            cur = self._conn.execute(
                """SELECT id, machine_id, timestamp, payload_json, source, buffered_at
                   FROM buffered_readings WHERE replayed = 0
                   ORDER BY timestamp ASC LIMIT ?""",
                (limit,),
            )
            rows = cur.fetchall()
        columns = ["id", "machine_id", "timestamp", "payload_json", "source", "buffered_at"]
        out = []
        for row in rows:
            d = dict(zip(columns, row))
            d["payload"] = json.loads(d["payload_json"])
            out.append(d)
        return out

    def mark_replayed(self, ids: list[int]) -> None:
        """Set replayed=1 for the given row IDs in a single UPDATE."""
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            self._conn.execute(
                f"UPDATE buffered_readings SET replayed = 1 WHERE id IN ({placeholders})",
                ids,
            )
            self._conn.commit()

    def pending_count(self) -> int:
        """Return count of unreplayed rows."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM buffered_readings WHERE replayed = 0"
            )
            return cur.fetchone()[0]

    def oldest_buffered_at(self) -> Optional[datetime]:
        """Return buffered_at of the oldest unreplayed row, or None."""
        with self._lock:
            cur = self._conn.execute(
                """SELECT buffered_at FROM buffered_readings
                   WHERE replayed = 0 ORDER BY timestamp ASC LIMIT 1"""
            )
            row = cur.fetchone()
        if not row:
            return None
        try:
            return datetime.fromisoformat(row[0].replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    def purge_replayed(self, older_than_days: int = 7) -> int:
        """Delete replayed rows older than N days. Returns count deleted."""
        with self._lock:
            cur = self._conn.execute(
                """DELETE FROM buffered_readings
                   WHERE replayed = 1 AND datetime(buffered_at) < datetime('now', '-' || ? || ' days')""",
                (older_than_days,),
            )
            deleted = cur.rowcount
            self._conn.commit()
        return deleted

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()
