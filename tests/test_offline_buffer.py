"""Unit tests for edge offline buffer and replay service (no Docker)."""

import tempfile
import unittest
from unittest.mock import Mock

from edge.offline_buffer import OfflineBuffer
from edge.replay_service import ReplayService


class TestOfflineBuffer(unittest.TestCase):
    def test_buffer_write_and_read(self) -> None:
        buf = OfflineBuffer(db_path=":memory:")
        ts = ["2025-01-01T00:00:0{}Z".format(i) for i in range(1, 6)]
        for i, t in enumerate(ts):
            buf.write("m1", t, {"v": i}, "test")
        rows = buf.read_pending(limit=10)
        self.assertEqual(len(rows), 5)
        for i, r in enumerate(rows):
            self.assertEqual(r["timestamp"], ts[i])
        self.assertEqual(buf.pending_count(), 5)
        buf.close()

    def test_mark_replayed(self) -> None:
        buf = OfflineBuffer(db_path=":memory:")
        for i in range(10):
            buf.write("m1", "2025-01-01T00:00:0{}Z".format(i), {"v": i}, "test")
        rows = buf.read_pending(10)
        first_five_ids = [r["id"] for r in rows[:5]]
        buf.mark_replayed(first_five_ids)
        remaining = buf.read_pending(10)
        self.assertEqual(len(remaining), 5)
        self.assertEqual(buf.pending_count(), 5)
        buf.close()

    def test_replay_service_success(self) -> None:
        buf = OfflineBuffer(db_path=":memory:")
        for i in range(10):
            buf.write("m1", "2025-01-01T00:00:0{}Z".format(i), {"v": i}, "test")
        published: list = []
        def mock_publish(payload: dict) -> None:
            published.append(payload)
        svc = ReplayService(buf, mock_publish, interval_seconds=60, purge_days=7)
        svc.replay_pending()
        self.assertEqual(len(published), 10)
        self.assertEqual(published, [{"v": i} for i in range(10)])
        self.assertEqual(buf.pending_count(), 0)
        self.assertEqual(len(buf.read_pending(10)), 0)
        buf.close()

    def test_replay_service_failure_is_safe(self) -> None:
        buf = OfflineBuffer(db_path=":memory:")
        for i in range(10):
            buf.write("m1", "2025-01-01T00:00:0{}Z".format(i), {"v": i}, "test")
        call_count = 0
        def mock_publish(_: dict) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 5:
                raise ConnectionError("simulated")
        svc = ReplayService(buf, mock_publish, interval_seconds=60, purge_days=7)
        svc.replay_pending()
        self.assertEqual(call_count, 5)
        self.assertEqual(buf.pending_count(), 10)
        self.assertEqual(len(buf.read_pending(10)), 10)
        buf.close()

    def test_purge_replayed(self) -> None:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        path = tmp.name
        tmp.close()
        try:
            buf = OfflineBuffer(db_path=path)
            for i in range(20):
                buf.write("m1", "2025-01-01T00:00:0{}Z".format(i), {"v": i}, "test")
            rows = buf.read_pending(20)
            buf.mark_replayed([r["id"] for r in rows])
            import sqlite3
            conn = sqlite3.connect(path)
            conn.execute(
                "UPDATE buffered_readings SET buffered_at = datetime('now', '-10 days') WHERE replayed = 1"
            )
            conn.commit()
            conn.close()
            n = buf.purge_replayed(older_than_days=7)
            self.assertEqual(n, 20)
            self.assertEqual(buf.pending_count(), 0)
            buf.close()
        finally:
            import os
            try:
                os.unlink(path)
            except OSError:
                pass
