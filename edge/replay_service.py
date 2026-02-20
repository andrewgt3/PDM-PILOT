"""
Replay service for edge adapters.
Drains offline buffer in timestamp order when is_online; at-least-once delivery.
"""

import logging
import threading
from typing import Callable, Optional

from edge.offline_buffer import OfflineBuffer

logger = logging.getLogger(__name__)


class ReplayService:
    """
    Replays buffered readings via a publish callable when is_online.
    Runs replay_pending() in a background thread at interval_seconds.
    """

    def __init__(
        self,
        buffer: OfflineBuffer,
        publish_callable: Callable[[dict], None],
        interval_seconds: float = 30,
        purge_days: int = 7,
    ) -> None:
        self._buffer = buffer
        self._publish = publish_callable
        self._interval = interval_seconds
        self._purge_days = purge_days
        self._is_online = True
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False
        self._lock = threading.Lock()

    @property
    def is_online(self) -> bool:
        return self._is_online

    @is_online.setter
    def is_online(self, value: bool) -> None:
        self._is_online = value

    def replay_pending(self) -> None:
        """
        Read up to 500 pending rows, publish each; if all succeed, mark replayed.
        If any fail, do not mark (will retry next call). Then purge old replayed rows.
        """
        batch = self._buffer.read_pending(limit=500)
        if not batch:
            try:
                self._buffer.purge_replayed(older_than_days=self._purge_days)
            except Exception as e:
                logger.warning("Purge replayed failed: %s", e)
            return
        ids = [r["id"] for r in batch]
        try:
            for r in batch:
                payload = r.get("payload") or r.get("payload_json")
                if isinstance(payload, str):
                    import json
                    payload = json.loads(payload)
                self._publish(payload)
            self._buffer.mark_replayed(ids)
        except Exception as e:
            logger.warning("Replay failed, will retry: %s", e)
            return
        try:
            self._buffer.purge_replayed(older_than_days=self._purge_days)
        except Exception as e:
            logger.warning("Purge replayed failed: %s", e)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            if self._is_online:
                try:
                    self.replay_pending()
                except Exception as e:
                    logger.warning("Replay loop error: %s", e)
            if self._stop_event.wait(timeout=self._interval):
                break

    def start_replay_loop(self, interval_seconds: float | None = None) -> None:
        """Start background replay loop. Idempotent."""
        if interval_seconds is not None:
            self._interval = interval_seconds
        with self._lock:
            if self._started:
                return
            self._started = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the replay thread to exit and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 2)
            self._thread = None
        with self._lock:
            self._started = False
