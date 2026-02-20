#!/usr/bin/env python3
"""
Siemens S7 PLC Adapter
Connects to a Siemens S7-1500 PLC via native S7 protocol (python-snap7) and publishes
normalized telemetry to the same Redis channel as the ABB adapter.

Structural pattern matches abb_adapter.py: outer reconnect loop with exponential backoff,
inner poll loop, same Redis publish channel (sensor_stream), same error handling approach.

Usage:
    python siemens_s7_adapter.py
    python siemens_s7_adapter.py --once   # one read, print JSON, exit

Configuration:
    SIEMENS_PLC_IP, SIEMENS_PLC_RACK, SIEMENS_PLC_SLOT, SIEMENS_DB_NUMBER in .env / config.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict

import redis
import snap7
from snap7.util import get_bool, get_dword, get_real

from config import get_settings
from edge.offline_buffer import OfflineBuffer
from edge.replay_service import ReplayService

# Load settings (same pattern as abb_adapter)
settings = get_settings()
siemens_settings = settings.siemens_s7
redis_settings = settings.redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("siemens_s7_adapter")

# Redis connection (same pattern as abb_adapter)
r = redis.Redis(
    host=redis_settings.host,
    port=redis_settings.port,
    password=redis_settings.password.get_secret_value() if redis_settings.password else None,
    decode_responses=True,
)

# DB layout: S7 big-endian; snap7 returns bytearray, util getters handle byte order.
# DBD[0]   = REAL (4)  -> motor_current (Amps)
# DBD[4]   = DWORD (4) -> cycle_timer_ms
# DBX[8.0] = BOOL      -> photo_eye_trigger
# DBX[8.1] = BOOL      -> fault_active
# DBD[10]  = REAL (4)  -> conveyor_speed (m/s)
# DBD[14]  = REAL (4)  -> motor_temperature_c (Celsius)
# DBD[18]  = REAL (4)  -> vibration_rms (mm/s)
DB_READ_START = 0
DB_READ_SIZE = 22

# Polling interval (100 ms = 10 Hz, configurable to match ABB)
DEFAULT_POLL_INTERVAL_S = 0.1

# Exponential backoff: 2s, 4s, 8s, ..., max 60s
BACKOFF_INITIAL_S = 2.0
BACKOFF_MAX_S = 60.0


def _parse_db_bytes(data: bytearray) -> Dict[str, Any]:
    """
    Parse raw DB bytes into telemetry dict. S7 uses big-endian; snap7.util getters handle it.
    """
    if data is None or len(data) < DB_READ_SIZE:
        raise ValueError(f"Need at least {DB_READ_SIZE} bytes, got {len(data) if data else 0}")
    return {
        "motor_current": get_real(data, 0),
        "cycle_timer_ms": get_dword(data, 4),
        "photo_eye_trigger": get_bool(data, 8, 0),
        "fault_active": get_bool(data, 8, 1),
        "conveyor_speed": get_real(data, 10),
        "motor_temperature_c": get_real(data, 14),
        "vibration_rms": get_real(data, 18),
    }


def build_payload(telemetry: Dict[str, Any], machine_id: str = "SIEMENS-PLC-001") -> Dict[str, Any]:
    """Build same top-level structure as ABB: machine_id, timestamp, source, telemetry nested."""
    return {
        "machine_id": machine_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "siemens_s7",
        "telemetry": telemetry,
    }


class SiemensS7Adapter:
    """
    Adapter for Siemens S7-1500 via python-snap7.
    Same structural pattern as ABB adapter: connect loop, poll loop, normalized JSON to Redis.
    """

    def __init__(
        self,
        plc_ip: str | None = None,
        rack: int | None = None,
        slot: int | None = None,
        db_number: int | None = None,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        redis_client: redis.Redis | None = None,
    ):
        self.plc_ip = plc_ip or siemens_settings.plc_ip
        self.rack = rack if rack is not None else siemens_settings.plc_rack
        self.slot = slot if slot is not None else siemens_settings.plc_slot
        self.db_number = db_number if db_number is not None else siemens_settings.db_number
        self.poll_interval_s = poll_interval_s
        self._redis = redis_client or r
        self._client: snap7.Client | None = None
        self._backoff_sec = BACKOFF_INITIAL_S
        self._offline_buffer: OfflineBuffer | None = None
        self._replay_service: ReplayService | None = None
        if settings.edge.offline_buffer_enabled:
            self._offline_buffer = OfflineBuffer(db_path="data/offline_buffer_s7.db")
            def _publish(payload_dict: Dict[str, Any]) -> None:
                self._redis.publish("sensor_stream", json.dumps(payload_dict))
            self._replay_service = ReplayService(
                self._offline_buffer,
                _publish,
                interval_seconds=settings.edge.offline_replay_interval_seconds,
                purge_days=settings.edge.offline_buffer_purge_days,
            )
            self._replay_service.is_online = True

    def _connect(self) -> bool:
        """Establish connection to PLC. Returns True if connected."""
        try:
            if self._client is None:
                self._client = snap7.Client()
            self._client.connect(self.plc_ip, self.rack, self.slot)
            logger.info(
                "Connected to Siemens S7 PLC at %s (rack=%s, slot=%s, db=%s)",
                self.plc_ip,
                self.rack,
                self.slot,
                self.db_number,
            )
            self._backoff_sec = BACKOFF_INITIAL_S
            return True
        except Exception as e:
            logger.error(
                "Siemens S7 connection failed: %s (plc_ip=%s, rack=%s, slot=%s)",
                e,
                self.plc_ip,
                self.rack,
                self.slot,
            )
            return False

    def _disconnect(self) -> None:
        """Disconnect from PLC (same error handling approach as ABB)."""
        if self._client is None:
            return
        try:
            self._client.disconnect()
        except Exception as e:
            logger.warning("Disconnect error: %s", e)
        self._client = None

    def _read_db(self) -> bytearray | None:
        """Read DB block; returns raw bytes or None on failure."""
        if self._client is None:
            return None
        try:
            data = self._client.db_read(self.db_number, DB_READ_START, DB_READ_SIZE)
            if data is None or len(data) < DB_READ_SIZE:
                logger.error(
                    "Siemens DB read returned insufficient data (expected %s, got %s)",
                    DB_READ_SIZE,
                    len(data) if data else 0,
                )
                return None
            return data
        except Exception as e:
            logger.error("Siemens DB read failed: %s (db_number=%s)", e, self.db_number)
            return None

    def _publish(self, payload: Dict[str, Any]) -> None:
        """Publish normalized JSON to Redis sensor_stream; on failure buffer offline."""
        try:
            self._redis.publish("sensor_stream", json.dumps(payload))
            if self._replay_service is not None:
                self._replay_service.start_replay_loop()
        except Exception as e:
            logger.error("Redis publish failed: %s", e)
            if self._offline_buffer is not None and self._replay_service is not None:
                self._replay_service.is_online = False
                machine_id = payload.get("machine_id", "unknown")
                ts = payload.get("timestamp", datetime.now(timezone.utc).isoformat())
                row_id = self._offline_buffer.write(machine_id, ts, payload, "siemens_s7")
                n = self._offline_buffer.pending_count()
                logger.warning("Offline — buffered reading %s. Pending count: %s", row_id, n)

    def test_read(self) -> Dict[str, Any] | None:
        """
        Read one sample and return it without starting the polling loop.
        Used by verify_connections.py. Connects, reads once, disconnects.
        Returns payload dict (machine_id, timestamp, source, telemetry) or None on failure.
        """
        if not self._connect():
            return None
        try:
            data = self._read_db()
            if data is None:
                return None
            telemetry = _parse_db_bytes(data)
            return build_payload(telemetry, machine_id="SIEMENS-PLC-001")
        finally:
            self._disconnect()

    def run_forever(self) -> None:
        """
        Main loop: connect with exponential backoff (2s, 4s, 8s, ..., max 60s),
        poll at poll_interval_s, publish normalized payload; on read/connection error,
        disconnect and retry with backoff (same design as ABB outer/inner loop).
        """
        logger.info(
            "Starting Siemens S7 Adapter for %s (db=%s, poll_interval=%.2fs)",
            self.plc_ip,
            self.db_number,
            self.poll_interval_s,
        )

        while True:
            if not self._connect():
                logger.info("Retrying in %.1f seconds...", self._backoff_sec)
                time.sleep(self._backoff_sec)
                self._backoff_sec = min(self._backoff_sec * 2, BACKOFF_MAX_S)
                continue
            if self._replay_service is not None:
                self._replay_service.is_online = True
                self._replay_service.start_replay_loop()

            while True:
                try:
                    data = self._read_db()
                    if data is None:
                        break
                    telemetry = _parse_db_bytes(data)
                    payload = build_payload(telemetry)
                    self._publish(payload)
                except Exception as e:
                    logger.error("Error reading/publishing data: %s", e)
                    break

                time.sleep(self.poll_interval_s)

            self._disconnect()
            logger.info("Retrying in %.1f seconds...", self._backoff_sec)
            time.sleep(self._backoff_sec)
            self._backoff_sec = min(self._backoff_sec * 2, BACKOFF_MAX_S)


def main() -> None:
    """Entry point: run adapter forever (same pattern as abb_adapter main loop)."""
    adapter = SiemensS7Adapter()
    adapter.run_forever()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        adapter = SiemensS7Adapter()
        result = adapter.test_read()
        if result is not None:
            print(json.dumps(result, indent=2))
        else:
            print("{}", file=sys.stderr)
            sys.exit(1)
    else:
        main()
