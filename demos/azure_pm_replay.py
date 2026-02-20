#!/usr/bin/env python3
"""
Azure PM Dataset Replay Engine.

Loads merged Azure PM data from TimescaleDB (sensor_readings, cwru_features),
replays it with time compression (e.g. 1 month = 1 hour), publishes to Redis
sensor_stream and optionally MQTT UNS. Annotates known failures to
events/failure_ground_truth for validation.

Usage:
    python -m demos.azure_pm_replay --speed-multiplier 720 --start-from "2023-01-01"
    python mock_fleet_streamer.py --source azure_pm --speed-multiplier 720 --start-from "2023-01-01"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Project root for imports
_BASE = Path(__file__).resolve().parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

logger = logging.getLogger(__name__)

REDIS_CHANNEL = "sensor_stream"
FAILURE_GROUND_TRUTH_TOPIC = "PlantAGI/Demo/events/failure_ground_truth"


def _get_redis():
    import redis
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD") or None
    return redis.Redis(host=host, port=port, password=password, decode_responses=True)


def _get_db_connection():
    from config import get_settings
    s = get_settings().database
    import psycopg2
    return psycopg2.connect(
        host=s.host,
        port=s.port,
        dbname=s.name,
        user=s.user,
        password=s.password.get_secret_value(),
        connect_timeout=10,
    )


def _load_failure_set(conn, start_from: datetime | None) -> set[tuple[str, datetime]]:
    """Load (machine_id, timestamp) where failure_class = 1 for O(1) lookup."""
    cur = conn.cursor()
    if start_from:
        cur.execute(
            """
            SELECT machine_id, timestamp FROM cwru_features
            WHERE failure_class = 1 AND timestamp >= %s
            """,
            (start_from,),
        )
    else:
        cur.execute(
            "SELECT machine_id, timestamp FROM cwru_features WHERE failure_class = 1"
        )
    rows = cur.fetchall()
    cur.close()
    # Normalize timestamps to naive for comparison (sensor_readings may use same tz)
    out = set()
    for machine_id, ts in rows:
        if hasattr(ts, "replace"):
            ts = ts.replace(tzinfo=None) if ts.tzinfo else ts
        out.add((str(machine_id), ts))
    return out


def _parse_vibration_raw(raw) -> list:
    """Turn DB vibration_raw (JSON array or scalar) into a list for payload."""
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
            return v if isinstance(v, list) else [float(v)]
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(raw, list):
        return [float(x) for x in raw]
    return [float(raw)]


class AzurePMReplayEngine:
    """
    Replays Azure PM sensor_readings from TimescaleDB with time compression.
    Publishes to Redis sensor_stream and optionally MQTT UNS; emits
    failure_ground_truth for known failure timestamps.
    """

    def __init__(
        self,
        *,
        speed_multiplier: float = 720.0,
        start_from: str | None = None,
        redis_url: str | None = None,
        redis_host: str | None = None,
        redis_port: int | None = None,
        redis_password: str | None = None,
        use_mqtt: bool = False,
        config_path: str | None = None,
    ):
        self.speed_multiplier = max(0.001, float(speed_multiplier))
        self.start_from_str = start_from
        self.redis_url = redis_url
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_password = redis_password
        self.start_from_dt: datetime | None = None
        if start_from:
            try:
                self.start_from_dt = datetime.fromisoformat(
                    start_from.replace("Z", "+00:00")
                )
                if self.start_from_dt.tzinfo is None:
                    self.start_from_dt = self.start_from_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                self.start_from_dt = None
        self.use_mqtt = use_mqtt
        self.config_path = config_path or os.getenv(
            "STATION_CONFIG_PATH", str(_BASE / "pipeline" / "station_config.json")
        )
        self._failure_set: set[tuple[str, datetime]] = set()
        self._mqtt = None
        self._uns_mapper = None

    def _init_mqtt(self) -> None:
        if not self.use_mqtt:
            return
        try:
            from config import get_settings
            from mqtt_client import MQTTPublisher
            from uns_mapper import UNSMapper
            cfg = get_settings().mqtt
            self._mqtt = MQTTPublisher()
            if self._mqtt.connect():
                self._uns_mapper = UNSMapper(config_path=self.config_path)
                logger.info("MQTT and UNS mapper initialized for replay")
            else:
                self._mqtt = None
        except Exception as e:
            logger.warning("MQTT init failed for replay: %s", e)
            self._mqtt = None
            self._uns_mapper = None

    def load_rows(self):
        """Yield (machine_id, timestamp, payload_dict) from sensor_readings ordered by timestamp."""
        conn = _get_db_connection()
        try:
            self._failure_set = _load_failure_set(conn, self.start_from_dt)
            logger.info("Loaded %d failure (machine_id, timestamp) for ground truth", len(self._failure_set))

            cur = conn.cursor()
            if self.start_from_dt:
                # Use naive for comparison if DB stores without tz
                start_naive = self.start_from_dt.replace(tzinfo=None) if self.start_from_dt.tzinfo else self.start_from_dt
                cur.execute(
                    """
                    SELECT machine_id, timestamp, rotational_speed, temperature, torque, tool_wear, vibration_raw
                    FROM sensor_readings
                    WHERE timestamp >= %s
                    ORDER BY timestamp
                    """,
                    (start_naive,),
                )
            else:
                cur.execute(
                    """
                    SELECT machine_id, timestamp, rotational_speed, temperature, torque, tool_wear, vibration_raw
                    FROM sensor_readings
                    ORDER BY timestamp
                    """
                )
            while True:
                row = cur.fetchone()
                if not row:
                    break
                machine_id, ts, rot, temp, torque, tool_wear, vib_raw = row
                machine_id = str(machine_id)
                if hasattr(ts, "isoformat"):
                    ts_iso = ts.isoformat() if ts.tzinfo else ts.isoformat() + "Z"
                else:
                    ts_iso = str(ts)
                vibration_list = _parse_vibration_raw(vib_raw)
                payload = {
                    "machine_id": machine_id,
                    "timestamp": ts_iso,
                    "rotational_speed": float(rot) if rot is not None else None,
                    "temperature": float(temp) if temp is not None else None,
                    "torque": float(torque) if torque is not None else None,
                    "tool_wear": float(tool_wear) if tool_wear is not None else None,
                    "vibration_raw": vibration_list,
                }
                yield machine_id, ts, payload
            cur.close()
        finally:
            conn.close()

    def _get_redis_client(self):
        """Build Redis client from redis_url or host/port/password (or env)."""
        import redis
        if self.redis_url:
            return redis.from_url(self.redis_url, decode_responses=True)
        host = self.redis_host or os.getenv("REDIS_HOST", "localhost")
        port = self.redis_port if self.redis_port is not None else int(os.getenv("REDIS_PORT", "6379"))
        password = self.redis_password or os.getenv("REDIS_PASSWORD") or None
        return redis.Redis(host=host, port=port, password=password, decode_responses=True)

    def run(self) -> None:
        """Run replay loop: sleep by compressed time, publish each row to Redis (and optional MQTT)."""
        redis_client = self._get_redis_client()
        self._init_mqtt()

        start_wall = time.monotonic()
        t0_hist = None
        published = 0
        failures_emitted = 0

        for machine_id, ts_hist, payload in self.load_rows():
            # First row: set t0
            if t0_hist is None:
                t0_hist = ts_hist
                if hasattr(t0_hist, "tzinfo") and t0_hist.tzinfo:
                    t0_hist = t0_hist.replace(tzinfo=None)

            # Compressed sleep: (ts_hist - t0_hist) / speed_multiplier = elapsed real
            ts_naive = ts_hist.replace(tzinfo=None) if hasattr(ts_hist, "tzinfo") and ts_hist.tzinfo else ts_hist
            delta_hist = (ts_naive - t0_hist).total_seconds()
            target_elapsed_real = delta_hist / self.speed_multiplier
            elapsed_real = time.monotonic() - start_wall
            sleep_secs = target_elapsed_real - elapsed_real
            if sleep_secs > 0:
                time.sleep(sleep_secs)

            # Publish to Redis (primary path for stream_consumer)
            try:
                redis_client.publish(REDIS_CHANNEL, json.dumps(payload))
                published += 1
            except Exception as e:
                logger.warning("Redis publish failed: %s", e)
                break

            # Optional MQTT UNS
            if self._mqtt and self._mqtt.is_connected and self._uns_mapper:
                try:
                    for topic, value, retain in self._uns_mapper.explode(machine_id, payload):
                        self._mqtt.publish(topic, value, qos=1, retain=retain)
                except Exception as e:
                    logger.debug("MQTT UNS publish failed (non-fatal): %s", e)

            # Failure ground truth
            if (machine_id, ts_naive) in self._failure_set:
                gt_payload = {
                    "machine_id": machine_id,
                    "timestamp": payload["timestamp"],
                    "failure": True,
                }
                try:
                    redis_client.publish("events/failure_ground_truth", json.dumps(gt_payload))
                except Exception:
                    pass
                if self._mqtt and self._mqtt.is_connected:
                    try:
                        self._mqtt.publish(FAILURE_GROUND_TRUTH_TOPIC, gt_payload, qos=1, retain=True)
                        failures_emitted += 1
                    except Exception:
                        pass

            if published % 10000 == 0 and published > 0:
                logger.info("Replay progress: %d published, %d failure_ground_truth", published, failures_emitted)

        if self._mqtt:
            try:
                self._mqtt.disconnect()
            except Exception:
                pass
        logger.info("Replay finished: %d messages published", published)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Azure PM dataset replay: stream historical sensor_readings with time compression"
    )
    parser.add_argument(
        "--speed-multiplier",
        type=float,
        default=720.0,
        help="Time compression: 720 = 1 month per hour (default)",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        help='Start replay from date, e.g. "2023-01-01"',
    )
    parser.add_argument(
        "--redis-url",
        type=str,
        default=None,
        help="Redis URL (optional; uses REDIS_HOST/REDIS_PORT if not set)",
    )
    parser.add_argument(
        "--mqtt",
        action="store_true",
        help="Enable MQTT UNS publish and failure_ground_truth topic",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to station_config.json for UNS mapping",
    )
    args = parser.parse_args()

    engine = AzurePMReplayEngine(
        speed_multiplier=args.speed_multiplier,
        start_from=args.start_from,
        redis_url=args.redis_url,
        use_mqtt=args.mqtt,
        config_path=args.config,
    )
    try:
        engine.run()
        return 0
    except KeyboardInterrupt:
        print("\nReplay stopped by user.", file=sys.stderr)
        return 0
    except Exception as e:
        logger.exception("Replay failed")
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
