#!/usr/bin/env python3
"""
Golden Pipeline — Stage 2: Cleansing / Validation Service
==========================================================
Consumes raw sensor data from Redis Stream `raw_sensor_data`,
performs outlier detection and schema validation, then publishes
cleaned records to `clean_sensor_data`.

Outlier Detection:
  - Z-score on temperature, rotational_speed, torque (|z| > 3 → dropped)
  - Moving average smoothing on vibration_raw

Schema Validation:
  - Rejects packets missing required fields

Redis Streams:
  IN:  raw_sensor_data   (consumer group: cleansing_group)
  OUT: clean_sensor_data
"""

import os
import sys
import json
import time
import signal
import logging
from collections import deque
from typing import Optional

import redis
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

INPUT_STREAM = "raw_sensor_data"
OUTPUT_STREAM = "clean_sensor_data"
CONSUMER_GROUP = "cleansing_group"
CONSUMER_NAME = os.getenv("CONSUMER_NAME", "cleansing-worker-1")

# Z-score parameters
ZSCORE_THRESHOLD = float(os.getenv("ZSCORE_THRESHOLD", "3.0"))
WINDOW_SIZE = int(os.getenv("STATS_WINDOW_SIZE", "200"))

# Required fields for a valid sensor packet
REQUIRED_FIELDS = {"timestamp", "machine_id", "rotational_speed", "temperature", "torque"}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("cleansing-svc")

# =============================================================================
# GRACEFUL SHUTDOWN
# =============================================================================

_running = True


def _shutdown_handler(signum, frame):
    global _running
    logger.info("Shutdown signal received (signal=%s).", signum)
    _running = False


signal.signal(signal.SIGINT, _shutdown_handler)
signal.signal(signal.SIGTERM, _shutdown_handler)


# =============================================================================
# STATISTICS TRACKER (Rolling window for Z-score)
# =============================================================================


class RollingStats:
    """Maintains a rolling window of values for Z-score computation."""

    def __init__(self, window_size: int = WINDOW_SIZE):
        self._window_size = window_size
        self._windows: dict[str, deque] = {}

    def update(self, field: str, value: float) -> Optional[float]:
        """Add a value and return the Z-score, or None if insufficient data."""
        if field not in self._windows:
            self._windows[field] = deque(maxlen=self._window_size)
        win = self._windows[field]
        win.append(value)

        if len(win) < 10:
            return None  # Not enough data to compute meaningful Z-score

        arr = np.array(win)
        mean = arr.mean()
        std = arr.std()
        if std < 1e-9:
            return 0.0
        return float((value - mean) / std)


# =============================================================================
# REDIS CONNECTION
# =============================================================================


def get_redis() -> redis.Redis:
    """Connect to Redis with retry."""
    while _running:
        try:
            r = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                decode_responses=True,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            r.ping()
            logger.info("Connected to Redis at %s:%s", REDIS_HOST, REDIS_PORT)
            return r
        except redis.ConnectionError as exc:
            logger.warning("Redis not ready (%s). Retrying in 3s…", exc)
            time.sleep(3)
    sys.exit(0)


def ensure_consumer_group(r: redis.Redis):
    """Create the consumer group if it doesn't exist."""
    try:
        r.xgroup_create(INPUT_STREAM, CONSUMER_GROUP, id="0", mkstream=True)
        logger.info("Created consumer group '%s' on '%s'", CONSUMER_GROUP, INPUT_STREAM)
    except redis.ResponseError as exc:
        if "BUSYGROUP" in str(exc):
            logger.info("Consumer group '%s' already exists.", CONSUMER_GROUP)
        else:
            raise


# =============================================================================
# VALIDATION & CLEANING
# =============================================================================


def validate_schema(payload: dict) -> bool:
    """Check that all required fields are present."""
    missing = REQUIRED_FIELDS - set(payload.keys())
    if missing:
        logger.warning(
            "Dropping packet: missing fields %s | machine_id=%s",
            missing,
            payload.get("machine_id", "unknown"),
        )
        return False
    return True


def smooth_vibration(vibration_raw: list, kernel_size: int = 5) -> list:
    """Apply simple moving-average smoothing to vibration data."""
    if not vibration_raw or len(vibration_raw) < kernel_size:
        return vibration_raw
    arr = np.array(vibration_raw, dtype=np.float64)
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(arr, kernel, mode="same")
    return smoothed.tolist()


# =============================================================================
# MAIN LOOP
# =============================================================================


def run():
    """Consume from raw_sensor_data, clean/validate, publish to clean_sensor_data."""
    r = get_redis()
    ensure_consumer_group(r)
    stats = RollingStats(window_size=WINDOW_SIZE)

    total_read = 0
    total_passed = 0
    total_dropped = 0

    logger.info(
        "Cleansing service started | in=%s | out=%s | z-threshold=%.1f",
        INPUT_STREAM,
        OUTPUT_STREAM,
        ZSCORE_THRESHOLD,
    )

    while _running:
        try:
            # Read from stream (block up to 2 seconds)
            messages = r.xreadgroup(
                CONSUMER_GROUP,
                CONSUMER_NAME,
                {INPUT_STREAM: ">"},
                count=50,
                block=2000,
            )

            if not messages:
                continue

            for stream_name, entries in messages:
                for msg_id, fields in entries:
                    total_read += 1

                    try:
                        payload = json.loads(fields.get("data", "{}"))
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON in message %s. Dropping.", msg_id)
                        r.xack(INPUT_STREAM, CONSUMER_GROUP, msg_id)
                        total_dropped += 1
                        continue

                    # 1. Schema validation
                    if not validate_schema(payload):
                        r.xack(INPUT_STREAM, CONSUMER_GROUP, msg_id)
                        total_dropped += 1
                        continue

                    # 2. Z-score outlier detection
                    is_outlier = False
                    for field in ("temperature", "rotational_speed", "torque"):
                        value = payload.get(field)
                        if value is None:
                            continue
                        z = stats.update(field, float(value))
                        if z is not None and abs(z) > ZSCORE_THRESHOLD:
                            logger.info(
                                "Outlier detected: %s=%.2f (z=%.2f) | machine=%s. Dropping.",
                                field,
                                value,
                                z,
                                payload.get("machine_id"),
                            )
                            is_outlier = True
                            break

                    if is_outlier:
                        r.xack(INPUT_STREAM, CONSUMER_GROUP, msg_id)
                        total_dropped += 1
                        continue

                    # 3. Smooth vibration data
                    if "vibration_raw" in payload:
                        payload["vibration_raw"] = smooth_vibration(
                            payload["vibration_raw"]
                        )

                    # 4. Add cleansing metadata
                    payload["cleansed"] = True
                    payload["cleansed_by"] = CONSUMER_NAME

                    # 5. Publish to clean stream
                    r.xadd(
                        OUTPUT_STREAM,
                        {"data": json.dumps(payload)},
                        maxlen=50_000,
                    )
                    r.xack(INPUT_STREAM, CONSUMER_GROUP, msg_id)
                    total_passed += 1

                    if total_read % 100 == 0:
                        logger.info(
                            "Progress | read=%d passed=%d dropped=%d (%.1f%% pass rate)",
                            total_read,
                            total_passed,
                            total_dropped,
                            (total_passed / total_read * 100) if total_read > 0 else 0,
                        )

        except redis.ConnectionError as exc:
            logger.warning("Redis connection lost (%s). Reconnecting in 5s…", exc)
            time.sleep(5)
            r = get_redis()
            ensure_consumer_group(r)
        except Exception as exc:
            logger.error("Unexpected error: %s", exc, exc_info=True)
            time.sleep(2)

    logger.info(
        "Cleansing service stopped. read=%d passed=%d dropped=%d",
        total_read,
        total_passed,
        total_dropped,
    )


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    run()
