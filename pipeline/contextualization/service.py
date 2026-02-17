#!/usr/bin/env python3
"""
Golden Pipeline — Stage 3: Contextualization Service (Digital Twin)
====================================================================
Consumes cleaned sensor data from `clean_sensor_data`, enriches each
record by mapping raw machine IDs to human-readable asset metadata
using station_config.json, then publishes to `contextualized_data`.

Implements Store-and-Forward: if the persistence service (Stage 4)
reports unhealthy via Redis key, this service buffers records in a
Redis List and retries on a backoff schedule.

Redis Streams:
  IN:  clean_sensor_data      (consumer group: context_group)
  OUT: contextualized_data
"""

import os
import sys
import json
import time
import signal
import logging
from datetime import datetime, timezone

import redis

# =============================================================================
# CONFIGURATION
# =============================================================================

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

INPUT_STREAM = "clean_sensor_data"
OUTPUT_STREAM = "contextualized_data"
CONSUMER_GROUP = "context_group"
CONSUMER_NAME = os.getenv("CONSUMER_NAME", "context-worker-1")

# Store-and-forward buffer
BUFFER_KEY = "buffer:contextualized"
BUFFER_MAX_SIZE = int(os.getenv("BUFFER_MAX_SIZE", "10000"))
BUFFER_FLUSH_INTERVAL = int(os.getenv("BUFFER_FLUSH_INTERVAL", "10"))  # seconds
PERSISTENCE_HEALTH_KEY = "system:health:persistence"

# Station config path
STATION_CONFIG_PATH = os.getenv("STATION_CONFIG_PATH", "/app/station_config.json")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("context-svc")

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
# STATION CONFIG (Digital Twin Mapping)
# =============================================================================


def load_station_config(path: str) -> dict:
    """Load the digital twin mapping from station_config.json."""
    try:
        with open(path, "r") as f:
            config = json.load(f)
        mappings = config.get("node_mappings", {})
        logger.info("Loaded %d station mappings from %s", len(mappings), path)
        return mappings
    except FileNotFoundError:
        logger.warning("Station config not found at %s. Using empty mappings.", path)
        return {}
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in station config: %s", exc)
        return {}


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
# CONTEXTUALIZATION
# =============================================================================


def contextualize(payload: dict, mappings: dict) -> dict:
    """Enrich a sensor record with digital twin metadata."""
    machine_id = payload.get("machine_id", "")
    twin = mappings.get(machine_id, {})

    payload["asset_name"] = twin.get("asset_name", machine_id)
    payload["shop"] = twin.get("shop", "Unknown")
    payload["line"] = twin.get("line", "Unknown")
    payload["equipment_type"] = twin.get("equipment_type", "Unknown")
    payload["criticality"] = twin.get("criticality", "medium")
    payload["opc_node_id"] = twin.get("opc_node_id", "")
    payload["contextualized_at"] = datetime.now(timezone.utc).isoformat()

    return payload


# =============================================================================
# STORE-AND-FORWARD
# =============================================================================


def is_persistence_healthy(r: redis.Redis) -> bool:
    """Check if the persistence service has reported healthy recently."""
    try:
        ts = r.get(PERSISTENCE_HEALTH_KEY)
        if ts is None:
            return True  # Assume healthy if no key (persistence hasn't started yet)
        age = time.time() - float(ts)
        return age < 60  # Healthy if heartbeat within 60s
    except Exception:
        return True  # Assume healthy on error


def buffer_record(r: redis.Redis, payload: dict):
    """Buffer a contextualized record in Redis List for later replay."""
    try:
        r.rpush(BUFFER_KEY, json.dumps(payload))
        # Trim buffer if it grows beyond max size (drop oldest)
        r.ltrim(BUFFER_KEY, -BUFFER_MAX_SIZE, -1)
    except Exception as exc:
        logger.error("Failed to buffer record: %s", exc)


def flush_buffer(r: redis.Redis) -> int:
    """Attempt to flush buffered records to the output stream."""
    flushed = 0
    while _running:
        record = r.lpop(BUFFER_KEY)
        if record is None:
            break
        try:
            r.xadd(
                OUTPUT_STREAM,
                {"data": record},
                maxlen=50_000,
            )
            flushed += 1
        except Exception as exc:
            # Push back if we can't publish
            r.lpush(BUFFER_KEY, record)
            logger.warning("Buffer flush interrupted: %s", exc)
            break
    return flushed


# =============================================================================
# MAIN LOOP
# =============================================================================


def run():
    """Consume from clean_sensor_data, contextualize, and publish."""
    r = get_redis()
    ensure_consumer_group(r)
    mappings = load_station_config(STATION_CONFIG_PATH)

    total_read = 0
    total_published = 0
    last_buffer_flush = time.time()

    logger.info(
        "Contextualization service started | in=%s | out=%s | mappings=%d",
        INPUT_STREAM,
        OUTPUT_STREAM,
        len(mappings),
    )

    while _running:
        try:
            # Periodically try to flush the store-and-forward buffer
            if time.time() - last_buffer_flush > BUFFER_FLUSH_INTERVAL:
                buf_len = r.llen(BUFFER_KEY) or 0
                if buf_len > 0 and is_persistence_healthy(r):
                    flushed = flush_buffer(r)
                    if flushed > 0:
                        logger.info("Flushed %d buffered records.", flushed)
                last_buffer_flush = time.time()

            # Read from stream
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
                        continue

                    # Contextualize
                    enriched = contextualize(payload, mappings)

                    # Check if persistence is healthy
                    if is_persistence_healthy(r):
                        r.xadd(
                            OUTPUT_STREAM,
                            {"data": json.dumps(enriched)},
                            maxlen=50_000,
                        )
                        total_published += 1
                    else:
                        # Store-and-forward: buffer locally
                        buffer_record(r, enriched)
                        logger.debug(
                            "Persistence unhealthy — buffered record for %s",
                            enriched.get("machine_id"),
                        )

                    r.xack(INPUT_STREAM, CONSUMER_GROUP, msg_id)

                    if total_read % 100 == 0:
                        buf_len = r.llen(BUFFER_KEY) or 0
                        logger.info(
                            "Progress | read=%d published=%d buffered=%d",
                            total_read,
                            total_published,
                            buf_len,
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
        "Contextualization service stopped. read=%d published=%d",
        total_read,
        total_published,
    )


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    run()
