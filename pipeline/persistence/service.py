#!/usr/bin/env python3
"""
Golden Pipeline — Stage 4: Persistence Service
================================================
Dedicated writer that consumes contextualized sensor data from the
`contextualized_data` Redis Stream and batch-inserts into TimescaleDB.

Sets a health flag in Redis (`system:health:persistence`) so the
Contextualization service (Stage 3) knows if the DB is available for
store-and-forward decisions.

Redis Streams:
  IN: contextualized_data  (consumer group: persistence_group)
  OUT: TimescaleDB (sensor_readings table)
"""

import os
import sys
import json
import time
import signal
import logging
from datetime import datetime

import redis
import psycopg2
from psycopg2 import extras, OperationalError

# =============================================================================
# CONFIGURATION
# =============================================================================

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "pdm_timeseries")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")

INPUT_STREAM = "contextualized_data"
CONSUMER_GROUP = "persistence_group"
CONSUMER_NAME = os.getenv("CONSUMER_NAME", "persistence-worker-1")
PERSISTENCE_HEALTH_KEY = "system:health:persistence"

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
HEARTBEAT_INTERVAL = 15  # seconds

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("persistence-svc")

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
# DATABASE CONNECTION
# =============================================================================


def get_db_connection():
    """Create PostgreSQL/TimescaleDB connection with retry."""
    while _running:
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                connect_timeout=10,
            )
            conn.autocommit = False
            logger.info("Connected to TimescaleDB at %s:%s/%s", DB_HOST, DB_PORT, DB_NAME)
            return conn
        except OperationalError as exc:
            logger.warning("DB not ready (%s). Retrying in 5s…", exc)
            time.sleep(5)
    sys.exit(0)


def ensure_tables(conn):
    """Create the sensor_readings table if it doesn't exist."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensor_readings (
                timestamp       TIMESTAMPTZ NOT NULL,
                machine_id      TEXT NOT NULL,
                rotational_speed DOUBLE PRECISION,
                temperature      DOUBLE PRECISION,
                torque           DOUBLE PRECISION,
                tool_wear        DOUBLE PRECISION,
                vibration_raw    TEXT,
                asset_name       TEXT,
                shop             TEXT,
                line             TEXT,
                equipment_type   TEXT,
                criticality      TEXT,
                UNIQUE(machine_id, timestamp)
            );
        """)
        conn.commit()
        logger.info("Table 'sensor_readings' ensured.")
    except Exception as exc:
        conn.rollback()
        logger.warning("Table creation skipped (may already exist): %s", exc)
    finally:
        cursor.close()


# =============================================================================
# BATCH INSERT
# =============================================================================


def bulk_insert(conn, records: list) -> int:
    """Bulk insert sensor records into TimescaleDB. Returns count inserted."""
    if not records:
        return 0

    cursor = conn.cursor()
    query = """
        INSERT INTO sensor_readings
            (timestamp, machine_id, rotational_speed, temperature, torque,
             tool_wear, vibration_raw, asset_name, shop, line, equipment_type, criticality)
        VALUES %s
        ON CONFLICT (machine_id, timestamp) DO NOTHING
    """

    values = [
        (
            r.get("timestamp", datetime.now().isoformat()),
            r.get("machine_id", "unknown"),
            r.get("rotational_speed"),
            r.get("temperature"),
            r.get("torque"),
            r.get("tool_wear"),
            json.dumps(r.get("vibration_raw", [])),
            r.get("asset_name", ""),
            r.get("shop", ""),
            r.get("line", ""),
            r.get("equipment_type", ""),
            r.get("criticality", "medium"),
        )
        for r in records
    ]

    try:
        extras.execute_values(cursor, query, values, page_size=100)
        conn.commit()
        return len(values)
    except Exception as exc:
        conn.rollback()
        logger.error("Bulk insert failed: %s", exc)
        raise
    finally:
        cursor.close()


# =============================================================================
# MAIN LOOP
# =============================================================================


def run():
    """Consume from contextualized_data, batch-insert into TimescaleDB."""
    r = get_redis()
    ensure_consumer_group(r)
    conn = get_db_connection()
    ensure_tables(conn)

    buffer: list = []
    total_read = 0
    total_inserted = 0
    last_heartbeat = 0

    logger.info(
        "Persistence service started | in=%s | batch_size=%d",
        INPUT_STREAM,
        BATCH_SIZE,
    )

    while _running:
        try:
            # Heartbeat — tell other services we're alive
            if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
                try:
                    r.set(PERSISTENCE_HEALTH_KEY, str(int(time.time())), ex=60)
                    last_heartbeat = time.time()
                except Exception as hb_err:
                    logger.warning("Heartbeat failed: %s", hb_err)

            # Read from stream
            messages = r.xreadgroup(
                CONSUMER_GROUP,
                CONSUMER_NAME,
                {INPUT_STREAM: ">"},
                count=50,
                block=2000,
            )

            if not messages:
                # Flush any partial batch on idle
                if buffer:
                    try:
                        inserted = bulk_insert(conn, buffer)
                        total_inserted += inserted
                        logger.info(
                            "Flushed partial batch: %d records (total=%d)",
                            inserted,
                            total_inserted,
                        )
                        buffer.clear()
                    except (OperationalError, Exception) as db_err:
                        logger.warning("DB write failed, will retry: %s", db_err)
                        conn = get_db_connection()
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

                    buffer.append(payload)
                    r.xack(INPUT_STREAM, CONSUMER_GROUP, msg_id)

                    # Flush when batch is full
                    if len(buffer) >= BATCH_SIZE:
                        try:
                            inserted = bulk_insert(conn, buffer)
                            total_inserted += inserted
                            logger.info(
                                "Batch inserted: %d records (total=%d)",
                                inserted,
                                total_inserted,
                            )
                            buffer.clear()
                        except OperationalError as db_err:
                            logger.warning(
                                "DB connection lost during insert: %s. Reconnecting…",
                                db_err,
                            )
                            conn = get_db_connection()
                            # Retry the insert
                            try:
                                inserted = bulk_insert(conn, buffer)
                                total_inserted += inserted
                                buffer.clear()
                            except Exception:
                                logger.error(
                                    "Retry insert failed. %d records in buffer.",
                                    len(buffer),
                                )

        except redis.ConnectionError as exc:
            logger.warning("Redis connection lost (%s). Reconnecting in 5s…", exc)
            time.sleep(5)
            r = get_redis()
            ensure_consumer_group(r)
        except Exception as exc:
            logger.error("Unexpected error: %s", exc, exc_info=True)
            time.sleep(2)

    # Final flush on shutdown
    if buffer:
        logger.info("Final flush: %d records", len(buffer))
        try:
            bulk_insert(conn, buffer)
        except Exception as exc:
            logger.error("Final flush failed: %s", exc)

    if conn:
        conn.close()

    logger.info(
        "Persistence service stopped. read=%d inserted=%d",
        total_read,
        total_inserted,
    )


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    run()
