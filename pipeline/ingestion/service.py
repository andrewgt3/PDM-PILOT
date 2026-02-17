#!/usr/bin/env python3
"""
Golden Pipeline — Stage 1: Ingestion Service
=============================================
Standalone service that polls sensor data and publishes raw JSON
to the `raw_sensor_data` Redis Stream.

In production: polls OPC UA server via asyncua.
In demo/dev: replays training_data.pkl at 10Hz.

Redis Stream Topic: raw_sensor_data
"""

import os
import sys
import json
import time
import signal
import logging
from datetime import datetime, timezone

import redis
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
STREAM_TOPIC = "raw_sensor_data"
PUBLISH_RATE_HZ = float(os.getenv("PUBLISH_RATE_HZ", "10"))
DATA_FILE = os.getenv("DATA_FILE", "training_data.pkl")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("ingestion-svc")

# =============================================================================
# AUTOMOTIVE PLANT TOPOLOGY
# Realistic manufacturing plant structure with generic equipment names
# =============================================================================

FLEET_TOPOLOGY = [
    # BODY SHOP - Welding and Assembly
    {
        "id": "WB-001", "name": "6-Axis Welder #1", "shop": "Body Shop",
        "line": "Underbody Weld Cell", "type": "Spot Welder", "health": "healthy"
    },
    {
        "id": "WB-002", "name": "6-Axis Welder #2", "shop": "Body Shop",
        "line": "Underbody Weld Cell", "type": "Spot Welder", "health": "healthy"
    },
    {
        "id": "WB-003", "name": "Seam Sealer Robot", "shop": "Body Shop",
        "line": "Sealer Cell", "type": "Sealer", "health": "warning"
    },
    # PAINT SHOP
    {
        "id": "PT-001", "name": "E-Coat Pump #1", "shop": "Paint Shop",
        "line": "E-Coat Line", "type": "Chemical Pump", "health": "healthy"
    },
    {
        "id": "PT-002", "name": "Spray Robot #1", "shop": "Paint Shop",
        "line": "Topcoat Booth", "type": "Paint Robot", "health": "warning"
    },
    # GENERAL ASSEMBLY
    {
        "id": "GA-001", "name": "Chassis Carrier #1", "shop": "General Assembly",
        "line": "Trim Line", "type": "Carrier", "health": "healthy"
    },
    {
        "id": "GA-002", "name": "Torque Gun Station", "shop": "General Assembly",
        "line": "Chassis Line", "type": "Torque Tool", "health": "healthy"
    },
    # MACHINING
    {
        "id": "CNC-001", "name": "HAAS VF-2 Mill", "shop": "Machining",
        "line": "Line A (Precision)", "type": "CNC Mill", "health": "healthy"
    },
    {
        "id": "CNC-002", "name": "Mazak Lathe", "shop": "Machining",
        "line": "Line A (Precision)", "type": "CNC Lathe", "health": "warning"
    },
    # FINAL ASSEMBLY
    {
        "id": "TS-001", "name": "Torque Station #1", "shop": "Final Assembly",
        "line": "Chassis Line", "type": "Torque Tool", "health": "warning"
    },
    {
        "id": "LA-003", "name": "Lift Assist #3", "shop": "Final Assembly",
        "line": "Chassis Line", "type": "Ergonomic Lift", "health": "healthy"
    },
    # CRITICAL EQUIPMENT
    {
        "id": "CV-100", "name": "Main Conveyor Drive", "shop": "Final Assembly",
        "line": "Chassis Line", "type": "Conveyor Motor", "health": "critical"
    },
]


# =============================================================================
# GRACEFUL SHUTDOWN
# =============================================================================

_running = True


def _shutdown_handler(signum, frame):
    global _running
    logger.info("Shutdown signal received (signal=%s). Draining…", signum)
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


# =============================================================================
# DATA LOADING
# =============================================================================


def load_sensor_data(filepath: str) -> tuple:
    """Load training data and split by health status."""
    try:
        df = pd.read_pickle(filepath)
        logger.info("Loaded %d records from %s", len(df), filepath)
    except FileNotFoundError:
        logger.error("Data file %s not found. Exiting.", filepath)
        sys.exit(1)

    if "label" in df.columns:
        healthy = df[df["label"] == 0]
        faulty = df[df["label"] == 1]
    else:
        healthy = df
        faulty = df.sample(frac=0.2)

    logger.info("Healthy pool: %d | Faulty pool: %d", len(healthy), len(faulty))
    return healthy, faulty


# =============================================================================
# PAYLOAD CREATION
# =============================================================================

# Vibration simulation parameters
SAMPLE_RATE = 12_000
N_SAMPLES = 4096


def _generate_vibration(health: str) -> list:
    """Generate synthetic vibration waveform based on equipment health."""
    t = np.linspace(0, N_SAMPLES / SAMPLE_RATE, N_SAMPLES, endpoint=False)
    base = np.sin(2 * np.pi * 100 * t) * 0.3  # 100 Hz fundamental

    if health == "healthy":
        noise = np.random.normal(0, 0.05, N_SAMPLES)
        return (base + noise).tolist()
    elif health == "warning":
        # Add bearing fault frequency
        fault = np.sin(2 * np.pi * 236.4 * t) * 0.15
        noise = np.random.normal(0, 0.08, N_SAMPLES)
        return (base + fault + noise).tolist()
    else:  # critical
        fault1 = np.sin(2 * np.pi * 236.4 * t) * 0.35
        fault2 = np.sin(2 * np.pi * 162.2 * t) * 0.20
        noise = np.random.normal(0, 0.15, N_SAMPLES)
        return (base + fault1 + fault2 + noise).tolist()


def create_payload(
    row: pd.Series,
    index: int,
    healthy_pool: pd.DataFrame,
    faulty_pool: pd.DataFrame,
) -> dict:
    """Create JSON-serializable payload from DataFrame row."""
    equip = FLEET_TOPOLOGY[index % len(FLEET_TOPOLOGY)]
    health = equip["health"]

    # Select appropriate sample based on health profile
    if health == "critical":
        sample = faulty_pool.iloc[np.random.randint(len(faulty_pool))]
    elif health == "warning":
        pool = healthy_pool if np.random.random() > 0.5 else faulty_pool
        sample = pool.iloc[np.random.randint(len(pool))]
    else:
        sample = healthy_pool.iloc[np.random.randint(len(healthy_pool))]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine_id": equip["id"],
        "rotational_speed": float(sample.get("rotational_speed", 1800.0)),
        "temperature": float(sample.get("air_temperature", 300.0)),
        "torque": float(sample.get("torque", 40.0)),
        "tool_wear": float(sample.get("tool_wear", 0)),
        "vibration_raw": _generate_vibration(health),
        "source": "ingestion-svc",
    }


# =============================================================================
# MAIN PUBLISHING LOOP
# =============================================================================


def run():
    """Main publishing loop — pushes raw sensor data to Redis Stream."""
    r = get_redis()
    healthy_pool, faulty_pool = load_sensor_data(DATA_FILE)
    interval = 1.0 / PUBLISH_RATE_HZ
    cycle = 0
    published = 0

    logger.info(
        "Starting ingestion loop at %.1f Hz → stream '%s'",
        PUBLISH_RATE_HZ,
        STREAM_TOPIC,
    )

    while _running:
        try:
            for idx, (_, row) in enumerate(healthy_pool.iterrows()):
                if not _running:
                    break

                payload = create_payload(row, idx, healthy_pool, faulty_pool)

                # XADD to Redis Stream (auto-generated ID with *)
                r.xadd(
                    STREAM_TOPIC,
                    {"data": json.dumps(payload)},
                    maxlen=50_000,  # Cap stream length for edge memory
                )
                published += 1

                if published % 100 == 0:
                    logger.info(
                        "Published %d records | cycle=%d | stream=%s",
                        published,
                        cycle,
                        STREAM_TOPIC,
                    )

                time.sleep(interval)

            cycle += 1
            logger.info("Completed cycle %d (%d total records)", cycle, published)

        except redis.ConnectionError as exc:
            logger.warning("Redis connection lost (%s). Reconnecting in 5s…", exc)
            time.sleep(5)
            r = get_redis()
        except Exception as exc:
            logger.error("Unexpected error: %s. Retrying in 5s…", exc, exc_info=True)
            time.sleep(5)

    logger.info("Ingestion service stopped. Total published: %d", published)


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    run()
