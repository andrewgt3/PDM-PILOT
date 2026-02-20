#!/usr/bin/env python3
"""
Redis Stream Publisher for GAIA Predictive Maintenance

Simulates real-time sensor data streaming by:
- Loading training_data.pkl
- Publishing JSON records to Redis channel at 10Hz
- Optionally publishing to MQTT via UNS (ISA-95 topic hierarchy)

Author: Senior Backend Engineer
"""

import json
import logging
import os
import time

import numpy as np
import pandas as pd
import redis
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MQTT is optional; use config for toggle. UNS mapper for ISA-95 topic hierarchy.
try:
    from config import get_settings
    from mqtt_client import MQTTPublisher
    from uns_mapper import UNSMapper
except ImportError:
    get_settings = None
    MQTTPublisher = None
    UNSMapper = None

logger = logging.getLogger("stream_publisher")


# =============================================================================
# CONFIGURATION (Redis and rate from config)
# =============================================================================
from config import get_settings
_settings = get_settings()
# Set to "1" or "true" to start ABB + Siemens S7 adapters in background threads
# alongside the publisher (all publish to sensor_stream).
RUN_LIVE_ADAPTERS = os.getenv('RUN_LIVE_ADAPTERS', '').lower() in ('1', 'true', 'yes')

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
        "id": "WB-003", "name": "Frame Welder", "shop": "Body Shop",
        "line": "Underbody Weld Cell", "type": "MIG Welder", "health": "healthy"
    },
    # STAMPING SHOP - Metal Forming
    {
        "id": "HP-200", "name": "Hydraulic Press 2000T", "shop": "Stamping",
        "line": "Press Line 1", "type": "Hydraulic Press", "health": "healthy"
    },
    {
        "id": "TD-450", "name": "Transfer Die Unit", "shop": "Stamping",
        "line": "Press Line 1", "type": "Transfer Die", "health": "healthy"
    },
    # PAINT SHOP - Surface Treatment
    {
        "id": "PR-101", "name": "Paint Robot #1", "shop": "Paint Shop",
        "line": "Sealer Line", "type": "Paint Applicator", "health": "healthy"
    },
    {
        "id": "CO-050", "name": "Curing Oven", "shop": "Paint Shop",
        "line": "Sealer Line", "type": "Thermal Oven", "health": "warning"
    },
    # FINAL ASSEMBLY - Vehicle Completion
    {
        "id": "TS-001", "name": "Torque Station #1", "shop": "Final Assembly",
        "line": "Chassis Line", "type": "Torque Tool", "health": "warning"
    },
    {
        "id": "LA-003", "name": "Lift Assist #3", "shop": "Final Assembly",
        "line": "Chassis Line", "type": "Ergonomic Lift", "health": "healthy"
    },
    # CRITICAL EQUIPMENT - For demo purposes
    {
        "id": "CV-100", "name": "Main Conveyor Drive", "shop": "Final Assembly",
        "line": "Chassis Line", "type": "Conveyor Motor", "health": "critical"
    },
]


def get_redis_connection():
    """Create Redis connection from config."""
    s = get_settings()
    pw = s.redis.password.get_secret_value() if s.redis.password else None
    return redis.Redis(
        host=s.redis.host,
        port=s.redis.port,
        password=pw,
        decode_responses=True,
    )


def load_sensor_data(filepath: str = 'training_data.pkl'):
    """Load training data and split by health status."""
    print(f"Loading data from {filepath}...")
    df = pd.read_pickle(filepath)
    
    # Split into healthy and faulty pools
    healthy = df[df['machine_failure'] == 0].reset_index(drop=True)
    faulty = df[df['machine_failure'] == 1].reset_index(drop=True)
    
    print(f"  Loaded {len(df)} samples")
    print(f"    Healthy: {len(healthy)} samples")
    print(f"    Faulty:  {len(faulty)} samples")
    
    return healthy, faulty


def create_json_payload(row: pd.Series, index: int, healthy_pool: pd.DataFrame, faulty_pool: pd.DataFrame) -> dict:
    """
    Create JSON-serializable payload from DataFrame row.
    
    Uses FLEET_TOPOLOGY to assign realistic automotive equipment IDs.
    Health profiles based on topology configuration:
    - healthy: Always use healthy samples
    - warning: 50% healthy, 50% faulty
    - critical: Always use faulty samples
    """
    # Select equipment from topology based on index
    equipment_idx = index % len(FLEET_TOPOLOGY)
    equipment = FLEET_TOPOLOGY[equipment_idx]
    
    # Select appropriate sample based on equipment health profile
    health_status = equipment.get("health", "healthy")
    
    if health_status == "healthy":
        source_row = healthy_pool.sample(n=1, random_state=index).iloc[0]
    elif health_status == "warning":
        is_healthy = (index % 2) == 0
        source_row = healthy_pool.sample(n=1, random_state=index).iloc[0] if is_healthy else faulty_pool.sample(n=1, random_state=index).iloc[0]
    else:  # critical
        source_row = faulty_pool.sample(n=1, random_state=index).iloc[0]
    
    # Convert numpy array to list for JSON serialization
    vibration = source_row['raw_vibration']
    if isinstance(vibration, np.ndarray):
        vibration_list = vibration.tolist()
    else:
        vibration_list = list(vibration)
    
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine_id": equipment["id"],
        "machine_name": equipment["name"],
        "shop": equipment["shop"],
        "line": equipment["line"],
        "equipment_type": equipment["type"],
        "rotational_speed": float(source_row.get('rotational_speed', 1800.0)) + np.random.uniform(-50, 50),
        "temperature": float(source_row.get('temperature', 70.0)) + np.random.uniform(-5, 15),
        "torque": float(source_row.get('torque', 40.0)) if 'torque' in source_row else 40.0,
        "tool_wear": float(source_row.get('tool_wear', 0.1)) if 'tool_wear' in source_row else 0.1,
        "vibration_raw": vibration_list,
        "sequence_id": index,
        "machine_failure": int(source_row.get('machine_failure', 0))
    }
    
    return payload


def _run_abb_adapter_thread():
    """Run ABB adapter in a background thread (same as: python abb_adapter.py)."""
    import asyncio
    import abb_adapter
    asyncio.run(abb_adapter.main())


def _run_siemens_adapter_thread():
    """Run Siemens S7 adapter in a background thread (same as: python siemens_s7_adapter.py)."""
    from siemens_s7_adapter import SiemensS7Adapter
    SiemensS7Adapter().run_forever()


def start_live_adapters():
    """
    Start ABB and Siemens S7 adapters alongside the publisher.
    Call this at startup to have both PLC adapters publishing to sensor_stream
    in background threads (same channel as this publisher).
    """
    import threading
    t_abb = threading.Thread(target=_run_abb_adapter_thread, name="ABBAdapter", daemon=True)
    t_siemens = threading.Thread(target=_run_siemens_adapter_thread, name="SiemensS7Adapter", daemon=True)
    t_abb.start()
    t_siemens.start()
    print("  ABB adapter (OPC-UA) and Siemens S7 adapter (snap7) started in background.")


def publish_stream(loop_forever: bool = True):
    """
    Main publishing loop.

    Args:
        loop_forever: If True, continuously loop through data
    """
    print("=" * 70)
    print("GAIA STREAM PUBLISHER")
    print("Redis Pub/Sub Sensor Data Simulator")
    print("=" * 70)
    
    # Optional: start ABB + Siemens S7 adapters (same interface as running abb_adapter.py + siemens_s7_adapter.py)
    if RUN_LIVE_ADAPTERS:
        start_live_adapters()

    # Connect to Redis
    try:
        r = get_redis_connection()
        r.ping()
        s = get_settings()
        interval = 1.0 / s.redis.publish_rate_hz
        print(f"\n✓ Connected to Redis at {s.redis.host}:{s.redis.port}")
        print(f"  Channel: {s.redis.redis_channel}")
        print(f"  Publish Rate: {s.redis.publish_rate_hz} Hz ({interval*1000:.0f}ms interval)")
    except redis.ConnectionError as e:
        print(f"\n❌ Redis connection failed: {e}")
        print("\nPlease ensure Redis is running:")
        print("  docker run -d -p 6379:6379 redis")
        print("  or")
        print("  redis-server")
        return

    # Optional MQTT publisher (additive to Redis; internal API still consumes from Redis)
    mqtt_publisher = None
    mqtt_enabled = False
    if get_settings is not None and MQTTPublisher is not None:
        mqtt_cfg = get_settings().mqtt
        mqtt_enabled = mqtt_cfg.publish_enabled
        if mqtt_enabled:
            try:
                mqtt_publisher = MQTTPublisher()
                if mqtt_publisher.connect():
                    print(f"MQTT connected to {mqtt_cfg.broker_host}:{mqtt_cfg.broker_port}")
                else:
                    print("⚠ MQTT broker unavailable; continuing without MQTT publish")
                    mqtt_publisher = None
            except Exception as e:
                logger.warning("MQTT setup failed: %s; continuing without MQTT", e)
                mqtt_publisher = None
        else:
            print("  MQTT publish disabled (MQTT_PUBLISH_ENABLED=false)")

    # UNS mapper for ISA-95 topic hierarchy (machines -> enterprise/site/area/cell/asset/category/metric)
    uns_mapper = None
    if mqtt_publisher and mqtt_publisher.is_connected and UNSMapper is not None:
        try:
            uns_mapper = UNSMapper(config_path="pipeline/station_config.json")
        except Exception as e:
            logger.warning("UNS mapper init failed: %s; MQTT will use flat topics", e)

    # Confirm both transports before starting loop
    print("\n  Transports: Redis (active)" + (" | MQTT (active)" if mqtt_publisher and mqtt_publisher.is_connected else " | MQTT (off)"))

    # Load data
    healthy_pool, faulty_pool = load_sensor_data()
    n_samples = len(healthy_pool) + len(faulty_pool)
    
    print(f"\n" + "-" * 70)
    print("STREAMING STARTED")
    print("-" * 70)
    print(f"Press Ctrl+C to stop\n")
    
    # Publishing loop
    messages_sent = 0
    start_time = time.time()
    idx = 0
    
    try:
        while True:
            # Generate payload (uses idx to select machine and determine health profile)
            payload = create_json_payload(None, idx, healthy_pool, faulty_pool)
            
            # Publish to Redis channel (internal API consumes from here)
            json_payload = json.dumps(payload)
            r.publish(get_settings().redis.redis_channel, json_payload)

            # Parallel MQTT publish via UNS (ISA-95 topic hierarchy); non-blocking, failures logged only
            if mqtt_publisher and mqtt_publisher.is_connected:
                try:
                    machine_id = payload.get("machine_id", "unknown")
                    if uns_mapper is not None:
                        for topic, value, retain in uns_mapper.explode(machine_id, payload):
                            mqtt_publisher.publish(topic, value, qos=1, retain=retain)
                    else:
                        topic = f"machines/{machine_id}/telemetry"
                        mqtt_publisher.publish(topic, payload, qos=1, retain=False)
                except Exception as e:
                    logger.warning("MQTT publish failed (non-blocking): %s", e)

            messages_sent += 1
            idx += 1
            elapsed = time.time() - start_time
            rate = messages_sent / elapsed if elapsed > 0 else 0
            
            # Progress indicator
            if messages_sent % 100 == 0:
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"Sent: {messages_sent:,} | "
                      f"Rate: {rate:.1f} msg/s | "
                      f"Machine: {payload['machine_id']} | "
                      f"Failure: {payload['machine_failure']}")
            
            # Rate limiting
            time.sleep(1.0 / get_settings().redis.publish_rate_hz)

            
    except KeyboardInterrupt:
        if mqtt_publisher is not None:
            try:
                mqtt_publisher.disconnect()
            except Exception:
                pass
        print(f"\n\n" + "-" * 70)
        print("STREAMING STOPPED")
        print("-" * 70)
        elapsed = time.time() - start_time
        print(f"  Total Messages: {messages_sent:,}")
        print(f"  Duration: {elapsed:.1f} seconds")
        print(f"  Average Rate: {messages_sent/elapsed:.1f} msg/s")


def main():
    """Entry point."""
    publish_stream(loop_forever=True)


if __name__ == "__main__":
    main()
