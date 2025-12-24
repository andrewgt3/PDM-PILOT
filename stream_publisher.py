#!/usr/bin/env python3
"""
Redis Stream Publisher for GAIA Predictive Maintenance

Simulates real-time sensor data streaming by:
- Loading training_data.pkl
- Publishing JSON records to Redis channel at 10Hz

Author: Senior Backend Engineer
"""

import os
import json
import time
import redis
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)
REDIS_CHANNEL = 'sensor_stream'

PUBLISH_RATE_HZ = 10.0  # 10 messages per second
PUBLISH_INTERVAL = 1.0 / PUBLISH_RATE_HZ  # 0.1 seconds

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
    """Create Redis connection."""
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True
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
    
    # Connect to Redis
    try:
        r = get_redis_connection()
        r.ping()
        print(f"\n✓ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        print(f"  Channel: {REDIS_CHANNEL}")
        print(f"  Publish Rate: {PUBLISH_RATE_HZ} Hz ({PUBLISH_INTERVAL*1000:.0f}ms interval)")
    except redis.ConnectionError as e:
        print(f"\n❌ Redis connection failed: {e}")
        print("\nPlease ensure Redis is running:")
        print("  docker run -d -p 6379:6379 redis")
        print("  or")
        print("  redis-server")
        return
    
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
            
            # Publish to Redis channel
            json_payload = json.dumps(payload)
            r.publish(REDIS_CHANNEL, json_payload)
            
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
            time.sleep(PUBLISH_INTERVAL)

            
    except KeyboardInterrupt:
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
