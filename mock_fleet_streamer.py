"""
Direct Fleet Streamer (Mock OPC UA Client)
===========================================
Simulates the exact behavior of the OPC UA fleet (Chaos Monkey)
and writes directly to the database.

With --source azure_pm: uses AzurePMReplayEngine to replay historical
Azure PM data from TimescaleDB to Redis (and optional MQTT) with time compression.

Workaround for asyncua library compatibility issues.
"""

import argparse
import os
import time
import random
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# Configuration (required only for synthetic mode)
DB_CONNECTION = os.getenv("DATABASE_URL")

# ENHANCED CONFIGURATION: Unique baselines for realism
ROBOT_CONFIG = {
    "ROBOT_1": {
        "base_vib": 0.14, "std_vib": 0.02, 
        "base_temp": 58.0, "std_temp": 0.5,
        "base_trq": 48.0, "std_trq": 2.5,
        "desc": "Heavy Load / Older Model"
    },
    "ROBOT_2": {
        "base_vib": 0.08, "std_vib": 0.01, 
        "base_temp": 50.0, "std_temp": 0.3, # Runs cooler
        "base_trq": 42.0, "std_trq": 1.5,
        "desc": "High Precision / New"
    },
    "ROBOT_3": {
        "base_vib": 0.12, "std_vib": 0.015,
        "base_temp": 54.0, "std_temp": 0.4,
        "base_trq": 45.0, "std_trq": 2.0,
        "desc": "Standard Payload"
    },
    "ROBOT_4": {
        "base_vib": 0.11, "std_vib": 0.015,
        "base_temp": 53.0, "std_temp": 0.4,
        "base_trq": 46.0, "std_trq": 2.2,
        "desc": "Standard Payload"
    }
}
ROBOT_IDS = list(ROBOT_CONFIG.keys())

# Chaos Monkey Config
CYCLE_DURATION_HEALTHY = 45
CYCLE_DURATION_FAILURE = 30


def run_azure_pm_replay(
    speed_multiplier: float,
    start_from: str | None,
    redis_host: str | None = None,
    redis_port: int | None = None,
    redis_password: str | None = None,
    redis_url: str | None = None,
) -> None:
    """Run Azure PM replay engine instead of synthetic stream."""
    from demos.azure_pm_replay import AzurePMReplayEngine
    print("=" * 80)
    print("AZURE PM REPLAY MODE")
    print("=" * 80)
    print(f"Speed: {speed_multiplier}x (1 month = {720 / speed_multiplier:.2f} hours)")
    if start_from:
        print(f"Start from: {start_from}")
    if redis_url:
        # Avoid printing password
        from urllib.parse import urlparse
        u = urlparse(redis_url)
        target = f"{u.hostname or 'redis'}:{u.port or 6379}"
    else:
        target = f"{redis_host or os.getenv('REDIS_HOST', 'localhost')}:{redis_port or os.getenv('REDIS_PORT', '6379')}"
    print(f"Publishing to Redis sensor_stream at {target}...")
    engine = AzurePMReplayEngine(
        speed_multiplier=speed_multiplier,
        start_from=start_from,
        redis_url=redis_url,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_password=redis_password,
        use_mqtt=os.getenv("MQTT_PUBLISH_ENABLED", "").lower() in ("1", "true", "yes"),
    )
    engine.run()


def main_synthetic():
    if not DB_CONNECTION:
        raise ValueError("DATABASE_URL environment variable is required for synthetic mode")
    print("=" * 80)
    print("DIRECT FLEET STREAMER (Simulation Mode)")
    print("=" * 80)
    print("Starting data stream to TimescaleDB...")

    engine = create_engine(DB_CONNECTION)
    
    # State
    is_cascade = False
    last_toggle = time.time()
    next_toggle_duration = CYCLE_DURATION_HEALTHY
    
    # Accumulators for organic drift (Brownian noise)
    drift = {rid: {'vib': 0.0, 'temp': 0.0} for rid in ROBOT_IDS}
    
    while True:
        # 1. Chaos Monkey Logic
        now = time.time()
        if now - last_toggle > next_toggle_duration:
            is_cascade = not is_cascade
            last_toggle = now
            if is_cascade:
                print(f"\n🚨 TRIGGERING CASCADE FAILURE! (Duration: {CYCLE_DURATION_FAILURE}s)")
                next_toggle_duration = CYCLE_DURATION_FAILURE
            else:
                print(f"\n✅ SYSTEM HEALTHY. (Duration: {CYCLE_DURATION_HEALTHY}s)")
                next_toggle_duration = CYCLE_DURATION_HEALTHY
        
        # 2. Generate Data
        batch = []
        timestamp = datetime.utcnow()
        
        for r_id in ROBOT_IDS:
            conf = ROBOT_CONFIG[r_id]
            
            # Update drift (Organic movement)
            drift[r_id]['vib'] += random.uniform(-0.005, 0.005)
            drift[r_id]['temp'] += random.uniform(-0.1, 0.1)
            
            # Clamp drift
            drift[r_id]['vib'] = max(-0.02, min(0.02, drift[r_id]['vib']))
            drift[r_id]['temp'] = max(-2.0, min(2.0, drift[r_id]['temp']))
            
            # Base Noise
            v_noise = random.gauss(0, conf['std_vib'])
            t_noise = random.gauss(0, conf['std_temp'])
            q_noise = random.gauss(0, conf['std_trq'])
            
            if not is_cascade:
                # NORMAL OPERATION
                vib = conf['base_vib'] + v_noise + drift[r_id]['vib']
                trq = conf['base_trq'] + q_noise
                tmp = conf['base_temp'] + t_noise + drift[r_id]['temp']
            else:
                # FAILURE MODES
                if r_id == "ROBOT_1": # Source (Catastrophic)
                    vib = 2.5 + random.uniform(-0.2, 0.2) 
                    trq = 180.0 + random.uniform(-10, 10) 
                    tmp = 85.0 + random.uniform(0, 0.5)   
                elif r_id in ["ROBOT_2", "ROBOT_3"]: # Victims (Sympathetic Vibration)
                    # Victims now react differently based on their mass/config
                    damping = 0.5 if r_id == "ROBOT_2" else 0.8
                    vib = conf['base_vib'] + (random.uniform(0.1, 0.3) * damping)
                    trq = 0.0 # Emergency Stop
                    tmp = conf['base_temp'] - 2.0 # Cooling down
                elif r_id == "ROBOT_4": # Compensator (Overworked)
                    vib = conf['base_vib'] * 3.0 + v_noise # 3x vibration (Struggle)
                    trq = conf['base_trq'] * 1.5 + q_noise # 1.5x Torque (Heavy Load)
                    tmp = conf['base_temp'] + 10.0 + t_noise # Overheating
            
            # RUL Logic
            if vib < 0.3: rul = 5000 + random.randint(-100, 100) # Variance in RUL too
            elif vib < 0.7: rul = 150 + random.randint(-20, 20)
            elif vib < 1.5: rul = 48 + random.randint(-5, 5)
            else: rul = 12
            
            batch.append({
                'timestamp': timestamp,
                'asset_id': r_id,
                'vibration_x': round(vib, 4),
                'motor_temp_c': round(tmp, 2),
                'joint_1_torque': round(trq, 2),
                'rul_hours': rul
            })
            
            # Print minimal status for Robot 1
            if r_id == "ROBOT_1":
                 status = "🔴" if is_cascade else "🟢"
                 print(f"{status} {r_id}: Vib={vib:.2f}g", end=" | ")

        print(f"Update @ {timestamp.strftime('%H:%M:%S')}", end="\r")
        
        # 3. Write to DB
        try:
            df = pd.DataFrame(batch)
            df.to_sql('sensors', engine, if_exists='append', index=False)
        except Exception as e:
            print(f"❌ DB Error: {e}")

        time.sleep(0.5) # 2Hz


def main():
    parser = argparse.ArgumentParser(description="Fleet streamer: synthetic chaos-monkey or Azure PM replay")
    parser.add_argument(
        "--source",
        choices=["synthetic", "azure_pm"],
        default="synthetic",
        help="Data source: synthetic (default) or azure_pm replay",
    )
    parser.add_argument(
        "--speed-multiplier",
        type=float,
        default=720.0,
        help="Replay time compression (azure_pm only): 720 = 1 month per hour",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        help='Start replay from date (azure_pm only), e.g. "2023-01-01"',
    )
    parser.add_argument(
        "--redis-host",
        type=str,
        default=None,
        help="Redis host (e.g. test server IP). Uses REDIS_HOST env if not set.",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=None,
        help="Redis port (default 6379). Uses REDIS_PORT env if not set.",
    )
    parser.add_argument(
        "--redis-password",
        type=str,
        default=None,
        help="Redis password if required. Uses REDIS_PASSWORD env if not set.",
    )
    parser.add_argument(
        "--redis-url",
        type=str,
        default=None,
        help="Full Redis URL (e.g. redis://user:pass@host:6379/0). Overrides --redis-host/port/password.",
    )
    args = parser.parse_args()

    if args.source == "azure_pm":
        try:
            run_azure_pm_replay(
                args.speed_multiplier,
                args.start_from,
                redis_host=args.redis_host,
                redis_port=args.redis_port,
                redis_password=args.redis_password,
                redis_url=args.redis_url,
            )
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    try:
        main_synthetic()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
