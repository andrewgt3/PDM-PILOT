#!/usr/bin/env python3
"""
OPC UA Client Adapter ("The Gateway")
Connects Industrial PLCs to the Gaia AI Pipeline.

Architecture:
[PLC / OPC Server] --(OPC UA)--> [THIS ADAPTER] --(JSON/Redis)--> [AI Engine]
"""

import asyncio
import logging
import json
import redis
import os
import numpy as np
from asyncua import Client
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPC_SERVER_URL = os.getenv('OPC_SERVER_URL', "opc.tcp://localhost:4840/freeopcua/server/")
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_CHANNEL = 'sensor_stream'

# Subscriptions
NAMESPACE_URI = "http://gaia.predictive.maintenance"

def get_redis_connection():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

async def main():
    print("=" * 70)
    print("GAIA INDUSTRIAL GATEWAY (OPC UA CLIENT)")
    print(f"Target PLC: {OPC_SERVER_URL}")
    print("=" * 70)

    r = get_redis_connection()
    
    # Retry Loop: Keep trying to connect if PLC is offline
    while True:
        try:
            client = Client(url=OPC_SERVER_URL)
            async with client:
                print(f"✓ Connected to PLC at {OPC_SERVER_URL}")
                
                # Get Namespace Index
                idx = await client.get_namespace_index(NAMESPACE_URI)
                print(f"  Namespace Index for '{NAMESPACE_URI}': {idx}")
                
                # Get Objects Node
                objects = client.nodes.objects
                plant = await objects.get_child([f"{idx}:Plant_Detroit"])
                
                print("  Scanning for machines...")
                machines = []
                for i in range(1, 11):
                    m_id = f"M_{i:03d}"
                    try:
                        node = await plant.get_child([f"{idx}:{m_id}"])
                        machines.append({
                            "id": m_id,
                            "node": node,
                            "tags": {
                                "vib": await node.get_child([f"{idx}:Vibration_Raw"]),
                                "speed": await node.get_child([f"{idx}:Speed_RPM"]),
                                "temp": await node.get_child([f"{idx}:Temperature_C"]),
                                "status": await node.get_child([f"{idx}:Status_Code"])
                            }
                        })
                    except Exception as e:
                        print(f"  ⚠ Could not find node for {m_id}: {e}")

                print(f"  ✓ Found {len(machines)} machines. Starting polling loop...")
                print("-" * 70)

                # Polling Loop
                while True:
                    for m in machines:
                        # Read Tags
                        vib_json = await m["tags"]["vib"].read_value()
                        speed = await m["tags"]["speed"].read_value()
                        temp = await m["tags"]["temp"].read_value()
                        
                        # Transform to Gaia Internal Schema
                        payload = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "machine_id": m["id"],
                            "rotational_speed": float(speed),
                            "temperature": float(temp),
                            "vibration_raw": json.loads(vib_json),
                            "torque": 45.0, # Placeholder
                            "tool_wear": 0.0, # Placeholder
                        }
                        
                        # Publish to Redis
                        r.publish(REDIS_CHANNEL, json.dumps(payload))
                    
                    # Rate Limiting (10Hz)
                    await asyncio.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nGateway Stopped.")
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"⚠ Connection failed {repr(e)}. Retrying in 5s...")
            await asyncio.sleep(5.0)

if __name__ == "__main__":
    asyncio.run(main())
