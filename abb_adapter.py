#!/usr/bin/env python3
"""
ABB Robot Adapter
Connects a specific ABB Robot to the Gaia AI Pipeline via OPC UA.

Usage:
    python abb_adapter.py
    
Configuration:
    Set ABB_IP_ADDRESS, ABB_PORT, etc. in .env or via environment variables.
"""

import asyncio
import logging
import json
import redis
from asyncua import Client
from datetime import datetime, timezone
from typing import Dict, Any, List
from pydantic import BaseModel, Field, ValidationError

from config import get_settings

# Load settings
settings = get_settings()
abb_settings = settings.abb
redis_settings = settings.redis

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("abb_adapter")

# Redis Connection
r = redis.Redis(
    host=redis_settings.host,
    port=redis_settings.port,
    password=redis_settings.password.get_secret_value() if redis_settings.password else None,
    decode_responses=True
)

class TelemetryData(BaseModel):
    """Telemetry data schema for the robot."""
    timestamp: str = Field(..., description="ISO8601 timestamp")
    machine_id: str = Field(..., description="Machine identifier")
    rotational_speed: float = Field(..., description="Speed")
    torque: float = Field(..., description="Torque")
    joints: List[float] = Field(default=[], description="Joint angles")
    status: str = Field(default="UNKNOWN", description="Robot status")

async def main():
    logger.info(f"Starting ABB Adapter for {abb_settings.url}...")
    
    while True:
        try:
            async with Client(url=abb_settings.url) as client:
                logger.info("Connected to ABB Robot OPC UA Server")
                
                # Resolve Nodes
                # Note: In a real scenario, these might be specific numeric NodeIDs
                # We use the configured strings from settings
                try:
                    node_speed = client.get_node(abb_settings.node_speed)
                    node_torque = client.get_node(abb_settings.node_torque)
                    node_joints = client.get_node(abb_settings.node_joints)
                except Exception as e:
                    logger.error(f"Failed to resolve nodes: {e}. Check NodeID configuration.")
                    await asyncio.sleep(5)
                    continue

                logger.info("Nodes resolved. Starting data loop...")
                
                while True:
                    try:
                        # Read values
                        speed_val = await node_speed.read_value()
                        torque_val = await node_torque.read_value()
                        joints_val = await node_joints.read_value()
                        
                        # Create payload
                        payload = TelemetryData(
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            machine_id="ABB-ROBOT-001",
                            rotational_speed=float(speed_val) if speed_val is not None else 0.0,
                            torque=float(torque_val) if torque_val is not None else 0.0,
                            joints=joints_val if isinstance(joints_val, list) else [],
                            status="ACTIVE"
                        )
                        
                        # Publish to Redis
                        r.publish("sensor_stream", payload.model_dump_json())
                        
                        # Optional: Print every 10th packet to stdout for debugging
                        # print(f"Published: {payload.json()}")
                        
                    except Exception as e:
                        logger.error(f"Error reading/publishing data: {e}")
                        break # Reconnect
                    
                    await asyncio.sleep(0.5) # 2Hz polling

        except ConnectionRefusedError:
            logger.error("Connection refused. Is the robot reachable?")
        except Exception as e:
            logger.error(f"Connection error: {e}")
        
        logger.info("Retrying in 10 seconds...")
        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
