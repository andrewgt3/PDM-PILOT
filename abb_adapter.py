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
from edge.offline_buffer import OfflineBuffer
from edge.replay_service import ReplayService

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

    offline_buffer = None
    replay_service = None
    if settings.edge.offline_buffer_enabled:
        offline_buffer = OfflineBuffer(db_path="data/offline_buffer_abb.db")
        def _publish(payload_dict: dict) -> None:
            r.publish("sensor_stream", json.dumps(payload_dict))
        replay_service = ReplayService(
            offline_buffer,
            _publish,
            interval_seconds=settings.edge.offline_replay_interval_seconds,
            purge_days=settings.edge.offline_buffer_purge_days,
        )
        replay_service.is_online = True

    while True:
        try:
            async with Client(url=abb_settings.url) as client:
                logger.info("Connected to ABB Robot OPC UA Server")
                if replay_service is not None:
                    replay_service.is_online = True
                    replay_service.start_replay_loop()
                
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
                        payload_dict = payload.model_dump()

                        # Publish to Redis; on failure buffer offline
                        try:
                            r.publish("sensor_stream", json.dumps(payload_dict))
                            if replay_service is not None:
                                replay_service.start_replay_loop()
                        except Exception as pub_err:
                            if offline_buffer is not None and replay_service is not None:
                                replay_service.is_online = False
                                ts = payload_dict.get("timestamp", datetime.now(timezone.utc).isoformat())
                                row_id = offline_buffer.write(
                                    payload_dict.get("machine_id", "unknown"),
                                    ts,
                                    payload_dict,
                                    "abb",
                                )
                                n = offline_buffer.pending_count()
                                logger.warning(
                                    "Offline — buffered reading %s. Pending count: %s",
                                    row_id,
                                    n,
                                )
                            else:
                                raise pub_err
                        
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
