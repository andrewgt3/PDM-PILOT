#!/usr/bin/env python3
"""
OPC UA Client Adapter ("The Gateway")
Connects Industrial PLCs to the Gaia AI Pipeline.

Architecture:
[PLC / OPC Server] --(OPC UA)--> [THIS ADAPTER] --(JSON/Redis/HTTP)--> [AI Engine]

Features:
- JWT authentication for enterprise API endpoints
- TelemetryData Pydantic validation (drops bad packets instead of crashing)
- Offline buffering with deque (max 10,000 packets) for network resilience
"""

import asyncio
import logging
import json
import redis
import numpy as np
import requests
from asyncua import Client
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from collections import deque
from pydantic import BaseModel, Field, ValidationError

from config import get_settings

# Load settings
settings = get_settings()

# Configuration (from EdgeSettings)
OPC_SERVER_URL = settings.edge.opc_server_url
REDIS_HOST = settings.redis.host
REDIS_PORT = settings.redis.port
REDIS_CHANNEL = 'sensor_stream'

# API Configuration
API_BASE_URL = settings.edge.api_base_url
API_USERNAME = settings.edge.api_username
API_PASSWORD = settings.edge.api_password.get_secret_value() if settings.edge.api_password else None

# Buffer Configuration
BUFFER_MAX_SIZE = settings.edge.buffer_max_size
BUFFER_BATCH_SIZE = settings.edge.buffer_batch_size

# Subscriptions
NAMESPACE_URI = settings.edge.namespace_uri

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("gaia_edge_client")


# =============================================================================
# TELEMETRY DATA SCHEMA (Embedded for standalone edge agent)
# =============================================================================

class TelemetryData(BaseModel):
    """
    Validated telemetry data from OPC UA nodes.
    
    Embedded in edge agent for standalone operation (no network dependency
    for validation). Mirrors schemas/telemetry.py.
    """
    timestamp: str = Field(..., description="ISO8601 timestamp")
    machine_id: str = Field(..., min_length=1, max_length=50, description="Machine identifier")
    rotational_speed: float = Field(..., ge=0, le=10000, description="Speed in RPM")
    temperature: float = Field(..., ge=-40, le=200, description="Temperature in Celsius")
    vibration_raw: List[float] = Field(..., min_length=1, description="Raw vibration samples")
    torque: float = Field(default=0.0, ge=0, le=1000, description="Torque in Nm")
    tool_wear: float = Field(default=0.0, ge=0, le=1, description="Tool wear ratio [0-1]")
    
    class Config:
        extra = "allow"  # Allow additional fields for extensibility


# =============================================================================
# GaiaEdgeClient - JWT Authenticated API Client with Offline Buffer
# =============================================================================

class GaiaEdgeClient:
    """
    Edge client for Gaia Predictive API with JWT authentication.
    
    Features:
    - Automatic authentication via /api/enterprise/token
    - Automatic token refresh on 401 Unauthorized
    - Push telemetry data to /api/telemetry
    - Offline buffer (deque) to prevent data loss during network outages
    
    Usage:
        client = GaiaEdgeClient()
        client.authenticate()
        client.push_data({"machine_id": "M_001", "temperature": 45.2})
    """
    
    def __init__(
        self,
        base_url: str = None,
        username: str = None,
        password: str = None,
        buffer_max_size: int = BUFFER_MAX_SIZE,
    ):
        """
        Initialize the Gaia Edge Client.
        
        Args:
            base_url: API base URL (defaults to GAIA_API_URL env var)
            username: API username (defaults to API_USERNAME env var)
            password: API password (defaults to API_PASSWORD env var)
            buffer_max_size: Maximum offline buffer size (default 10,000)
        """
        self.base_url = (base_url or API_BASE_URL).rstrip('/')
        self.username = username or API_USERNAME
        self.password = password or API_PASSWORD
        
        # Validate credentials
        if not self.username or not self.password:
            raise ValueError(
                "API credentials not configured. "
                "Set API_USERNAME and API_PASSWORD environment variables."
            )
        
        self.access_token: Optional[str] = None
        self.session = requests.Session()
        
        # Offline buffer for network resilience
        self.buffer: deque = deque(maxlen=buffer_max_size)
        self.api_healthy: bool = True
        
        logger.info(f"GaiaEdgeClient initialized for {self.base_url}")
    
    def authenticate(self) -> bool:
        """
        Authenticate with the Gaia API and obtain an access token.
        
        POSTs to /api/enterprise/token with form-encoded credentials.
        
        Returns:
            True if authentication succeeded, False otherwise.
        """
        token_url = f"{self.base_url}/api/enterprise/token"
        
        try:
            response = self.session.post(
                token_url,
                data={
                    "username": self.username,
                    "password": self.password,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10,
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get("access_token")
                
                if self.access_token:
                    logger.info("Successfully authenticated with Gaia API")
                    self.api_healthy = True
                    return True
                else:
                    logger.error("Authentication response missing access_token")
                    return False
            else:
                logger.error(
                    f"Authentication failed: {response.status_code} - {response.text}"
                )
                return False
                
        except requests.ConnectionError as e:
            logger.warning(f"API unreachable during auth: {e}")
            self.api_healthy = False
            return False
        except requests.RequestException as e:
            logger.error(f"Authentication request failed: {e}")
            return False
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Return authorization headers with Bearer token."""
        if not self.access_token:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
    
    def push_data(self, payload: Dict[str, Any], retry_on_401: bool = True) -> bool:
        """
        Push telemetry data to /api/telemetry.
        
        Handles:
        - 401 Unauthorized: Re-authenticate and retry once
        - ConnectionError/500: Buffer data for later retry
        
        Args:
            payload: Telemetry data dictionary to send.
            retry_on_401: Whether to retry on 401 (default: True).
        
        Returns:
            True if data was accepted, False otherwise (buffered on failure).
        """
        telemetry_url = f"{self.base_url}/api/telemetry"
        
        # Ensure we have a token
        if not self.access_token:
            logger.info("No access token, authenticating first...")
            if not self.authenticate():
                self._add_to_buffer(payload)
                return False
        
        try:
            response = self.session.post(
                telemetry_url,
                json=payload,
                headers=self._get_auth_headers(),
                timeout=5,
            )
            
            if response.status_code in (200, 201, 202):
                logger.debug(f"Telemetry pushed: {payload.get('machine_id', 'unknown')}")
                self.api_healthy = True
                return True
            
            elif response.status_code == 401 and retry_on_401:
                # Token expired or invalid - re-authenticate and retry once
                logger.warning("Received 401 Unauthorized, re-authenticating...")
                
                if self.authenticate():
                    return self.push_data(payload, retry_on_401=False)
                else:
                    self._add_to_buffer(payload)
                    return False
            
            elif response.status_code >= 500:
                # Server error - buffer and retry later
                logger.warning(f"Server error {response.status_code}, buffering data")
                self.api_healthy = False
                self._add_to_buffer(payload)
                return False
            
            else:
                logger.error(
                    f"Telemetry push failed: {response.status_code} - {response.text}"
                )
                return False
                
        except requests.ConnectionError as e:
            # Network unreachable - buffer data
            logger.warning(f"API unreachable, buffering data: {e}")
            self.api_healthy = False
            self._add_to_buffer(payload)
            return False
        except requests.RequestException as e:
            logger.error(f"Telemetry push request failed: {e}")
            self._add_to_buffer(payload)
            return False
    
    def _add_to_buffer(self, payload: Dict[str, Any]) -> None:
        """Add payload to offline buffer."""
        self.buffer.append(payload)
        if len(self.buffer) % 100 == 0:
            logger.info(f"Buffer size: {len(self.buffer)}/{self.buffer.maxlen}")
    
    def flush_buffer(self, batch_size: int = BUFFER_BATCH_SIZE) -> int:
        """
        Attempt to send buffered data to the API.
        
        Args:
            batch_size: Number of items to send per call.
        
        Returns:
            Number of items successfully sent.
        """
        if not self.buffer or not self.api_healthy:
            return 0
        
        sent_count = 0
        items_to_send = min(batch_size, len(self.buffer))
        
        for _ in range(items_to_send):
            if not self.buffer:
                break
            
            payload = self.buffer[0]  # Peek at front
            
            if self.push_data(payload, retry_on_401=True):
                self.buffer.popleft()  # Remove only on success
                sent_count += 1
            else:
                # API became unhealthy, stop flushing
                break
        
        if sent_count > 0:
            logger.info(f"Flushed {sent_count} items from buffer. Remaining: {len(self.buffer)}")
        
        return sent_count
    
    def has_buffered_data(self) -> bool:
        """Check if there's data in the offline buffer."""
        return len(self.buffer) > 0
    
    def close(self):
        """Close the HTTP session."""
        self.session.close()
        if self.buffer:
            logger.warning(f"Client closed with {len(self.buffer)} unsent items in buffer")
        else:
            logger.info("GaiaEdgeClient session closed")


# =============================================================================
# LEGACY FUNCTIONS
# =============================================================================

def get_redis_connection():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    print("=" * 70)
    print("GAIA INDUSTRIAL GATEWAY (OPC UA CLIENT)")
    print(f"Target PLC: {OPC_SERVER_URL}")
    print("=" * 70)

    r = get_redis_connection()
    
    # Initialize authenticated API client (optional - for HTTP push)
    api_client = None
    if API_USERNAME and API_PASSWORD:
        try:
            api_client = GaiaEdgeClient()
            api_client.authenticate()
            print("✓ Authenticated with Gaia API")
        except Exception as e:
            print(f"⚠ API authentication failed: {e}. Using Redis only.")
    
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
                    # FLUSH BUFFER: If API is healthy and we have buffered data, send it first
                    if api_client and api_client.api_healthy and api_client.has_buffered_data():
                        api_client.flush_buffer(batch_size=BUFFER_BATCH_SIZE)
                    
                    for m in machines:
                        try:
                            # Read Tags from OPC UA
                            vib_json = await m["tags"]["vib"].read_value()
                            speed = await m["tags"]["speed"].read_value()
                            temp = await m["tags"]["temp"].read_value()
                            
                            # Parse vibration data
                            vibration_data = json.loads(vib_json) if isinstance(vib_json, str) else vib_json
                            
                            # VALIDATE with Pydantic TelemetryData schema
                            try:
                                payload = TelemetryData(
                                    timestamp=datetime.now(timezone.utc).isoformat(),
                                    machine_id=m["id"],
                                    rotational_speed=float(speed),
                                    temperature=float(temp),
                                    vibration_raw=vibration_data if isinstance(vibration_data, list) else [float(vibration_data)],
                                    torque=45.0,  # Placeholder
                                    tool_wear=0.0,  # Placeholder
                                )
                                
                                payload_dict = payload.model_dump()
                                
                            except ValidationError as e:
                                # BAD DATA: Log error and DROP the packet (don't crash)
                                logger.error(
                                    f"Invalid telemetry data from {m['id']}: {e.error_count()} errors. "
                                    f"Dropping packet. Errors: {e.errors()}"
                                )
                                continue  # Skip to next machine
                            
                            # Publish to Redis
                            r.publish(REDIS_CHANNEL, json.dumps(payload_dict))
                            
                            # Also push via HTTP API if configured
                            if api_client:
                                api_client.push_data(payload_dict)
                        
                        except Exception as e:
                            logger.error(f"Error reading OPC UA tags for {m['id']}: {e}")
                            continue
                    
                    # Rate Limiting (10Hz)
                    await asyncio.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nGateway Stopped.")
            if api_client:
                api_client.close()
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"⚠ Connection failed {repr(e)}. Retrying in 5s...")
            await asyncio.sleep(5.0)

if __name__ == "__main__":
    asyncio.run(main())
