"""
OPC UA Fleet Client - Real-Time Data Ingestion to TimescaleDB
==============================================================
Subscribes to OPC UA server and streams live data to database.
(Polling Version - Robust Fallback)

Author: PlantAGI Team
"""

import os
import asyncio
import logging
from datetime import datetime
from asyncua import Client
from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPCUA_ENDPOINT = "opc.tcp://localhost:4840/freeopcua/server/"
DB_CONNECTION = os.getenv('DATABASE_URL')
if not DB_CONNECTION:
    raise ValueError("DATABASE_URL environment variable is required")
ROBOT_IDS = ["ROBOT_1", "ROBOT_2", "ROBOT_3", "ROBOT_4"]

class FleetDataStreamer:
    def __init__(self):
        self.engine = create_engine(DB_CONNECTION)
        self.client = None
        self.robot_nodes = {}
        
    async def connect_opcua(self):
        """Connect to OPC UA server and discover robot nodes."""
        print("=" * 80)
        print("FLEET DATA STREAMER - OPC UA → TimescaleDB (Polling Mode)")
        print("=" * 80)
        print()
        
        self.client = Client(url=OPCUA_ENDPOINT)
        await self.client.connect()
        print(f"✓ Connected to OPC UA server: {OPCUA_ENDPOINT}")
        
        # Get namespace index
        uri = "http://gaiapredictive.com/fleet"
        idx = await self.client.get_namespace_index(uri)
        
        # Discover robot nodes
        objects = self.client.nodes.objects
        
        for robot_id in ROBOT_IDS:
            robot_obj = await objects.get_child([f"{idx}:{robot_id}"])
            
            # Get sensor nodes
            vib_node = await robot_obj.get_child([f"{idx}:Vibration_X"])
            trq_node = await robot_obj.get_child([f"{idx}:Torque_J1"])
            tmp_node = await robot_obj.get_child([f"{idx}:Motor_Temp"])
            
            self.robot_nodes[robot_id] = {
                'vibration': vib_node,
                'torque': trq_node,
                'temperature': tmp_node
            }
            
            print(f"  ✓ Discovered {robot_id}")
        
        print()
        print(f"✓ All {len(ROBOT_IDS)} robots connected")
        print()
        
    async def stream_data(self):
        """Main streaming loop - reads OPC UA data and writes to database."""
        print("Starting real-time data stream...")
        print("=" * 80)
        print()
        
        batch = []
        batch_size = 1  # Write immediately for "live" feel in demo
        
        while True:
            timestamp = datetime.now()
            
            # Read all robot sensors
            for robot_id, nodes in self.robot_nodes.items():
                try:
                    vib = await nodes['vibration'].read_value()
                    trq = await nodes['torque'].read_value()
                    tmp = await nodes['temperature'].read_value()
                    
                    # Calculate simple RUL estimate
                    if vib < 0.3:
                        rul_estimate = 5000  # Healthy
                    elif vib < 0.7:
                        rul_estimate = 150   # Warning
                    elif vib < 1.5:
                        rul_estimate = 48    # Critical
                    else:
                        rul_estimate = 12    # Imminent failure
                    
                    batch.append({
                        'timestamp': timestamp,
                        'asset_id': robot_id,
                        'vibration_x': round(vib, 4),
                        'motor_temp_c': round(tmp, 2),
                        'joint_1_torque': round(trq, 2),
                        'rul_hours': rul_estimate
                    })
                    
                    # Print status
                    status = "🟢" if vib < 0.3 else "🟡" if vib < 0.7 else "🔴"
                    print(f"{status} {robot_id}: Vib={vib:.3f}g  Torque={trq:.1f}Nm  Temp={tmp:.1f}°C")
                    
                except Exception as e:
                    print(f"⚠️  Error reading {robot_id}: {e}")
            
            # Write to database
            if batch:
                try:
                    df = pd.DataFrame(batch)
                    df.to_sql('sensors', self.engine, if_exists='append', index=False)
                    batch = []
                except Exception as e:
                    print(f"❌ Database write error: {e}")
                    batch = []
            
            await asyncio.sleep(0.5)  # 2Hz sampling rate
    
    async def run(self):
        """Main entry point."""
        try:
            await self.connect_opcua()
            await self.stream_data()
        except Exception as e:
            print(f"\n❌ Streaming error: {e}")
        finally:
            if self.client:
                await self.client.disconnect()
                print("\n✓ Disconnected from OPC UA server")

async def main():
    streamer = FleetDataStreamer()
    await streamer.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Streaming stopped by user")
