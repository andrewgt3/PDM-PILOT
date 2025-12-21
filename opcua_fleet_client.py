import asyncio
import logging
from asyncua import Client
from sqlalchemy import create_engine, text
import datetime

# --- CONFIGURATION ---
DB_CONNECTION = "postgresql://postgres:password@localhost:5432/pdm_timeseries"
OPC_URL = "opc.tcp://localhost:4840/freeopcua/server/"
NAMESPACE_URI = "http://gaiapredictive.com/fleet"

# Setup DB Engine
engine = create_engine(DB_CONNECTION)

class SubHandler:
    """
    Subscription Handler. 
    This function triggers EVERY time a value changes on the server.
    """
    def __init__(self, asset_id, sensor_type):
        self.asset_id = asset_id
        self.sensor_type = sensor_type

    def datachange_notification(self, node, val, data):
        # We received a value. Insert into DB immediately.
        # Note: In production, you would batch these. For MVP, direct insert is fine.
        timestamp = datetime.datetime.now()
        
        # We need to be careful. The DB expects a row with ALL columns.
        # Since OPC sends updates individually, we will Upsert or simplified log.
        # For this MVP, we will print to console and do a simplified update query.
        
        print(f"📥 {self.asset_id} | {self.sensor_type}: {val:.4f}")
        
        # Construct a targeted update/insert query
        # (Simplified: We assume a row exists for this second, or we insert new)
        # For high-speed ingest, we typically buffer. 
        # Here we just execute a raw insert for demonstration.
        
        try:
             with engine.connect() as conn:
                conn.execute(text(f"""
                    INSERT INTO sensors (timestamp, asset_id, {self.sensor_type}, rul_hours)
                    VALUES (:ts, :asset, :val, 1000)
                    ON CONFLICT (timestamp, asset_id) 
                    DO UPDATE SET {self.sensor_type} = :val
                """), {"ts": timestamp, "asset": self.asset_id, "val": val})
                conn.commit()
        except Exception as e:
            print(f"DB Error: {e}")

async def main():
    print(f"🔌 Connecting to Gaia Virtual Fleet at {OPC_URL}...")
    
    async with Client(url=OPC_URL) as client:
        # Get Namespace Index
        idx = await client.get_namespace_index(NAMESPACE_URI)
        print(f"✅ Connected! Namespace Index: {idx}")
        
        # Create Subscription
        handler = SubHandler("null", "null") # Base handler
        sub = await client.create_subscription(500, handler)
        
        # Scan for Robots and Subscribe
        objects = client.nodes.objects
        # We look for children in our namespace
        children = await objects.get_children()
        
        print("🔍 Scanning for Assets...")
        for child in children:
            name = await child.read_browse_name()
            if name.Name.startswith("ROBOT"):
                asset_id = name.Name
                print(f"   -> Found {asset_id}. Subscribing to tags...")
                
                # Find Sensors
                vars = await child.get_children()
                for v in vars:
                    v_name = (await v.read_browse_name()).Name
                    
                    # Map OPC Tag Name to DB Column Name
                    db_col = None
                    if "Vibration" in v_name: db_col = "vibration_x"
                    if "Torque" in v_name: db_col = "joint_1_torque"
                    if "Temp" in v_name: db_col = "motor_temp_c"
                    
                    if db_col:
                        # Create a specific handler for this tag
                        h = SubHandler(asset_id, db_col)
                        await sub.subscribe_data_change(v, h)
        
        print("🚀 Ingestion Active. Press Ctrl+C to stop.")
        # Keep client alive
        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped.")
