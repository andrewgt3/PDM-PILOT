"""
Fleet ETL Pipeline - Load Multi-Robot Data into TimescaleDB
============================================================
Loads fleet_sensor_data.csv and fleet_events.csv into database.

Author: PlantAGI Team
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Configuration
DB_CONNECTION = os.getenv('DATABASE_URL')
if not DB_CONNECTION:
    raise ValueError("DATABASE_URL environment variable is required")
SENSOR_FILE = "fleet_sensor_data.csv"
EVENTS_FILE = "fleet_events.csv"

def load_fleet_data():
    """Load fleet data into TimescaleDB."""
    print("=" * 80)
    print("FLEET ETL PIPELINE")
    print("=" * 80)
    print()
    
    # Connect to database
    engine = create_engine(DB_CONNECTION)
    
    # Create tables
    print("Creating tables...")
    with engine.connect() as conn:
        # Drop existing tables
        conn.execute(text("DROP TABLE IF EXISTS sensors CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS events CASCADE"))
        conn.commit()
        
        # Create sensors table with asset_id
        conn.execute(text("""
        CREATE TABLE sensors (
            timestamp TIMESTAMPTZ NOT NULL,
            asset_id TEXT NOT NULL,
            vibration_x DOUBLE PRECISION,
            vibration_y DOUBLE PRECISION,
            vibration_z DOUBLE PRECISION,
            joint_1_torque DOUBLE PRECISION,
            joint_2_torque DOUBLE PRECISION,
            joint_3_torque DOUBLE PRECISION,
            motor_temp_c DOUBLE PRECISION,
            current_draw_a DOUBLE PRECISION,
            rul_hours DOUBLE PRECISION
        )
        """))
        
        # Create hypertable
        conn.execute(text("SELECT create_hypertable('sensors', 'timestamp')"))
        
        # Create events table
        conn.execute(text("""
        CREATE TABLE events (
            timestamp TIMESTAMPTZ NOT NULL,
            event_type TEXT,
            severity TEXT,
            affected_assets TEXT,
            description TEXT
        )
        """))
        
        conn.commit()
        print("✓ Tables created")
    
    # Load sensor data using COPY (much faster for bulk inserts)
    print()
    print("Loading sensor data...")
    df_sensors = pd.read_csv(SENSOR_FILE)
    df_sensors['timestamp'] = pd.to_datetime(df_sensors['timestamp'])
    
    print(f"  Rows to load: {len(df_sensors):,}")
    print("  Using COPY for fast bulk insert...")
    
    # Use pg COPY which is much faster
    from io import StringIO
    import psycopg2
    
    buffer = StringIO()
    df_sensors.to_csv(buffer, index=False, header=False)
    buffer.seek(0)
    
    # Get raw psycopg2 connection
    conn = engine.raw_connection()
    try:
        cur = conn.cursor()
        cur.copy_expert("COPY sensors FROM STDIN WITH CSV", buffer)
        conn.commit()
        cur.close()
    finally:
        conn.close()
    
    print("✓ Sensor data loaded")
    
    # Load events
    print()
    print("Loading events...")
    df_events = pd.read_csv(EVENTS_FILE)
    df_events['timestamp'] = pd.to_datetime(df_events['timestamp'])
    
    print(f"  Events to load: {len(df_events)}")
    df_events.to_sql('events', engine, if_exists='append', index=False)
    print("✓ Events loaded")
    
    # Verify
    print()
    print("=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM sensors"))
        sensor_count = result.fetchone()[0]
        
        result = conn.execute(text("SELECT COUNT(DISTINCT asset_id) FROM sensors"))
        asset_count = result.fetchone()[0]
        
        result = conn.execute(text("SELECT COUNT(*) FROM events"))
        event_count = result.fetchone()[0]
        
        print(f"✓ Sensors table: {sensor_count:,} rows")
        print(f"✓ Unique assets: {asset_count}")
        print(f"✓ Events table: {event_count} events")
    
    print()
    print("=" * 80)
    print("✅ FLEET DATA LOADED SUCCESSFULLY")
    print("=" * 80)
    print()
    print("Next step: Refresh dashboard at http://localhost:8501")

if __name__ == "__main__":
    load_fleet_data()
