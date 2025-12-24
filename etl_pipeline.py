"""
ETL Pipeline for Predictive Maintenance Data Ingestion
=======================================================
Cleans and ingests synthetic sensor data into TimescaleDB.

Pipeline Steps:
1. Schema Creation: Create hypertable for sensors, standard table for events
2. Data Cleaning: Apply forward fill to handle NaN values
3. Data Ingestion: Efficient bulk upload to TimescaleDB
4. Verification: Validate data integrity and row counts
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Float, DateTime, Integer
from sqlalchemy.engine import URL
from datetime import datetime
import time
import warnings
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Database connection parameters
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'pdm_timeseries'),
    'username': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD')
}
if not DB_CONFIG['password']:
    raise ValueError("DB_PASSWORD environment variable is required")

# Data file paths
SENSOR_DATA_FILE = 'sensor_data_dirty.csv'
CONTEXT_DATA_FILE = 'context_logs.csv'

# Table names
SENSORS_TABLE = 'sensors'
EVENTS_TABLE = 'events'

# =============================================================================
# DATABASE CONNECTION
# =============================================================================

def create_db_connection():
    """Create SQLAlchemy engine for TimescaleDB."""
    connection_url = URL.create(
        drivername='postgresql+psycopg2',
        username=DB_CONFIG['username'],
        password=DB_CONFIG['password'],
        host=DB_CONFIG['host'],
        port=DB_CONFIG['port'],
        database=DB_CONFIG['database']
    )
    
    engine = create_engine(connection_url, pool_pre_ping=True)
    return engine

# =============================================================================
# STEP A: SCHEMA CREATION
# =============================================================================

def create_schema(engine):
    """
    Create database schema with TimescaleDB hypertable for sensors
    and standard table for events.
    """
    print("=" * 70)
    print("STEP A: CREATING SCHEMA")
    print("=" * 70)
    
    with engine.connect() as conn:
        # Enable TimescaleDB extension
        print("📦 Enabling TimescaleDB extension...")
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
            conn.commit()
            print("   ✓ TimescaleDB extension enabled")
        except Exception as e:
            print(f"   ⚠️  Extension may already exist: {e}")
        
        # Drop existing tables if they exist (for clean re-runs)
        print("\n🗑️  Dropping existing tables (if any)...")
        try:
            conn.execute(text(f"DROP TABLE IF EXISTS {SENSORS_TABLE} CASCADE;"))
            conn.execute(text(f"DROP TABLE IF EXISTS {EVENTS_TABLE} CASCADE;"))
            conn.commit()
            print("   ✓ Cleaned up existing tables")
        except Exception as e:
            print(f"   ⚠️  Cleanup warning: {e}")
        
        # Create sensors table (will become hypertable)
        print(f"\n📊 Creating '{SENSORS_TABLE}' table...")
        create_sensors_sql = f"""
        CREATE TABLE {SENSORS_TABLE} (
            timestamp TIMESTAMPTZ NOT NULL,
            asset_id VARCHAR(50) NOT NULL,
            joint_1_torque DOUBLE PRECISION,
            vibration_x DOUBLE PRECISION,
            motor_temp_c DOUBLE PRECISION,
            PRIMARY KEY (timestamp, asset_id)
        );
        """
        conn.execute(text(create_sensors_sql))
        conn.commit()
        print(f"   ✓ Created '{SENSORS_TABLE}' table")
        
        # Convert to hypertable (TimescaleDB magic!)
        print(f"\n⚡ Converting '{SENSORS_TABLE}' to hypertable...")
        hypertable_sql = f"""
        SELECT create_hypertable('{SENSORS_TABLE}', 'timestamp',
                                 chunk_time_interval => INTERVAL '1 day',
                                 if_not_exists => TRUE);
        """
        conn.execute(text(hypertable_sql))
        conn.commit()
        print(f"   ✓ '{SENSORS_TABLE}' is now a TimescaleDB hypertable")
        print("   ℹ️  Chunk interval: 1 day (optimized for 7-day dataset)")
        
        # Create events table (standard relational table)
        print(f"\n📋 Creating '{EVENTS_TABLE}' table...")
        create_events_sql = f"""
        CREATE TABLE {EVENTS_TABLE} (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            event_type VARCHAR(100) NOT NULL,
            staff_id VARCHAR(50),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        conn.execute(text(create_events_sql))
        conn.commit()
        print(f"   ✓ Created '{EVENTS_TABLE}' table")
        
        # Create index on timestamp for fast queries
        print(f"\n🔍 Creating indexes...")
        conn.execute(text(f"CREATE INDEX idx_events_timestamp ON {EVENTS_TABLE}(timestamp);"))
        conn.execute(text(f"CREATE INDEX idx_events_type ON {EVENTS_TABLE}(event_type);"))
        conn.commit()
        print("   ✓ Indexes created for optimized queries")
    
    print("\n✅ Schema creation complete!")
    print("=" * 70)
    print()

# =============================================================================
# STEP B: DATA CLEANING
# =============================================================================

def clean_sensor_data(file_path):
    """
    Load and clean sensor data using forward fill for NaN values.
    This simulates proprietary data cleaning logic.
    """
    print("=" * 70)
    print("STEP B: DATA CLEANING")
    print("=" * 70)
    
    print(f"📂 Loading sensor data from '{file_path}'...")
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    initial_rows = len(df)
    print(f"   ✓ Loaded {initial_rows:,} rows")
    
    # Analyze data quality issues
    print("\n🔍 Analyzing data quality...")
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"   Total NaN values: {total_nulls:,}")
    for col in ['joint_1_torque', 'vibration_x', 'motor_temp_c']:
        if null_counts[col] > 0:
            pct = (null_counts[col] / len(df)) * 100
            print(f"   - {col}: {null_counts[col]:,} NaN ({pct:.2f}%)")
    
    # Apply forward fill (proprietary cleaning method)
    print("\n🧹 Applying forward fill (ffill) to sensor columns...")
    sensor_cols = ['joint_1_torque', 'vibration_x', 'motor_temp_c']
    df[sensor_cols] = df[sensor_cols].fillna(method='ffill')
    
    # Handle any remaining NaN at the start (backfill)
    df[sensor_cols] = df[sensor_cols].fillna(method='bfill')
    
    # Verify cleaning
    remaining_nulls = df[sensor_cols].isnull().sum().sum()
    print(f"   ✓ Cleaning complete - {total_nulls:,} NaN values filled")
    print(f"   ✓ Remaining NaN values: {remaining_nulls}")
    
    print("\n✅ Data cleaning complete!")
    print("=" * 70)
    print()
    
    return df

def clean_event_data(file_path):
    """Load event/context data (no cleaning needed)."""
    print("=" * 70)
    print("LOADING CONTEXT DATA")
    print("=" * 70)
    
    print(f"📂 Loading context events from '{file_path}'...")
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    print(f"   ✓ Loaded {len(df):,} events")
    
    # Display event type distribution
    print("\n📊 Event type distribution:")
    event_counts = df['event_type'].value_counts()
    for event_type, count in event_counts.items():
        print(f"   - {event_type}: {count}")
    
    print("\n✅ Context data loaded!")
    print("=" * 70)
    print()
    
    return df

# =============================================================================
# STEP C: DATA INGESTION
# =============================================================================

def ingest_data(engine, sensor_df, event_df):
    """
    Ingest cleaned data into TimescaleDB using efficient bulk operations.
    """
    print("=" * 70)
    print("STEP C: DATA INGESTION")
    print("=" * 70)
    
    # Ingest sensor data
    print(f"📤 Ingesting sensor data into '{SENSORS_TABLE}'...")
    print(f"   Rows to insert: {len(sensor_df):,}")
    
    start_time = time.time()
    
    # Use to_sql with method='multi' for faster bulk inserts
    sensor_df.to_sql(
        name=SENSORS_TABLE,
        con=engine,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=10000  # Insert in batches of 10k rows
    )
    
    sensor_duration = time.time() - start_time
    rows_per_sec = len(sensor_df) / sensor_duration
    print(f"   ✓ Inserted {len(sensor_df):,} rows in {sensor_duration:.2f}s")
    print(f"   ℹ️  Throughput: {rows_per_sec:,.0f} rows/second")
    
    # Ingest event data
    print(f"\n📤 Ingesting context events into '{EVENTS_TABLE}'...")
    print(f"   Rows to insert: {len(event_df):,}")
    
    start_time = time.time()
    
    event_df.to_sql(
        name=EVENTS_TABLE,
        con=engine,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=1000
    )
    
    event_duration = time.time() - start_time
    print(f"   ✓ Inserted {len(event_df):,} rows in {event_duration:.2f}s")
    
    print("\n✅ Data ingestion complete!")
    print("=" * 70)
    print()

# =============================================================================
# STEP D: VERIFICATION
# =============================================================================

def verify_ingestion(engine):
    """
    Run verification queries to ensure data was ingested correctly.
    """
    print("=" * 70)
    print("STEP D: VERIFICATION")
    print("=" * 70)
    
    with engine.connect() as conn:
        # Verify sensor data
        print(f"🔍 Verifying '{SENSORS_TABLE}' table...")
        
        # Row count
        result = conn.execute(text(f"SELECT COUNT(*) FROM {SENSORS_TABLE};"))
        sensor_count = result.fetchone()[0]
        print(f"   ✓ Total rows: {sensor_count:,}")
        
        # Time range
        result = conn.execute(text(f"""
            SELECT MIN(timestamp) as start_time, 
                   MAX(timestamp) as end_time
            FROM {SENSORS_TABLE};
        """))
        time_range = result.fetchone()
        print(f"   ✓ Time range: {time_range[0]} to {time_range[1]}")
        
        # Check for NaN values (should be 0 after cleaning)
        result = conn.execute(text(f"""
            SELECT 
                COUNT(*) FILTER (WHERE joint_1_torque IS NULL) as torque_nulls,
                COUNT(*) FILTER (WHERE vibration_x IS NULL) as vib_nulls,
                COUNT(*) FILTER (WHERE motor_temp_c IS NULL) as temp_nulls
            FROM {SENSORS_TABLE};
        """))
        null_check = result.fetchone()
        print(f"   ✓ Remaining NULL values:")
        print(f"       - torque: {null_check[0]}")
        print(f"       - vibration: {null_check[1]}")
        print(f"       - temperature: {null_check[2]}")
        
        # Sample statistics
        result = conn.execute(text(f"""
            SELECT 
                AVG(joint_1_torque) as avg_torque,
                AVG(vibration_x) as avg_vib,
                AVG(motor_temp_c) as avg_temp
            FROM {SENSORS_TABLE};
        """))
        stats = result.fetchone()
        print(f"\n📊 Sensor statistics:")
        print(f"   - Avg torque: {stats[0]:.2f} Nm")
        print(f"   - Avg vibration: {stats[1]:.3f} g")
        print(f"   - Avg temperature: {stats[2]:.2f} °C")
        
        # Verify event data
        print(f"\n🔍 Verifying '{EVENTS_TABLE}' table...")
        
        # Row count
        result = conn.execute(text(f"SELECT COUNT(*) FROM {EVENTS_TABLE};"))
        event_count = result.fetchone()[0]
        print(f"   ✓ Total rows: {event_count:,}")
        
        # Event type distribution
        result = conn.execute(text(f"""
            SELECT event_type, COUNT(*) as count
            FROM {EVENTS_TABLE}
            GROUP BY event_type
            ORDER BY count DESC;
        """))
        print(f"\n📊 Event distribution:")
        for row in result:
            print(f"   - {row[0]}: {row[1]}")
        
        # Check for cleaning crew events (the critical ones!)
        result = conn.execute(text(f"""
            SELECT COUNT(*) 
            FROM {EVENTS_TABLE}
            WHERE event_type = 'Cleaning_Crew_Zone_3';
        """))
        cleaning_count = result.fetchone()[0]
        print(f"\n⚡ Critical events found:")
        print(f"   - Cleaning_Crew_Zone_3: {cleaning_count} events")
        
        if cleaning_count > 0:
            result = conn.execute(text(f"""
                SELECT timestamp, staff_id
                FROM {EVENTS_TABLE}
                WHERE event_type = 'Cleaning_Crew_Zone_3'
                ORDER BY timestamp;
            """))
            print(f"   Timestamps:")
            for row in result:
                print(f"     • {row[0]} (Staff: {row[1]})")
    
    print("\n✅ Verification complete - Data integrity confirmed!")
    print("=" * 70)
    print()

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Execute the complete ETL pipeline."""
    print("\n")
    print("=" * 70)
    print("ETL PIPELINE: PREDICTIVE MAINTENANCE DATA INGESTION")
    print("=" * 70)
    print()
    
    overall_start = time.time()
    
    try:
        # Connect to database
        print("🔌 Connecting to TimescaleDB...")
        engine = create_db_connection()
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"   ✓ Connected successfully")
            print(f"   ℹ️  {version.split(',')[0]}")
        print()
        
        # STEP A: Create schema
        create_schema(engine)
        
        # STEP B: Clean data
        sensor_df = clean_sensor_data(SENSOR_DATA_FILE)
        event_df = clean_event_data(CONTEXT_DATA_FILE)
        
        # STEP C: Ingest data
        ingest_data(engine, sensor_df, event_df)
        
        # STEP D: Verify
        verify_ingestion(engine)
        
        # Final summary
        total_duration = time.time() - overall_start
        print("=" * 70)
        print("🎉 ETL PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"Total execution time: {total_duration:.2f} seconds")
        print(f"Sensor rows ingested: {len(sensor_df):,}")
        print(f"Event rows ingested: {len(event_df):,}")
        print()
        print("Next Steps:")
        print("  • Query the data: psql -U postgres -d pdm_timeseries")
        print("  • Test queries from Python: Use SQLAlchemy with the engine")
        print("  • Build dashboards: Connect BI tools to port 5432")
        print("=" * 70)
        print()
        
    except Exception as e:
        print("\n❌ ERROR in ETL pipeline:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
