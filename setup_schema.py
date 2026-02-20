#!/usr/bin/env python3
"""
Database Schema Setup for GAIA Predictive Maintenance

Creates PostgreSQL tables for:
- sensor_readings: Raw telemetry and vibration data
- cwru_features: Computed signal processing features

Author: Senior Backend Engineer
"""

import os
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# DATABASE CONNECTION
# =============================================================================
def get_db_connection():
    """
    Create PostgreSQL connection from environment variables.
    
    Expected .env variables:
        DATABASE_URL or individual: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    """
    database_url = os.getenv('DATABASE_URL')
    
    if database_url:
        return psycopg2.connect(database_url)
    else:
        return psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'gaia'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'postgres')
        )


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

# Table 1: Raw sensor readings
SENSOR_READINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS sensor_readings (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    machine_id VARCHAR(50) NOT NULL,
    rotational_speed FLOAT,
    temperature FLOAT,
    torque FLOAT,
    tool_wear FLOAT,
    vibration_raw JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
    
    -- Indexes for common queries
    CONSTRAINT sensor_readings_machine_timestamp_idx 
        UNIQUE (machine_id, timestamp)
);

-- Create index for time-series queries
CREATE INDEX IF NOT EXISTS idx_sensor_readings_timestamp 
    ON sensor_readings (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_sensor_readings_machine_id 
    ON sensor_readings (machine_id);

CREATE INDEX IF NOT EXISTS idx_sensor_readings_machine_time 
    ON sensor_readings (machine_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_sensor_readings_tenant 
    ON sensor_readings (tenant_id);

CREATE INDEX IF NOT EXISTS idx_readings_tenant_machine 
    ON sensor_readings (tenant_id, machine_id);
"""

# Table 2: Computed CWRU features (26 features + prediction)
CWRU_FEATURES_SCHEMA = """
CREATE TABLE IF NOT EXISTS cwru_features (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    machine_id VARCHAR(50) NOT NULL,
    
    -- FFT Features (16)
    peak_freq_1 FLOAT,
    peak_freq_2 FLOAT,
    peak_freq_3 FLOAT,
    peak_freq_4 FLOAT,
    peak_freq_5 FLOAT,
    peak_amp_1 FLOAT,
    peak_amp_2 FLOAT,
    peak_amp_3 FLOAT,
    peak_amp_4 FLOAT,
    peak_amp_5 FLOAT,
    low_band_power FLOAT,
    mid_band_power FLOAT,
    high_band_power FLOAT,
    spectral_entropy FLOAT,
    spectral_kurtosis FLOAT,
    total_power FLOAT,
    
    -- Envelope Features (5)
    bpfo_amp FLOAT,
    bpfi_amp FLOAT,
    bsf_amp FLOAT,
    ftf_amp FLOAT,
    sideband_strength FLOAT,
    
    -- Degradation Features (1)
    degradation_score FLOAT,
    degradation_score_smoothed FLOAT,
    
    -- Telemetry Features (4)
    rotational_speed FLOAT,
    temperature FLOAT,
    torque FLOAT,
    tool_wear FLOAT,
    
    -- Model Prediction
    failure_prediction FLOAT,
    failure_class INT,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
    
    -- Indexes
    CONSTRAINT cwru_features_machine_timestamp_idx 
        UNIQUE (machine_id, timestamp)
);

-- Create indexes for analytics queries
CREATE INDEX IF NOT EXISTS idx_cwru_features_timestamp 
    ON cwru_features (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_cwru_features_machine_id 
    ON cwru_features (machine_id);

CREATE INDEX IF NOT EXISTS idx_cwru_features_failure 
    ON cwru_features (failure_prediction DESC);

CREATE INDEX IF NOT EXISTS idx_cwru_features_machine_time 
    ON cwru_features (machine_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_cwru_features_tenant 
    ON cwru_features (tenant_id);

CREATE INDEX IF NOT EXISTS idx_cwru_tenant_machine 
    ON cwru_features (tenant_id, machine_id);
"""

# Additional table for model metadata
MODEL_METADATA_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_metadata (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    training_date TIMESTAMPTZ NOT NULL,
    n_features INT,
    feature_columns JSONB,
    hyperparameters JSONB,
    cv_metrics JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""


def setup_database():
    """
    Execute all schema creation scripts.
    
    Safe operation: Uses CREATE TABLE IF NOT EXISTS.
    """
    print("=" * 70)
    print("GAIA DATABASE SCHEMA SETUP")
    print("=" * 70)
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        print("\nConnected to PostgreSQL database.")
        
        # Create sensor_readings table
        print("\n[1/3] Creating 'sensor_readings' table...")
        cursor.execute(SENSOR_READINGS_SCHEMA)
        print("  ✓ sensor_readings table ready")
        
        # Create cwru_features table
        print("\n[2/3] Creating 'cwru_features' table...")
        cursor.execute(CWRU_FEATURES_SCHEMA)
        print("  ✓ cwru_features table ready")
        
        # Create model_metadata table
        print("\n[3/3] Creating 'model_metadata' table...")
        cursor.execute(MODEL_METADATA_SCHEMA)
        print("  ✓ model_metadata table ready")
        
        # Commit all changes
        conn.commit()
        
        # Verify tables exist
        print("\n" + "-" * 70)
        print("VERIFICATION")
        print("-" * 70)
        
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('sensor_readings', 'cwru_features', 'model_metadata')
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        print("\nCreated tables:")
        for table in tables:
            # Use sql.Identifier for safe table name handling (fixes bandit B608)
            count_query = sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table[0]))
            cursor.execute(count_query)
            count = cursor.fetchone()[0]
            print(f"  • {table[0]}: {count} rows")
        
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 70)
        print("SCHEMA SETUP COMPLETE")
        print("=" * 70)
        
    except psycopg2.OperationalError as e:
        print(f"\n❌ Database connection failed: {e}")
        print("\nPlease ensure:")
        print("  1. PostgreSQL is running")
        print("  2. .env file contains valid credentials:")
        print("     DATABASE_URL=postgresql://user:pass@host:5432/dbname")
        print("     or")
        print("     DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    setup_database()
