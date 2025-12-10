import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        dbname=os.getenv("DT_POSTGRES_DB", "pdm_timeseries"),
        user=os.getenv("DT_POSTGRES_USER", "postgres"),
        password=os.getenv("DT_POSTGRES_PASSWORD", "password")
    )

def setup_extra_schemas():
    conn = get_db_connection()
    conn.autocommit = True
    cursor = conn.cursor()
    
    print("Creating additional tables...")

    # 1. CMAPSS (NASA RUL)
    # Standard format: unit, time_cycles, settings 1-3, sensors 1-21
    # We will prioritize a few key columns for the MVP schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rul_nasa_data (
            timestamp TIMESTAMPTZ NOT NULL,
            unit_id INT NOT NULL,
            cycle INT NOT NULL,
            setting_1 DOUBLE PRECISION,
            setting_2 DOUBLE PRECISION,
            sensor_2 DOUBLE PRECISION,
            sensor_3 DOUBLE PRECISION,
            sensor_4 DOUBLE PRECISION,
            sensor_7 DOUBLE PRECISION,
            sensor_11 DOUBLE PRECISION,
            sensor_12 DOUBLE PRECISION
        );
    """)
    # Hypertable (partition by timestamp, or cycle if we mapped cycle to time)
    # We'll stick to timestamp for TimescaleDB best practice
    try:
        cursor.execute("SELECT create_hypertable('rul_nasa_data', 'timestamp', if_not_exists => TRUE);")
        print(" - 'rul_nasa_data' hypertable created.")
    except Exception as e:
        print(f" - Notice: {e}")

    # 2. SCANIA APS
    # Standard: class (neg/pos), anonymized features (aa_000, ...)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scania_benchmarks (
            timestamp TIMESTAMPTZ NOT NULL,
            truck_id TEXT NOT NULL,
            class_label TEXT,
            aa_000 DOUBLE PRECISION,
            ab_000 DOUBLE PRECISION,
            ac_000 DOUBLE PRECISION
        );
    """)
    try:
        cursor.execute("SELECT create_hypertable('scania_benchmarks', 'timestamp', if_not_exists => TRUE);")
        print(" - 'scania_benchmarks' hypertable created.")
    except Exception as e:
        print(f" - Notice: {e}")

    conn.close()

if __name__ == "__main__":
    setup_extra_schemas()
