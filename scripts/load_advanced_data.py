import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Config
CMAPSS_FILE = "data/CMAPSS/train_FD001.txt"
SCANIA_FILE = "data/SCANIA/aps_failure_training_set.csv"

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        dbname=os.getenv("DT_POSTGRES_DB", "pdm_timeseries"),
        user=os.getenv("DT_POSTGRES_USER", "postgres"),
        password=os.getenv("DT_POSTGRES_PASSWORD", "password")
    )

def setup_tables(conn):
    print("--- Setting up Schemas ---")
    cursor = conn.cursor()
    
    # 1. CMAPSS (NASA RUL)
    # Hypertable for NASA RUL data
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
    # Attempt to create hypertable (idempotent check)
    try:
        cursor.execute("SELECT create_hypertable('rul_nasa_data', 'timestamp', if_not_exists => TRUE);")
        print("✅ 'rul_nasa_data' hypertable ready.")
    except Exception as e:
        print(f"ℹ️ Hypertable check: {e}")

    # 2. SCANIA APS
    # Hypertable for Scania Truck data
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
        print("✅ 'scania_benchmarks' hypertable ready.")
    except Exception as e:
        print(f"ℹ️ Hypertable check: {e}")
        
    conn.commit()

def load_cmapss(conn):
    print("\n--- Loading CMAPSS Data ---")
    if not os.path.exists(CMAPSS_FILE):
        print(f"❌ File not found: {CMAPSS_FILE}")
        return

    try:
        col_names = ['unit_id', 'cycle', 'setting_1', 'setting_2', 'setting_3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
        df = pd.read_csv(CMAPSS_FILE, sep=r"\s+", header=None, names=col_names)
    except Exception as e:
        print(f"❌ Error reading CMAPSS: {e}")
        return

    # Synthetic Timestamps (Backfill from NOW)
    now = datetime.now()
    start_time = now - timedelta(hours=len(df))
    timestamps = [start_time + timedelta(hours=i) for i in range(len(df))]
    df['timestamp'] = timestamps

    # Select Columns matching schema
    columns_to_insert = ['timestamp', 'unit_id', 'cycle', 'setting_1', 'setting_2', 's2', 's3', 's4', 's7', 's11', 's12']
    data = df[columns_to_insert].copy()
    
    # Convert timestamps to python datetime
    data['timestamp'] = data['timestamp'].apply(lambda x: x.to_pydatetime() if pd.notnull(x) else None)
    
    list_of_tuples = [tuple(x) for x in data.to_numpy().tolist()]
    
    print(f"Inserting {len(list_of_tuples)} CMAPSS records...")
    cursor = conn.cursor()
    try:
        query = """
            INSERT INTO rul_nasa_data (timestamp, unit_id, cycle, setting_1, setting_2, sensor_2, sensor_3, sensor_4, sensor_7, sensor_11, sensor_12)
            VALUES %s
        """
        execute_values(cursor, query, list_of_tuples)
        conn.commit()
        print("✅ CMAPSS Data Inserted.")
    except Exception as e:
        print(f"❌ CMAPSS Insert Error: {e}")
        conn.rollback()

def load_scania(conn):
    print("\n--- Loading SCANIA Data ---")
    if not os.path.exists(SCANIA_FILE):
        print(f"❌ File not found: {SCANIA_FILE}")
        return

    try:
        df = pd.read_csv(SCANIA_FILE, na_values="na")
    except Exception as e:
        print(f"❌ Error reading SCANIA: {e}")
        return
    
    # Synthetic Data
    now = datetime.now()
    start_time = now - timedelta(hours=len(df))
    timestamps = [start_time + timedelta(hours=i) for i in range(len(df))]
    df['timestamp'] = timestamps
    df['truck_id'] = [f"T-{i}" for i in range(len(df))]

    # Prepare Data
    data = df[['timestamp', 'truck_id', 'class', 'aa_000', 'ab_000', 'ac_000']].copy()
    
    # Handle NaNs -> None (for SQL NULL)
    data = data.astype(object)
    data = data.where(pd.notnull(data), None)
    
    list_of_tuples = [tuple(x) for x in data.to_numpy().tolist()]
    
    print(f"Inserting {len(list_of_tuples)} SCANIA records...")
    cursor = conn.cursor()
    try:
        query = """
            INSERT INTO scania_benchmarks (timestamp, truck_id, class_label, aa_000, ab_000, ac_000)
            VALUES %s
        """
        execute_values(cursor, query, list_of_tuples)
        conn.commit()
        print("✅ SCANIA Data Inserted.")
    except Exception as e:
        print(f"❌ SCANIA Insert Error: {e}")
        conn.rollback()

def main():
    conn = get_db_connection()
    try:
        setup_tables(conn)
        load_cmapss(conn)
        load_scania(conn)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
