import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

DATA_FILE = "data/CMAPSS/train_FD001.txt"

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        dbname=os.getenv("DT_POSTGRES_DB", "pdm_timeseries"),
        user=os.getenv("DT_POSTGRES_USER", "postgres"),
        password=os.getenv("DT_POSTGRES_PASSWORD", "password")
    )

def load_cmapss():
    print(f"Reading {DATA_FILE}...")
    try:
        # CMAPSS is space-separated, no header
        col_names = ['unit_id', 'cycle', 'setting_1', 'setting_2', 'setting_3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
        df = pd.read_csv(DATA_FILE, sep=r"\s+", header=None, names=col_names)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    print(f"Loaded {len(df)} records.")
    
    # Generate Synthetic Timestamps based on cycle (1 hour overlap per cycle for simplicity)
    # Each unit starts 1 year ago? Let's just do sequential back from now.
    # Group by unit, iterate cycles.
    
    # Actually, simpler MVP approach: just assign timestamps backwards from NOW.
    # But ensuring order: unit 1 cycle 1 is oldest.
    
    now = datetime.now()
    # Assign timestamp = Now - (Max_Cycles - Current_Cycle) * 1 hour (as example)
    timestamps = []
    
    # Just linear backfill for speed in MVP
    start_time = now - timedelta(hours=len(df))
    timestamps = [start_time + timedelta(hours=i) for i in range(len(df))]
    
    df['timestamp'] = timestamps

    # Prepare for DB
    # Schema: timestamp, unit_id, cycle, setting_1, setting_2, sensor_2, sensor_3, sensor_4, sensor_7, sensor_11, sensor_12
    # Selecting subset as defined in setup_extra_schemas
    
    columns_to_insert = ['timestamp', 'unit_id', 'cycle', 'setting_1', 'setting_2', 's2', 's3', 's4', 's7', 's11', 's12']
    data = df[columns_to_insert].copy()
    
    # Convert timestamps to pydatetime !important
    data['timestamp'] = data['timestamp'].dt.to_pydatetime()
    
    list_of_tuples = [tuple(x) for x in data.to_numpy().tolist()]
    
    print(f"Inserting {len(list_of_tuples)} records into 'rul_nasa_data'...")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = """
            INSERT INTO rul_nasa_data (timestamp, unit_id, cycle, setting_1, setting_2, sensor_2, sensor_3, sensor_4, sensor_7, sensor_11, sensor_12)
            VALUES %s
        """
        execute_values(cursor, query, list_of_tuples)
        conn.commit()
        print("✅ CMAPSS Data Loaded.")
    except Exception as e:
        print(f"❌ Error inserting: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    load_cmapss()
