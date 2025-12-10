import os
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

DATA_FILE = "data/SCANIA/aps_failure_training_set.csv"

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        dbname=os.getenv("DT_POSTGRES_DB", "pdm_timeseries"),
        user=os.getenv("DT_POSTGRES_USER", "postgres"),
        password=os.getenv("DT_POSTGRES_PASSWORD", "password")
    )

def load_scania():
    print(f"Reading {DATA_FILE}...")
    try:
        # 'na' is used for missing values in this dataset
        df = pd.read_csv(DATA_FILE, na_values="na")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    print(f"Loaded {len(df)} records.")
    
    # Generate Synthetic Truck IDs and Timestamps
    # Similar approach: linear timestamps back from now
    now = datetime.now()
    start_time = now - timedelta(hours=len(df))
    timestamps = [start_time + timedelta(hours=i) for i in range(len(df))]
    
    df['timestamp'] = timestamps
    # Truck ID: generic T-{index}
    df['truck_id'] = [f"T-{i}" for i in range(len(df))]

    # Prepare for DB
    # Schema: timestamp, truck_id, class_label, aa_000, ab_000, ac_000
    # Map 'class' -> 'class_label'
    
    data = df[['timestamp', 'truck_id', 'class', 'aa_000', 'ab_000', 'ac_000']].copy()
    
    # Handle NaNs: Convert to None (which psycopg2 handles as NULL)
    # Since numpy NaNs can be tricky with lists, replace with Python None
    # object conversion is safest for mixed types (timestamps + strings + floats + None)
    data = data.astype(object)
    data = data.where(pd.notnull(data), None)
    
    # Ensure timestamps are pydatetime (after astype(object) they might be ok, but let's be sure)
    # If they became None, that's bad, but our generation ensured they exist.
    # Re-apply explicit datetime for safety if not null
    # data['timestamp'] = data['timestamp'].apply(lambda x: x.to_pydatetime() if pd.notnull(x) else None)
    # Actually, simplistic approach:
    
    # Convert back to list of tuples
    list_of_tuples = [tuple(x) for x in data.to_numpy().tolist()]
    
    print(f"Inserting {len(list_of_tuples)} records into 'scania_benchmarks'...")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = """
            INSERT INTO scania_benchmarks (timestamp, truck_id, class_label, aa_000, ab_000, ac_000)
            VALUES %s
        """
        execute_values(cursor, query, list_of_tuples)
        conn.commit()
        print("✅ SCANIA Data Loaded.")
    except Exception as e:
        print(f"❌ Error inserting: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    load_scania()
