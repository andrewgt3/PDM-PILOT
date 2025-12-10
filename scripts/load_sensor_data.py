import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Config
CSV_FILE = "data/data/ai4i2020.csv"
BATCH_SIZE = 1000

def dsn():
    return f"host=localhost port=5432 dbname={os.getenv('DT_POSTGRES_DB', 'pdm_timeseries')} " \
           f"user={os.getenv('DT_POSTGRES_USER', 'postgres')} " \
           f"password={os.getenv('DT_POSTGRES_PASSWORD', 'password')}"

def load_data():
    print(f"Reading {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"❌ File not found: {CSV_FILE}")
        return

    # Column Mapping
    # 'Product ID' -> machine_id
    # 'Air temperature [K]' -> temperature_air
    # 'Rotational speed [rpm]' -> rotational_speed
    # 'Tool wear [min]' -> tool_wear
    
    # 1. Cleaning: Forward Fill
    print("Scrubbing data (Forward Fill)...")
    df.ffill(inplace=True)
    
    # 2. Add Timestamp
    # Generating synthetic timestamps for the sake of the demo, 
    # going back in time from now, 1 minute per record.
    now = datetime.now()
    timestamps = [now - timedelta(minutes=i) for i in range(len(df))]
    # Reverse so oldest is first
    timestamps.reverse()
    df['timestamp'] = timestamps

    # 3. Prepare for Insert
    # Select and Rename
    data_to_insert = df[['timestamp', 'Product ID', 'Rotational speed [rpm]', 'Air temperature [K]', 'Torque [Nm]', 'Tool wear [min]']].copy()
    data_to_insert.columns = ['timestamp', 'machine_id', 'rotational_speed', 'temperature_air', 'torque', 'tool_wear']
    
    # Convert timestamps to native Python datetime objects (avoids numpy.datetime64 error)
    # data_to_insert['timestamp'] is likely datetime64[ns], convert to object(pydatetime)
    data_to_insert['timestamp'] = data_to_insert['timestamp'].dt.to_pydatetime()

    # Convert entire dataframe to list of standard Python tuples (handling numpy floats for safety)
    # numpy types are usually adapted, but explicit conversion is safer
    list_of_tuples = [tuple(x) for x in data_to_insert.to_numpy().tolist()]

    print(f"Connecting to DB to insert {len(list_of_tuples)} records...")
    
    conn = psycopg2.connect(dsn())
    cursor = conn.cursor()
    
    try:
        insert_query = """
            INSERT INTO sensor_readings (timestamp, machine_id, rotational_speed, temperature_air, torque, tool_wear)
            VALUES %s
        """
        
        # Batch Insert
        total_inserted = 0
        for i in range(0, len(list_of_tuples), BATCH_SIZE):
            batch = list_of_tuples[i:i + BATCH_SIZE]
            execute_values(cursor, insert_query, batch)
            total_inserted += len(batch)
            print(f" - Inserted {total_inserted}/{len(list_of_tuples)}")
        
        conn.commit()
        print("✅ Data Load Complete.")
        
        # Also populate Assets table for integrity
        assets = df[['Product ID', 'Type']].drop_duplicates()
        print(f"Populating Assets table ({len(assets)} machines)...")
        asset_tuples = [tuple(x) for x in assets.to_records(index=False)]
        
        execute_values(cursor, """
            INSERT INTO assets (machine_id, asset_type)
            VALUES %s
            ON CONFLICT (machine_id) DO NOTHING
        """, asset_tuples)
        conn.commit()
        print("✅ Assets Loaded.")

    except Exception as e:
        print(f"❌ Error: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    load_data()
