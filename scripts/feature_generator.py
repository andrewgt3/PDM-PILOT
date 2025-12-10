import os
import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Configuration
# Picking a machine ID that definitely has data from our load script
TARGET_MACHINE_ID = "M24859" 
WINDOW_MINUTES = 15

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        dbname=os.getenv("DT_POSTGRES_DB", "pdm_timeseries"),
        user=os.getenv("DT_POSTGRES_USER", "postgres"),
        password=os.getenv("DT_POSTGRES_PASSWORD", "password")
    )

def calculate_features(machine_id):
    conn = get_db_connection()
    
    # Determine the latest timestamp for this machine to anchor our "Last 15 minutes" window
    # This handles cases where data might be historical or timezone-shifted relative to NOW()
    max_time_query = "SELECT MAX(timestamp) FROM sensor_readings WHERE machine_id = %s"
    cursor = conn.cursor()
    cursor.execute(max_time_query, (machine_id,))
    latest_ts = cursor.fetchone()[0]
    cursor.close()

    if not latest_ts:
        print(f"❌ No data found for machine {machine_id}")
        return

    print(f"Latest data point: {latest_ts}")

    query = """
        SELECT timestamp, rotational_speed, temperature_air 
        FROM sensor_readings 
        WHERE machine_id = %s 
          AND timestamp >= %s - INTERVAL '15 minutes'
        ORDER BY timestamp ASC;
    """
    
    print(f"Querying data for Machine: {machine_id} (Window: 15 mins before latest)...")
    
    try:
        df = pd.read_sql(query, conn, params=(machine_id, latest_ts))
        
        if df.empty:
            print("⚠️ No data found in the last 15 minutes. (Did you just load the data?)")
            return

        print(f"✅ Retrieved {len(df)} records.")

        # 1. Rolling RMS of Rotational Speed
        # RMS = sqrt(mean(x^2))
        # We calculate it as a single scalar for the window (as implied by "calculate... features", 
        # usually for inference you want the feature vector for that window).
        # If "Rolling" implies a time-series of RMS values, we'd use .rolling().
        # Given "output the calculated features" (plural but implies summary for the window), 
        # I will output the scalar feature for this specific 15-minute window.
        
        rpm_values = df['rotational_speed']
        rms_rpm = np.sqrt(np.mean(rpm_values**2))
        
        # 2. Skewness of Air Temperature
        temp_values = df['temperature_air']
        skew_temp = temp_values.skew()
        
        print("\n--- Calculated Features ---")
        print(f"Machine ID: {machine_id}")
        print(f"Window Size: {len(df)} records")
        print(f"feature_rpm_rms:   {rms_rpm:.4f}")
        print(f"feature_temp_skew: {skew_temp:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    calculate_features(TARGET_MACHINE_ID)
