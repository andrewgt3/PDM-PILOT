
import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import numpy as np
from scipy.stats import linregress

load_dotenv()

# Configuration
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = os.getenv("DT_POSTGRES_DB", "pdm_timeseries")
DB_USER = os.getenv("DT_POSTGRES_USER", "postgres")
DB_PASS = os.getenv("DT_POSTGRES_PASSWORD", "password")

FEATURE_STORE_PATH = "data/feature_store.csv"

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )

# Slope Helper (Vectorized would be faster, but Rolling Apply is robust)
def calc_slope(y):
    # Assume constant time step for simplicity (1 unit steps), or could use time index
    # AI4I is roughly sequential. 
    # linregress(x, y)[0]
    n = len(y)
    if n < 2: return 0.0
    x = np.arange(n)
    # Manual Linear Regression Slope: (N*Σxy - ΣxΣy) / (N*Σx² - (Σx)²)
    # Faster than scipy for rolling
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x * x)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    return slope

def calculate_advanced_features():
    print("🚀 Starting Advanced Feature Engineering (Phase 2)...")
    
    conn = get_db_connection()
    # ... existing load code ...
    print("✅ Database Connected")
    
    # Load Sensor Readings
    # Assuming 'timestamp' exists. If AI4I lacks real timestamps, we might have synthesized them on load.
    # If not, we'll assume sequential rows are 1 hour apart for the sake of the "8 Hour" requirement.
    query = """
        SELECT timestamp, machine_id, rotational_speed, temperature_air, torque, tool_wear
        FROM sensor_readings
        ORDER BY machine_id, timestamp ASC;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"📊 Loaded {len(df)} records from DB.")
    
    if df.empty:
        print("❌ No data found.")
        return

    # Load Labels from CSV (Source of Truth for Targets)
    csv_path = "data/data/ai4i2020.csv"
    if os.path.exists(csv_path):
        print(f"📥 Loading labels from {csv_path}...")
        labels_df = pd.read_csv(csv_path)
        # Ensure mapping columns match
        # DB: machine_id, CSV: Product ID
        labels_map = labels_df[['Product ID', 'Machine failure']].rename(
            columns={'Product ID': 'machine_id', 'Machine failure': 'machine_failure'}
        )
        
        # Merge Labels into Sensor Data
        df = df.merge(labels_map, on='machine_id', how='left')
        print(f"✅ Merged labels. missing inputs: {df['machine_failure'].isna().sum()}")
        df['machine_failure'] = df['machine_failure'].fillna(0).astype(int)
    else:
        print("⚠️ CSV Labels not found. Feature store will lack target 'machine_failure'.")

    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Set index for rolling
    df.set_index('timestamp', inplace=True)
    
    # Feature Config
    sensor_cols = ['rotational_speed', 'temperature_air', 'torque', 'tool_wear']
    
    features_list = []
    
    print("⚙️ Computing Rolling 8H (Mean, Std, Skew) + 24H Trends...")
    
    # Group by Machine ID and Rolling
    for machine_id, group in df.groupby('machine_id'):
        # Sort just in case
        group = group.sort_index()
        
        # 8 Hour Window
        roller_8h = group[sensor_cols].rolling(window='8h', min_periods=1)
        
        # Mean
        roll_mean = roller_8h.mean().add_suffix('_roll_mean_8h')
        
        # Std
        roll_std = roller_8h.std().fillna(0).add_suffix('_roll_std_8h')
        
        # Skew
        roll_skew = roller_8h.skew().fillna(0).add_suffix('_roll_skew_8h')
        
        # 24 Hour Trend (Slope)
        # Using raw=True might pass numpy array for speed if we used apply, 
        # but for simplicity we rely on pandas rolling. 
        # Note: Apply with custom function on time-based rolling is tricky in some pandas versions.
        # Fallback: Resample to fixed steps or assume index is regular if density is high.
        # Given AI4I is row-based (10k rows), let's assume 'window=24' steps if time doesn't work well with apply,
        # but we set index to timestamp. Let's try 24h.
        
        # slope calc
        # rolling().apply() on time-series index passes the values in the window.
        # We use the raw=True for speed.
        roller_24h = group[sensor_cols].rolling(window='24h', min_periods=2)
        roll_slope = roller_24h.apply(calc_slope, raw=True).fillna(0).add_suffix('_trend_slope_24h')
        
        # CSLM (Contextual Feature)
        # In AI4I, tool_wear is accumulative.
        cslm = group[['tool_wear']].rename(columns={'tool_wear': 'CSLM_cycles'})
        
        # Combine
        combined = pd.concat([group, roll_mean, roll_std, roll_skew, roll_slope, cslm], axis=1)
        combined['machine_id'] = machine_id # Restore column
        features_list.append(combined)
        
    final_df = pd.concat(features_list)
    final_df.reset_index(inplace=True) # Bring timestamp back
    
    # Save to Feature Store (CSV Cache)
    os.makedirs(os.path.dirname(FEATURE_STORE_PATH), exist_ok=True)
    final_df.to_csv(FEATURE_STORE_PATH, index=False)
    print(f"✅ Features cached to {FEATURE_STORE_PATH}")
    
    # Save to TimescaleDB (Feature Store Table)
    from sqlalchemy import create_engine
    
    # Create SQLAlchemy Engine
    try:
        db_url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(db_url)
        
        print("💾 Saving to TimescaleDB table 'sensor_features'...")
        final_df.to_sql('sensor_features', engine, if_exists='replace', index=False, chunksize=1000)
        print("✅ Successfully saved traits to DB.")
        
    except Exception as e:
        print(f"❌ Failed to save to DB: {e}")
        print("⚠️ Continuing with CSV cache only.")

    print("Sample Columns:", final_df.columns.tolist())

if __name__ == "__main__":
    calculate_advanced_features()
