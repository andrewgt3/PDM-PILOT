
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

FEATURE_STORE_PATH = "data/feature_store.csv"
MODEL_PATH = "models/rul_transfer_model.pkl"

def train_rul_transfer_model():
    print("🚀 Starting RUL Transfer Model Training...")
    
    # Configuration
    DB_HOST = "localhost"
    DB_PORT = "5432"
    DB_NAME = os.getenv("DT_POSTGRES_DB", "pdm_timeseries")
    DB_USER = os.getenv("DT_POSTGRES_USER", "postgres")
    DB_PASS = os.getenv("DT_POSTGRES_PASSWORD", "password")
    
    from sqlalchemy import create_engine
    
    # 1. Load Advanced Features from DB
    try:
        db_url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(db_url)
        print("📥 Fetching features from TimescaleDB 'sensor_features'...")
        df = pd.read_sql("SELECT * FROM sensor_features", engine)
        print(f"📊 Loaded {len(df)} records from Feature Store.")
    except Exception as e:
        print(f"❌ Failed to load from DB: {e}")
        return
    
    # 2. Construct Proxy Target (RUL)
    # Since we don't have labeled RUL for AI4I, we simulate it for the Transfer Learning demo.
    # We assume 'Time to Last Timestamp' for each machine is the RUL.
    # In a real Transfer scenario, we'd train on NASA and apply here, but the prompt asks to "modify THE training script"
    # implying we train ON these features.
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate RUL: Max Time - Current Time (per machine)
    # This assumes the dataset ends at failure or a common censorship point.
    max_times = df.groupby('machine_id')['timestamp'].transform('max')
    df['RUL_seconds'] = (max_times - df['timestamp']).dt.total_seconds()
    df['RUL_days'] = df['RUL_seconds'] / (3600 * 24)
    
    # 3. Select Features
    # User requested to "exclusively use these new, time-aggregated features"
    feature_cols = [
        c for c in df.columns 
        if '_8h' in c or 'slope_24h' in c or 'CSLM' in c
    ]
    print(f"🎯 Training on {len(feature_cols)} Advanced Features: {feature_cols}")
    
    X = df[feature_cols]
    y = df['RUL_days']
    
    # Handle NaN from rolling (first few rows)
    # Drop rows with NaNs
    clean_idx = X.dropna().index
    X = X.loc[clean_idx]
    y = y.loc[clean_idx]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train Model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"✅ Training Complete.")
    print(f"📉 RMSE: {rmse:.4f} days")
    print(f"📈 R2 Score: {r2:.4f}")
    
    # 6. Save Model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_rul_transfer_model()
