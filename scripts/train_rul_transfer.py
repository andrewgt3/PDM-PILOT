import os
import pandas as pd
import joblib
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from dotenv import load_dotenv

load_dotenv()

# Config
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/rul_model.pkl"

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        dbname=os.getenv("DT_POSTGRES_DB", "pdm_timeseries"),
        user=os.getenv("DT_POSTGRES_USER", "postgres"),
        password=os.getenv("DT_POSTGRES_PASSWORD", "password")
    )

def train_rul_model():
    print("Connecting to Database...")
    conn = get_db_connection()
    
    # Query Data (NASA RUL)
    query = """
        SELECT unit_id, cycle, setting_1, setting_2, sensor_2, sensor_3, sensor_4, sensor_7, sensor_11, sensor_12
        FROM rul_nasa_data
    """
    
    try:
        df = pd.read_sql(query, conn)
    except Exception as e:
        print(f"❌ Error querying data: {e}")
        conn.close()
        return
    finally:
        conn.close()

    if df.empty:
        print("❌ No data found in 'rul_nasa_data'.")
        return

    print(f"Loaded {len(df)} records.")

    # Calculate RUL Target
    print("Calculating RUL labels...")
    
    # RUL = Max_Cycle - Current_Cycle
    max_cycles = df.groupby('unit_id')['cycle'].max().rename('max_cycle')
    df = df.merge(max_cycles, on='unit_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    
    # Features
    # Note: Mock data might miss some sensors, so we limit to guaranteed ones + robust dropna
    features = ['setting_1', 'setting_2', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7']
    target = 'RUL'
    
    # Preprocessing: Drop NaNs
    df.dropna(subset=features + [target], inplace=True)
    
    if df.empty:
        print("❌ Data Empty after dropping NaNs!")
        return

    X = df[features]
    y = df[target]
    
    # Train/Test Split
    test_size = 0.2
    if len(df) < 20: 
        test_size = 0.2 # Small dataset logic handled by sklearn mostly ok, or just trust standard
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    print(f"Training GradientBoostingRegressor on {len(X_train)} records...")
    
    # Train
    reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    reg.fit(X_train, y_train)
    
    # Evaluate
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Save
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(reg, MODEL_PATH)
    print("✅ RUL Model saved successfully.")

if __name__ == "__main__":
    train_rul_model()
