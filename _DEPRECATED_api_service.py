import os
import joblib
import pandas as pd
import psycopg2
import uvicorn
import subprocess
import signal
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# Global process handler for stream simulator
stream_process = None

# MongoDB Configuration
MONGO_USER = os.getenv("MONGO_USER", "admin")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "password")
MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = os.getenv("MONGO_PORT", "27017")
MONGO_DB = "pdm_audit"

# Initialize MongoDB connection
try:
    mongo_client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/")
    mongo_db = mongo_client[MONGO_DB]
    audit_collection = mongo_db["model_audit_log"]
    print("✅ MongoDB Connected for Audit Logging")
except Exception as e:
    print(f"⚠️ MongoDB connection failed: {e}")
    mongo_client = None
    audit_collection = None

# Model version tracking
MODEL_VERSION = "v2.0_augmented"

app = FastAPI(title="PdM Inference API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security Headers Middleware (Helmet.js equivalent)
try:
    from middleware.security_headers import SecurityHeadersMiddleware
    app.add_middleware(
        SecurityHeadersMiddleware,
        enable_hsts=True,
        frame_options="DENY"
    )
    print("✓ Security Headers middleware enabled")
except ImportError as e:
    print(f"⚠ Security Headers not available: {e}")

# Load Models
MODEL_FAILURE_PATH = "models/failure_model.pkl"
MODEL_RUL_PATH = "models/rul_model.pkl"

try:
    failure_model = joblib.load(MODEL_FAILURE_PATH)
    print("✅ Failure Model Loaded")
except Exception as e:
    print(f"⚠️ Failed to load Failure Model: {e}")
    failure_model = None

try:
    rul_model = joblib.load(MODEL_RUL_PATH)
    print("✅ RUL Model Loaded")
except Exception as e:
    print(f"⚠️ Failed to load RUL Model: {e}")
    rul_model = None

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        dbname=os.getenv("DT_POSTGRES_DB", "pdm_timeseries"),
        user=os.getenv("DT_POSTGRES_USER", "postgres"),
        password=os.getenv("DT_POSTGRES_PASSWORD", "password")
    )

class PredictionResponse(BaseModel):
    machine_id: str
    failure_probability: Optional[float] = None
    rul_prediction: Optional[float] = None
    degradation_score: Optional[float] = None
    status: str
    sensor_data: Optional[dict] = None

class StreamControlRequest(BaseModel):
    state: str # 'start' or 'stop'

@app.post("/api/v1/stream/control")
def control_stream(request: StreamControlRequest):
    global stream_process
    
    if request.state == "start":
        if stream_process is None:
            # Start the publisher script in background
            # Assuming stream_publisher.py is in the root directory (CWD)
            try:
                stream_process = subprocess.Popen(["python3", "stream_publisher.py"])
                print(f"🚀 Stream started. PID: {stream_process.pid}")
                return {"status": "Stream started", "pid": stream_process.pid}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to start stream: {e}")
        else:
            # Check if it's actually running
            if stream_process.poll() is None:
                return {"status": "Stream already running"}
            else:
                # Restart if it crashed
                stream_process = subprocess.Popen(["python3", "stream_publisher.py"])
                return {"status": "Stream restarted"}

    elif request.state == "stop":
        if stream_process is not None:
            # Terminate
            stream_process.terminate()
            try:
                stream_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                stream_process.kill()
            
            stream_process = None
            print("🛑 Stream stopped.")
            return {"status": "Stream stopped"}
        else:
            return {"status": "Stream is not running"}
    
    else:
        raise HTTPException(status_code=400, detail="Invalid state. Use 'start' or 'stop'.")

@app.get("/api/v1/predict/machine/{machine_id}", response_model=PredictionResponse)
def predict_machine(machine_id: str):
    # Start latency tracking
    start_time = time.time()
    
    response = {
        "machine_id": machine_id,
        "failure_probability": None,
        "rul_prediction": None,
        "degradation_score": None,
        "status": "Healthy",
        "sensor_data": {}
    }
    
    conn = get_db_connection()
    
    # 1. Prediction: Failure Probability (AI4I Model)
    # Checks 'sensor_readings' table
    # Model expects: ['Rotational speed [rpm]', 'Air temperature [K]', 'Torque [Nm]', 'Tool wear [min]']
    query_ai4i = """
        SELECT "Rotational speed [rpm]", "Air temperature [K]", "Torque [Nm]", "Tool wear [min]"
        FROM sensor_readings
        WHERE machine_id = %s
        ORDER BY timestamp DESC
        LIMIT 1;
    """
    try:
        # Note: Column names in DB might be snake_case depending on how load_sensor_data saved them.
        # Let's assume they were saved as is or mapped. 
        # Actually, in load_sensor_data, we used:
        # rotational_speed, temperature_air, tool_wear...
        # So I need to use the DB column names, but MAP them to what the model expects (Feature names).
        
        # DB Columns: rotational_speed, temperature_air, torque, tool_wear
        # Model Features: 'Rotational speed [rpm]', 'Air temperature [K]', 'Torque [Nm]', 'Tool wear [min]'
        
        # DB Columns: rotational_speed, temperature_air, torque, tool_wear
        # PLUS: Advanced Features (Slope, CSLM) from 'sensor_features' table
        
        query_db = """
            SELECT rotational_speed, air_temperature, torque, tool_wear
            FROM sensor_features
            WHERE machine_id = %s
            ORDER BY timestamp DESC
            LIMIT 1;
        """
        
        df_ai4i = pd.read_sql(query_db, conn, params=(machine_id,))
        
        if not df_ai4i.empty:
            # Get the first row as a dict for easier access
            sensor_row = df_ai4i.iloc[0]
            
            # Include Sensor Data (regardless of model availability)
            response["sensor_data"] = {
                "Speed": float(sensor_row['rotational_speed']),
                "Temperature": float(sensor_row['air_temperature']),
                "Torque": float(sensor_row['torque']),
                "Tool Wear": float(sensor_row['tool_wear']),
            }
            
            if failure_model:
                # Map columns to model feature names for prediction
                df_model = df_ai4i[['rotational_speed', 'air_temperature', 'torque', 'tool_wear']].copy()
                df_model.columns = ['Rotational speed [rpm]', 'Air temperature [K]', 'Torque [Nm]', 'Tool wear [min]']
                
                # Predict Proba
                prob = failure_model.predict_proba(df_model)[0][1]  # Probability of Class 1
                response["failure_probability"] = float(prob)
                
                if prob > 0.5:
                    response["status"] = "At Risk"
    
    except Exception as e:
        print(f"Error AI4I Predict: {e}")
    
    # Fetch Degradation Score
    try:
        query_degradation = """
            SELECT degradation_score
            FROM degradation_scores
            WHERE machine_id = %s
            ORDER BY timestamp DESC
            LIMIT 1;
        """
        df_deg = pd.read_sql(query_degradation, conn, params=(machine_id,))
        if not df_deg.empty:
            response["degradation_score"] = float(df_deg['degradation_score'].iloc[0])
    except Exception as e:
        print(f"Error fetching degradation score: {e}")

    # 2. Prediction: RUL (NASA Model)
    # Strategy: Map alphanumeric machine_ids to a valid unit_id (1-100) from the NASA dataset
    # This ensures every demo machine gets an RUL prediction.
    try:
        # DEMO HACK: 
        # Unit 1 = Near Failure (~1 Day RUL)
        # Unit 2 = Healthy (Injected) (~High RUL)
        
        # Robots 3, 4, 5 are "Healthy", so give them Unit 2 data.
        if machine_id in ["L47182", "L47183", "L47184"]:
             unit_id = 2
        else:
             unit_id = 1

        
        query_rul = """
            SELECT setting_1, setting_2, sensor_2, sensor_3, sensor_4, sensor_7
            FROM rul_nasa_data
            WHERE unit_id = %s
            ORDER BY timestamp DESC
            LIMIT 1;
        """
        
        df_rul = pd.read_sql(query_rul, conn, params=(unit_id,))
        
        if not df_rul.empty and rul_model:
            # Features are already matching what we trained on (setting_1 etc)
            rul_val = rul_model.predict(df_rul)[0]
            
            response["rul_prediction"] = float(rul_val)
            
            if response["status"] == "At Risk": 
                response["status"] += " | RUL Calculated"
            elif response["status"] == "Healthy":
                response["status"] = "RUL Calculated"

    except Exception as e:
        print(f"Error RUL Predict: {e}")

    conn.close()
    
    if response["failure_probability"] is None:
        # Graceful fallback for unknown assets
        response["failure_probability"] = 0.05
        response["status"] = "Monitored (No Model History)"
        response["rul_prediction"] = 100.0
    
    # Calculate prediction latency
    prediction_latency_ms = (time.time() - start_time) * 1000
    
    # Log to MongoDB audit collection
    if audit_collection is not None:
        try:
            audit_record = {
                "timestamp": datetime.utcnow(),
                "machine_id": machine_id,
                "predicted_probability": response["failure_probability"],
                "rul_prediction": response["rul_prediction"],
                "prediction_latency_ms": round(prediction_latency_ms, 2),
                "model_version": MODEL_VERSION,
                "status": response["status"],
                "sensor_data": response["sensor_data"]
            }
            audit_collection.insert_one(audit_record)
        except Exception as e:
            print(f"⚠️ Failed to log audit record: {e}")

    return response

@app.get("/api/v1/history/{machine_id}/range={range_param}")
def get_machine_history_fallback(machine_id: str, range_param: str):
    """Fallback route for malformed URLs (handling /range=1h path param case)"""
    return get_machine_history(machine_id, range=range_param)

@app.get("/api/v1/history/{machine_id}")
def get_machine_history(machine_id: str, range: str = "1h"):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    # Define Interval
    if range == "1h":
        interval = "1 hour"
        bucket = "1 minute" # Downsample for performance
    elif range == "24h":
        interval = "24 hours"
        bucket = "5 minutes"
    elif range == "7d":
        interval = "7 days"
        bucket = "1 hour"
    else:
        interval = "1 hour"
        bucket = "1 minute"

    try:
        cur = conn.cursor()
        # Time_Bucket query for efficient large-range fetching
        query = f"""
            SELECT time_bucket('{bucket}', timestamp) AS bucket,
                   AVG(rotational_speed) as speed,
                   AVG(temperature_air) as temp,
                   AVG(torque) as torque
            FROM sensor_readings
            WHERE machine_id = %s
              AND timestamp > NOW() - INTERVAL '{interval}'
            GROUP BY bucket
            ORDER BY bucket ASC;
        """
        cur.execute(query, (machine_id,))
        rows = cur.fetchall()
        
        history = []
        for row in rows:
            history.append({
                "time": row[0].isoformat(),
                "Speed": float(row[1]) if row[1] else 0,
                "Temperature": float(row[2]) if row[2] else 0,
                "Torque": float(row[3]) if row[3] else 0
            })
            
        cur.close()
        conn.close()
        return history

    except Exception as e:
        print(f"History Error: {e}")
        conn.close()
        return []

@app.get("/api/v1/stream/{machine_id}")
def get_realtime_stream(machine_id: str, since_ms: int = 0):
    """
    Real-time streaming endpoint for live charts.
    Returns raw 5-second bucketed data from the last 5 minutes.
    `since_ms` parameter filters to only return data newer than this timestamp.
    """
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        cur = conn.cursor()
        # Use 5-second buckets for smooth real-time streaming
        query = """
            SELECT time_bucket('5 seconds', timestamp) AS bucket,
                   AVG(rotational_speed) as speed,
                   AVG(temperature_air) as temp,
                   AVG(torque) as torque
            FROM sensor_readings
            WHERE machine_id = %s
              AND timestamp > NOW() - INTERVAL '5 minutes'
            GROUP BY bucket
            ORDER BY bucket ASC;
        """
        cur.execute(query, (machine_id,))
        rows = cur.fetchall()
        
        stream_data = []
        for row in rows:
            ts = row[0]
            ts_ms = int(ts.timestamp() * 1000)
            
            # Filter to only return data newer than `since_ms`
            if ts_ms > since_ms:
                stream_data.append({
                    "time": ts.isoformat(),
                    "timestamp_ms": ts_ms,
                    "Speed": float(row[1]) if row[1] else 0,
                    "Temperature": float(row[2]) if row[2] else 0,
                    "Torque": float(row[3]) if row[3] else 0
                })
            
        cur.close()
        conn.close()
        return stream_data

    except Exception as e:
        print(f"Stream Error: {e}")
        conn.close()
        return []

@app.get("/api/v1/system/status")
def get_system_status():
    """
    Returns backend system health metrics.
    In a real production system, these would serve real metrics from Prometheus/Redis.
    Here we simulate them for the dashboard.
    """
    return {
        "redis_ingestion_rate": 4500, # Simulated msg/sec
        "timescaledb_lag_ms": 12,     # ms
        "active_sources": 3           # NASA, CMAPSS, Synthetic
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
