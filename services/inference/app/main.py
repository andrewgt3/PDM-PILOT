import os
import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="PdM Inference Service")

DB_HOST = os.getenv("DB_HOST", "timescaledb")
DB_NAME = os.getenv("DB_NAME", "pdm_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASSWORD", "password")

class PredictionRequest(BaseModel):
    sensor_id: str
    machine_id: Optional[str] = None

class PredictionResponse(BaseModel):
    sensor_id: str
    rul_hours: float
    failure_probability: float
    status: str

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        return conn
    except Exception as e:
        print(f"DB Connection failed: {e}")
        return None

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict_maintenance(request: PredictionRequest):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cursor = conn.cursor()
        
        # Fetch last 5 readings
        cursor.execute("""
            SELECT temperature, vibration, rpm 
            FROM sensor_readings 
            WHERE sensor_id = %s 
            ORDER BY time DESC 
            LIMIT 5
        """, (request.sensor_id,))
        readings = cursor.fetchall()
        
        if not readings:
            # Fallback or error if no data (returning safe default for MVP)
            return PredictionResponse(
                sensor_id=request.sensor_id,
                rul_hours=999.0,
                failure_probability=0.0,
                status="NO_DATA"
            )

        # Simple Heuristic / "Model"
        # If avg vibration > 3.0, high risk.
        avg_vibration = sum([r[1] for r in readings]) / len(readings)
        avg_temp = sum([r[0] for r in readings]) / len(readings)
        
        failure_prob = 0.0
        if avg_vibration > 4.0:
            failure_prob = 0.8
        elif avg_vibration > 2.0:
            failure_prob = 0.4
        elif avg_temp > 85.0:
            failure_prob = 0.6
            
        rul = 1000 * (1 - failure_prob)
        
        status = "HEALTHY"
        if failure_prob > 0.7:
            status = "CRITICAL"
        elif failure_prob > 0.3:
            status = "WARNING"
            
        return PredictionResponse(
            sensor_id=request.sensor_id,
            rul_hours=round(rul, 2),
            failure_probability=round(failure_prob, 2),
            status=status
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()
