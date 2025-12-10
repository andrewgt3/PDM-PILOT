import os
import joblib
import pandas as pd
import psycopg2
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="PdM Inference API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    status: str
    sensor_data: Optional[dict] = None

@app.get("/api/v1/predict/machine/{machine_id}", response_model=PredictionResponse)
def predict_machine(machine_id: str):
    response = {
        "machine_id": machine_id,
        "failure_probability": None,
        "rul_prediction": None,
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
        
        query_db = """
            SELECT rotational_speed, temperature_air, torque, tool_wear
            FROM sensor_readings
            WHERE machine_id = %s
            ORDER BY timestamp DESC
            LIMIT 1;
        """
        
        df_ai4i = pd.read_sql(query_db, conn, params=(machine_id,))
        
        if not df_ai4i.empty and failure_model:
            # Map columns to model feature names
            df_ai4i.columns = ['Rotational speed [rpm]', 'Air temperature [K]', 'Torque [Nm]', 'Tool wear [min]']
            
            # Predict Proba
            prob = failure_model.predict_proba(df_ai4i)[0][1] # Probability of Class 1
            response["failure_probability"] = float(prob)
            
            # Include Sensor Data
            response["sensor_data"] = {
                "Rotational Speed": float(df_ai4i.iloc[0]['Rotational speed [rpm]']),
                "Temperature": float(df_ai4i.iloc[0]['Air temperature [K]']),
                "Torque": float(df_ai4i.iloc[0]['Torque [Nm]']),
                "Tool Wear": float(df_ai4i.iloc[0]['Tool wear [min]'])
            }
            
            if prob > 0.5:
                response["status"] = "At Risk"

    except Exception as e:
        print(f"Error AI4I Predict: {e}")

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
    
    if response["failure_probability"] is None and response["rul_prediction"] is None:
        raise HTTPException(status_code=404, detail="Machine ID not found or no data available.")

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
