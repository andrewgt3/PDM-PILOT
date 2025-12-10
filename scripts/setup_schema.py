import os
import psycopg2
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL Connection
def get_pg_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        dbname=os.getenv("DT_POSTGRES_DB", "pdm_timeseries"),
        user=os.getenv("DT_POSTGRES_USER", "postgres"),
        password=os.getenv("DT_POSTGRES_PASSWORD", "password")
    )

def setup_postgres():
    print("Setting up PostgreSQL/TimescaleDB Schema...")
    conn = get_pg_connection()
    conn.autocommit = True
    cursor = conn.cursor()

    try:
        # Enable Extensions
        cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
        print(" - TimescaleDB extension enabled.")

        # 1. Assets Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assets (
                machine_id TEXT PRIMARY KEY,
                asset_type TEXT
            );
        """)
        print(" - 'assets' table created.")

        # 2. Sensor Readings Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensor_readings (
                timestamp TIMESTAMPTZ NOT NULL,
                machine_id TEXT NOT NULL,
                rotational_speed DOUBLE PRECISION,
                temperature_air DOUBLE PRECISION,
                torque DOUBLE PRECISION,
                tool_wear DOUBLE PRECISION
            );
        """)
        print(" - 'sensor_readings' table created.")

        # Convert to Hypertable (Hypertable creation might fail if already exists, handle gracefully)
        try:
            cursor.execute("SELECT create_hypertable('sensor_readings', 'timestamp', if_not_exists => TRUE);")
            print(" - 'sensor_readings' converted to hypertable.")
        except Exception as e:
            print(f" - Hypertable notice: {e}")

    except Exception as e:
        print(f"❌ Error in Postgres setup: {e}")
    finally:
        conn.close()

def setup_mongo():
    print("Setting up MongoDB...")
    try:
        user = os.getenv("MONGO_USER", "admin")
        pwd = os.getenv("MONGO_PASSWORD", "password")
        uri = f"mongodb://{user}:{pwd}@localhost:27017/"
        
        client = MongoClient(uri)
        db = client["pdm_logs"] # Database name for logs
        
        # Create collection explicitly (optional in Mongo, but good for verification)
        if "maintenance_logs" not in db.list_collection_names():
            db.create_collection("maintenance_logs")
            print(" - 'maintenance_logs' collection created.")
        else:
            print(" - 'maintenance_logs' collection already exists.")

    except Exception as e:
        print(f"❌ Error in Mongo setup: {e}")

if __name__ == "__main__":
    setup_postgres()
    setup_mongo()
    print("✅ Schema setup complete.")
