import os
import sys
import redis
import psycopg2
from pymongo import MongoClient
from dotenv import load_dotenv

# Load env vars
load_dotenv()

def test_timescale():
    print("Testing TimescaleDB (PostgreSQL)... ", end="")
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            dbname=os.getenv("DT_POSTGRES_DB", "pdm_timeseries"),
            user=os.getenv("DT_POSTGRES_USER", "postgres"),
            password=os.getenv("DT_POSTGRES_PASSWORD", "password")
        )
        conn.close()
        print("✅ OK")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_redis():
    print("Testing Redis... ", end="")
    try:
        r = redis.Redis(host="localhost", port=6379, db=0)
        r.ping()
        print("✅ OK")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_mongo():
    print("Testing MongoDB... ", end="")
    try:
        user = os.getenv("MONGO_USER", "admin")
        pwd = os.getenv("MONGO_PASSWORD", "password")
        uri = f"mongodb://{user}:{pwd}@localhost:27017/"
        
        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        client.admin.command('ping')
        print("✅ OK")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

if __name__ == "__main__":
    print("--- Checking Backend Services ---")
    results = [test_timescale(), test_redis(), test_mongo()]
    
    if all(results):
        print("\nAll foundational services are reachable! 🚀")
        sys.exit(0)
    else:
        print("\nSome services failed to connect.")
        sys.exit(1)
