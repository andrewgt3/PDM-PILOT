import os
from sqlalchemy import create_engine, text
import pandas as pd
from datetime import datetime
import pytz
from dotenv import load_dotenv

load_dotenv()

# DB Connection
conn_str = os.getenv('DATABASE_URL')
if not conn_str:
    raise ValueError("DATABASE_URL environment variable is required")
engine = create_engine(conn_str)

print(f"Python Current Time: {datetime.now()}")
print(f"Python UTC Time: {datetime.utcnow()}")

with engine.connect() as conn:
    # Check database timezone
    db_now = conn.execute(text("SELECT NOW()")).scalar()
    print(f"Database NOW(): {db_now}")
    
    # Check latest sensor timestamp
    latest = conn.execute(text("SELECT MAX(timestamp) FROM sensors")).scalar()
    print(f"Latest sensor timestamp: {latest}")
    
    # Calculate difference
    if latest:
        time_diff = db_now - latest
        print(f"Time difference: {time_diff}")
    
    # Try the dashboard query
    query = """
    SELECT DISTINCT ON (asset_id) 
        asset_id, timestamp, vibration_x, motor_temp_c, rul_hours
    FROM sensors 
    WHERE timestamp > NOW() - INTERVAL '1 hour'
    ORDER BY asset_id, timestamp DESC
    """
    result = conn.execute(text(query)).fetchall()
    print(f"\nDashboard query returned {len(result)} rows")
    
    # Check with different time window
    query2 = """
    SELECT COUNT(*) FROM sensors 
    WHERE timestamp > NOW() - INTERVAL '24 hours'
    """
    count_24h = conn.execute(text(query2)).scalar()
    print(f"Rows in last 24 hours: {count_24h}")
