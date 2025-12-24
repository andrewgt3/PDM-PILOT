import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

conn_str = os.getenv('DATABASE_URL')
if not conn_str:
    raise ValueError("DATABASE_URL environment variable is required")
engine = create_engine(conn_str)

# This is the exact query from dashboard.py line 74-79
query_fleet = """
SELECT DISTINCT ON (asset_id) 
    asset_id, timestamp, vibration_x, motor_temp_c, rul_hours
FROM sensors 
WHERE timestamp > NOW() - INTERVAL '1 hour'
ORDER BY asset_id, timestamp DESC
"""

with engine.connect() as conn:
    result = conn.execute(text(query_fleet)).fetchall()
    print(f"Dashboard query returned {len(result)} rows:\n")
    for row in result:
        print(f"{row[0]}: RUL={row[4]}h, Vib={row[2]:.4f}g, Temp={row[3]:.2f}°C")
