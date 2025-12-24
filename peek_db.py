
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

try:
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("SELECT * FROM cwru_features ORDER BY timestamp DESC LIMIT 1;")
    row = cursor.fetchone()
    print("Latest Row:", row)

    conn.close()
except Exception as e:
    print(f"Error: {e}")
