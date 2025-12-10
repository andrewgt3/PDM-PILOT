import os
import json
import logging
import time
import psutil # Not used but good for monitoring
from kafka import KafkaConsumer # type: ignore
from kafka.errors import NoBrokersAvailable # type: ignore
import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("processing-service")

# Config
BOOTSTRAP_SERVERS = os.getenv("REDPANDA_BROKERS", "localhost:19092").split(",")
DB_HOST = os.getenv("DB_HOST", "timescaledb")
DB_NAME = os.getenv("DB_NAME", "pdm_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASSWORD", "password")
SENSOR_TOPIC = "sensor-readings"
LOGS_TOPIC = "maintenance-logs"

# Load Model (Global to avoid reloading)
# In production, this would be optimized or served via a separate model server.
logger.info("Loading binding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("Model loaded.")

def create_db_connection():
    while True:
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASS
            )
            return conn
        except Exception as e:
            logger.warning(f"DB not ready ({e}), retrying...")
            time.sleep(5)

def create_consumer():
    while True:
        try:
            consumer = KafkaConsumer(
                SENSOR_TOPIC, LOGS_TOPIC,
                bootstrap_servers=BOOTSTRAP_SERVERS,
                group_id="processing-group",
                auto_offset_reset='earliest',
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            return consumer
        except NoBrokersAvailable:
            logger.warning("Redpanda not ready, retrying...")
            time.sleep(5)

def process_messages():
    conn = create_db_connection()
    consumer = create_consumer()
    
    cursor = conn.cursor()
    
    logger.info("Starting message processing...")
    
    for message in consumer:
        topic = message.topic
        data = message.value
        
        try:
            if topic == SENSOR_TOPIC:
                # Insert into Hypertable
                cursor.execute(
                    """
                    INSERT INTO sensor_readings (time, sensor_id, temperature, vibration, pressure, rpm)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (data['timestamp'], data['sensor_id'], data['temperature'], 
                     data['vibration'], data['pressure'], data['rpm'])
                )
            
            elif topic == LOGS_TOPIC:
                # Vectorize and Insert
                log_text = data['log_text']
                embedding = model.encode(log_text).tolist()
                
                cursor.execute(
                    """
                    INSERT INTO maintenance_logs (machine_id, log_text, timestamp, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (data['machine_id'], log_text, data['timestamp'], str(embedding)) 
                    # pgvector expects '[...]' string or list depending on adapter. str(list) works for psycopg2 usually with pgvector cast.
                    # Or use execute args directly if psycopg2 sees it as array, but vector extension usually requires special handling or string format.
                    # String format '[1,2,3]' is safest without registering adapter explicitly.
                )
            
            conn.commit()
            # logger.info(f"Processed message from {topic}")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            conn.rollback()

if __name__ == "__main__":
    process_messages()
