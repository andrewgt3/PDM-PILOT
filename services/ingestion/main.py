import os
import time
import json
import random
import logging
from kafka import KafkaProducer # type: ignore
from kafka.errors import NoBrokersAvailable # type: ignore
from pydantic import BaseModel
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingestion-service")

# Configuration
BOOTSTRAP_SERVERS = os.getenv("REDPANDA_BROKERS", "localhost:19092").split(",")
SENSOR_TOPIC = "sensor-readings"
LOGS_TOPIC = "maintenance-logs"

class SensorReading(BaseModel):
    sensor_id: str
    timestamp: str
    temperature: float
    vibration: float
    pressure: float
    rpm: float

class MaintenanceLog(BaseModel):
    machine_id: str
    log_text: str
    timestamp: str

def create_producer():
    retries = 0
    while retries < 10:
        try:
            producer = KafkaProducer(
                bootstrap_servers=BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            logger.info("Connected to Redpanda!")
            return producer
        except NoBrokersAvailable:
            logger.warning("Redpanda not ready, retrying in 5s...")
            time.sleep(5)
            retries += 1
    raise Exception("Could not connect to Redpanda")

def generate_mock_data(producer):
    sensor_ids = ["S-001", "S-002", "S-003", "S-004"]
    machine_ids = ["M-A", "M-B"]
    
    logger.info("Starting mock data generation...")
    
    while True:
        # 1. Generate Sensor Readings
        for sid in sensor_ids:
            reading = SensorReading(
                sensor_id=sid,
                timestamp=datetime.utcnow().isoformat(),
                temperature=round(random.uniform(50.0, 90.0), 2),
                vibration=round(random.uniform(0.0, 5.0), 2),
                pressure=round(random.uniform(10.0, 30.0), 2),
                rpm=round(random.uniform(1000, 3000), 2)
            )
            producer.send(SENSOR_TOPIC, reading.model_dump())
            
        # 2. Randomly Generate Logs (less frequent)
        if random.random() < 0.1: # 10% chance per loop
            mid = random.choice(machine_ids)
            log = MaintenanceLog(
                machine_id=mid,
                log_text=f"Routine check: Machine {mid} operating within normal parameters.",
                timestamp=datetime.utcnow().isoformat()
            )
            producer.send(LOGS_TOPIC, log.model_dump())
            logger.info(f"Produced log for {mid}")

        logger.info(f"Produced sensor batch at {datetime.utcnow()}")
        time.sleep(2) # Simulating 0.5Hz data rate for MVP

if __name__ == "__main__":
    producer = create_producer()
    generate_mock_data(producer)
