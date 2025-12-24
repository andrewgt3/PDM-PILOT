#!/usr/bin/env python3
"""
Redis Stream Consumer for GAIA Predictive Maintenance

Subscribes to Redis sensor_stream channel and:
- Batches incoming messages (100 per batch)
- Bulk inserts raw readings to PostgreSQL sensor_readings
- Real-time feature extraction via SignalProcessor
- Inserts computed features to cwru_features table

Author: Senior Backend Engineer
"""

import os
import json
import time
import redis
import psycopg2
from psycopg2 import extras
import numpy as np
import joblib
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

# Import custom modules
from advanced_features import SignalProcessor, SAMPLE_RATE, N_SAMPLES

# Load environment variables
load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)
REDIS_CHANNEL = 'sensor_stream'

BATCH_SIZE = 100  # Buffer size before bulk insert


def get_redis_connection():
    """Create Redis pub/sub connection."""
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True
    )
    return r


def get_db_connection():
    """Create PostgreSQL connection."""
    database_url = os.getenv('DATABASE_URL')
    
    if database_url:
        return psycopg2.connect(database_url)
    else:
        return psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'gaia'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'postgres')
        )


def load_ml_model():
    """Load trained XGBoost model for real-time inference."""
    try:
        pipeline = joblib.load('gaia_model.pkl')
        return pipeline['model'], pipeline['scaler'], pipeline['feature_columns']
    except Exception as e:
        print(f"  ⚠ Model not loaded: {e}")
        return None, None, None


class StreamConsumer:
    """
    Real-time stream consumer with batch processing.
    
    Features:
    - Redis pub/sub subscription
    - Batch buffering (100 messages)
    - PostgreSQL bulk inserts
    - Real-time feature extraction
    - ML inference on every message
    """
    
    def __init__(self):
        """Initialize consumer with connections and processors."""
        # Signal processor for feature extraction
        self.processor = SignalProcessor(sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES)
        
        # ML model for predictions
        self.model, self.scaler, self.feature_columns = load_ml_model()
        
        # Message buffer for batch processing
        self.buffer: List[Dict] = []
        
        # Stats
        self.messages_received = 0
        self.batches_processed = 0
        self.features_computed = 0
    
    def process_message(self, message_data: str) -> Dict:
        """Parse incoming JSON message."""
        return json.loads(message_data)
    
    def extract_features_and_predict(self, payload: Dict) -> Dict:
        """
        Extract 26 features and run ML inference.
        
        Args:
            payload: Message with vibration_raw array
            
        Returns:
            Dictionary with features and prediction
        """
        # Get vibration array
        vibration = np.array(payload['vibration_raw'])
        
        # Prepare telemetry
        telemetry = {
            'rotational_speed': payload.get('rotational_speed', 1800.0),
            'temperature': payload.get('temperature', 70.0),
            'torque': payload.get('torque', 40.0),
            'tool_wear': payload.get('tool_wear', 0.1)
        }
        
        # Extract 26 features
        features = self.processor.process_signal(vibration, telemetry)
        
        # Run ML prediction if model is available
        if self.model is not None:
            # Prepare feature vector
            feature_vector = np.array([[features.get(col, 0.0) for col in self.feature_columns]])
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale and predict
            feature_scaled = self.scaler.transform(feature_vector)
            proba = self.model.predict_proba(feature_scaled)[0]
            
            features['failure_prediction'] = float(proba[1])
            features['failure_class'] = int(proba[1] > 0.5)
        else:
            features['failure_prediction'] = 0.0
            features['failure_class'] = 0
        
        # Add metadata
        features['timestamp'] = payload['timestamp']
        features['machine_id'] = payload['machine_id']
        
        return features
    
    def bulk_insert_readings(self, conn, readings: List[Dict]):
        """
        Bulk insert raw sensor readings to PostgreSQL.
        
        Uses psycopg2's execute_values for efficient batch insert.
        """
        if not readings:
            return
        
        cursor = conn.cursor()
        
        insert_sql = """
            INSERT INTO sensor_readings 
                (timestamp, machine_id, rotational_speed, temperature, torque, tool_wear, vibration_raw)
            VALUES %s
            ON CONFLICT (machine_id, timestamp) DO NOTHING
        """
        
        values = [
            (
                r['timestamp'],
                r['machine_id'],
                r.get('rotational_speed'),
                r.get('temperature'),
                r.get('torque'),
                r.get('tool_wear'),
                json.dumps(r.get('vibration_raw', []))
            )
            for r in readings
        ]
        
        extras.execute_values(cursor, insert_sql, values, page_size=100)
        conn.commit()
        cursor.close()
    
    def insert_features(self, conn, features: Dict):
        """
        Insert computed features to cwru_features table.
        Applies exponential smoothing to degradation_score for stable RUL calculations.
        """
        cursor = conn.cursor()
        
        # Get previous smoothed degradation score for this machine
        cursor.execute("""
            SELECT degradation_score_smoothed 
            FROM cwru_features 
            WHERE machine_id = %s 
            ORDER BY timestamp DESC 
            LIMIT 1
        """, (features['machine_id'],))
        
        result = cursor.fetchone()
        previous_smoothed = result[0] if result else features['degradation_score']
        
        # Apply exponential moving average (EMA)
        # Alpha = 0.15 means more weight on history (85%) vs current reading (15%)
        # This provides ~6.6x smoothing window
        alpha = 0.15
        degradation_smoothed = alpha * features['degradation_score'] + (1 - alpha) * previous_smoothed
        degradation_smoothed = float(degradation_smoothed)
        
        insert_sql = """
            INSERT INTO cwru_features (
                timestamp, machine_id,
                peak_freq_1, peak_freq_2, peak_freq_3, peak_freq_4, peak_freq_5,
                peak_amp_1, peak_amp_2, peak_amp_3, peak_amp_4, peak_amp_5,
                low_band_power, mid_band_power, high_band_power,
                spectral_entropy, spectral_kurtosis, total_power,
                bpfo_amp, bpfi_amp, bsf_amp, ftf_amp, sideband_strength,
                degradation_score, degradation_score_smoothed,
                rotational_speed, temperature, torque, tool_wear,
                failure_prediction, failure_class
            ) VALUES (
                %(timestamp)s, %(machine_id)s,
                %(peak_freq_1)s, %(peak_freq_2)s, %(peak_freq_3)s, %(peak_freq_4)s, %(peak_freq_5)s,
                %(peak_amp_1)s, %(peak_amp_2)s, %(peak_amp_3)s, %(peak_amp_4)s, %(peak_amp_5)s,
                %(low_band_power)s, %(mid_band_power)s, %(high_band_power)s,
                %(spectral_entropy)s, %(spectral_kurtosis)s, %(total_power)s,
                %(bpfo_amp)s, %(bpfi_amp)s, %(bsf_amp)s, %(ftf_amp)s, %(sideband_strength)s,
                %(degradation_score)s, %(degradation_score_smoothed)s,
                %(rotational_speed)s, %(temperature)s, %(torque)s, %(tool_wear)s,
                %(failure_prediction)s, %(failure_class)s
            )
            ON CONFLICT (machine_id, timestamp) DO UPDATE SET
                failure_prediction = EXCLUDED.failure_prediction,
                failure_class = EXCLUDED.failure_class,
                degradation_score_smoothed = EXCLUDED.degradation_score_smoothed
        """
        
        # Add smoothed value to features dict
        features['degradation_score_smoothed'] = degradation_smoothed
        
        cursor.execute(insert_sql, features)
        conn.commit()
        cursor.close()
    
    def run(self):
        """
        Main consumer loop.
        
        Subscribes to Redis channel and processes messages in real-time.
        """
        print("=" * 70)
        print("GAIA STREAM CONSUMER")
        print("Real-Time Feature Extraction & Persistence")
        print("=" * 70)
        
        # Connect to Redis
        try:
            r = get_redis_connection()
            r.ping()
            pubsub = r.pubsub()
            pubsub.subscribe(REDIS_CHANNEL)
            print(f"\n✓ Subscribed to Redis channel: {REDIS_CHANNEL}")
        except redis.ConnectionError as e:
            print(f"\n❌ Redis connection failed: {e}")
            return
        
        # Connect to PostgreSQL
        try:
            conn = get_db_connection()
            print(f"✓ Connected to PostgreSQL")
        except Exception as e:
            print(f"\n❌ Database connection failed: {e}")
            return
        
        # Model status
        if self.model is not None:
            print(f"✓ ML Model loaded ({len(self.feature_columns)} features)")
        else:
            print(f"⚠ ML Model not available - predictions disabled")
        
        print(f"\nBatch Size: {BATCH_SIZE} messages")
        print(f"\n" + "-" * 70)
        print("CONSUMING STREAM")
        print("-" * 70)
        print(f"Press Ctrl+C to stop\n")
        
        start_time = time.time()
        
        try:
            for message in pubsub.listen():
                if message['type'] != 'message':
                    continue
                
                # Parse message
                payload = self.process_message(message['data'])
                self.messages_received += 1
                
                # Add to buffer for batch insert
                self.buffer.append(payload)
                
                # Real-time feature extraction and prediction
                features = self.extract_features_and_predict(payload)
                self.features_computed += 1
                
                # Insert features immediately (real-time)
                try:
                    self.insert_features(conn, features)
                except Exception as e:
                    print(f"  ⚠ Feature insert error: {e}")
                
                # Batch insert raw readings when buffer is full
                if len(self.buffer) >= BATCH_SIZE:
                    try:
                        self.bulk_insert_readings(conn, self.buffer)
                        self.batches_processed += 1
                        
                        elapsed = time.time() - start_time
                        rate = self.messages_received / elapsed if elapsed > 0 else 0
                        
                        print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                              f"Batch {self.batches_processed} | "
                              f"Messages: {self.messages_received:,} | "
                              f"Features: {self.features_computed:,} | "
                              f"Rate: {rate:.1f} msg/s | "
                              f"Last Pred: {features['failure_prediction']:.2%}")
                        
                        self.buffer.clear()
                        
                    except Exception as e:
                        print(f"  ⚠ Batch insert error: {e}")
                        self.buffer.clear()
                        
        except KeyboardInterrupt:
            # Final batch insert
            if self.buffer:
                self.bulk_insert_readings(conn, self.buffer)
            
            print(f"\n\n" + "-" * 70)
            print("CONSUMER STOPPED")
            print("-" * 70)
            elapsed = time.time() - start_time
            print(f"  Messages Received: {self.messages_received:,}")
            print(f"  Features Computed: {self.features_computed:,}")
            print(f"  Batches Processed: {self.batches_processed}")
            print(f"  Duration: {elapsed:.1f} seconds")
            print(f"  Average Rate: {self.messages_received/elapsed:.1f} msg/s")
            
        finally:
            pubsub.close()
            conn.close()


def main():
    """Entry point."""
    consumer = StreamConsumer()
    consumer.run()


if __name__ == "__main__":
    main()
