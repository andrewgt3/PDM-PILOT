#!/usr/bin/env python3
"""
Redis Stream Consumer for GAIA Predictive Maintenance

Subscribes to Redis sensor_stream channel and:
- Batches incoming messages (100 per batch)
- Bulk inserts raw readings to PostgreSQL sensor_readings
- Real-time feature extraction via SignalProcessor
- Inserts computed features to cwru_features table

Refactored for:
- Graceful Shutdown (Signal Handling)
- Enterprise Logging (Structcal)
- Configuration Management (config.py)
- Connection Resilience (Retry Logic)

Author: Senior Backend Engineer
"""

import signal
import sys
import json
import time
import redis
import psycopg2
from psycopg2 import extras, OperationalError
import numpy as np
import joblib
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

# Enterprise Imports
import structlog
from config import get_settings
from logger import configure_logging, get_logger

# Custom Modules
from advanced_features import SignalProcessor, SAMPLE_RATE, N_SAMPLES

# =============================================================================
# INITIALIZATION
# =============================================================================

# Configure structured logging
configure_logging(environment="production", log_level="INFO")
logger = get_logger("stream_consumer")

# Load settings
settings = get_settings()

REDIS_CHANNEL = 'sensor_stream'
BATCH_SIZE = 100

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_redis_connection() -> redis.Redis:
    """Create Redis pub/sub connection using config settings."""
    return redis.Redis(
        host=settings.redis.host,
        port=settings.redis.port,
        password=settings.redis.password.get_secret_value() if settings.redis.password else None,
        decode_responses=True,
        socket_timeout=settings.redis.socket_timeout,
        retry_on_timeout=True
    )

def get_db_connection():
    """Create PostgreSQL connection using config settings."""
    return psycopg2.connect(
        host=settings.database.host,
        port=settings.database.port,
        database=settings.database.name,
        user=settings.database.user,
        password=settings.database.password.get_secret_value(),
        connect_timeout=10
    )

def load_ml_model() -> Tuple[Any, Any, Any]:
    """Load trained XGBoost model for real-time inference."""
    try:
        pipeline = joblib.load('gaia_model.pkl')
        model = pipeline.get('model')
        scaler = pipeline.get('scaler')
        feature_columns = pipeline.get('feature_columns')
        
        if model and scaler and feature_columns:
            logger.info("ml_model_loaded", 
                       features_count=len(feature_columns), 
                       model_type=type(model).__name__)
            return model, scaler, feature_columns
        else:
             logger.warning("ml_model_incomplete", detail="Pipeline file loaded but missing components")
             return None, None, None
             
    except FileNotFoundError:
        logger.warning("ml_model_not_found", detail="gaia_model.pkl not found, skipping inference")
        return None, None, None
    except Exception as e:
        logger.error("ml_model_load_failed", error=str(e))
        return None, None, None

# =============================================================================
# CONSUMER CLASS
# =============================================================================

class StreamConsumer:
    """
    Real-time stream consumer with batch processing, resilience, and graceful shutdown.
    """
    
    def __init__(self):
        """Initialize consumer resources."""
        self.processor = SignalProcessor(sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES)
        self.model, self.scaler, self.feature_columns = load_ml_model()
        
        # Buffer
        self.buffer: List[Dict] = []
        
        # Statistics
        self.messages_received = 0
        self.batches_processed = 0
        self.features_computed = 0
        
        # Loop control
        self.running = True
        
    def shutdown_handler(self, signum, frame):
        """Handle OS signals (SIGINT/SIGTERM) to trigger graceful shutdown."""
        logger.info("shutdown_signal_received", signal=signum)
        self.running = False

    def process_message(self, message_data: str) -> Dict:
        """Parse incoming JSON message."""
        try:
            return json.loads(message_data)
        except json.JSONDecodeError as e:
            logger.error("json_decode_error", error=str(e), data=message_data[:50])
            raise

    def extract_features_and_predict(self, payload: Dict) -> Dict:
        """Extract features and run ML inference."""
        # 1. Prepare Inputs
        vibration = np.array(payload.get('vibration_raw', []))
        if len(vibration) == 0:
             logger.warning("empty_vibration_data", machine_id=payload.get("machine_id"))
             
        telemetry = {
            'rotational_speed': payload.get('rotational_speed', 1800.0),
            'temperature': payload.get('temperature', 70.0),
            'torque': payload.get('torque', 40.0),
            'tool_wear': payload.get('tool_wear', 0.1)
        }
        
        # 2. Extract Signal Features
        # Note: process_signal might print errors internally if we didn't refactor it, 
        # but capturing exceptions here is safer.
        features = self.processor.process_signal(vibration, telemetry)
        
        # 3. ML Inference
        if self.model is not None and len(vibration) > 0:
            try:
                # Construct feature vector matching training columns
                feature_vector = np.array([[features.get(col, 0.0) for col in self.feature_columns]])
                # Handle any NaNs or Infinities
                feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Scale and Predict
                feature_scaled = self.scaler.transform(feature_vector)
                proba = self.model.predict_proba(feature_scaled)[0]
                
                features['failure_prediction'] = float(proba[1])
                features['failure_class'] = int(proba[1] > 0.5)
            except Exception as e:
                logger.error("inference_failed", error=str(e))
                features['failure_prediction'] = 0.0
                features['failure_class'] = 0
        else:
            features['failure_prediction'] = 0.0
            features['failure_class'] = 0
        
        # 4. Add Metadata
        features['timestamp'] = payload.get('timestamp', datetime.now().isoformat())
        features['machine_id'] = payload.get('machine_id', 'unknown')
        
        return features

    def bulk_insert_readings(self, conn, readings: List[Dict]):
        """Bulk insert raw readings."""
        if not readings:
            return
            
        cursor = conn.cursor()
        query = """
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
        
        try:
            extras.execute_values(cursor, query, values, page_size=100)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error("bulk_insert_failed", error=str(e))
            raise
        finally:
            cursor.close()

    def insert_features(self, conn, features: Dict):
        """Insert features with exponential smoothing."""
        cursor = conn.cursor()
        try:
            # 1. Fetch previous score for smoothing
            cursor.execute("""
                SELECT degradation_score_smoothed 
                FROM cwru_features 
                WHERE machine_id = %s 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (features['machine_id'],))
            
            result = cursor.fetchone()
            previous_smoothed = result[0] if result else features.get('degradation_score', 0.0)
            
            # 2. Calculate EMA
            alpha = 0.15
            current_score = features.get('degradation_score', 0.0)
            degradation_smoothed = alpha * current_score + (1 - alpha) * previous_smoothed
            degradation_smoothed = float(degradation_smoothed)
            
            # 3. Update feature dict
            features['degradation_score_smoothed'] = degradation_smoothed
            
            # 4. Insert
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
            cursor.execute(insert_sql, features)
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            logger.error("feature_insert_failed", error=str(e))
            # Don't raise here to avoid stopping the loop for a single row error
        finally:
            cursor.close()

    def run(self):
        """Main loop with connection resilience."""
        logger.info("consumer_startup")
        
        # Register Signals
        signal.signal(signal.SIGINT, self.shutdown_handler)
        signal.signal(signal.SIGTERM, self.shutdown_handler)
        
        while self.running:
            pubsub = None
            conn = None
            
            try:
                # 1. Establish Connections
                logger.info("connecting_to_services")
                
                r = get_redis_connection()
                r.ping() # Check connection
                pubsub = r.pubsub()
                pubsub.subscribe(REDIS_CHANNEL)
                
                conn = get_db_connection()
                
                logger.info("connected", redis_channel=REDIS_CHANNEL)
                
                start_time = time.time()
                last_heartbeat = 0
                
                # 2. Message Processing Loop
                while self.running:
                    try:
                        # HEARTBEAT (Every 30s)
                        if time.time() - last_heartbeat > 30:
                            try:
                                r.set("system:health:consumer", int(time.time()), ex=60)
                                last_heartbeat = time.time()
                                logger.debug("heartbeat_sent")
                            except Exception as hb_error:
                                logger.warning("heartbeat_failed", error=str(hb_error))

                        # Non-blocking check for messages with 1s timeout
                        # This allows the loop to cycle and check self.running regularly
                        message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)

                        
                        if message is None:
                            continue
                            
                        if message['type'] != 'message':
                            continue
                        

                        # BEGIN PROCESSING
                        try:
                            # Process Payload
                            payload = self.process_message(message['data'])
                            self.messages_received += 1
                            
                            # Add to Buffer
                            self.buffer.append(payload)
                            
                            # Compute features & Persist
                            features = self.extract_features_and_predict(payload)
                            self.features_computed += 1
                            self.insert_features(conn, features)
                            
                            # Flush Buffer if full
                            if len(self.buffer) >= BATCH_SIZE:
                                batch_start = time.perf_counter()
                                self.bulk_insert_readings(conn, self.buffer)
                                processing_time_ms = (time.perf_counter() - batch_start) * 1000
                                
                                self.batches_processed += 1
                                
                                # Log batch_processed with count and processing_time
                                logger.info(
                                    "batch_processed",
                                    count=len(self.buffer),
                                    processing_time_ms=round(processing_time_ms, 2),
                                    batch_number=self.batches_processed,
                                    total_messages=self.messages_received,
                                )
                                
                                self.buffer.clear()
                                
                        except (redis.ConnectionError, redis.TimeoutError, OperationalError) as e:
                            # Critical connection error -> Re-raise to trigger reconnection
                            raise
                        except Exception as e:
                            # POISON PILL PROTECTION
                            # Capture malformed data, log it, and push to DLQ
                            logger.error("processing_failed", error=str(e), action="push_to_dlq")
                            try:
                                r.lpush('sensor_stream:dlq', message['data'])
                            except Exception as dlq_error:
                                logger.critical("dlq_push_failed", error=str(dlq_error))
                            
                            # Continue to next message
                            continue

                    except (redis.ConnectionError, redis.TimeoutError, OperationalError):
                        # Propagate connection errors to the main connection retry loop
                        raise
                    except Exception as e:
                        # Log unexpected loop errors (e.g. get_message failure) and continue
                        logger.error("unexpected_loop_error", error=str(e))

                        
            except (redis.ConnectionError, redis.TimeoutError, OperationalError) as e:
                # Log connection_lost event for monitoring
                logger.warning(
                    "connection_lost",
                    error_type=type(e).__name__,
                    error=str(e),
                    retry_in_seconds=5,
                )
                time.sleep(5)
                
            except Exception as e:
                logger.critical("unexpected_crash", error=str(e), action="retrying_in_5s")
                time.sleep(5)
                
            finally:
                # Cleanup before retry or exit
                if pubsub:
                    try: pubsub.close()
                    except: pass
                if conn:
                    try: conn.close()
                    except: pass
                    
        # =============================================================================
        # SHUTDOWN SEQUENCE
        # =============================================================================
        logger.info("shutdown_initiated")
        
        # Final Flush
        if self.buffer:
            logger.info("flushing_final_buffer", count=len(self.buffer))
            try:
                # Re-connect specifically for flush if needed, or use existing if valid
                # For safety, create a fresh connection for the final flush
                final_conn = get_db_connection()
                self.bulk_insert_readings(final_conn, self.buffer)
                final_conn.close()
                logger.info("final_flush_success")
            except Exception as e:
                logger.error("final_flush_failed", error=str(e))
        
        logger.info("consumer_stopped_cleanly")

if __name__ == "__main__":
    consumer = StreamConsumer()
    consumer.run()
