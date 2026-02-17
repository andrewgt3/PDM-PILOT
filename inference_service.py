#!/usr/bin/env python3
"""
Decoupled Inference Service
===========================
Consumes 'clean_features' from Redis, predicts RUL, and publishes 'inference_results'.

Workflow:
1. Subscribe to Redis Stream `clean_features`.
2. On new message:
   - Parse Station ID and Features.
   - Load appropriate model (via station_config.json).
   - Normalize features (scaler.pkl).
   - Predict RUL (xgb_rul_v1.json).
   - Calculate Health Score (0-100%).
3. Publish to Redis Stream `inference_results`.

Usage:
    python inference_service.py
"""

import sys
import json
import time
import pickle
import logging
import redis
import xgboost as xgb
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "data" / "models"
CONFIG_FILE = BASE_DIR / "station_config.json"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

REDIS_HOST = "localhost"
REDIS_PORT = 6379
INPUT_STREAM = "clean_features"
OUTPUT_STREAM = "inference_results"
CONSUMER_GROUP = "inference_group"
CONSUMER_NAME = "inference_worker_1"

# Feature order MUST match training - load from rul_features.txt if available
RUL_FEATURES_PATH = MODELS_DIR / "rul_features.txt"
FEATURES = ['rms', 'kurtosis', 'p2p', 'skewness', 'crest', 'shape']  # fallback
MAX_RUL_MINUTES = 24 * 60 * 7  # Cap health score at 1 week (arbitrary baseline)

# Physics confirmation: reduce false alarms by requiring degradation/rms_ratio evidence
IMMINENT_THRESHOLD_MIN = 24 * 60  # RUL < 24h = imminent
DEGRADATION_THRESHOLD = 0.03   # deg > this indicates real degradation
RMS_RATIO_THRESHOLD = 1.0      # rms_ratio_baseline > this indicates rise from healthy

def _physics_confirmed_imminent(features_dict: dict, rul_pred_min: float) -> bool:
    """
    Only flag imminent failure when RUL < 24h AND physics evidence supports it.
    Reduces false alarms (pred < 24h on healthy machines) while keeping ~99% recall.
    """
    if rul_pred_min >= IMMINENT_THRESHOLD_MIN:
        return False
    # Get or compute degradation_score (0-1, higher = more degraded)
    deg = features_dict.get("degradation_score")
    if deg is None:
        rms = features_dict.get("rms", 0.0)
        deg = max(0.0, min(1.0, (float(rms) - 0.07) / 0.7))
    # rms_ratio_baseline: current RMS / healthy baseline (>1 = rising from healthy)
    rms_ratio = features_dict.get("rms_ratio_baseline")
    if rms_ratio is None:
        rms_ratio = 1.0  # no run context = assume no rise
    physics_confirm = (float(deg) > DEGRADATION_THRESHOLD) or (float(rms_ratio) > RMS_RATIO_THRESHOLD)
    return physics_confirm


# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("inference")

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

class InferenceEngine:
    def __init__(self):
        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.scaler = None
        self.models = {}  # Cache for loaded models
        self.station_map = {}
        
        self.load_artifacts()
        self.setup_redis()

    def load_artifacts(self):
        """Loads Scaler, Feature List, and Station Config."""
        # 1. Load Scaler
        if not SCALER_PATH.exists():
            logger.error("Scaler not found at %s", SCALER_PATH)
            sys.exit(1)
            
        with open(SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info("Loaded Scaler.")

        # 2. Load feature list (physics + refinery model)
        self.features = list(FEATURES)
        if RUL_FEATURES_PATH.exists():
            with open(RUL_FEATURES_PATH, 'r') as f:
                self.features = [line.strip() for line in f if line.strip()]
            logger.info("Loaded %d RUL features from rul_features.txt", len(self.features))

        # 2. Load Station Config
        if not CONFIG_FILE.exists():
            logger.error("Config not found at %s", CONFIG_FILE)
            sys.exit(1)
            
        with open(CONFIG_FILE, 'r') as f:
            self.station_map = json.load(f)
        logger.info("Loaded Station Map: %s", list(self.station_map.keys()))

    def get_model(self, station_id):
        """Lazy-loads XGBoost model for a station."""
        model_file = self.station_map.get(station_id)
        if not model_file:
            logger.warning("No model mapped for station %s", station_id)
            return None
            
        if station_id not in self.models:
            model_path = MODELS_DIR / model_file
            if not model_path.exists():
                logger.error("Model file %s not found", model_path)
                return None
                
            model = xgb.Booster()
            model.load_model(model_path)
            self.models[station_id] = model
            logger.info("Loaded model %s for %s", model_file, station_id)
            
        return self.models[station_id]

    def setup_redis(self):
        """Creates consumer group if not exists."""
        try:
            self.redis.xgroup_create(INPUT_STREAM, CONSUMER_GROUP, id="0", mkstream=True)
            logger.info("Created Consumer Group: %s", CONSUMER_GROUP)
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info("Consumer Group %s already exists.", CONSUMER_GROUP)
            else:
                logger.error("Redis Error: %s", e)

    def process_message(self, message_id, data):
        """Core Inference Logic."""
        try:
            station_id = data.get('station_id')
            payload_str = data.get('features')
            
            if not station_id or not payload_str:
                logger.warning("Invalid message format: %s", data)
                return

            features_dict = json.loads(payload_str)
            
            # 1. Extract Feature Vector (Ensure Order, use defaults for missing)
            DEFAULTS = {"time_pct": 0.5, "run_length_minutes": 10000}  # ~7 days default run length
            vector = []
            for f in self.features:
                val = features_dict.get(f)
                if val is None:
                    val = DEFAULTS.get(f, 0.0)
                vector.append(float(val))
            
            # 2. Normalize
            # Reshape to (1, -1) because scaler expects 2D array
            normalized_vector = self.scaler.transform([vector])
            
            # 3. Get Model & Predict
            model = self.get_model(station_id)
            if not model:
                return

            # XGBoost Booster expects DMatrix
            dmatrix = xgb.DMatrix(normalized_vector, feature_names=self.features)
            rul_log = model.predict(dmatrix)[0]
            # Model trained on log1p(RUL); convert back to minutes
            rul_pred = max(0.0, float(np.expm1(rul_log)))
            
            # 4. Calculate Health Score
            # 100% = Max RUL, 0% = 0 RUL
            health_score = max(0, min(100, (rul_pred / MAX_RUL_MINUTES) * 100))

            # 5. Imminent failure flag (physics-confirmed to reduce false alarms)
            imminent_failure = _physics_confirmed_imminent(features_dict, rul_pred)
            
            # 6. Publish Result
            result_payload = {
                "station_id": station_id,
                "timestamp": features_dict.get('timestamp', time.time()),
                "rul_minutes": round(float(rul_pred), 2),
                "health_score": round(float(health_score), 2),
                "imminent_failure": bool(imminent_failure),
                "model_used": self.station_map[station_id]
            }
            
            self.redis.xadd(OUTPUT_STREAM, result_payload)
            logger.info("Prediction: Station=%s | RUL=%.1f min | Health=%.1f%% | Imminent=%s",
                        station_id, rul_pred, health_score, imminent_failure)
            
            # Acknowledge message
            self.redis.xack(INPUT_STREAM, CONSUMER_GROUP, message_id)

        except Exception as e:
            logger.error("Processing failed for %s: %s", message_id, e, exc_info=True)

    def run(self):
        """Main Loop."""
        logger.info("Inference Service Started. Waiting for features...")
        while True:
            try:
                # Block for 2 seconds waiting for new messages
                messages = self.redis.xreadgroup(
                    CONSUMER_GROUP, CONSUMER_NAME, {INPUT_STREAM: ">"}, count=1, block=2000
                )
                
                if not messages:
                    continue
                    
                for stream, msg_list in messages:
                    for message_id, data in msg_list:
                        self.process_message(message_id, data)
                        
            except KeyboardInterrupt:
                logger.info("Stopping service...")
                break
            except Exception as e:
                logger.error("Loop Error: %s", e)
                time.sleep(1)

if __name__ == "__main__":
    engine = InferenceEngine()
    engine.run()
