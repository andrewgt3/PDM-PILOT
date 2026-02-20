#!/usr/bin/env python3
"""
Feature Extraction Service (Refinery)
=====================================
Processes raw vibration data files to extract health indicators.

Features:
- RMS, Kurtosis, Skewness, Peak-to-Peak, Crest Factor, Shape Factor.
- Validates input quality (rejects short files, flatlines, NaNs).
- Appends to `data/processed/features.csv`.

Usage:
    python refinery.py <filepath>
"""

import sys
import logging
import csv
import numpy as np
from scipy.stats import skew, kurtosis
from pathlib import Path
from datetime import datetime, timezone

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent
PROCESSED_DATA_FILE = BASE_DIR / "data" / "processed" / "features.csv"
MIN_SAMPLES = 20480

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("refinery_events.log")
    ]
)
logger = logging.getLogger("refinery")

# =============================================================================
# CORE LOGIC
# =============================================================================

def load_data(filepath: Path) -> np.ndarray:
    """Loads raw ASCII vibration data (one sample per line)."""
    try:
        data = np.loadtxt(filepath)
        return data
    except Exception as e:
        logger.error("Failed to read file %s: %s", filepath, e)
        return None

def validate_data(data: np.ndarray, filepath: Path) -> bool:
    """Checks data quality."""
    if data is None:
        return False

    # Check length
    if len(data) < MIN_SAMPLES:
        logger.warning("REJECTED: File %s has too few samples (%d < %d)", 
                       filepath.name, len(data), MIN_SAMPLES)
        return False

    # Check for NaNs or Infs
    if not np.isfinite(data).all():
        logger.warning("REJECTED: File %s contains NaN or Inf values", filepath.name)
        return False

    # Check for flatline (all zeros or constant value)
    if np.std(data) == 0:
        logger.warning("REJECTED: File %s is a flatline (zero variance)", filepath.name)
        return False

    return True

def extract_features(data: np.ndarray) -> dict:
    """Calculates time-domain health indicators."""
    # Ensure 1D array
    data = data.flatten()

    # 1. RMS (Root Mean Square) - Energy
    rms = np.sqrt(np.mean(data**2))

    # 2. Peak-to-Peak - Amplitude Range
    p2p = np.ptp(data)

    # 3. Kurtosis - Impulsiveness (Fisher's definition, normal=0)
    # Using Fisher=False gives pearson (normal=3), default moves it to 0
    kurt = kurtosis(data, fisher=False) 

    # 4. Skewness - Asymmetry
    skew_val = skew(data)

    # 5. Crest Factor - Peak / RMS 
    # (Impact detection)
    peak_abs = np.max(np.abs(data))
    crest = peak_abs / rms if rms > 0 else 0

    # 6. Shape Factor - RMS / Mean Abs
    mean_abs = np.mean(np.abs(data))
    shape = rms / mean_abs if mean_abs > 0 else 0

    return {
        "rms": round(rms, 4),
        "p2p": round(p2p, 4),
        "kurtosis": round(kurt, 4),
        "skewness": round(skew_val, 4),
        "crest": round(crest, 4),
        "shape": round(shape, 4)
    }

def save_features(features: dict, filename: str):
    """Appends feature row to CSV."""
    file_exists = PROCESSED_DATA_FILE.exists()
    
    # Prepare row
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        **features
    }
    
    # Define column order
    fieldnames = ["timestamp", "filename", "rms", "p2p", "kurtosis", "skewness", "crest", "shape"]

    try:
        with open(PROCESSED_DATA_FILE, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row)
            logger.info("Features saved to %s", PROCESSED_DATA_FILE)

    except Exception as e:
        logger.error("Failed to write to CSV: %s", e)

# =============================================================================
# REDIS PUBLISHING
# =============================================================================

import redis
import json
import time

from config import get_settings

STREAM_KEY = "clean_features"

def publish_to_redis(features: dict, station_id="Bearing_1"):
    """Publishes features to Redis Stream for Inference Service (Redis from config)."""
    try:
        s = get_settings()
        pw = s.redis.password.get_secret_value() if s.redis.password else None
        r = redis.Redis(host=s.redis.host, port=s.redis.port, password=pw, decode_responses=True)
        
        # Add timestamp if not present
        if "timestamp" not in features:
            features["timestamp"] = time.time()
            
        payload = {
            "station_id": station_id,
            "features": json.dumps(features)
        }
        
        r.xadd(STREAM_KEY, payload)
        logger.info("Published to Redis stream: %s", STREAM_KEY)
        
    except Exception as e:
        logger.error("Failed to publish to Redis: %s", e)

# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python refinery.py <filepath>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    
    logger.info("Starting refinement for: %s", input_path.name)
    
    # 1. Load
    raw_data = load_data(input_path)
    
    # 2. Validate
    if validate_data(raw_data, input_path):
        # 3. Extract
        feats = extract_features(raw_data)
        logger.info("Extracted features: %s", feats)
        
        # 4. Save to CSV
        save_features(feats, input_path.name)
        
        # 5. Publish to Redis (Bridge to Inference)
        # Assuming Bearing_1 for now, or derive from filename
        station_id = "Bearing_1" 
        publish_to_redis(feats, station_id)
        
    else:
        logger.warning("Processing aborted for %s due to validation failure.", input_path.name)
