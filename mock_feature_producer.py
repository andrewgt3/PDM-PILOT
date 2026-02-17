#!/usr/bin/env python3
"""
Mock Feature Producer
=====================
Simulates the "Refinery" by pushing feature vectors to Redis.
Used to verify the Inference Service.

Usage:
    python mock_feature_producer.py
"""

import redis
import json
import time
import random

REDIS_HOST = "localhost"
REDIS_PORT = 6379
STREAM_KEY = "clean_features"

def produce_mock_features():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    
    stations = ["Bearing_1", "Bearing_2"]
    
    print(f"Producing mock features to stream: {STREAM_KEY}")
    
    try:
        while True:
            for station_id in stations:
                # Simulate feature vector (RMS, Kurtosis, P2P, Skew, Crest, Shape)
                # Randomize slightly to see varying predictions
                features = {
                    "rms": random.uniform(0.5, 5.0),
                    "kurtosis": random.uniform(2.5, 8.0),
                    "p2p": random.uniform(1.0, 10.0),
                    "skewness": random.uniform(-0.5, 0.5),
                    "crest": random.uniform(1.5, 4.0),
                    "shape": random.uniform(1.0, 1.5),
                    "timestamp": time.time()
                }
                
                payload = {
                    "station_id": station_id,
                    "features": json.dumps(features)
                }
                
                msg_id = r.xadd(STREAM_KEY, payload)
                print(f"Sent features for {station_id}: RMS={features['rms']:.2f}")
                
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nStopping mock producer.")

if __name__ == "__main__":
    produce_mock_features()
