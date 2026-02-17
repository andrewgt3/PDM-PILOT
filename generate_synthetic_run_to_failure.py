import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create synthetic "run-to-failure" data
# 10 steps, 1 minute apart
# RMS increases linearly from 1.0 to 10.0
# Kurtosis increases from 3.0 to 12.0

start_time = datetime.now()
data = []

for i in range(10):
    timestamp = start_time + timedelta(minutes=i)
    rms = 1.0 + i  # Increasing RMS
    kurtosis = 3.0 + i  # Increasing Kurtosis
    
    data.append({
        "timestamp": timestamp.isoformat(),
        "filename": f"file_{i:03d}.txt",
        "rms": rms,
        "p2p": rms * 2,
        "kurtosis": kurtosis,
        "skewness": 0.1,
        "crest": 2.0,
        "shape": 1.1
    })

df = pd.DataFrame(data)
df.to_csv("data/processed/features.csv", index=False)
print("Generated synthetic run-to-failure data in data/processed/features.csv")
