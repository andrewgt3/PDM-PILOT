import pickle
import pandas as pd
import numpy as np

# Load training data
with open('training_data.pkl', 'rb') as f:
    df = pickle.load(f)

print("DataFrame Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nFirst 5 rows (selected columns):")

# Inspect key columns
cols_to_show = ['machine_failure', 'rotational_speed', 'temperature']
if 'machine_failure' in df.columns:
    print(f"\nFailure Distribution:")
    print(df['machine_failure'].value_counts())
    print(f"\nPercentage: {df['machine_failure'].value_counts(normalize=True) * 100}")

# Check vibration data
if 'raw_vibration' in df.columns:
    print(f"\nVibration Data:")
    vib_sample = df['raw_vibration'].iloc[0]
    print(f"  Type: {type(vib_sample)}")
    print(f"  Length: {len(vib_sample) if hasattr(vib_sample, '__len__') else 'N/A'}")
    if isinstance(vib_sample, np.ndarray):
        print(f"  Max amplitude: {np.max(np.abs(vib_sample)):.4f}")
        print(f"  Mean amplitude: {np.mean(np.abs(vib_sample)):.4f}")

print("\nData types:")
print(df.dtypes)
