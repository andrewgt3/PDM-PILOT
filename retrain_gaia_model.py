#!/usr/bin/env python3
"""
Retrain GAIA failure prediction model with correct feature scaling.

This script retrains the XGBClassifier on training_data.pkl to fix the
issue where all machines show 96.5% failure probability.
"""

import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from advanced_features import SignalProcessor, SAMPLE_RATE, N_SAMPLES

print("="*70)
print("GAIA MODEL RETRAINING SCRIPT")
print("=" *70)

# Load training data
print("\n[1/5] Loading training data...")
with open('training_data.pkl', 'rb') as f:
    df = pd.read_pickle(f)

print(f"  Loaded {len(df)} samples")
print(f"  Healthy: {len(df[df['machine_failure'] == 0])}")
print(f"  Faulty: {len(df[df['machine_failure'] == 1])}")

# Initialize feature extractor
print("\n[2/5] Extracting features from vibration signals...")
processor = SignalProcessor(sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES)

features_list = []
for idx, row in df.iterrows():
    vibration = row['raw_vibration']
    telemetry = {
        'rotational_speed': row.get('rotational_speed', 1800.0),
        'temperature': row.get('temperature', 70.0),
        'torque': row.get('torque', 40.0),
        'tool_wear': row.get('tool_wear', 0.1)
    }
    
    features = processor.process_signal(vibration, telemetry)
    features_list.append(features)
    
    if (idx + 1) % 200 == 0:
        print(f"  Processed {idx + 1}/{len(df)} samples...")

# Convert to DataFrame
features_df = pd.DataFrame(features_list)
feature_columns = list(features_df.columns)

# Note: The model expects 28 features but we have 26
# We need to add 2 missing features to match the expected schema
if len(feature_columns) == 26:
    print("\n  Adding missing features to match model schema...")
    # Add placeholder features (these might be missing from our extractor)
    features_df['peak_freq_6'] = 0.0  
    features_df['peak_amp_6'] = 0.0
    feature_columns = list(features_df.columns)
    print(f"  Total features: {len(feature_columns)}")

X = features_df[feature_columns].values
y = df['machine_failure'].values

print(f"\n[3/5] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)} samples")
print(f"  Test: {len(X_test)} samples")

# Scale features
print(f"\n[4/5] Training scaler and model...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost Classifier
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*70)
print("MODEL PERFORMANCE")
print("="*70)
print(f"\nAccuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Healthy', 'Faulty']))

# Show prediction distribution
print("\nPrediction Distribution on Test Set:")
print(f"  Predicted Healthy: {(y_pred == 0).sum()} ({(y_pred == 0).sum()/len(y_pred)*100:.1f}%)")
print(f"  Predicted Faulty: {(y_pred == 1).sum()} ({(y_pred == 1).sum()/len(y_pred)*100:.1f}%)")

# Save model
print(f"\n[5/5] Saving model to gaia_model.pkl...")
pipeline = {
    'model': model,
    'scaler': scaler,
    'feature_columns': feature_columns
}

joblib.dump(pipeline, 'gaia_model.pkl')
print("  ✓ Model saved successfully")

print("\n" + "="*70)
print("RETRAINING COMPLETE")
print("="*70)
print("\nNext steps:")
print("  1. Restart the API server")
print("  2. Restart the stream consumer")
print("  3. Refresh the dashboard")
