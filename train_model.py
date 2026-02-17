#!/usr/bin/env python3
"""
XGBoost Trainer (The Brain)
===========================
Trains the RUL prediction model using labeled feature data.

Steps:
1. Load `labeled_features.csv`.
2. Split into Train/Test sets.
3. Normalize features (StandardScaler) - SAVES SCALER.
4. Train XGBoost Regressor - SAVES MODEL.
5. Evaluate performance (RMSE).

Usage:
    python train_model.py
"""

import sys
import pandas as pd
import numpy as np
import pickle
import logging
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "labeled_features.csv"
MODELS_DIR = BASE_DIR / "data" / "models"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
MODEL_PATH = MODELS_DIR / "xgb_rul_v1.json"

FEATURES = ['rms', 'kurtosis', 'p2p', 'skewness', 'crest', 'shape']
TARGET = 'rul_minutes'

# Ensure models dir exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("trainer")

# =============================================================================
# CORE LOGIC
# =============================================================================

def load_and_prep_data():
    """Loads data, handles NaNs, and splits into X/y."""
    if not INPUT_FILE.exists():
        logger.error("Data file not found: %s", INPUT_FILE)
        sys.exit(1)
        
    df = pd.read_csv(INPUT_FILE)
    logger.info("Loaded %d rows from %s", len(df), INPUT_FILE)
    
    # Drop rows with NaNs in features (refinery might produce them on flatlines)
    df = df.dropna(subset=FEATURES)
    
    X = df[FEATURES]
    y = df[TARGET]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_pipeline(X_train, X_test, y_train, y_test):
    """Trains Scaler and XGBoost Model."""
    
    # 1. Fit Scaler
    logger.info("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info("Saved scaler to %s", SCALER_PATH)
    
    # 2. Train Model
    logger.info("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Save Model (Use underlying booster to avoid sklearn mixin issues)
    model.get_booster().save_model(MODEL_PATH)
    logger.info("Saved model to %s", MODEL_PATH)
    
    # 3. Evaluate
    predictions = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    logger.info("-" * 30)
    logger.info(f"Model Performance (RMSE): {rmse:.4f} minutes")
    logger.info("-" * 30)
    
    # Feature Importance
    importances = model.feature_importances_
    logger.info("Feature Importance:")
    for name, imp in zip(FEATURES, importances):
        logger.info(f"{name.ljust(15)}: {imp:.4f}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prep_data()
    train_pipeline(X_train, X_test, y_train, y_test)
