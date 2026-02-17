"""
XGBoost RUL Model Training Script
==================================
Trains an XGBoost regression model to predict Remaining Useful Life (RUL)
from sensor readings.

Features: vibration, vibration_rate, time_pct
Target: RUL (Remaining Useful Life in hours)

Author: ML Engineering Team
"""

import pandas as pd
import numpy as np
import xgboost as xgb
# train_test_split removed to prevent data leakage in time-series
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import os
import hashlib
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_FILE = "rul_training_data.csv"
MODEL_OUTPUT_DIR = "models"
MODEL_VERSION = 1
RANDOM_STATE = 42
TEST_SIZE = 0.2

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(filepath):
    """Load training data from CSV."""
    print("=" * 80)
    print("LOADING TRAINING DATA")
    print("=" * 80)
    print()
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df):,} samples from {filepath}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Robots: {df['robot_id'].nunique()}")
    print()
    
    return df

# =============================================================================
# PREPROCESSING
# =============================================================================

def prepare_features(df):
    """
    Split data into features (X) and target (y).
    Validates features against shared ModelFeatures schema to prevent training-serving skew.
    """
    from schemas.ml import ModelFeatures, validate_training_features
    
    print("=" * 80)
    print("PREPROCESSING")
    print("=" * 80)
    print()
    
    # Target variable
    target_col = 'RUL'
    
    # Get expected feature columns from shared schema
    expected_features = ModelFeatures.feature_names()
    
    # Check which expected features are present
    available_features = [col for col in expected_features if col in df.columns]
    missing_features = [col for col in expected_features if col not in df.columns]
    
    if missing_features:
        print(f"⚠ Warning: {len(missing_features)} expected features missing from training data:")
        print(f"   {missing_features}")
        print()
    
    # Use only the features from the shared schema (in order)
    feature_cols = available_features
    
    if len(feature_cols) != ModelFeatures.expected_count():
        print(f"⚠ Warning: Training with {len(feature_cols)}/{ModelFeatures.expected_count()} expected features")
    else:
        print(f"✓ All {ModelFeatures.expected_count()} expected features present")
    
    print(f"Features: {feature_cols}")
    print(f"Target: {target_col}")
    print()
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Validate feature schema
    try:
        validate_training_features(feature_cols)
        print("✓ Feature schema validation passed")
    except ValueError as e:
        print(f"⚠ Feature schema validation: {e}")
    print()
    
    # Statistics
    print("Feature Statistics:")
    print(X.describe())
    print()
    
    print("Target Statistics:")
    print(f"  Min RUL: {y.min():.1f} hours")
    print(f"  Max RUL: {y.max():.1f} hours")
    print(f"  Mean RUL: {y.mean():.1f} hours")
    print(f"  Std RUL: {y.std():.1f} hours")
    print()
    
    return X, y, feature_cols

# =============================================================================
# TRAIN/TEST SPLIT (Chronological + Group Separation)
# =============================================================================

def split_data(df, X, y, test_size=TEST_SIZE):
    """
    Split data chronologically with robot_id group separation.
    
    This prevents data leakage by:
    1. Sorting by timestamp (no future data predicting past).
    2. Ensuring no robot_id in test set appears in training set (GroupKFold logic).
    
    Args:
        df: Original dataframe with 'time' and 'robot_id' columns
        X: Feature dataframe
        y: Target series
        test_size: Fraction of data for testing (default 0.2)
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("=" * 80)
    print("TRAIN/TEST SPLIT (Chronological + Group Separation)")
    print("=" * 80)
    print()
    
    # 1. Sort data by timestamp
    df_sorted = df.sort_values('time').reset_index(drop=True)
    sorted_indices = df_sorted.index
    
    # 2. Calculate split point
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    # 3. Get robot_ids in each split
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    train_robots = set(train_df['robot_id'].unique())
    test_robots = set(test_df['robot_id'].unique())
    
    # 4. Check for overlap (GroupKFold logic)
    overlap = train_robots.intersection(test_robots)
    if overlap:
        print(f"⚠ Warning: {len(overlap)} robot_ids appear in both sets")
        print(f"  Removing overlapping robots from test set to ensure generalization...")
        
        # Remove overlapping robots from test set
        test_df = test_df[~test_df['robot_id'].isin(overlap)]
        test_robots = set(test_df['robot_id'].unique())
        
        if len(test_df) == 0:
            raise ValueError("No samples remain after removing overlapping robots. Not enough unique robots.")
    
    # 5. Reconstruct X and y from sorted indices
    # We need to use original dataframe indices to slice X and y correctly
    train_original_indices = df_sorted.iloc[:split_idx].index
    test_original_indices = test_df.index
    
    # Reindex X and y to match sorted order, then slice
    X_sorted = X.iloc[df.sort_values('time').index].reset_index(drop=True)
    y_sorted = y.iloc[df.sort_values('time').index].reset_index(drop=True)
    
    X_train = X_sorted.iloc[:split_idx]
    X_test = X_sorted.iloc[test_df.index]
    y_train = y_sorted.iloc[:split_idx]
    y_test = y_sorted.iloc[test_df.index]
    
    # Statistics
    print(f"✓ Chronological split: First {(1-test_size)*100:.0f}% for training")
    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    print()
    print(f"✓ Robot separation:")
    print(f"  Training robots: {len(train_robots)}")
    print(f"  Test robots: {len(test_robots)}")
    print(f"  Overlap: {len(overlap) if overlap else 0} (removed from test)")
    print()
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_model(X_train, y_train):
    """
    Train XGBoost regression model for RUL prediction.
    """
    print("=" * 80)
    print("TRAINING XGBOOST MODEL")
    print("=" * 80)
    print()
    
    # XGBoost Regressor with optimized hyperparameters
    model = xgb.XGBRegressor(
        objective='reg:squarederror',  # Regression task
        n_estimators=200,               # Number of trees
        max_depth=6,                    # Tree depth
        learning_rate=0.05,             # Learning rate
        subsample=0.8,                  # Row sampling
        colsample_bytree=0.8,           # Column sampling
        min_child_weight=3,             # Minimum samples in leaf
        gamma=0,                        # Regularization
        random_state=RANDOM_STATE,
        n_jobs=-1                       # Use all CPU cores
    )
    
    print("Hyperparameters:")
    print(f"  n_estimators: {model.n_estimators}")
    print(f"  max_depth: {model.max_depth}")
    print(f"  learning_rate: {model.learning_rate}")
    print()
    
    print("Training model...")
    model.fit(
        X_train, y_train,
        verbose=False
    )
    
    print("✓ Training complete!")
    print()
    
    return model

# =============================================================================
# MODEL EVALUATION
# =============================================================================

# Business cost constants
COST_FALSE_POSITIVE = 100    # Early alarm (unnecessary maintenance)
COST_FALSE_NEGATIVE = 1000   # Missed failure (catastrophic breakdown)
RUL_FAILURE_THRESHOLD = 24   # Hours - below this is considered "imminent failure"

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model performance with standard metrics and custom business cost.
    
    Business Cost Metric:
    - False Positive (model predicts failure when there's time): Cost = 100 units
    - False Negative (model misses imminent failure): Cost = 1000 units
    """
    from logger import configure_logging, get_logger
    
    # Setup logging
    configure_logging(environment="development", log_level="INFO")
    logger = get_logger("train_rul_model")
    
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    print()
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Standard Metrics - Training
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Standard Metrics - Test
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    # ==========================================================================
    # BUSINESS COST METRIC
    # ==========================================================================
    # Convert to binary classification for business cost:
    # - "Failure imminent" = RUL < threshold
    # - "Safe" = RUL >= threshold
    
    y_test_binary = (y_test.values < RUL_FAILURE_THRESHOLD).astype(int)
    y_pred_binary = (y_test_pred < RUL_FAILURE_THRESHOLD).astype(int)
    
    # Count FP and FN
    false_positives = ((y_pred_binary == 1) & (y_test_binary == 0)).sum()
    false_negatives = ((y_pred_binary == 0) & (y_test_binary == 1)).sum()
    true_positives = ((y_pred_binary == 1) & (y_test_binary == 1)).sum()
    true_negatives = ((y_pred_binary == 0) & (y_test_binary == 0)).sum()
    
    # Calculate total business cost
    total_cost = (false_positives * COST_FALSE_POSITIVE) + (false_negatives * COST_FALSE_NEGATIVE)
    cost_per_sample = total_cost / len(y_test) if len(y_test) > 0 else 0
    
    # Print standard metrics
    print("Training Set Performance:")
    print(f"  MAE:  {train_mae:.2f} hours")
    print(f"  RMSE: {train_rmse:.2f} hours")
    print(f"  R²:   {train_r2:.4f}")
    print()
    
    print("Test Set Performance:")
    print(f"  MAE:  {test_mae:.2f} hours")
    print(f"  RMSE: {test_rmse:.2f} hours")
    print(f"  R²:   {test_r2:.4f}")
    print()
    
    # Print business cost analysis
    print("-" * 80)
    print("BUSINESS COST ANALYSIS (Failure Threshold: {} hours)".format(RUL_FAILURE_THRESHOLD))
    print("-" * 80)
    print(f"  True Positives:  {true_positives} (correctly predicted failures)")
    print(f"  True Negatives:  {true_negatives} (correctly predicted safe)")
    print(f"  False Positives: {false_positives} (early alarms @ ${COST_FALSE_POSITIVE}/each)")
    print(f"  False Negatives: {false_negatives} (missed failures @ ${COST_FALSE_NEGATIVE}/each)")
    print()
    print(f"  Total Business Cost: ${total_cost:,}")
    print(f"  Cost per Sample:     ${cost_per_sample:.2f}")
    print()
    
    # Log to structlog
    logger.info(
        "model_evaluation_complete",
        test_mae=round(test_mae, 2),
        test_rmse=round(test_rmse, 2),
        test_r2=round(test_r2, 4),
        false_positives=int(false_positives),
        false_negatives=int(false_negatives),
        total_business_cost=int(total_cost),
        cost_per_sample=round(cost_per_sample, 2),
    )
    
    # Accuracy statement
    print("=" * 80)
    print(f"✓ Model Accuracy: ±{test_mae:.1f} hours (Mean Absolute Error)")
    print(f"✓ Business Cost: ${total_cost:,} (lower is better)")
    print("=" * 80)
    print()
    
    return {
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives),
        'total_business_cost': int(total_cost),
        'cost_per_sample': float(cost_per_sample),
    }

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def print_feature_importance(model, feature_names):
    """Print feature importance scores."""
    print("=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)
    print()
    
    importance = model.feature_importances_
    
    # Sort by importance
    indices = np.argsort(importance)[::-1]
    
    print("Features ranked by importance:")
    for i, idx in enumerate(indices, 1):
        print(f"  {i}. {feature_names[idx]}: {importance[idx]:.4f}")
    print()

# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_model(model, metrics, feature_names, df):
    """
    Save trained model with versioned filename and metadata sidecar.
    
    Args:
        model: Trained XGBoost model
        metrics: Dictionary of evaluation metrics
        feature_names: List of feature column names
        df: Original training dataframe (for hash computation)
    """
    print("=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    print()
    
    # Ensure models directory exists
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    # Generate versioned filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_str = datetime.now().strftime("%Y%m%d")
    model_filename = f"rul_model_v{MODEL_VERSION}_{date_str}.json"
    model_filepath = os.path.join(MODEL_OUTPUT_DIR, model_filename)
    
    # Save XGBoost model
    model.save_model(model_filepath)
    print(f"✓ Model saved to: {model_filepath}")
    
    # Compute SHA256 hash of training data for reproducibility
    df_bytes = df.to_csv(index=False).encode('utf-8')
    data_hash = hashlib.sha256(df_bytes).hexdigest()
    
    # Build comprehensive metadata
    metadata = {
        'model_info': {
            'type': 'XGBoost Regressor',
            'version': MODEL_VERSION,
            'target': 'RUL',
            'features': feature_names,
        },
        'training_metrics': {
            'train_mae': float(metrics['train_mae']),
            'train_rmse': float(metrics['train_rmse']),
            'train_r2': float(metrics['train_r2']),
            'test_mae': float(metrics['test_mae']),
            'test_rmse': float(metrics['test_rmse']),
            'test_r2': float(metrics['test_r2']),
        },
        'hyperparameters': {
            'n_estimators': int(model.n_estimators),
            'max_depth': int(model.max_depth),
            'learning_rate': float(model.learning_rate),
            'subsample': float(model.subsample),
            'colsample_bytree': float(model.colsample_bytree),
            'min_child_weight': int(model.min_child_weight),
        },
        'reproducibility': {
            'training_data_sha256': data_hash,
            'training_data_file': DATA_FILE,
            'random_state': RANDOM_STATE,
        },
        'timestamp': timestamp,
        'created_at': datetime.now().isoformat(),
    }
    
    # Save sidecar metadata file
    meta_filename = f"rul_model_v{MODEL_VERSION}_{date_str}_meta.json"
    meta_filepath = os.path.join(MODEL_OUTPUT_DIR, meta_filename)
    
    with open(meta_filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to: {meta_filepath}")
    print(f"✓ Dataset hash: {data_hash[:16]}...")
    print()
    
    return model_filepath, meta_filepath

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Execute the complete training pipeline."""
    
    # 1. Load data
    df = load_data(DATA_FILE)
    
    # 2. Prepare features
    X, y, feature_names = prepare_features(df)
    
    # 3. Split data (chronological with group separation)
    X_train, X_test, y_train, y_test = split_data(df, X, y)
    
    # 4. Train model
    model = train_model(X_train, y_train)
    
    # 5. Evaluate model
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # 6. Feature importance
    print_feature_importance(model, feature_names)
    
    # 7. Save model with versioning
    model_path, meta_path = save_model(model, metrics, feature_names, df)
    
    # Final summary
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print()
    print(f"✓ Model file: {model_path}")
    print(f"✓ Metadata file: {meta_path}")
    print(f"✓ Test MAE: ±{metrics['test_mae']:.1f} hours")
    print(f"✓ R² Score: {metrics['test_r2']:.4f}")
    print()
    print("Next Steps:")
    print(f"  1. Load model: model = xgb.XGBRegressor(); model.load_model('{model_path}')")
    print("  2. Predict RUL: rul = model.predict([[vibration, vib_rate, time_pct]])")
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
