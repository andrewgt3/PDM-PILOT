
import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, classification_report
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# Configuration
MODEL_PATH = "models/failure_model.pkl"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = os.getenv("DT_POSTGRES_DB", "pdm_timeseries")
DB_USER = os.getenv("DT_POSTGRES_USER", "postgres")
DB_PASS = os.getenv("DT_POSTGRES_PASSWORD", "password")

def optimize_model():
    print("🚀 Starting Model Optimization (F1 Maximization)...")
    
    # 1. Load Data from DB
    try:
        db_url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(db_url)
        print("📥 Fetching features from TimescaleDB 'sensor_features'...")
        df = pd.read_sql("SELECT * FROM sensor_features", engine)
        print(f"📊 Loaded {len(df)} records.")
    except Exception as e:
        print(f"❌ Failed to load from DB: {e}")
        return

    # 2. Advanced Feature Selection
    feature_cols = [
        c for c in df.columns 
        if '_8h' in c or 'slope_24h' in c or 'CSLM' in c
    ]
    print(f"🎯 Optimization Focus: {len(feature_cols)} Features")
    
    X = df[feature_cols]
    y = df['machine_failure']
    
    # Clean NaNs
    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]
    
    # Check Balance
    num_pos = y.sum()
    num_neg = len(y) - num_pos
    balance_ratio = num_neg / num_pos if num_pos > 0 else 1.0
    print(f"⚖️ Raw Balance Ratio: {balance_ratio:.2f}")

    # 3. Grid Search Configuration
    # User requested: scale_pos_weight + hypertuning
    param_grid = {
        'scale_pos_weight': [10.0, 20.0, balance_ratio], # Tune weight
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [100, 200],
        'subsample': [0.8], # Prevent overfitting
        'colsample_bytree': [0.8]
    }
    
    print(f"🔍 Starting GridSearchCV (Target: F1-Score)... Params: {param_grid.keys()}")
    
    xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
    
    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='f1',
        cv=3, # 3-Fold Cross-Validation during tuning
        verbose=1,
        n_jobs=-1
    )
    
    grid.fit(X, y)
    
    print("\n✅ Optimization Complete!")
    print(f"🏆 Best F1-Score: {grid.best_score_:.4f}")
    print(f"⚙️ Best Parameters: {grid.best_params_}")
    
    # 4. Save Optimized Model
    best_model = grid.best_estimator_
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"💾 Optimized model saved to {MODEL_PATH}")
    
    # 5. Output for Audit Script Update
    # I will print the params so I can update the audit script manually or programmatically.
    return grid.best_params_

if __name__ == "__main__":
    optimize_model()
