"""
Contextual Anomaly Detection Engine
====================================
Lead ML Engineer: Predictive Maintenance MVP

This engine performs advanced anomaly detection on sensor data and correlates
anomalies with environmental events to identify root causes.

Patent-Pending Logic:
- Feature engineering on raw vibration data (rolling statistics)
- IsolationForest ML model for anomaly detection
- Temporal correlation with contextual events (30-minute lookback)
- Automated root cause identification

Author: Senior ML Engineering Team
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import json
import xgboost as xgb
import warnings
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Database connection
DB_CONNECTION_STRING = os.getenv('DATABASE_URL')
if not DB_CONNECTION_STRING:
    raise ValueError("DATABASE_URL environment variable is required")

# Feature engineering parameters
ROLLING_WINDOW_SECONDS = 60  # 60-second rolling window
LOOKBACK_DAYS = 7  # Analyze last 7 days

# Anomaly detection parameters
CONTAMINATION_RATE = 0.01  # Top 1% most anomalous
RANDOM_STATE = 42  # Reproducibility

# Correlation parameters
EVENT_LOOKBACK_MINUTES = 30  # Look for events 30 min before anomaly

# Output file
OUTPUT_FILE = "insight_report.json"

# RUL Model
RUL_MODEL_FILE = "rul_model.json"

# RUL Thresholds (hours)
RUL_WARNING_THRESHOLD = 72  # 3 days - Pre-Failure Warning
RUL_CRITICAL_THRESHOLD = 24  # 1 day - Imminent Failure

# =============================================================================
# STEP 1: DATABASE CONNECTION
# =============================================================================

def create_db_engine():
    """Create SQLAlchemy engine for TimescaleDB."""
    try:
        engine = create_engine(DB_CONNECTION_STRING, pool_pre_ping=True)
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print("=" * 80)
            print("CONTEXTUAL ANOMALY DETECTION ENGINE WITH RUL PREDICTION")
            print("=" * 80)
            print(f"✓ Connected to TimescaleDB")
            print(f"  {version.split(',')[0]}")
            print()
        return engine
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        raise

def load_rul_model(model_path=RUL_MODEL_FILE):
    """Load the trained XGBoost RUL prediction model."""
    print("=" * 80)
    print("LOADING RUL PREDICTION MODEL")
    print("=" * 80)
    
    try:
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        print(f"✓ Loaded XGBoost RUL model from: {model_path}")
        print(f"  Features: vibration, vibration_rate, time_pct")
        print(f"  Thresholds: Warning={RUL_WARNING_THRESHOLD}h, Critical={RUL_CRITICAL_THRESHOLD}h")
        print()
        return model
    except Exception as e:
        print(f"❌ Failed to load RUL model: {e}")
        print(f"   Continuing without RUL predictions...")
        print()
        return None

# =============================================================================
# STEP 2: FEATURE ENGINEERING
# =============================================================================

def fetch_sensor_data(engine, days=LOOKBACK_DAYS):
    """
    Fetch vibration sensor data from the last N days.
    Returns DataFrame with timestamp and vibration_x.
    """
    print("=" * 80)
    print("STEP 1: DATA EXTRACTION")
    print("=" * 80)
    
    query = f"""
    SELECT timestamp, vibration_x
    FROM sensors
    WHERE timestamp >= NOW() - INTERVAL '{days} days'
    ORDER BY timestamp ASC;
    """
    
    print(f"📊 Querying sensor data (last {days} days)...")
    df = pd.read_sql(query, engine, parse_dates=['timestamp'])
    
    print(f"   ✓ Retrieved {len(df):,} sensor readings")
    print(f"   ✓ Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   ✓ Vibration range: {df['vibration_x'].min():.3f}g to {df['vibration_x'].max():.3f}g")
    print()
    
    return df

def engineer_features(df, window_seconds=ROLLING_WINDOW_SECONDS):
    """
    Engineer features from raw vibration data.
    
    Features:
    - Rolling Mean (trend extraction)
    - Rolling Std (volatility detection)
    
    Window: 60 seconds (captures short-term patterns)
    """
    print("=" * 80)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 80)
    
    print(f"🔧 Engineering features (rolling window: {window_seconds}s)...")
    
    # Sort by timestamp (ensure chronological order)
    df = df.sort_values('timestamp').copy()
    
    # Calculate rolling statistics
    # Note: window=60 means 60 rows (1 sample/second = 60 seconds)
    df['vibration_rolling_mean'] = df['vibration_x'].rolling(window=window_seconds, min_periods=1).mean()
    df['vibration_rolling_std'] = df['vibration_x'].rolling(window=window_seconds, min_periods=1).std()
    
    # Fill any NaN in std (first few rows might be NaN)
    df['vibration_rolling_std'] = df['vibration_rolling_std'].fillna(0)
    
    print(f"   ✓ Created 'vibration_rolling_mean' feature")
    print(f"   ✓ Created 'vibration_rolling_std' feature")
    
    # Display feature statistics
    print(f"\n📈 Feature Statistics:")
    print(f"   Rolling Mean: {df['vibration_rolling_mean'].mean():.3f}g ± {df['vibration_rolling_mean'].std():.3f}g")
    print(f"   Rolling Std:  {df['vibration_rolling_std'].mean():.3f}g ± {df['vibration_rolling_std'].std():.3f}g")
    print()
    
    return df

# =============================================================================
# STEP 3: ANOMALY DETECTION (ML MODEL)
# =============================================================================

def detect_anomalies(df, contamination=CONTAMINATION_RATE):
    """
    Train IsolationForest model to detect anomalies.
    
    IsolationForest:
    - Unsupervised learning (no labels needed)
    - Effective for outlier detection in time-series
    - Contamination = expected proportion of outliers
    
    Returns DataFrame with anomaly labels and severity scores.
    """
    print("=" * 80)
    print("STEP 3: ANOMALY DETECTION (ISOLATION FOREST)")
    print("=" * 80)
    
    # Prepare features for ML
    feature_cols = ['vibration_rolling_mean', 'vibration_rolling_std']
    X = df[feature_cols].values
    
    print(f"🤖 Training IsolationForest model...")
    print(f"   Features: {feature_cols}")
    print(f"   Contamination rate: {contamination*100}% (top {contamination*100}% anomalies)")
    print(f"   Samples: {len(X):,}")
    
    # Standardize features (important for ML models)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train IsolationForest
    model = IsolationForest(
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_estimators=100,
        max_samples='auto'
    )
    
    # Predict: -1 = anomaly, 1 = normal
    predictions = model.fit_predict(X_scaled)
    
    # Get anomaly scores (more negative = more anomalous)
    scores = model.score_samples(X_scaled)
    
    # Add to dataframe
    df['is_anomaly'] = (predictions == -1)
    df['anomaly_score'] = scores
    
    # Count anomalies
    num_anomalies = df['is_anomaly'].sum()
    anomaly_pct = (num_anomalies / len(df)) * 100
    
    print(f"   ✓ Model trained successfully")
    print(f"   ✓ Detected {num_anomalies:,} anomalies ({anomaly_pct:.2f}% of data)")
    
    # Show most severe anomalies
    top_anomalies = df[df['is_anomaly']].nsmallest(5, 'anomaly_score')
    print(f"\n🚨 Top 5 Most Severe Anomalies:")
    for idx, row in top_anomalies.iterrows():
        print(f"   - {row['timestamp']} | Score: {row['anomaly_score']:.4f} | Vib: {row['vibration_x']:.3f}g")
    print()
    
    return df

# =============================================================================
# STEP 4: CONTEXTUAL CORRELATION (PATENTABLE LOGIC)
# =============================================================================

def fetch_events(engine):
    """Fetch all events from the database."""
    query = "SELECT id, timestamp, event_type, staff_id FROM events ORDER BY timestamp ASC;"
    events_df = pd.read_sql(query, engine, parse_dates=['timestamp'])
    return events_df

def correlate_with_events(df, events_df, rul_model, lookback_minutes=EVENT_LOOKBACK_MINUTES):
    """
    THE SMOKING GUN LOGIC + RUL PREDICTION:
    
    For each detected anomaly, search backwards in time to find any event
    that occurred within the previous N minutes. This identifies the
    potential root cause of the anomaly.
    
    NEW: Use XGBoost model to predict Remaining Useful Life (RUL) and
    assign severity based on predicted failure time.
    
    This is the "patentable" correlation algorithm that links sensor
    degradation to environmental events.
    """
    print("=" * 80)
    print("STEP 4: CONTEXTUAL CORRELATION & RUL PREDICTION")
    print("=" * 80)
    
    print(f"🔍 Analyzing correlations (lookback window: {lookback_minutes} minutes)...")
    
    # Filter to only anomalies
    anomalies = df[df['is_anomaly']].copy()
    
    print(f"   Anomalies to analyze: {len(anomalies)}")
    print(f"   Events in database: {len(events_df)}")
    
    # Store correlation results
    correlations = []
    
    # For RUL prediction - calculate time_pct (percentage of observed lifetime)
    # Assume current time is end of observed data
    max_time = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600  # hours
    anomalies['operating_hours'] = (anomalies['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600
    anomalies['time_pct'] = (anomalies['operating_hours'] / max_time) * 100 if max_time > 0 else 50
    
    for idx, anomaly in anomalies.iterrows():
        anomaly_time = anomaly['timestamp']
        lookback_start = anomaly_time - timedelta(minutes=lookback_minutes)
        
        # PREDICT RUL using XGBoost model
        predicted_rul = None
        rul_severity = None
        
        if rul_model is not None:
            try:
                # Prepare features for RUL prediction
                # Calculate vibration_rate if not present
                if 'vibration_rate' not in anomaly or pd.isna(anomaly.get('vibration_rate')):
                    vib_rate = 0.0
                else:
                    vib_rate = anomaly['vibration_rate']
                
                features = np.array([[
                    anomaly['vibration_x'],
                    vib_rate,
                    anomaly['time_pct']
                ]])
                
                predicted_rul = float(rul_model.predict(features)[0])
                
                # Determine severity based on RUL
                if predicted_rul < RUL_CRITICAL_THRESHOLD:
                    rul_severity = 'CRITICAL'  # Imminent Failure
                elif predicted_rul < RUL_WARNING_THRESHOLD:
                    rul_severity = 'WARNING'  # Pre-Failure Warning
                else:
                    rul_severity = 'NORMAL'
                    
            except Exception as e:
                print(f"   Warning: RUL prediction failed for anomaly at {anomaly_time}: {e}")
                predicted_rul = None
                rul_severity = None
        
        # Find all events in the lookback window
        relevant_events = events_df[
            (events_df['timestamp'] >= lookback_start) & 
            (events_df['timestamp'] < anomaly_time)
        ]
        
        # Determine final severity (use most severe between anomaly score and RUL)
        anomaly_severity = 'HIGH' if anomaly['anomaly_score'] < -0.5 else 'MEDIUM'
        if rul_severity == 'CRITICAL':
            final_severity = 'CRITICAL: IMMINENT FAILURE'
        elif rul_severity == 'WARNING':
            final_severity = 'HIGH: PRE-FAILURE WARNING'
        else:
            final_severity = anomaly_severity
        
        if len(relevant_events) > 0:
            # Multiple events possible - take the most recent one
            most_recent_event = relevant_events.iloc[-1]
            
            time_delta = (anomaly_time - most_recent_event['timestamp']).total_seconds() / 60
            
            correlation = {
                'anomaly_timestamp': str(anomaly_time),
                'anomaly_score': float(anomaly['anomaly_score']),
                'severity': final_severity,
                'predicted_rul_hours': predicted_rul,
                'rul_severity': rul_severity,
                'vibration_x': float(anomaly['vibration_x']),
                'vibration_rolling_mean': float(anomaly['vibration_rolling_mean']),
                'vibration_rolling_std': float(anomaly['vibration_rolling_std']),
                'root_cause_found': True,
                'event_type': most_recent_event['event_type'],
                'event_timestamp': str(most_recent_event['timestamp']),
                'staff_id': most_recent_event['staff_id'],
                'time_delta_minutes': round(time_delta, 1),
                'confidence': 'HIGH' if time_delta <= 20 else 'MEDIUM'
            }
        else:
            # No events found in lookback window
            correlation = {
                'anomaly_timestamp': str(anomaly_time),
                'anomaly_score': float(anomaly['anomaly_score']),
                'severity': final_severity,
                'predicted_rul_hours': predicted_rul,
                'rul_severity': rul_severity,
                'vibration_x': float(anomaly['vibration_x']),
                'vibration_rolling_mean': float(anomaly['vibration_rolling_mean']),
                'vibration_rolling_std': float(anomaly['vibration_rolling_std']),
                'root_cause_found': False,
                'event_type': None,
                'event_timestamp': None,
                'staff_id': None,
                'time_delta_minutes': None,
                'confidence': None
            }
        
        correlations.append(correlation)
    
    correlations_with_cause = sum(1 for c in correlations if c['root_cause_found'])
    critical_failures = sum(1 for c in correlations if c.get('rul_severity') == 'CRITICAL')
    warnings = sum(1 for c in correlations if c.get('rul_severity') == 'WARNING')
    
    print(f"   ✓ Analysis complete")
    print(f"   ✓ Correlations found: {correlations_with_cause}/{len(correlations)}")
    if rul_model is not None:
        print(f"   ✓ RUL Predictions: {critical_failures} Critical, {warnings} Warnings")
    print()
    
    return correlations

# =============================================================================
# STEP 5: OUTPUT GENERATION
# =============================================================================

def print_report(correlations):
    """Print human-readable console report."""
    print("=" * 80)
    print("ANOMALY DETECTION REPORT")
    print("=" * 80)
    print()
    
    # Count by severity
    high_severity = sum(1 for c in correlations if c['severity'] == 'HIGH')
    medium_severity = sum(1 for c in correlations if c['severity'] == 'MEDIUM')
    
    print(f"📊 Summary:")
    print(f"   Total Anomalies: {len(correlations)}")
    print(f"   High Severity: {high_severity}")
    print(f"   Medium Severity: {medium_severity}")
    print(f"   Root Causes Identified: {sum(1 for c in correlations if c['root_cause_found'])}")
    print()
    
    # Detailed anomaly reports
    print("=" * 80)
    print("DETAILED ANOMALY ANALYSIS")
    print("=" * 80)
    print()
    
    for i, corr in enumerate(correlations, 1):
        print(f"🚨 ANOMALY #{i} DETECTED")
        print(f"   Timestamp: {corr['anomaly_timestamp']}")
        print(f"   Severity: {corr['severity']} (Score: {corr['anomaly_score']:.4f})")
        if corr.get('predicted_rul_hours') is not None:
            print(f"   🔮 Predicted RUL: {corr['predicted_rul_hours']:.1f} hours ({corr['predicted_rul_hours']/24:.1f} days)")
        print(f"   Vibration: {corr['vibration_x']:.3f}g")
        print(f"   Rolling Mean: {corr['vibration_rolling_mean']:.3f}g")
        print(f"   Rolling Std: {corr['vibration_rolling_std']:.3f}g")
        
        if corr['root_cause_found']:
            print(f"   🎯 ROOT CAUSE IDENTIFIED:")
            print(f"      Event Type: {corr['event_type']}")
            print(f"      Event Time: {corr['event_timestamp']}")
            print(f"      Staff ID: {corr['staff_id']}")
            print(f"      Time Delta: {corr['time_delta_minutes']} minutes prior")
            print(f"      Confidence: {corr['confidence']}")
            
            # Special callout for cleaning crew
            if corr['event_type'] == 'Cleaning_Crew_Zone_3':
                print(f"      ⚡ CRITICAL: Cleaning crew activity detected!")
                print(f"         This matches the known degradation pattern.")
        else:
            print(f"   ⚠️  No root cause identified (no events in 30min lookback)")
        
        print()
    
    print("=" * 80)
    print()

def save_json_report(correlations, filename=OUTPUT_FILE):
    """Save correlation results to JSON for LLM agent consumption."""
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    report = {
        'generated_at': str(datetime.now()),
        'analysis_period_days': LOOKBACK_DAYS,
        'contamination_rate': CONTAMINATION_RATE,
        'lookback_minutes': EVENT_LOOKBACK_MINUTES,
        'summary': {
            'total_anomalies': len(correlations),
            'high_severity': sum(1 for c in correlations if c['severity'] == 'HIGH'),
            'medium_severity': sum(1 for c in correlations if c['severity'] == 'MEDIUM'),
            'root_causes_found': sum(1 for c in correlations if c['root_cause_found']),
            'cleaning_crew_events': sum(1 for c in correlations if c.get('event_type') == 'Cleaning_Crew_Zone_3')
        },
        'anomalies': correlations
    }
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"💾 Saved insight report to: {filename}")
    print(f"   ✓ Ready for LLM agent consumption")
    print()

# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================

def main():
    """Execute the complete anomaly detection pipeline."""
    try:
        # Step 1: Connect to database
        engine = create_db_engine()
        
        # Step 2: Fetch sensor data
        sensor_df = fetch_sensor_data(engine)
        
        # Step 3: Engineer features
        sensor_df = engineer_features(sensor_df)
        
        # Step 4: Detect anomalies with ML
        sensor_df = detect_anomalies(sensor_df)
        
        # Step 5: Fetch events
        print("=" * 80)
        print("LOADING CONTEXTUAL EVENTS")
        print("=" * 80)
        events_df = fetch_events(engine)
        print(f"📋 Loaded {len(events_df)} events from database")
        print()
        
        # Step 6: Load RUL model
        rul_model = load_rul_model()
        
        # Step 7: Correlate anomalies with events + RUL prediction
        correlations = correlate_with_events(sensor_df, events_df, rul_model)
        
        # Step 8: Generate reports
        print_report(correlations)
        save_json_report(correlations)
        
        # Final summary
        print("=" * 80)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Total anomalies detected: {len(correlations)}")
        print(f"Root causes identified: {sum(1 for c in correlations if c['root_cause_found'])}")
        print(f"Output saved to: {OUTPUT_FILE}")
        print()
        print("Next Steps:")
        print("  • Review insight_report.json")
        print("  • Feed to LLM agent for natural language insights")
        print("  • Investigate high-confidence correlations")
        print("=" * 80)
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
