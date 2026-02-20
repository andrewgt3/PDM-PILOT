#!/usr/bin/env python3
"""
XGBoost / FLAML Trainer (The Brain)
====================================
Trains the RUL prediction model (NASA data) or anomaly model (custom robot/PLC data).
With --data-source custom and --automl (default), uses FLAML AutoML.

Usage:
    # NASA pipeline (existing): labeled_features.csv, RUL target, saves xgb_rul_v1.json
    python train_model.py --data-source nasa

    # Custom pipeline (FLAML AutoML): time-aware split, save to models/gaia_model_*.pkl
    python train_model.py --data-source custom --data-path /path/to/export.csv
    python train_model.py --data-source custom --data-path /path/to/export.csv --no-automl
"""

import argparse
import json
import sys
import pandas as pd
import numpy as np
import pickle
import logging
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "labeled_features.csv"
MODELS_DIR = BASE_DIR / "data" / "models"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
MODEL_PATH = MODELS_DIR / "xgb_rul_v1.json"

# Custom (robot/PLC) paths — do not overwrite NASA model
CUSTOM_SCALER_PATH = MODELS_DIR / "scaler_custom.pkl"
CUSTOM_MODEL_PATH = MODELS_DIR / "gaia_model_custom.pkl"

# FLAML / AutoML: save to project root models/ (gaia_model_*.pkl + metadata sidecar)
GAIA_MODELS_DIR = BASE_DIR / "models"
GAIA_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# NASA feature set (CWRU / turbofan)
FEATURES = ['rms', 'kurtosis', 'p2p', 'skewness', 'crest', 'shape']
TARGET = 'rul_minutes'

# Custom feature set (from RobotFeatureExtractor / config/feature_config.yaml)
CUSTOM_FEATURES = [
    "torque_mean", "torque_std", "torque_max", "torque_p95",
    "torque_deviation_zscore", "speed_normalized_torque",
    "cycle_time_drift_pct", "temp_rate_of_change", "thermal_torque_ratio",
]
CUSTOM_TARGET = "anomaly"

# Default threshold rules for weak-supervision when using custom data
DEFAULT_ANOMALY_RULES = {
    "torque_deviation_zscore": 2.5,
    "cycle_time_drift_pct": 0.05,
}

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
# NASA PIPELINE
# =============================================================================

def load_and_prep_nasa():
    """Load labeled_features.csv, split into X/y for RUL."""
    if not INPUT_FILE.exists():
        logger.error("Data file not found: %s", INPUT_FILE)
        sys.exit(1)
    df = pd.read_csv(INPUT_FILE)
    logger.info("Loaded %d rows from %s", len(df), INPUT_FILE)
    df = df.dropna(subset=FEATURES)
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_nasa_pipeline(X_train, X_test, y_train, y_test):
    """Scaler + XGB Regressor for RUL; save to MODEL_PATH / SCALER_PATH."""
    logger.info("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info("Saved scaler to %s", SCALER_PATH)

    logger.info("Training XGBoost Regressor (RUL)...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    model.get_booster().save_model(str(MODEL_PATH))
    logger.info("Saved model to %s", MODEL_PATH)

    predictions = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    logger.info("Model Performance (RMSE): %.4f minutes", rmse)
    for name, imp in zip(FEATURES, model.feature_importances_):
        logger.info("%s: %.4f", name.ljust(15), imp)

# =============================================================================
# CUSTOM (ROBOT/PLC) PIPELINE
# =============================================================================

def load_and_prep_custom(data_path: str):
    """Ingest CSV -> normalize -> RobotFeatureExtractor -> label_by_threshold -> X/y."""
    from pipeline.ingestion.custom_data_ingestor import ingest
    from pipeline.cleansing.robot_feature_extractor import RobotFeatureExtractor
    from labeling_engine import label_by_threshold

    path = Path(data_path)
    if not path.exists():
        logger.error("Data file not found: %s", path)
        sys.exit(1)

    raw = ingest(str(path))
    if raw.empty:
        logger.error("Ingestion produced no rows.")
        sys.exit(1)
    extractor = RobotFeatureExtractor()
    features_df = extractor.extract(raw)
    if features_df.empty:
        logger.error("Feature extraction produced no rows.")
        sys.exit(1)

    labeled = label_by_threshold(features_df, rules=DEFAULT_ANOMALY_RULES, anomaly_col=CUSTOM_TARGET)
    # Use only columns that exist and are in CUSTOM_FEATURES
    use_cols = [c for c in CUSTOM_FEATURES if c in labeled.columns]
    labeled = labeled.dropna(subset=use_cols)
    if labeled.empty:
        logger.error("No rows left after dropping NaNs in feature columns.")
        sys.exit(1)
    return labeled, use_cols


def load_and_prep_human_labels(data_path: str):
    """Load pre-labeled CSV from human-in-the-loop (columns + 'anomaly' or 'label')."""
    path = Path(data_path)
    if not path.exists():
        logger.error("Data file not found: %s", path)
        sys.exit(1)
    df = pd.read_csv(path)
    if "anomaly" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "anomaly"})
    if "anomaly" not in df.columns:
        logger.error("CSV must have 'anomaly' or 'label' column.")
        sys.exit(1)
    # Use numeric columns only (exclude anomaly)
    use_cols = [c for c in df.columns if c != "anomaly" and pd.api.types.is_numeric_dtype(df[c])]
    if not use_cols:
        logger.error("No numeric feature columns found.")
        sys.exit(1)
    df = df.dropna(subset=use_cols + ["anomaly"])
    if len(df) < 20:
        logger.error("Need at least 20 rows after dropna.")
        sys.exit(1)
    return df, use_cols


def _time_aware_split(labeled: pd.DataFrame, use_cols: list, test_frac: float = 0.2):
    """Time-aware split: sort by timestamp, last test_frac = test set (shuffle=False)."""
    if "timestamp" not in labeled.columns:
        # Fallback: random split
        return train_test_split(
            labeled[use_cols], labeled[CUSTOM_TARGET],
            test_size=test_frac, random_state=42,
        ) + (use_cols,)
    df = labeled.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    split_idx = int(n * (1 - test_frac))
    if split_idx == 0 or split_idx >= n:
        split_idx = max(1, n - 1)
    X_train = df.iloc[:split_idx][use_cols]
    X_test = df.iloc[split_idx:][use_cols]
    y_train = df.iloc[:split_idx][CUSTOM_TARGET]
    y_test = df.iloc[split_idx:][CUSTOM_TARGET]
    return X_train, X_test, y_train, y_test, use_cols

def train_custom_pipeline(X_train, X_test, y_train, y_test, feature_names: list):
    """Scaler + XGB Classifier for anomaly; save to CUSTOM_MODEL_PATH / CUSTOM_SCALER_PATH."""
    logger.info("Fitting StandardScaler (custom)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    with open(CUSTOM_SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info("Saved scaler to %s", CUSTOM_SCALER_PATH)

    logger.info("Training XGBoost Classifier (anomaly)...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train_scaled, y_train)
    with open(CUSTOM_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    logger.info("Saved model to %s", CUSTOM_MODEL_PATH)

    pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred)
    logger.info("Model Performance (accuracy): %.4f", acc)
    logger.info(classification_report(y_test, pred, target_names=["normal", "anomaly"]))
    for name, imp in zip(feature_names, model.feature_importances_):
        logger.info("%s: %.4f", name.ljust(25), imp)


# FLAML config for custom pipeline
FLAML_TIME_BUDGET = 300
FLAML_ESTIMATOR_LIST = ["xgboost", "lgbm", "rf", "lrl1"]


def train_custom_pipeline_automl(
    X_train, X_test, y_train, y_test, feature_names: list, dataset_source: str = "custom", staging_only: bool = False,
):
    """FLAML AutoML classification; save to models/gaia_model_{timestamp}.pkl + metadata JSON. If staging_only=True, save only to gaia_model_staging.pkl (do not overwrite production)."""
    from flaml import AutoML

    logger.info("Fitting StandardScaler (custom)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    automl = AutoML()
    flaml_config = {
        "task": "classification",
        "metric": "f1",
        "time_budget": FLAML_TIME_BUDGET,
        "estimator_list": FLAML_ESTIMATOR_LIST,
        "log_level": "info",
    }
    logger.info("Training FLAML AutoML (task=classification, metric=f1, time_budget=%ds)...", FLAML_TIME_BUDGET)
    automl.fit(
        X_train_scaled, y_train,
        **flaml_config,
    )

    best_estimator = automl.best_estimator
    best_config = automl.best_config
    model = automl.model

    # Predictions and metrics
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    if hasattr(model, "predict_proba"):
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_test_proba = y_test_pred.astype(float)

    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_auc = float(roc_auc_score(y_test, y_test_proba)) if roc_auc_score and len(np.unique(y_test)) > 1 else 0.0

    logger.info("--- FLAML results ---")
    logger.info("Best estimator: %s", type(best_estimator).__name__)
    logger.info("Best config: %s", best_config)
    logger.info("Train F1: %.4f", train_f1)
    logger.info("Test F1: %.4f", test_f1)
    logger.info("Test precision: %.4f", test_precision)
    logger.info("Test recall: %.4f", test_recall)
    logger.info("Test AUC: %.4f", test_auc)
    logger.info(classification_report(y_test, y_test_pred, target_names=["normal", "anomaly"]))

    # Probability calibration: use last 10% of training as calibration set
    n_train = len(X_train_scaled)
    cal_frac = 0.10
    split_cal = max(1, int(n_train * (1 - cal_frac)))
    X_cal = X_train_scaled[split_cal:]
    y_cal = y_train.iloc[split_cal:] if hasattr(y_train, "iloc") else y_train[split_cal:]
    from sklearn.calibration import CalibratedClassifierCV
    calibrated_model = CalibratedClassifierCV(best_estimator, method="isotonic", cv="prefit")
    calibrated_model.fit(X_cal, y_cal)
    # Calibration curve: expected vs actual failure rate by probability decile
    y_test_proba_cal = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    _report_calibration_curve(y_test.values, y_test_proba_cal, num_bins=10)
    # Use calibrated model for saving
    model = calibrated_model

    metrics = {
        "train_f1": float(train_f1),
        "test_f1": float(test_f1),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_auc": float(test_auc),
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    training_date = datetime.now(timezone.utc).isoformat()
    train_df = pd.DataFrame(X_train, columns=feature_names)
    metadata = {
        "model_type": type(best_estimator).__name__,
        "training_date": training_date,
        "feature_list": feature_names,
        "metrics": metrics,
        "dataset_source": dataset_source,
        "flaml_config": flaml_config,
        "feature_ranges": _feature_ranges_from_df(train_df),
        "feature_histograms": _feature_histograms_from_df(train_df, num_bins=10),
        "run_id": timestamp,
    }
    pipeline = {"model": model, "scaler": scaler, "feature_columns": feature_names}

    if staging_only:
        # Shadow deployment: save only to staging paths; do not overwrite production
        staging_pkl = GAIA_MODELS_DIR / "gaia_model_staging.pkl"
        staging_meta = GAIA_MODELS_DIR / "gaia_model_staging_metadata.json"
        GAIA_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(staging_pkl, "wb") as f:
            pickle.dump(pipeline, f)
        with open(staging_meta, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved staging model to %s (run_id=%s)", staging_pkl, timestamp)
        return staging_pkl, staging_meta

    # Save timestamped pkl and metadata
    pkl_path = GAIA_MODELS_DIR / f"gaia_model_{timestamp}.pkl"
    meta_path = GAIA_MODELS_DIR / f"gaia_model_{timestamp}_metadata.json"
    with open(pkl_path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info("Saved model to %s", pkl_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", meta_path)
    # Save calibrated classifier as gaia_model_calibrated.pkl
    calibrated_pkl = GAIA_MODELS_DIR / "gaia_model_calibrated.pkl"
    with open(calibrated_pkl, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info("Saved calibrated classifier to %s", calibrated_pkl)

    # Optional: export training sample for KS test in drift monitor (up to 1000 rows)
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        sample_df = train_df.iloc[:1000]
        sample_path = MODELS_DIR / "training_sample.csv"
        sample_df.to_csv(sample_path, index=False)
        logger.info("Saved training sample for drift to %s (%d rows)", sample_path, len(sample_df))
    except Exception as e:
        logger.warning("Could not save training_sample.csv: %s", e)

    # Also save as latest
    latest_pkl = GAIA_MODELS_DIR / "gaia_model_latest.pkl"
    latest_meta = GAIA_MODELS_DIR / "gaia_model_latest_metadata.json"
    with open(latest_pkl, "wb") as f:
        pickle.dump(pipeline, f)
    with open(latest_meta, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved latest to %s and %s", latest_pkl, latest_meta)
    return pkl_path, meta_path


def _feature_ranges_from_df(df: pd.DataFrame) -> dict:
    """Compute min/max per column for validation (expected value ranges)."""
    return {
        col: {"min": float(df[col].min()), "max": float(df[col].max())}
        for col in df.columns
        if df[col].dtype.kind in "fc"
    }


def _feature_histograms_from_df(df: pd.DataFrame, num_bins: int = 10) -> dict:
    """Compute num_bins-bin histogram counts per numeric column for PSI drift."""
    out = {}
    for col in df.columns:
        if df[col].dtype.kind not in "fc":
            continue
        vals = df[col].dropna()
        if len(vals) < 2:
            out[col] = [0] * num_bins
            continue
        counts, _ = np.histogram(vals, bins=num_bins)
        out[col] = counts.tolist()
    return out


def _report_calibration_curve(y_true, y_proba, num_bins=10):
    """Report calibration curve: expected vs actual failure rate at each probability decile."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    bins = np.linspace(0, 1, num_bins + 1)
    logger.info("--- Calibration curve (expected vs actual failure rate by decile) ---")
    for i in range(num_bins):
        low, high = bins[i], bins[i + 1]
        mask = (y_proba >= low) & (y_proba < high)
        if i == num_bins - 1:
            mask = (y_proba >= low) & (y_proba <= high)
        if mask.sum() == 0:
            continue
        expected = np.mean(y_proba[mask])
        actual = np.mean(y_true[mask])
        logger.info("  Decile %d [%.2f, %.2f]: expected=%.3f actual=%.3f (n=%d)", i + 1, low, high, expected, actual, mask.sum())

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train RUL (NASA) or anomaly (custom) model.")
    parser.add_argument(
        "--data-source",
        choices=["nasa", "custom", "human_labels"],
        default="nasa",
        help="Use 'nasa' for RUL pipeline; 'custom' for robot/PLC CSV; 'human_labels' for pre-labeled CSV.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to CSV when --data-source custom or human_labels.",
    )
    parser.add_argument(
        "--automl",
        action="store_true",
        default=True,
        help="Use FLAML AutoML for custom pipeline (default: True).",
    )
    parser.add_argument(
        "--no-automl",
        action="store_false",
        dest="automl",
        help="Disable FLAML; use manual XGBoost for custom pipeline.",
    )
    parser.add_argument(
        "--staging",
        action="store_true",
        help="Save only to gaia_model_staging.pkl (do not overwrite production). Used by retraining service.",
    )
    args = parser.parse_args()

    if args.data_source == "nasa":
        X_train, X_test, y_train, y_test = load_and_prep_nasa()
        train_nasa_pipeline(X_train, X_test, y_train, y_test)
        return

    if args.data_source == "custom":
        if not args.data_path:
            logger.error("--data-source custom requires --data-path /path/to/export.csv")
            sys.exit(1)
        labeled, use_cols = load_and_prep_custom(args.data_path)
        X_train, X_test, y_train, y_test, feature_names = _time_aware_split(labeled, use_cols, test_frac=0.2)
        logger.info("Time-aware split: last 20%% of timesteps = test set (shuffle=False)")
        if args.automl:
            train_custom_pipeline_automl(
                X_train, X_test, y_train, y_test, feature_names,
                dataset_source=args.data_path,
                staging_only=args.staging,
            )
        else:
            train_custom_pipeline(X_train, X_test, y_train, y_test, feature_names)
        return

    if args.data_source == "human_labels":
        if not args.data_path:
            logger.error("--data-source human_labels requires --data-path /path/to/export.csv")
            sys.exit(1)
        labeled, use_cols = load_and_prep_human_labels(args.data_path)
        X_train, X_test, y_train, y_test, feature_names = _time_aware_split(labeled, use_cols, test_frac=0.2)
        logger.info("Human-labels split: last 20%% = test set")
        train_custom_pipeline_automl(
            X_train, X_test, y_train, y_test, feature_names,
            dataset_source=args.data_path,
            staging_only=args.staging,
        )


if __name__ == "__main__":
    main()
