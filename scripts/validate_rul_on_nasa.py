#!/usr/bin/env python3
"""
NASA IMS RUL Model Validation Script
=====================================
Validates the RUL model on NASA IMS bearing data WITHOUT using that data for training.
Uses run-based holdout to prevent data leakage.

Method:
1. Load features.csv (NASA IMS processed by refinery)
2. Group by bearing run (gap > 24h = new run)
3. Compute actual RUL from file order (10-min intervals per NASA IMS spec)
4. Hold out last run(s) for testing - model was NOT trained on this
5. Load deployed model (xgb_rul_v1.json) and scaler
6. Predict on holdout runs, compare to actual RUL
7. Report MAE, RMSE, R², and failure prediction accuracy

No overfitting: Test data is from different bearing runs than training.
No hallucination: Ground truth RUL comes from file chronology (NASA IMS spec).

R² targets for RUL prediction:
  R² > 0.85  Strong (production-ready)
  R² > 0.70  Good (acceptable for early warning)
  R² > 0.50  Acceptable (needs improvement)
  R² < 0     Model worse than predicting mean (critical - fix holdout/features)
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths - prefer combined (NASA+FEMTO) if exists
BASE = Path(__file__).resolve().parent.parent
COMBINED_CSV = BASE / "data" / "processed" / "combined_features_physics.csv"
NASA_CSV = BASE / "data" / "processed" / "nasa_features_physics.csv"
FEATURES_CSV = COMBINED_CSV if COMBINED_CSV.exists() else NASA_CSV
RUL_FEATURES_PATH = BASE / "data" / "models" / "rul_features.txt"
SCALER_PATH = BASE / "data" / "models" / "scaler.pkl"
MODEL_PATH = BASE / "data" / "models" / "xgb_rul_v1.json"
INTERVAL_MINUTES = 10  # NASA IMS: 10-min between files
RUN_GAP_HOURS = 24    # Gap > 24h = new bearing run


def parse_nasa_filename(fname: str) -> pd.Timestamp:
    """Parse NASA IMS filename YYYY.MM.DD.HH.MM.SS to datetime."""
    try:
        parts = str(fname).split('.')
        if len(parts) == 6:
            return pd.Timestamp(
                int(parts[0]), int(parts[1]), int(parts[2]),
                int(parts[3]), int(parts[4]), int(parts[5])
            )
    except Exception:
        pass
    return pd.NaT


def load_features_list() -> list:
    """Load feature list from rul_features.txt."""
    if RUL_FEATURES_PATH.exists():
        with open(RUL_FEATURES_PATH, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    return ['rms', 'kurtosis', 'p2p', 'skewness', 'crest', 'shape']


def load_and_prepare_nasa_features(features: list) -> pd.DataFrame:
    """Load features CSV (nasa or combined) with run_id and RUL."""
    df = pd.read_csv(FEATURES_CSV)
    df["actual_rul_minutes"] = df["rul_minutes"]
    # Use run_idx for combined (numeric split), else run_id
    if "run_idx" in df.columns:
        df["_run_key"] = df["run_idx"]
    else:
        df["_run_key"] = df["run_id"]
    df = df.dropna(subset=features + ["actual_rul_minutes"])
    return df


def run_validation():
    """Run full validation and print report."""
    print("=" * 70)
    print("NASA IMS RUL MODEL VALIDATION")
    print("=" * 70)
    print()

    features = load_features_list()
    print(f"[0] Using {len(features)} features from rul_features.txt")

    # 1. Load NASA features with ground truth RUL
    if not FEATURES_CSV.exists():
        print(f"ERROR: {FEATURES_CSV} not found. Run build_nasa_features_with_physics.py first.")
        sys.exit(1)
    df = load_and_prepare_nasa_features(features)
    runs = np.sort(df["_run_key"].unique())
    print(f"[1] Loaded {len(df)} samples across {len(runs)} bearing runs ({FEATURES_CSV.name})")
    print(f"    Runs: {runs.tolist()}")
    print()

    # 2. Hold out runs for testing (random, not chronological - last runs can differ)
    n_holdout = max(1, len(runs) // 3)
    rng = np.random.RandomState(42)
    test_idx = rng.choice(len(runs), size=min(n_holdout, len(runs)), replace=False)
    test_runs = np.sort(runs[test_idx])
    train_runs = np.sort(np.setdiff1d(runs, test_runs))
    df_test = df[df["_run_key"].isin(test_runs)]
    df_train = df[df["_run_key"].isin(train_runs)]
    print(f"[2] Holdout split (no data leakage):")
    print(f"    Train runs: {train_runs.tolist()} ({len(df_train)} samples)")
    print(f"    Test runs:  {test_runs.tolist()} ({len(df_test)} samples)")
    print(f"    NOTE: Model was trained on labeled_features.csv (synthetic), NOT on train runs.")
    print(f"    We are testing generalization to unseen NASA bearing runs.")
    print()

    # 3. Load deployed model and scaler
    if not SCALER_PATH.exists():
        print(f"ERROR: Scaler not found at {SCALER_PATH}")
        sys.exit(1)
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))
    print(f"[3] Loaded model: {MODEL_PATH.name}")
    print()

    # 4. Predict on test runs
    X_test = df_test[features].values
    y_actual = df_test['actual_rul_minutes'].values
    X_scaled = scaler.transform(X_test)
    dmat = xgb.DMatrix(X_scaled, feature_names=features)
    y_pred_log = model.predict(dmat)
    # Model trained on log1p(RUL); convert back to minutes
    y_pred = np.maximum(0, np.expm1(y_pred_log))

    # 4b. Apply physics confirmation to reduce false alarms (matches inference_service)
    IMMINENT_MIN = 24 * 60
    DEG_THRESH = 0.03
    RMS_RATIO_THRESH = 1.0
    pred_raw_imminent = y_pred < IMMINENT_MIN
    deg = df_test["degradation_score"].values if "degradation_score" in df_test.columns else np.zeros(len(df_test))
    rms_ratio = df_test["rms_ratio_baseline"].values if "rms_ratio_baseline" in df_test.columns else np.ones(len(df_test))
    physics_confirm = (deg > DEG_THRESH) | (rms_ratio > RMS_RATIO_THRESH)
    pred_imminent_confirmed = pred_raw_imminent & physics_confirm
    print(f"[4] Predictions on {len(df_test)} holdout samples")
    print()

    # 5. Metrics
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)
    mape = np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-6))) * 100

    print("=" * 70)
    print("ACCURACY METRICS (Holdout NASA IMS Runs)")
    print("=" * 70)
    print(f"  MAE:  {mae:.2f} minutes")
    print(f"  RMSE: {rmse:.2f} minutes")
    print(f"  R²:   {r2:.4f}  {'✓' if r2 >= 0.7 else '✓ acceptable' if r2 >= 0.5 else '⚠ CRITICAL' if r2 < 0 else '⚠ LOW'}")
    print(f"  Target: R² > 0.70 good, > 0.85 strong")
    print(f"  MAPE: {mape:.1f}%")
    print()

    # 6. Failure prediction (binary: RUL < 24h = imminent failure)
    # With physics confirmation: only flag when deg>0.03 OR rms_ratio>1.0
    threshold_min = 24 * 60  # 24 hours in minutes
    actual_imminent = (y_actual < threshold_min).astype(int)
    pred_imminent = pred_imminent_confirmed.astype(int)  # physics-confirmed
    tp = ((pred_imminent == 1) & (actual_imminent == 1)).sum()
    tn = ((pred_imminent == 0) & (actual_imminent == 0)).sum()
    fp = ((pred_imminent == 1) & (actual_imminent == 0)).sum()
    fn = ((pred_imminent == 0) & (actual_imminent == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("FAILURE PREDICTION (RUL < 24h + physics confirmation: deg>0.03 OR rms_ratio>1.0)")
    print("-" * 70)
    print(f"  True Positives:  {tp}")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print(f"  F1:        {f1:.2%}")
    print()

    # 7. Per-run summary
    print("PER-RUN SUMMARY (Test Runs)")
    print("-" * 70)
    for run in test_runs:
        rdf = df_test[df_test["_run_key"] == run]
        y_a = rdf['actual_rul_minutes'].values
        idx = rdf.index
        pos = np.isin(df_test.index, idx)
        y_p = y_pred[pos]
        run_mae = mean_absolute_error(y_a, y_p)
        run_r2 = r2_score(y_a, y_p)
        print(f"  Run {run}: {len(rdf)} samples | MAE={run_mae:.0f} min | R²={run_r2:.3f}")
    print()

    # 8. Sanity: monotonicity (RUL should decrease as we approach failure)
    print("SANITY: RUL Monotonicity")
    print("-" * 70)
    for run in test_runs[:1]:
        rdf = df_test[df_test["_run_key"] == run].copy()
        if "filedate" in rdf.columns and rdf["filedate"].notna().any():
            rdf = rdf.sort_values("filedate")
        actual = rdf["actual_rul_minutes"].values
        preds = y_pred[df_test["_run_key"] == run]
        actual_decreasing = np.all(np.diff(actual) <= 0)
        print(f"  Actual RUL monotonically decreasing: {actual_decreasing}")
        print(f"  First 5 actual RUL:  {actual[:5].tolist()}")
        print(f"  First 5 pred RUL:    {preds[:5].round(1).tolist()}")
    print()

    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    metrics = {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }
    return metrics


def save_validation_metrics(metrics: dict) -> Path:
    """Save metrics to data/models/validation_metrics.json for Pipeline UI."""
    out = BASE / "data" / "models" / "validation_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out}")
    return out


def run_retrain_and_validate():
    """
    Retrain model on NASA train runs, validate on holdout runs.
    This shows achievable accuracy when properly trained on real data.
    """
    print("=" * 70)
    print("NASA IMS - RETRAIN ON REAL DATA & VALIDATE")
    print("=" * 70)
    print()

    features = load_features_list()
    df = load_and_prepare_nasa_features(features)
    runs = np.sort(df["_run_key"].unique())
    n_holdout = max(1, len(runs) // 3)
    rng = np.random.RandomState(42)
    test_idx = rng.choice(len(runs), size=min(n_holdout, len(runs)), replace=False)
    test_runs = np.sort(runs[test_idx])
    train_runs = np.sort(np.setdiff1d(runs, test_runs))
    df_test = df[df["_run_key"].isin(test_runs)]
    df_train = df[df["_run_key"].isin(train_runs)]

    # Log-transform target for training
    y_train = np.log1p(df_train['actual_rul_minutes'].values)
    y_actual = df_test['actual_rul_minutes'].values

    X_train = df_train[features].values
    X_test = df_test[features].values

    # Retrain
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, max_depth=6, learning_rate=0.03, random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = np.maximum(0, np.expm1(model.predict(X_test_s)))

    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)

    print(f"Trained on {len(df_train)} samples (runs {train_runs.tolist()})")
    print(f"Validated on {len(df_test)} samples (runs {test_runs.tolist()})")
    print()
    print("ACCURACY (when trained on NASA data):")
    print(f"  MAE:  {mae:.2f} minutes ({mae/60:.1f} hours)")
    print(f"  RMSE: {rmse:.2f} minutes")
    print(f"  R²:   {r2:.4f}  {'✓' if r2 >= 0.5 else '⚠ CRITICAL' if r2 < 0 else '⚠ LOW'}")
    print(f"  Target: R² > 0.70 good, > 0.85 strong")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate RUL model on holdout runs")
    parser.add_argument("--save", action="store_true", help="Save metrics to data/models/validation_metrics.json for Pipeline UI")
    args = parser.parse_args()

    metrics = run_validation()
    if args.save:
        save_validation_metrics(metrics)
    print()
    run_retrain_and_validate()
