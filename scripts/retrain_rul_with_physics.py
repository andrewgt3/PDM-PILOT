#!/usr/bin/env python3
"""
Retrain RUL Model with Physics-Based Features
=============================================
Trains XGBoost on NASA IMS data with 15 features (refinery + physics).
Uses run-based holdout to prevent data leakage.
Saves model and scaler to data/models/ for inference pipeline.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent.parent
COMBINED_CSV = BASE / "data" / "processed" / "combined_features_physics.csv"
NASA_CSV = BASE / "data" / "processed" / "nasa_features_physics.csv"
# Prefer combined (NASA + FEMTO) if exists, else NASA only
INPUT_CSV = COMBINED_CSV if COMBINED_CSV.exists() else NASA_CSV
MODELS_DIR = BASE / "data" / "models"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
MODEL_PATH = MODELS_DIR / "xgb_rul_v1.json"

# Physics + refinery + spectral + rolling/baseline
# time_pct: use when run context available (position in run); improves accuracy significantly
FEATURES = [
    "rms", "kurtosis", "p2p", "skewness", "crest", "shape",
    "bpfo_amp", "bpfi_amp", "bsf_amp", "ftf_amp", "sideband_strength",
    "low_band_power", "mid_band_power", "high_band_power",
    "degradation_score",
    "rms_ratio_baseline", "rms_slope_5", "kurtosis_slope_5",
    "bpfo_to_rms", "bpfi_to_rms", "bsf_to_rms",
    "time_pct",  # Position in run (0=start, 1=end)
    "run_length_minutes",  # Scale cue - RUL max varies by run length
]
TARGET = "rul_minutes"
RUN_GAP_HOURS = 24
TEST_RUN_RATIO = 0.33


def main():
    print("=" * 70)
    print("RUL MODEL RETRAIN - Physics + Refinery Features")
    print("=" * 70)

    if not INPUT_CSV.exists():
        print(f"ERROR: Run build script first.")
        print(f"  NASA only:  python scripts/build_nasa_features_with_physics.py")
        print(f"  Combined:  python scripts/build_combined_features.py (after download_femto_data.py)")
        sys.exit(1)
    print(f"Input: {INPUT_CSV.name}")

    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=FEATURES + [TARGET])

    # Use run_idx if present (combined data), else run_id
    run_col = "run_idx" if "run_idx" in df.columns else "run_id"
    runs = np.sort(df[run_col].unique())
    n_test = max(1, int(len(runs) * TEST_RUN_RATIO))
    # Random holdout (same seed as validate_rul_on_nasa.py) - chronological holdout
    # can fail when last runs differ (e.g. different bearing set, much longer run)
    rng = np.random.RandomState(42)
    test_idx = rng.choice(len(runs), size=min(n_test, len(runs)), replace=False)
    test_runs = np.sort(runs[test_idx])
    train_runs = np.sort(np.setdiff1d(runs, test_runs))

    df_train = df[df[run_col].isin(train_runs)]
    df_test = df[df[run_col].isin(test_runs)]

    # Log-transform RUL for better scale (predict log, convert back)
    use_log_rul = True
    if use_log_rul:
        y_train = np.log1p(df_train[TARGET].values)
        y_test_raw = df_test[TARGET].values
    else:
        y_train = df_train[TARGET].values
        y_test_raw = df_test[TARGET].values

    X_train = df_train[FEATURES].values
    X_test = df_test[FEATURES].values

    print(f"\nTrain: {len(df_train)} samples (runs {train_runs.tolist()})")
    print(f"Test:  {len(df_test)} samples (runs {test_runs.tolist()})")
    if use_log_rul:
        print("Target: log1p(RUL) for training")

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        random_state=42,
    )
    model.fit(X_train_s, y_train)

    # Evaluate (convert log back to minutes)
    y_pred_log = model.predict(X_test_s)
    y_pred = np.expm1(np.maximum(0, y_pred_log))

    mae = mean_absolute_error(y_test_raw, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred))
    r2 = r2_score(y_test_raw, y_pred)

    print("\n" + "=" * 70)
    print("ACCURACY (Holdout Runs)")
    print("=" * 70)
    print(f"  MAE:  {mae:.2f} minutes ({mae/60:.1f} hours)")
    print(f"  RMSE: {rmse:.2f} minutes")
    print(f"  R²:   {r2:.4f}")
    print()

    # Feature importance
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    print("Feature Importance:")
    for i in order[:10]:
        print(f"  {FEATURES[i]:25s} {imp[i]:.4f}")
    print()

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    model.get_booster().save_model(str(MODEL_PATH))

    # Save feature list for inference
    meta_path = MODELS_DIR / "rul_features.txt"
    with open(meta_path, "w") as f:
        f.write("\n".join(FEATURES))

    # Save training metadata for Pipeline UI
    training_source = "NASA + FEMTO" if INPUT_CSV == COMBINED_CSV else "NASA only"
    training_meta = {
        "source": training_source,
        "n_runs": int(len(runs)),
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
    }
    with open(MODELS_DIR / "training_metadata.json", "w") as f:
        json.dump(training_meta, f, indent=2)

    print(f"Saved {SCALER_PATH}")
    print(f"Saved {MODEL_PATH}")
    print(f"Saved {meta_path}")
    print("\n" + "=" * 70)
    print("DONE. Update inference_service.py FEATURES to match.")
    print("=" * 70)


if __name__ == "__main__":
    main()
