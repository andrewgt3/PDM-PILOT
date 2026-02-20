#!/usr/bin/env python3
"""
Golden Pipeline — Stage 5: Inference Service
==============================================
Decoupled XGBoost RUL (Remaining Useful Life) prediction engine.

Consumes cleaned sensor data from `clean_sensor_data`, extracts signal
features via SignalProcessor, runs ML inference, and publishes results
to `inference_results`.

This service is strictly decoupled:
  - No database dependency
  - No API dependency
  - Only reads from Redis Stream and writes to Redis Stream

Redis Streams:
  IN:  clean_sensor_data     (consumer group: inference_group)
  OUT: inference_results
"""

import os
import sys
import json
import time
import signal
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import redis
import numpy as np
import joblib

from config import get_settings

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_STREAM = "clean_sensor_data"
OUTPUT_STREAM = "inference_results"
CONSUMER_GROUP = "inference_group"
CONSUMER_NAME = os.getenv("CONSUMER_NAME", "inference-worker-1")

# Signal processing parameters
SAMPLE_RATE = 12_000
N_SAMPLES = 4096

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("inference-svc")

# =============================================================================
# GRACEFUL SHUTDOWN
# =============================================================================

_running = True


def _shutdown_handler(signum, frame):
    global _running
    logger.info("Shutdown signal received (signal=%s).", signum)
    _running = False


signal.signal(signal.SIGINT, _shutdown_handler)
signal.signal(signal.SIGTERM, _shutdown_handler)


# =============================================================================
# SIGNAL PROCESSOR (Embedded for standalone operation)
# =============================================================================
# Lightweight feature extraction without depending on the full
# advanced_features module. For production, mount the module or
# copy it into the container.
# =============================================================================


class SignalProcessor:
    """
    Lightweight signal processor for feature extraction.
    Extracts spectral, bearing fault, and degradation features
    from vibration waveforms.
    """

    # Standard bearing fault frequency ratios (6205-2RS deep groove)
    BPFO_RATIO = 3.05
    BPFI_RATIO = 4.95
    BSF_RATIO = 1.99
    FTF_RATIO = 0.40

    def __init__(self, sample_rate: int = SAMPLE_RATE, n_samples: int = N_SAMPLES):
        self.sample_rate = sample_rate
        self.n_samples = n_samples

    def process_signal(self, vibration: np.ndarray, telemetry: dict) -> dict:
        """Extract all features from a vibration signal + telemetry."""
        features = {}

        if len(vibration) < 64:
            # Return zeros for short signals
            return self._empty_features(telemetry)

        # FFT
        fft_vals = np.fft.rfft(vibration * np.hanning(len(vibration)))
        fft_mag = np.abs(fft_vals) / len(vibration)
        freqs = np.fft.rfftfreq(len(vibration), 1.0 / self.sample_rate)

        # Peak frequencies
        peak_indices = np.argsort(fft_mag)[-5:][::-1]
        for i, idx in enumerate(peak_indices):
            features[f"peak_freq_{i+1}"] = float(freqs[idx])
            features[f"peak_amp_{i+1}"] = float(fft_mag[idx])

        # Spectral band power
        low_mask = freqs < 500
        mid_mask = (freqs >= 500) & (freqs < 2000)
        high_mask = freqs >= 2000

        features["low_band_power"] = float(np.sum(fft_mag[low_mask] ** 2))
        features["mid_band_power"] = float(np.sum(fft_mag[mid_mask] ** 2))
        features["high_band_power"] = float(np.sum(fft_mag[high_mask] ** 2))
        features["total_power"] = float(np.sum(fft_mag ** 2))

        # Spectral entropy
        psd = fft_mag ** 2
        psd_norm = psd / (np.sum(psd) + 1e-12)
        features["spectral_entropy"] = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))

        # Spectral kurtosis
        psd_mean = np.mean(psd)
        psd_std = np.std(psd)
        if psd_std > 1e-12:
            features["spectral_kurtosis"] = float(
                np.mean(((psd - psd_mean) / psd_std) ** 4) - 3
            )
        else:
            features["spectral_kurtosis"] = 0.0

        # Bearing fault frequencies
        shaft_rpm = telemetry.get("rotational_speed", 1800.0)
        shaft_freq = shaft_rpm / 60.0
        bpfo = shaft_freq * self.BPFO_RATIO
        bpfi = shaft_freq * self.BPFI_RATIO
        bsf = shaft_freq * self.BSF_RATIO
        ftf = shaft_freq * self.FTF_RATIO

        features["bpfo_amp"] = self._amplitude_at_freq(freqs, fft_mag, bpfo)
        features["bpfi_amp"] = self._amplitude_at_freq(freqs, fft_mag, bpfi)
        features["bsf_amp"] = self._amplitude_at_freq(freqs, fft_mag, bsf)
        features["ftf_amp"] = self._amplitude_at_freq(freqs, fft_mag, ftf)
        features["sideband_strength"] = float(
            (features["bpfo_amp"] + features["bpfi_amp"]) / 2.0
        )

        # Degradation score (composite)
        fault_energy = (
            features["bpfo_amp"] + features["bpfi_amp"]
            + features["bsf_amp"] + features["ftf_amp"]
        )
        features["degradation_score"] = float(
            np.clip(fault_energy / 0.5, 0.0, 1.0)
        )

        # Add telemetry passthrough
        features["rotational_speed"] = telemetry.get("rotational_speed", 0.0)
        features["temperature"] = telemetry.get("temperature", 0.0)
        features["torque"] = telemetry.get("torque", 0.0)
        features["tool_wear"] = telemetry.get("tool_wear", 0.0)

        return features

    def _amplitude_at_freq(
        self, freqs: np.ndarray, magnitudes: np.ndarray, target_freq: float, bandwidth: float = 5.0
    ) -> float:
        """Get maximum amplitude within a bandwidth around the target frequency."""
        mask = (freqs >= target_freq - bandwidth) & (freqs <= target_freq + bandwidth)
        if np.any(mask):
            return float(np.max(magnitudes[mask]))
        return 0.0

    def _empty_features(self, telemetry: dict) -> dict:
        """Return a zeroed-out feature dict for signals too short to process."""
        features = {}
        for i in range(1, 6):
            features[f"peak_freq_{i}"] = 0.0
            features[f"peak_amp_{i}"] = 0.0
        for key in (
            "low_band_power", "mid_band_power", "high_band_power",
            "total_power", "spectral_entropy", "spectral_kurtosis",
            "bpfo_amp", "bpfi_amp", "bsf_amp", "ftf_amp",
            "sideband_strength", "degradation_score",
        ):
            features[key] = 0.0
        features["rotational_speed"] = telemetry.get("rotational_speed", 0.0)
        features["temperature"] = telemetry.get("temperature", 0.0)
        features["torque"] = telemetry.get("torque", 0.0)
        features["tool_wear"] = telemetry.get("tool_wear", 0.0)
        return features


# =============================================================================
# MODEL LOADING
# =============================================================================


def load_model() -> tuple:
    """Load the trained classification model pipeline (prefer gaia_model_calibrated.pkl in same dir)."""
    settings = get_settings()
    base_path = settings.model.resolved_gaia_model_path
    models_dir = base_path.parent
    path_calibrated = models_dir / "gaia_model_calibrated.pkl"
    path = path_calibrated if path_calibrated.exists() else base_path
    if path == path_calibrated:
        logger.info("Using calibrated classifier: %s", path)
    try:
        pipeline = joblib.load(str(path))
        model = pipeline.get("model")
        scaler = pipeline.get("scaler")
        feature_columns = pipeline.get("feature_columns")

        if model and scaler and feature_columns:
            logger.info(
                "Classification model loaded: %s (%d features) from %s",
                type(model).__name__,
                len(feature_columns),
                path,
            )
            return model, scaler, feature_columns
        else:
            logger.warning("Model pipeline missing components. Running without ML.")
            return None, None, None
    except FileNotFoundError:
        logger.warning("Model file %s not found. Running without ML.", path)
        return None, None, None
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        return None, None, None


def load_staging_model() -> Optional[tuple]:
    """Load staging model for shadow deployment. Returns (model, scaler, feature_columns) or None."""
    staging_path = get_settings().model.resolved_gaia_model_path.parent / "gaia_model_staging.pkl"
    if not staging_path.exists():
        return None
    try:
        pipeline = joblib.load(str(staging_path))
        model = pipeline.get("model")
        scaler = pipeline.get("scaler")
        feature_columns = pipeline.get("feature_columns")
        if model and scaler and feature_columns:
            logger.info("Staging model loaded from %s (shadow deployment)", staging_path)
            return (model, scaler, feature_columns)
        return None
    except Exception as exc:
        logger.warning("Failed to load staging model: %s", exc)
        return None


def load_rul_model() -> Optional[Any]:
    """Load RUL model: prefer rul_model_calibrated.pkl (MapieRegressor), else rul_model.pkl, else legacy XGB json."""
    settings = get_settings()
    models_dir = settings.model.resolved_rul_model_path.parent
    calibrated_path = models_dir / "rul_model_calibrated.pkl"
    if calibrated_path.exists():
        try:
            payload = joblib.load(str(calibrated_path))
            model = payload.get("model")
            if model is not None:
                logger.info("RUL calibrated model (MapieRegressor) loaded from %s", calibrated_path)
                return payload  # return full dict for scaler + feature_names
        except Exception as exc:
            logger.warning("Failed to load calibrated RUL model: %s", exc)
    rul_path = settings.model.resolved_rul_model_path
    if rul_path.exists():
        try:
            payload = joblib.load(str(rul_path))
            model = payload.get("model")
            if model is not None:
                logger.info("RUL model loaded from %s", rul_path)
                return payload
        except Exception as exc:
            logger.warning("Failed to load RUL model from %s: %s", rul_path, exc)
    try:
        import xgboost as xgb
        legacy_path = os.getenv("RUL_MODEL_LEGACY") or str(models_dir / "rul_model.json")
        if not os.path.exists(legacy_path):
            logger.warning("RUL model not found. Skipping RUL prediction.")
            return None
        model = xgb.XGBRegressor()
        model.load_model(legacy_path)
        logger.info("RUL model loaded from %s", legacy_path)
        return {"model": model, "scaler": None, "feature_names": None}
    except Exception as exc:
        logger.error("Failed to load RUL model: %s", exc)
        return None


# =============================================================================
# REDIS CONNECTION
# =============================================================================


def get_redis() -> redis.Redis:
    """Connect to Redis with retry (config from get_settings().redis)."""
    s = get_settings()
    pw = s.redis.password.get_secret_value() if s.redis.password else None
    while _running:
        try:
            r = redis.Redis(
                host=s.redis.host,
                port=s.redis.port,
                password=pw,
                decode_responses=True,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            r.ping()
            logger.info("Connected to Redis at %s:%s", s.redis.host, s.redis.port)
            return r
        except redis.ConnectionError as exc:
            logger.warning("Redis not ready (%s). Retrying in 3s…", exc)
            time.sleep(3)
    sys.exit(0)


def ensure_consumer_group(r: redis.Redis):
    """Create the consumer group if it doesn't exist."""
    try:
        r.xgroup_create(INPUT_STREAM, CONSUMER_GROUP, id="0", mkstream=True)
        logger.info("Created consumer group '%s' on '%s'", CONSUMER_GROUP, INPUT_STREAM)
    except redis.ResponseError as exc:
        if "BUSYGROUP" in str(exc):
            logger.info("Consumer group '%s' already exists.", CONSUMER_GROUP)
        else:
            raise


def _write_shadow_prediction(machine_id: str, production_prediction: float, staging_prediction: float, features_snapshot: dict) -> None:
    """Write one row to shadow_predictions table. Uses SHADOW_DB_DSN or DATABASE_URL (sync). No-op if unset or on error."""
    dsn = os.getenv("SHADOW_DB_DSN") or os.getenv("DATABASE_URL")
    if not dsn:
        return
    try:
        import psycopg2
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        agree = (production_prediction > 0.5) == (staging_prediction > 0.5)
        cur.execute(
            """INSERT INTO shadow_predictions (machine_id, production_prediction, staging_prediction, agree, features_snapshot_json)
               VALUES (%s, %s, %s, %s, %s)""",
            (machine_id, production_prediction, staging_prediction, agree, json.dumps(features_snapshot)),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as exc:
        logger.debug("Shadow DB write failed: %s", exc)


# =============================================================================
# INFERENCE PIPELINE
# =============================================================================


def _confidence_from_interval(rul_point: float, rul_lower: float, rul_upper: float) -> str:
    """Classify confidence as HIGH/MEDIUM/LOW from 80% interval width relative to point estimate.
    HIGH: width < 30% of point; MEDIUM: 30-60%; LOW: > 60%."""
    width = rul_upper - rul_lower
    denom = max(abs(rul_point), 1e-6)
    ratio = width / denom if denom else 1.0
    if ratio < 0.30:
        return "HIGH"
    if ratio <= 0.60:
        return "MEDIUM"
    return "LOW"


def run_inference(
    payload: dict,
    processor: SignalProcessor,
    model,
    scaler,
    feature_columns: list,
    rul_payload: Optional[dict],
    staging_pipeline: Optional[tuple] = None,
) -> dict:
    """Run full inference: feature extraction → classification → RUL prediction (calibrated when available)."""
    vibration = np.array(payload.get("vibration_raw", []))
    telemetry = {
        "rotational_speed": payload.get("rotational_speed", 1800.0),
        "temperature": payload.get("temperature", 70.0),
        "torque": payload.get("torque", 40.0),
        "tool_wear": payload.get("tool_wear", 0.1),
    }

    # 1. Feature extraction
    features = processor.process_signal(vibration, telemetry)
    in_training_distribution = True
    meta_path = str(get_settings().model.resolved_gaia_model_path.parent / "gaia_model_latest_metadata.json")

    # 2. Classification (failure prediction) — calibrated probabilities when using gaia_model_calibrated
    if model is not None and len(vibration) > 0:
        try:
            feature_vector = np.array(
                [[features.get(col, 0.0) for col in feature_columns]]
            )
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            from pipeline.inference.feature_validator import validate_features
            feature_dict = {col: float(feature_vector[0][i]) for i, col in enumerate(feature_columns)}
            valid, ood_warnings = validate_features(feature_dict, metadata_path=meta_path)
            in_training_distribution = valid and len(ood_warnings) == 0
            if ood_warnings:
                logger.warning(
                    "Feature validator flagged out-of-distribution features (potential drift): %s",
                    ood_warnings,
                )
            feature_scaled = scaler.transform(feature_vector)
            proba = model.predict_proba(feature_scaled)[0]
            failure_prob = float(proba[1])
            features["failure_prediction"] = failure_prob
            features["failure_probability"] = failure_prob
            features["failure_class"] = int(failure_prob > 0.5)
            features["health_score"] = float(max(0, min(100, (1.0 - failure_prob) * 100)))
            # Shadow: run staging model if provided (same feature_vector used for production)
            if staging_pipeline:
                try:
                    s_model, s_scaler, s_cols = staging_pipeline
                    s_vec = np.array([[features.get(c, 0.0) for c in s_cols]], dtype=float)
                    s_vec = np.nan_to_num(s_vec, nan=0.0, posinf=0.0, neginf=0.0)
                    s_vec = s_scaler.transform(s_vec)
                    s_proba = s_model.predict_proba(s_vec)[0]
                    features["_staging_failure_prob"] = float(s_proba[1])
                except Exception as s_exc:
                    logger.debug("Staging prediction failed: %s", s_exc)
        except Exception as exc:
            logger.error("Classification failed: %s", exc)
            features["failure_prediction"] = 0.0
            features["failure_probability"] = 0.0
            features["failure_class"] = 0
            features["health_score"] = 100.0
    else:
        features["failure_prediction"] = 0.0
        features["failure_probability"] = 0.0
        features["failure_class"] = 0
        features["health_score"] = 100.0

    # 3. RUL regression — use MapieRegressor.predict(X, alpha=0.20) when calibrated
    rul_payload = rul_payload or {}
    rul_model = rul_payload.get("model")
    rul_scaler = rul_payload.get("scaler")
    if rul_model is not None and len(vibration) > 0:
        try:
            rul_features = np.array([[
                features.get("rotational_speed", 0.0),
                features.get("temperature", 0.0),
                features.get("torque", 0.0),
                features.get("tool_wear", 0.0),
                features.get("degradation_score", 0.0),
            ]])
            rul_features = np.nan_to_num(rul_features, nan=0.0)
            if rul_scaler is not None:
                rul_features = rul_scaler.transform(rul_features)
            # MapieRegressor returns (y_pred, y_pis) with alpha; legacy returns array
            if hasattr(rul_model, "predict") and callable(getattr(rul_model, "predict", None)):
                try:
                    out = rul_model.predict(rul_features, alpha=0.20)
                    if isinstance(out, tuple) and len(out) == 2:
                        y_pred, y_pis = out
                        rul_hours = float(np.asarray(y_pred).flatten()[0])
                        rul_lower = float(y_pis[0, 0, 0])
                        rul_upper = float(y_pis[0, 1, 0])
                        features["rul_hours"] = max(0.0, rul_hours)
                        features["rul_days"] = max(0.0, rul_hours / 24.0)
                        features["rul_lower_80"] = max(0.0, rul_lower / 24.0)
                        features["rul_upper_80"] = max(0.0, rul_upper / 24.0)
                        features["confidence"] = _confidence_from_interval(rul_hours, rul_lower, rul_upper)
                    else:
                        rul_hours = float(np.asarray(out).flatten()[0])
                        rul_days = max(0.0, rul_hours / 24.0)
                        features["rul_hours"] = max(0.0, rul_hours)
                        features["rul_days"] = rul_days
                        features["rul_lower_80"] = rul_days * 0.75
                        features["rul_upper_80"] = rul_days * 1.25
                        features["confidence"] = _confidence_from_interval(rul_days, rul_days * 0.75, rul_days * 1.25)
                except TypeError:
                    rul_hours = float(rul_model.predict(rul_features)[0])
                    rul_days = max(0.0, rul_hours / 24.0)
                    features["rul_hours"] = max(0.0, rul_hours)
                    features["rul_days"] = rul_days
                    features["rul_lower_80"] = rul_days * 0.75
                    features["rul_upper_80"] = rul_days * 1.25
                    features["confidence"] = _confidence_from_interval(rul_days, rul_days * 0.75, rul_days * 1.25)
            else:
                rul_hours = float(rul_model.predict(rul_features)[0])
                rul_days = max(0.0, rul_hours / 24.0)
                features["rul_hours"] = max(0.0, rul_hours)
                features["rul_days"] = rul_days
                features["rul_lower_80"] = rul_days * 0.75
                features["rul_upper_80"] = rul_days * 1.25
                features["confidence"] = _confidence_from_interval(rul_days, rul_days * 0.75, rul_days * 1.25)
        except Exception as exc:
            logger.error("RUL prediction failed: %s", exc)
            features["rul_hours"] = -1.0
            features["rul_days"] = -1.0
            features["rul_lower_80"] = features["rul_upper_80"] = -1.0
            features["confidence"] = "LOW"
    else:
        features["rul_hours"] = -1.0
        features["rul_days"] = -1.0
        features["rul_lower_80"] = features["rul_upper_80"] = -1.0
        features["confidence"] = "LOW"
    features["in_training_distribution"] = in_training_distribution

    # 4. Metadata
    features["timestamp"] = payload.get("timestamp", datetime.now(timezone.utc).isoformat())
    features["machine_id"] = payload.get("machine_id", "unknown")
    features["inference_at"] = datetime.now(timezone.utc).isoformat()
    features["inference_svc_version"] = "1.0.0"

    return features


# =============================================================================
# MAIN LOOP
# =============================================================================


def run():
    """Consume from clean_sensor_data, run inference, publish to inference_results."""
    r = get_redis()
    ensure_consumer_group(r)

    # Load models
    processor = SignalProcessor(sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES)
    model, scaler, feature_columns = load_model()
    rul_model = load_rul_model()
    staging_pipeline = load_staging_model()

    total_read = 0
    total_inferred = 0

    logger.info(
        "Inference service started | in=%s | out=%s | model=%s | rul=%s",
        INPUT_STREAM,
        OUTPUT_STREAM,
        "loaded" if model else "NONE",
        "loaded" if rul_model else "NONE",
    )

    while _running:
        try:
            # Read from stream
            messages = r.xreadgroup(
                CONSUMER_GROUP,
                CONSUMER_NAME,
                {INPUT_STREAM: ">"},
                count=10,  # Smaller batch for low-latency inference
                block=2000,
            )

            if not messages:
                continue

            for stream_name, entries in messages:
                for msg_id, fields in entries:
                    total_read += 1

                    try:
                        payload = json.loads(fields.get("data", "{}"))
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON in message %s. Dropping.", msg_id)
                        r.xack(INPUT_STREAM, CONSUMER_GROUP, msg_id)
                        continue

                    # Run inference (rul_model is dict with model/scaler/feature_names or None)
                    result = run_inference(
                        payload, processor, model, scaler,
                        feature_columns or [], rul_model,
                        staging_pipeline=staging_pipeline,
                    )

                    # Shadow: log production vs staging and remove internal key before publish
                    if "_staging_failure_prob" in result:
                        machine_id = result.get("machine_id", "unknown")
                        features_snapshot = {c: result.get(c) for c in (feature_columns or []) if c in result}
                        _write_shadow_prediction(
                            machine_id,
                            result["failure_probability"],
                            result["_staging_failure_prob"],
                            features_snapshot,
                        )
                        del result["_staging_failure_prob"]

                    # Publish to inference_results stream
                    r.xadd(
                        OUTPUT_STREAM,
                        {"data": json.dumps(result)},
                        maxlen=50_000,
                    )

                    # Also cache latest result per machine for fast API lookup
                    machine_id = result.get("machine_id", "unknown")
                    r.set(
                        f"inference:latest:{machine_id}",
                        json.dumps(result),
                        ex=300,  # 5 min TTL
                    )

                    # Append to drift buffer (last 10k per machine) for PSI/KS
                    try:
                        snapshot = {
                            "features": {col: result.get(col) for col in (feature_columns or []) if col in result},
                            "failure_probability": result.get("failure_probability"),
                            "health_score": result.get("health_score"),
                            "timestamp": result.get("timestamp") or result.get("inference_at"),
                        }
                        buffer_key = f"inference:buffer:{machine_id}"
                        r.rpush(buffer_key, json.dumps(snapshot))
                        r.ltrim(buffer_key, 0, 9999)
                    except Exception as buf_err:
                        logger.warning("Drift buffer write failed: %s", buf_err)

                    r.xack(INPUT_STREAM, CONSUMER_GROUP, msg_id)
                    total_inferred += 1

                    if total_inferred % 50 == 0:
                        logger.info(
                            "Inference progress | read=%d inferred=%d | last_machine=%s prediction=%.3f rul=%.1fh",
                            total_read,
                            total_inferred,
                            machine_id,
                            result.get("failure_prediction", 0),
                            result.get("rul_hours", -1),
                        )

        except redis.ConnectionError as exc:
            logger.warning("Redis connection lost (%s). Reconnecting in 5s…", exc)
            time.sleep(5)
            r = get_redis()
            ensure_consumer_group(r)
        except Exception as exc:
            logger.error("Unexpected error: %s", exc, exc_info=True)
            time.sleep(2)

    logger.info(
        "Inference service stopped. read=%d inferred=%d",
        total_read,
        total_inferred,
    )


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    run()
