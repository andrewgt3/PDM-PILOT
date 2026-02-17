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

# =============================================================================
# CONFIGURATION
# =============================================================================

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

INPUT_STREAM = "clean_sensor_data"
OUTPUT_STREAM = "inference_results"
CONSUMER_GROUP = "inference_group"
CONSUMER_NAME = os.getenv("CONSUMER_NAME", "inference-worker-1")

MODEL_PATH = os.getenv("MODEL_PATH", "/app/gaia_model.pkl")
RUL_MODEL_PATH = os.getenv("RUL_MODEL_PATH", "/app/rul_model.json")

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
    """Load the trained classification model pipeline."""
    try:
        pipeline = joblib.load(MODEL_PATH)
        model = pipeline.get("model")
        scaler = pipeline.get("scaler")
        feature_columns = pipeline.get("feature_columns")

        if model and scaler and feature_columns:
            logger.info(
                "Classification model loaded: %s (%d features)",
                type(model).__name__,
                len(feature_columns),
            )
            return model, scaler, feature_columns
        else:
            logger.warning("Model pipeline missing components. Running without ML.")
            return None, None, None
    except FileNotFoundError:
        logger.warning("Model file %s not found. Running without ML.", MODEL_PATH)
        return None, None, None
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        return None, None, None


def load_rul_model() -> Optional[Any]:
    """Load the XGBoost RUL regression model."""
    try:
        import xgboost as xgb

        model = xgb.XGBRegressor()
        model.load_model(RUL_MODEL_PATH)
        logger.info("RUL model loaded from %s", RUL_MODEL_PATH)
        return model
    except FileNotFoundError:
        logger.warning("RUL model %s not found. Skipping RUL prediction.", RUL_MODEL_PATH)
        return None
    except Exception as exc:
        logger.error("Failed to load RUL model: %s", exc)
        return None


# =============================================================================
# REDIS CONNECTION
# =============================================================================


def get_redis() -> redis.Redis:
    """Connect to Redis with retry."""
    while _running:
        try:
            r = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                decode_responses=True,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            r.ping()
            logger.info("Connected to Redis at %s:%s", REDIS_HOST, REDIS_PORT)
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


# =============================================================================
# INFERENCE PIPELINE
# =============================================================================


def run_inference(
    payload: dict,
    processor: SignalProcessor,
    model,
    scaler,
    feature_columns: list,
    rul_model,
) -> dict:
    """Run full inference: feature extraction → classification → RUL prediction."""
    vibration = np.array(payload.get("vibration_raw", []))
    telemetry = {
        "rotational_speed": payload.get("rotational_speed", 1800.0),
        "temperature": payload.get("temperature", 70.0),
        "torque": payload.get("torque", 40.0),
        "tool_wear": payload.get("tool_wear", 0.1),
    }

    # 1. Feature extraction
    features = processor.process_signal(vibration, telemetry)

    # 2. Classification (failure prediction)
    if model is not None and len(vibration) > 0:
        try:
            feature_vector = np.array(
                [[features.get(col, 0.0) for col in feature_columns]]
            )
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            feature_scaled = scaler.transform(feature_vector)
            proba = model.predict_proba(feature_scaled)[0]
            features["failure_prediction"] = float(proba[1])
            features["failure_class"] = int(proba[1] > 0.5)
        except Exception as exc:
            logger.error("Classification failed: %s", exc)
            features["failure_prediction"] = 0.0
            features["failure_class"] = 0
    else:
        features["failure_prediction"] = 0.0
        features["failure_class"] = 0

    # 3. RUL regression
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
            rul_hours = float(rul_model.predict(rul_features)[0])
            features["rul_hours"] = max(0.0, rul_hours)
        except Exception as exc:
            logger.error("RUL prediction failed: %s", exc)
            features["rul_hours"] = -1.0
    else:
        features["rul_hours"] = -1.0

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

                    # Run inference
                    result = run_inference(
                        payload, processor, model, scaler,
                        feature_columns or [], rul_model,
                    )

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
