"""
Gaia Predictive — Drift Monitor Service.

Detects data drift (PSI, KS) and concept drift (rolling F1 vs baseline).
DriftMonitorService: save_training_distribution (model_baselines), run_drift_check (drift_reports).
"""

from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from logger import get_logger

logger = get_logger(__name__)

# Redis buffer key written by pipeline inference service
REDIS_BUFFER_PREFIX = "inference:buffer:"
REDIS_HOST = __import__("os").environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(__import__("os").environ.get("REDIS_PORT", "6379"))
BUFFER_MAX = 10_000
PSI_WARNING = 0.1
PSI_CRITICAL = 0.25
F1_DROP_WARNING = 0.15
F1_DROP_CRITICAL = 0.30
NUM_BINS = 10
KS_P_THRESHOLD = 0.05

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_MODELS_DIR = BASE_DIR / "data" / "models"
METADATA_PATH = MODELS_DIR / "gaia_model_latest_metadata.json"
TRAINING_SAMPLE_PATH = DATA_MODELS_DIR / "training_sample.csv"
GAIA_MODEL_PATH = MODELS_DIR / "gaia_model_latest.pkl"


class DriftReport(BaseModel):
    """Report from run_drift_check (persisted to drift_reports)."""
    machine_id: str
    checked_at: str
    drifted_features: list[str] = []
    worst_psi: float = 0.0
    prediction_drift_pct: Optional[float] = None
    overall_status: str  # STABLE | WARNING | CRITICAL
    recommended_action: str = ""
    # Legacy / optional
    data_drift_psi: Optional[dict] = None
    concept_drift_f1_delta: Optional[float] = None


class DriftMonitorService:
    """
    Saves training distributions to model_baselines and runs drift checks
    (PSI + optional prediction drift), persisting results to drift_reports.
    """

    def __init__(self):
        self._settings = get_settings()

    def save_training_distribution(self, machine_id: str, feature_df: pd.DataFrame, model_version: str = "v1") -> None:
        """
        Compute per-feature stats (mean, std, p5, p50, p95) from feature_df and persist
        to table model_baselines (machine_id, model_version, feature_stats_json, created_at).
        """
        numeric = feature_df.select_dtypes(include=[np.number])
        if numeric.empty:
            logger.warning("save_training_distribution: no numeric columns in feature_df")
            return
        stats = {}
        for col in numeric.columns:
            s = numeric[col].dropna()
            if len(s) < 2:
                continue
            stats[col] = {
                "mean": float(s.mean()),
                "std": float(s.std()) if s.std() > 0 else 1e-6,
                "p5": float(s.quantile(0.05)),
                "p50": float(s.quantile(0.50)),
                "p95": float(s.quantile(0.95)),
            }
        if not stats:
            return
        payload = json.dumps(stats)

        async def _insert():
            from database import get_db_context
            async with get_db_context() as db:
                await db.execute(
                    text("""
                        INSERT INTO model_baselines (machine_id, model_version, feature_stats_json, created_at)
                        VALUES (:machine_id, :model_version, :feature_stats_json::jsonb, NOW())
                        ON CONFLICT (machine_id, model_version) DO UPDATE SET
                            feature_stats_json = EXCLUDED.feature_stats_json,
                            created_at = NOW()
                    """),
                    {"machine_id": machine_id, "model_version": model_version, "feature_stats_json": payload},
                )
                await db.commit()

        try:
            asyncio.run(_insert())
            logger.info("save_training_distribution: saved baseline for machine_id=%s model_version=%s", machine_id, model_version)
        except Exception as e:
            logger.warning("save_training_distribution failed: %s", e)

    def run_drift_check(self, machine_id: str) -> DriftReport:
        """
        Data drift: query last N days cwru_features, load baseline from model_baselines,
        compute PSI per feature (10 bins). Prediction drift: last N days inference health_score
        vs baseline; flag if mean shifted > 20%. Persist to drift_reports; return DriftReport.
        """
        return asyncio.run(_run_drift_check_impl(self._settings, machine_id))


async def _run_drift_check_impl(settings: Any, machine_id: str) -> DriftReport:
    from database import get_db_context
    lookback_days = settings.drift.drift_check_lookback_days
    warning_psi = settings.drift.drift_warning_psi
    critical_psi = settings.drift.drift_critical_psi
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    checked_at = datetime.now(timezone.utc).isoformat()

    # Load baseline from model_baselines
    async with get_db_context() as db:
        r = await db.execute(
            text("SELECT machine_id, model_version, feature_stats_json, baseline_metadata FROM model_baselines WHERE machine_id = :mid ORDER BY created_at DESC LIMIT 1"),
            {"mid": machine_id},
        )
        row = r.mappings().fetchone()
    baseline_stats = None
    baseline_metadata = None
    baseline_model_version = None
    if row:
        baseline_model_version = row.get("model_version")
        fs = row["feature_stats_json"]
        baseline_stats = fs if isinstance(fs, dict) else (json.loads(fs) if fs else None)
        bm = row["baseline_metadata"]
        baseline_metadata = bm if isinstance(bm, dict) else (json.loads(bm) if bm else None)

    # Query cwru_features last N days (all columns; we'll intersect with baseline_stats)
    CWRU_FLOAT_COLS = [
        "peak_freq_1", "peak_freq_2", "peak_freq_3", "peak_freq_4", "peak_freq_5",
        "peak_amp_1", "peak_amp_2", "peak_amp_3", "peak_amp_4", "peak_amp_5",
        "low_band_power", "mid_band_power", "high_band_power",
        "spectral_entropy", "spectral_kurtosis", "total_power",
        "bpfo_amp", "bpfi_amp", "bsf_amp", "ftf_amp", "sideband_strength",
        "degradation_score", "degradation_score_smoothed",
        "rotational_speed", "temperature", "torque", "tool_wear",
    ]
    live_rows = []
    async with get_db_context() as db:
        result = await db.execute(
            text("SELECT * FROM cwru_features WHERE machine_id = :mid AND timestamp >= :since ORDER BY timestamp"),
            {"mid": machine_id, "since": since},
        )
        live_rows = [dict(r._mapping) for r in result.mappings().fetchall()]

    # PSI per feature
    worst_psi = 0.0
    drifted_features = []
    if baseline_stats and live_rows:
        for feat, train_s in baseline_stats.items():
            if feat.startswith("_"):
                continue
            if not isinstance(train_s, dict):
                continue
            live_vals = [r.get(feat) for r in live_rows if r.get(feat) is not None]
            if len(live_vals) < 10:
                continue
            live_vals = np.array(live_vals, dtype=float)
            lo = train_s.get("p5", train_s["mean"] - 3 * train_s.get("std", 1))
            hi = train_s.get("p95", train_s["mean"] + 3 * train_s.get("std", 1))
            if hi <= lo:
                hi = lo + 1e-6
            expected_pcts = [0.1] * NUM_BINS
            bins = np.linspace(lo, hi, NUM_BINS + 1)
            actual_counts = np.histogram(live_vals, bins=bins)[0]
            actual_pcts = (actual_counts / (actual_counts.sum() or 1)).tolist()
            psi_val = _psi(expected_pcts, actual_pcts)
            if psi_val > worst_psi:
                worst_psi = psi_val
            if psi_val >= critical_psi:
                drifted_features.append(feat)
            elif psi_val >= warning_psi and feat not in drifted_features:
                drifted_features.append(feat)

    # Prediction drift: mean health_score from Redis vs baseline
    prediction_drift_pct = None
    live_list = get_prediction_distribution(machine_id)
    if live_list:
        health_scores = [float(x.get("health_score", 0)) for x in live_list if x.get("health_score") is not None]
        if not health_scores:
            health_scores = [float(x.get("health_score", 0)) for x in live_list]
        current_mean = float(np.mean(health_scores)) if health_scores else None
        if current_mean is not None and baseline_metadata and isinstance(baseline_metadata.get("health_score_mean"), (int, float)):
            baseline_mean = float(baseline_metadata["health_score_mean"])
            if baseline_mean != 0:
                prediction_drift_pct = (current_mean - baseline_mean) / baseline_mean * 100.0
        if current_mean is not None and baseline_metadata is None and baseline_model_version is not None:
            async with get_db_context() as db:
                await db.execute(
                    text("UPDATE model_baselines SET baseline_metadata = :meta::jsonb WHERE machine_id = :mid AND model_version = :mv"),
                    {"meta": json.dumps({"health_score_mean": current_mean}), "mid": machine_id, "mv": baseline_model_version},
                )
                await db.commit()

    # Overall status
    overall_status = "STABLE"
    recommended_action = "None"
    if worst_psi >= critical_psi or (prediction_drift_pct is not None and abs(prediction_drift_pct) > 20):
        overall_status = "CRITICAL"
        recommended_action = "Retrain model or review features"
    elif worst_psi >= warning_psi or (prediction_drift_pct is not None and abs(prediction_drift_pct) > 20):
        overall_status = "WARNING"
        recommended_action = "Monitor"

    report = DriftReport(
        machine_id=machine_id,
        checked_at=checked_at,
        drifted_features=drifted_features,
        worst_psi=round(worst_psi, 4),
        prediction_drift_pct=round(prediction_drift_pct, 2) if prediction_drift_pct is not None else None,
        overall_status=overall_status,
        recommended_action=recommended_action,
    )

    async with get_db_context() as db:
        await db.execute(
            text("""
                INSERT INTO drift_reports (machine_id, checked_at, worst_psi, drifted_features_json, prediction_drift_pct, overall_status, recommended_action)
                VALUES (:mid, :checked_at, :worst_psi, :drifted_json::jsonb, :pred_drift, :status, :action)
            """),
            {
                "mid": machine_id,
                "checked_at": datetime.now(timezone.utc),
                "worst_psi": report.worst_psi,
                "drifted_json": json.dumps(report.drifted_features),
                "pred_drift": report.prediction_drift_pct,
                "status": report.overall_status,
                "action": report.recommended_action,
            },
        )
        await db.commit()
    return report


def get_prediction_distribution(machine_id: str) -> list[dict]:
    """Read last 10k prediction/feature snapshots from Redis buffer (written by inference service)."""
    try:
        import redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        key = f"{REDIS_BUFFER_PREFIX}{machine_id}"
        raw = r.lrange(key, 0, -1)
        if not raw:
            return []
        out = []
        for s in raw:
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError:
                continue
        return out
    except Exception as e:
        logger.warning("get_prediction_distribution failed: %s", e)
        return []


def get_training_distribution(machine_id: str) -> dict:
    """Load feature stats and optional sample from model metadata (and training_sample file if present)."""
    out: dict = {"feature_ranges": {}, "feature_list": [], "feature_histograms": {}, "metrics": {}, "training_sample": None}
    path = METADATA_PATH
    if not path.exists():
        path = DATA_MODELS_DIR / "gaia_model_latest_metadata.json"
    if not path.exists():
        return out
    try:
        with open(path) as f:
            meta = json.load(f)
        out["feature_ranges"] = meta.get("feature_ranges") or {}
        if isinstance(out["feature_ranges"], list):
            out["feature_ranges"] = {}
        out["feature_list"] = meta.get("feature_list") or []
        out["feature_histograms"] = meta.get("feature_histograms") or {}
        out["metrics"] = meta.get("metrics") or {}
    except Exception as e:
        logger.warning("Load metadata failed: %s", e)
    if TRAINING_SAMPLE_PATH.exists():
        try:
            out["training_sample"] = pd.read_csv(TRAINING_SAMPLE_PATH)
        except Exception as e:
            logger.warning("Load training_sample.csv failed: %s", e)
    return out


def _psi(expected_pcts: list[float], actual_pcts: list[float]) -> float:
    """Population Stability Index: sum (actual_i - expected_i) * ln(actual_i / expected_i)."""
    psi = 0.0
    for a, e in zip(actual_pcts, expected_pcts):
        if e <= 0 or a <= 0:
            continue
        psi += (a - e) * math.log(a / e)
    return float(psi)


def _run_data_drift(
    machine_id: str,
    training: dict,
    live_list: list[dict],
) -> tuple[dict, list[str]]:
    """Compute PSI and optionally KS; return (data_drift_psi dict, drifted_features list)."""
    psi_result: dict = {"max_psi": 0.0, "per_feature": {}}
    drifted: list[str] = []
    if not live_list:
        return psi_result, drifted
    feature_histograms = training.get("feature_histograms") or {}
    feature_ranges = training.get("feature_ranges") or {}
    if not isinstance(feature_ranges, dict):
        feature_ranges = {}
    feature_list = training.get("feature_list") or []
    feature_names = list(feature_ranges.keys()) or list(feature_histograms.keys()) or list(feature_list)
    if not feature_names:
        for d in live_list:
            feats = d.get("features") or d
            if isinstance(feats, dict):
                feature_names = [k for k in feats if k not in ("failure_probability", "health_score", "timestamp")]
                break
    for feat in feature_names:
        live_vals = []
        for d in live_list:
            feats = d.get("features") or d
            if isinstance(feats, dict) and feat in feats:
                try:
                    live_vals.append(float(feats[feat]))
                except (TypeError, ValueError):
                    pass
        if len(live_vals) < 10:
            continue
        live_vals = np.array(live_vals)
        rng = feature_ranges.get(feat) if isinstance(feature_ranges.get(feat), dict) else {}
        lo = rng.get("min", float(np.nanmin(live_vals)))
        hi = rng.get("max", float(np.nanmax(live_vals)))
        if hi <= lo:
            hi = lo + 1e-6
        hist_train = feature_histograms.get(feat)
        if isinstance(hist_train, list) and len(hist_train) == NUM_BINS:
            expected_counts = np.array(hist_train, dtype=float)
            expected_pcts = (expected_counts / (expected_counts.sum() or 1)).tolist()
        else:
            expected_pcts = [1.0 / NUM_BINS] * NUM_BINS
        bins = np.linspace(lo, hi, NUM_BINS + 1)
        actual_counts = np.histogram(live_vals, bins=bins)[0]
        actual_pcts = (actual_counts / (actual_counts.sum() or 1)).tolist()
        psi_val = _psi(expected_pcts, actual_pcts)
        psi_result["per_feature"][feat] = round(psi_val, 4)
        if psi_val > psi_result["max_psi"]:
            psi_result["max_psi"] = round(psi_val, 4)
        if psi_val > PSI_CRITICAL:
            drifted.append(feat)
        elif psi_val > PSI_WARNING and feat not in drifted:
            drifted.append(feat)
    if psi_result["max_psi"] == 0 and psi_result["per_feature"]:
        psi_result["max_psi"] = max(psi_result["per_feature"].values()) if psi_result["per_feature"] else 0
    # KS test if training sample available
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        ks_2samp = None
    training_sample = training.get("training_sample")
    if ks_2samp is not None and training_sample is not None and isinstance(training_sample, pd.DataFrame):
        for feat in feature_names:
            if feat not in training_sample.columns:
                continue
            train_vals = training_sample[feat].dropna().values
            live_vals = []
            for d in live_list:
                feats = d.get("features") or d
                if isinstance(feats, dict) and feat in feats:
                    try:
                        live_vals.append(float(feats[feat]))
                    except (TypeError, ValueError):
                        pass
            if len(train_vals) < 5 or len(live_vals) < 5:
                continue
            _, p_val = ks_2samp(train_vals, live_vals)
            if p_val < KS_P_THRESHOLD and feat not in drifted:
                drifted.append(feat)
    return psi_result, list(dict.fromkeys(drifted))


def _run_concept_drift_sync(machine_id: str, baseline_f1: float) -> Optional[float]:
    """Compute 7-day rolling F1 on labeled examples vs current model; return F1 delta (current - baseline)."""
    async def _async_run():
        from database import get_db_context
        async with get_db_context() as db:
            result = await db.execute(
                text("""
                    SELECT lt.feature_snapshot_json, l.label
                    FROM labeling_tasks lt
                    JOIN labels l ON l.task_id = lt.task_id
                    WHERE lt.machine_id = :mid AND lt.status = 'completed'
                    AND lt.created_at >= :since
                    ORDER BY lt.created_at
                """),
                {"mid": machine_id, "since": datetime.now(timezone.utc) - timedelta(days=7)},
            )
            rows = result.mappings().all()
        if len(rows) < 10:
            return None
        # Load current model
        if not GAIA_MODEL_PATH.exists():
            return None
        import joblib
        pipeline = joblib.load(GAIA_MODEL_PATH)
        model = pipeline.get("model")
        scaler = pipeline.get("scaler")
        feature_columns = pipeline.get("feature_columns") or []
        if not model or not feature_columns:
            return None
        y_true = []
        y_pred = []
        for r in rows:
            snap = r["feature_snapshot_json"] or {}
            if not isinstance(snap, dict):
                continue
            y_true.append(int(r["label"]))
            vec = np.array([[float(snap.get(c, 0.0)) for c in feature_columns]])
            vec = np.nan_to_num(vec, nan=0.0)
            if scaler is not None:
                vec = scaler.transform(vec)
            pred = model.predict(vec)[0] if hasattr(model, "predict") else int(model.predict_proba(vec)[0][1] > 0.5)
            y_pred.append(int(pred))
        if len(y_true) < 10:
            return None
        from sklearn.metrics import f1_score
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return float(f1 - baseline_f1)
    try:
        return asyncio.run(_async_run())
    except Exception as e:
        logger.warning("Concept drift computation failed: %s", e)
        return None


async def _run_drift_check_async(machine_id: str) -> DriftReport:
    from database import get_db_context
    training = get_training_distribution(machine_id)
    live_list = get_prediction_distribution(machine_id)
    checked_at = datetime.now(timezone.utc).isoformat()
    data_drift_psi, drifted_features = _run_data_drift(machine_id, training, live_list)
    baseline_f1 = float((training.get("metrics") or {}).get("test_f1") or (training.get("metrics") or {}).get("train_f1") or 0.5)
    concept_drift_f1_delta = _run_concept_drift_sync(machine_id, baseline_f1)
    overall_status = "OK"
    recommended_action = "None"
    if data_drift_psi.get("max_psi", 0) > PSI_CRITICAL or (concept_drift_f1_delta is not None and concept_drift_f1_delta < -F1_DROP_CRITICAL):
        overall_status = "CRITICAL"
        recommended_action = "Retrain model" if concept_drift_f1_delta is not None and concept_drift_f1_delta < -F1_DROP_CRITICAL else "Review features"
    elif data_drift_psi.get("max_psi", 0) > PSI_WARNING or (concept_drift_f1_delta is not None and concept_drift_f1_delta < -F1_DROP_WARNING):
        overall_status = "WARNING"
        recommended_action = "Monitor"
    report = DriftReport(
        machine_id=machine_id,
        checked_at=checked_at,
        data_drift_psi=data_drift_psi or None,
        drifted_features=drifted_features,
        concept_drift_f1_delta=concept_drift_f1_delta,
        overall_status=overall_status,
        recommended_action=recommended_action,
    )
    async with get_db_context() as db:
        await db.execute(
            text("""
                INSERT INTO model_drift_reports
                (machine_id, checked_at, data_drift_psi, drifted_features, concept_drift_f1_delta, overall_status, recommended_action, report_json)
                VALUES (:mid, :checked_at, :data_drift_psi, :drifted_features, :f1_delta, :status, :action, :report_json)
            """),
            {
                "mid": machine_id,
                "checked_at": datetime.now(timezone.utc),
                "data_drift_psi": json.dumps(report.data_drift_psi) if report.data_drift_psi else None,
                "drifted_features": json.dumps(report.drifted_features),
                "f1_delta": report.concept_drift_f1_delta,
                "status": report.overall_status,
                "action": report.recommended_action,
                "report_json": report.model_dump_json(),
            },
        )
        await db.commit()
    if report.overall_status == "CRITICAL":
        try:
            from services.notification_service import notify_critical
            notify_critical("drift", f"Drift CRITICAL for {machine_id}: {report.recommended_action}", report.model_dump())
        except Exception as e:
            logger.warning("Drift notify_critical failed: %s", e)
        try:
            from services.retraining_service import trigger_retraining
            trigger_retraining(machine_id, "drift_critical", data_path=None)
        except Exception as e:
            logger.warning("Drift trigger_retraining failed: %s", e)
    return report


def run_drift_check(machine_id: str) -> DriftReport:
    """Sync entrypoint: run data + concept drift check, persist report, return DriftReport."""
    return asyncio.run(_run_drift_check_async(machine_id))


def schedule_weekly_check(machine_ids: Optional[list[str]] = None) -> list[DriftReport]:
    """Run drift check for each machine. If machine_ids is None, derive from Redis inference:buffer keys."""
    if machine_ids is not None:
        ids = list(machine_ids)
    else:
        try:
            import redis
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            keys = r.keys(f"{REDIS_BUFFER_PREFIX}*")
            ids = [k.replace(REDIS_BUFFER_PREFIX, "") for k in keys if k.startswith(REDIS_BUFFER_PREFIX)]
        except Exception as e:
            logger.warning("schedule_weekly_check: could not get machine list: %s", e)
            ids = []
    reports = []
    for mid in ids:
        try:
            reports.append(run_drift_check(mid))
        except Exception as e:
            logger.warning("Drift check failed for %s: %s", mid, e)
    return reports
