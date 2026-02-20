"""
Automated retraining with shadow deployment: trigger training, save to Staging, notify for human-gated promotion.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from logger import get_logger

logger = get_logger(__name__)

REASONS = ("drift_critical", "label_milestone_50", "label_milestone_100", "label_milestone_200", "label_milestone_500", "scheduled_weekly")
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"
STAGING_META_PATH = MODELS_DIR / "gaia_model_staging_metadata.json"
LATEST_META_PATH = MODELS_DIR / "gaia_model_latest_metadata.json"


def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {"production_run_id": None, "staging_run_id": None, "staging_metrics": None, "previous_production_run_id": None}
    try:
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not load model registry: %s", e)
        return {"production_run_id": None, "staging_run_id": None, "staging_metrics": None, "previous_production_run_id": None}


def _save_registry(reg: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)


def trigger_retraining(machine_id: str, reason: str, data_path: Optional[str] = None) -> None:
    """
    Schedule retraining as a background job. Does not block.
    If data_path is None, exports labeled data for machine_id then runs training with --staging.
    """
    if reason not in REASONS:
        logger.warning("Unknown trigger reason %s", reason)
    logger.info("Triggering retraining for machine_id=%s reason=%s", machine_id, reason)
    import threading
    thread = threading.Thread(target=_run_retrain_background, args=(machine_id, reason, data_path), daemon=True)
    thread.start()


def _run_retrain_background(machine_id: str, reason: str, data_path: Optional[str]) -> None:
    try:
        asyncio.run(_export_and_train_async(machine_id, reason, data_path))
    except Exception as e:
        logger.exception("Retraining failed: %s", e)


async def _export_and_train_async(machine_id: str, reason: str, data_path: Optional[str]) -> None:
    csv_path = data_path
    if not csv_path:
        from database import get_db_context
        from labeling_engine import get_labeled_dataset
        async with get_db_context() as db:
            df = await get_labeled_dataset(db, machine_id, min_labels=50)
        if df is None or len(df) < 50:
            logger.warning("Not enough labeled data for %s (need 50); skipping retrain", machine_id)
            return
        out_dir = BASE_DIR / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        csv_path = str(out_dir / f"labeled_export_{machine_id}_{ts}.csv")
        df.to_csv(csv_path, index=False)
    _run_train_subprocess(csv_path, machine_id, reason)


def _run_train_subprocess(csv_path: str, machine_id: str, reason: str) -> None:
    cmd = [
        sys.executable, str(BASE_DIR / "train_model.py"),
        "--data-source", "human_labels",
        "--data-path", csv_path,
        "--staging",
    ]
    logger.info("Running training subprocess: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        logger.error("Training failed: stderr=%s", proc.stderr)
        return
    # Read staging metadata for run_id and metrics
    new_f1 = None
    run_id = None
    if STAGING_META_PATH.exists():
        try:
            with open(STAGING_META_PATH) as f:
                meta = json.load(f)
            new_f1 = meta.get("metrics", {}).get("test_f1")
            run_id = meta.get("run_id")
        except Exception as e:
            logger.warning("Could not read staging metadata: %s", e)
    current_f1 = None
    if LATEST_META_PATH.exists():
        try:
            with open(LATEST_META_PATH) as f:
                current_f1 = json.load(f).get("metrics", {}).get("test_f1")
        except Exception:
            pass
    reg = _load_registry()
    reg["staging_run_id"] = run_id
    reg["staging_metrics"] = {"test_f1": new_f1} if new_f1 is not None else None
    _save_registry(reg)
    try:
        from services.notification_service import notify_critical
        msg = f"New model ready for review: {run_id} — F1 {new_f1} vs current {current_f1}"
        notify_critical("retraining", msg, {"machine_id": machine_id, "reason": reason, "run_id": run_id, "new_f1": new_f1, "current_f1": current_f1})
    except Exception as e:
        logger.warning("Notification failed: %s", e)
    logger.info("Staging model saved. run_id=%s new_f1=%s current_f1=%s", run_id, new_f1, current_f1)


def promote_staging_to_production(run_id: str) -> bool:
    """Copy staging model to production (gaia_model_latest, gaia_model_calibrated). Returns True on success."""
    import shutil
    staging_pkl = MODELS_DIR / "gaia_model_staging.pkl"
    staging_meta = MODELS_DIR / "gaia_model_staging_metadata.json"
    if not staging_pkl.exists() or not staging_meta.exists():
        logger.warning("Staging model not found")
        return False
    reg = _load_registry()
    if reg.get("staging_run_id") != run_id:
        logger.warning("run_id %s does not match current staging %s", run_id, reg.get("staging_run_id"))
    latest_pkl = MODELS_DIR / "gaia_model_latest.pkl"
    latest_meta = MODELS_DIR / "gaia_model_latest_metadata.json"
    cal_pkl = MODELS_DIR / "gaia_model_calibrated.pkl"
    # Backup current production
    if latest_pkl.exists():
        backup_pkl = MODELS_DIR / "gaia_model_previous_production.pkl"
        shutil.copy2(latest_pkl, backup_pkl)
    if latest_meta.exists():
        with open(latest_meta) as f:
            old_meta = json.load(f)
        reg["previous_production_run_id"] = old_meta.get("run_id")
    # Promote staging to production
    shutil.copy2(staging_pkl, latest_pkl)
    shutil.copy2(staging_meta, latest_meta)
    if cal_pkl.exists() or staging_pkl.exists():
        shutil.copy2(staging_pkl, cal_pkl)
    reg["production_run_id"] = run_id
    reg["staging_run_id"] = None
    reg["staging_metrics"] = None
    _save_registry(reg)
    logger.info("Promoted staging run_id=%s to production", run_id)
    return True


def rollback_production() -> bool:
    """Revert production to previous version. Returns True if rollback was performed."""
    import shutil
    prev_pkl = MODELS_DIR / "gaia_model_previous_production.pkl"
    if not prev_pkl.exists():
        logger.warning("No previous production backup found")
        return False
    reg = _load_registry()
    latest_pkl = MODELS_DIR / "gaia_model_latest.pkl"
    latest_meta = MODELS_DIR / "gaia_model_latest_metadata.json"
    cal_pkl = MODELS_DIR / "gaia_model_calibrated.pkl"
    shutil.copy2(prev_pkl, latest_pkl)
    # Restore latest_meta from previous if we have it; else leave as-is
    reg["production_run_id"] = reg.get("previous_production_run_id")
    reg["previous_production_run_id"] = None
    _save_registry(reg)
    if cal_pkl.exists():
        shutil.copy2(prev_pkl, cal_pkl)
    logger.info("Rolled back production to previous version")
    return True


def get_staging_info() -> Optional[dict]:
    """Return current staging run_id and metrics if any."""
    reg = _load_registry()
    if not reg.get("staging_run_id"):
        return None
    return {"run_id": reg["staging_run_id"], "metrics": reg.get("staging_metrics")}
