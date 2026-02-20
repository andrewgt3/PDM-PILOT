"""
Prefect flows for scheduled and event-triggered maintenance jobs.

Flows: weekly drift check, new-machine onboarding, scheduled retraining, shadow evaluation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from prefect import flow, task

# Project root for station_config and imports
_BASE = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)
if str(_BASE) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_BASE))


def _station_config_path() -> Path:
    return Path(os.getenv("STATION_CONFIG_PATH", _BASE / "pipeline" / "station_config.json"))


def _get_machine_ids() -> list[str]:
    """Load machine IDs from station_config.json (node_mappings keys)."""
    path = _station_config_path()
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        node_mappings = data.get("node_mappings") or {}
        return list(node_mappings.keys())
    except Exception:
        return []


# -----------------------------------------------------------------------------
# Weekly drift check
# -----------------------------------------------------------------------------


@flow(name="weekly-drift-check")
def weekly_drift_check_flow() -> dict[str, Any]:
    """
    For each machine in station_config, run drift check; on CRITICAL trigger retraining.
    Saves summary to a JSON file (no DB table in first iteration).
    """
    machine_ids = _get_machine_ids()
    if not machine_ids:
        return {"critical_count": 0, "reports": [], "summary_path": None}

    reports: list[dict[str, Any]] = []
    critical_machines: list[str] = []

    for machine_id in machine_ids:
        try:
            report = _run_drift_check_task(machine_id)
            reports.append(report)
            if report.get("overall_status") == "CRITICAL":
                critical_machines.append(machine_id)
                _trigger_retraining_task(machine_id, "drift_critical")
        except Exception as e:
            reports.append({"machine_id": machine_id, "error": str(e)})

    summary = {
        "scheduled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "machine_ids": machine_ids,
        "reports": reports,
        "critical_count": len(critical_machines),
        "critical_machines": critical_machines,
    }
    out_path = _BASE / "data" / "drift_check_summaries"
    out_path.mkdir(parents=True, exist_ok=True)
    summary_file = out_path / f"weekly_{summary['scheduled_at'].replace(':', '-').replace(' ', '_')}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    summary["summary_path"] = str(summary_file)
    return summary


@task
def _run_drift_check_task(machine_id: str) -> dict[str, Any]:
    from services.drift_monitor_service import run_drift_check
    report = run_drift_check(machine_id)
    return report.model_dump() if hasattr(report, "model_dump") else dict(report)


@task
def _trigger_retraining_task(machine_id: str, reason: str) -> None:
    import subprocess
    import sys
    from config import get_settings
    settings = get_settings()
    if getattr(settings, "federated", None) and getattr(settings.federated, "fl_enabled", False):
        server_addr = settings.federated.fl_server_address
        cmd = [
            sys.executable, "-m", "federated.fl_client",
            "--server-address", server_addr,
            "--machine-id", machine_id,
        ]
        try:
            subprocess.run(cmd, cwd=str(_BASE), timeout=3600, check=False)
        except Exception as e:
            logger.warning("FL client run failed for %s: %s", machine_id, e)
        return
    from services.retraining_service import trigger_retraining
    trigger_retraining(machine_id, reason, data_path=None)


# -----------------------------------------------------------------------------
# New-machine onboarding
# -----------------------------------------------------------------------------

# Baseline feature columns (must match DataProfilerService.PROFILE_METRICS)
_BASELINE_METRICS = ["rotational_speed", "temperature", "torque", "tool_wear"]

POLL_INTERVAL_AWAIT_DATA = 300  # 5 minutes
TIMEOUT_AWAIT_DATA_SECONDS = 24 * 3600  # 24 hours
MIN_ROWS_ONE_HOUR = 60
BASELINE_HOURS = 48


def _update_onboarding_status(
    machine_id: str,
    current_step: Optional[str] = None,
    status: Optional[str] = None,
    error_message: Optional[str] = None,
    model_id: Optional[str] = None,
    completed_at: Optional[Any] = None,
) -> None:
    from services.onboarding_helpers import update_onboarding_status
    update_onboarding_status(
        machine_id=machine_id,
        current_step=current_step,
        status=status,
        error_message=error_message,
        model_id=model_id,
        completed_at=completed_at,
    )


@flow(name="new-machine-onboarding")
def onboarding_flow(machine_id: str) -> dict[str, Any]:
    """
    Onboard a new machine: await 1hr data, data quality check, 48h baseline, bootstrap IF model, activate, notify.
    """
    _update_onboarding_status(machine_id, current_step="await_minimum_data", status="IN_PROGRESS")
    try:
        await_minimum_data(machine_id, min_hours=1)
    except (TimeoutError, Exception) as e:
        err_msg = str(e) if "Timeout" in str(e) or "STALLED" in str(e) else f"await_minimum_data failed: {e}"
        _update_onboarding_status(
            machine_id,
            current_step="await_minimum_data",
            status="STALLED",
            error_message=err_msg[:500],
        )
        return {"machine_id": machine_id, "status": "STALLED", "error": err_msg}

    _update_onboarding_status(machine_id, current_step="run_data_quality_check", status="IN_PROGRESS")
    try:
        profile_result = run_data_quality_check(machine_id)
    except Exception as e:
        return {"machine_id": machine_id, "status": "PAUSED", "error": str(e)}

    _update_onboarding_status(machine_id, current_step="compute_healthy_baseline", status="IN_PROGRESS")
    baseline_stats = compute_healthy_baseline(machine_id, hours=BASELINE_HOURS)

    _update_onboarding_status(machine_id, current_step="train_bootstrap_model", status="IN_PROGRESS")
    model_id = train_bootstrap_model(machine_id, baseline_stats)

    _update_onboarding_status(machine_id, current_step="activate_model", status="IN_PROGRESS")
    activate_model(machine_id, model_id)

    send_onboarding_complete_notification(machine_id)
    return {"machine_id": machine_id, "status": "complete", "model_id": model_id}


@task(retries=3, retry_delay_seconds=300)
def await_minimum_data(machine_id: str, min_hours: int = 1) -> None:
    """Poll every 5 minutes until machine has at least 1 hour of data; timeout 24h then mark STALLED."""
    min_rows = MIN_ROWS_ONE_HOUR
    poll_interval = POLL_INTERVAL_AWAIT_DATA
    deadline = time.monotonic() + TIMEOUT_AWAIT_DATA_SECONDS
    from config import get_settings
    from services.onboarding_helpers import get_connection, update_onboarding_status
    s = get_settings().database
    import psycopg2
    conn_str = (
        f"host={s.host} port={s.port} dbname={s.name} user={s.user} "
        f"password={s.password.get_secret_value()} connect_timeout=5"
    )
    while time.monotonic() < deadline:
        try:
            conn = psycopg2.connect(conn_str)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT COUNT(*) FROM sensor_readings
                WHERE machine_id = %s AND timestamp >= NOW() - INTERVAL '1 hour'
                """,
                (machine_id,),
            )
            n = cur.fetchone()[0]
            cur.close()
            conn.close()
            if n >= min_rows:
                return
        except Exception:
            pass
        time.sleep(poll_interval)
    update_onboarding_status(
        machine_id,
        current_step="await_minimum_data",
        status="STALLED",
        error_message="Timeout: insufficient data in 24h",
    )
    raise TimeoutError(f"await_minimum_data: machine_id={machine_id} did not reach {min_rows} rows within 24h")


@task
def run_data_quality_check(machine_id: str) -> dict[str, Any]:
    """Run DataProfilerService.profile(machine_id, hours=1). If CRITICAL: notify, set PAUSED, raise."""
    from database import get_db_context
    from services.data_profiler_service import DataProfilerService
    from services.notification_service import notify_critical

    async def _profile() -> Any:
        async with get_db_context() as db:
            svc = DataProfilerService(db)
            return await svc.profile(machine_id, hours=1)

    profile = asyncio.run(_profile())
    overall = getattr(profile, "overall_status", None) or (profile.get("overall_status") if isinstance(profile, dict) else None)
    metrics_list = getattr(profile, "metrics", None) or (profile.get("metrics") or [])
    if overall == "CRITICAL":
        msg = f"Data quality CRITICAL for machine {machine_id}; onboarding paused"
        notify_critical("onboarding", msg, {"machine_id": machine_id, "overall_status": overall})
        _update_onboarding_status(machine_id, status="PAUSED", error_message=msg)
        raise RuntimeError(msg)
    for m in metrics_list:
        status = getattr(m, "status", None) or (m.get("status") if isinstance(m, dict) else None)
        if status == "CRITICAL":
            msg = f"Data quality CRITICAL for machine {machine_id} (metric {getattr(m, 'metric_name', m.get('metric_name', '?'))}); onboarding paused"
            notify_critical("onboarding", msg, {"machine_id": machine_id})
            _update_onboarding_status(machine_id, status="PAUSED", error_message=msg)
            raise RuntimeError(msg)
    return profile.model_dump() if hasattr(profile, "model_dump") else (profile if isinstance(profile, dict) else {})


@task
def compute_healthy_baseline(machine_id: str, hours: int = 48) -> dict[str, Any]:
    """Query last 48h from sensor_readings; compute mean, std, p5, p95 per metric; write machine_baseline_stats."""
    from services.onboarding_helpers import get_connection
    import numpy as np

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT rotational_speed, temperature, torque, tool_wear
            FROM sensor_readings
            WHERE machine_id = %s AND timestamp >= NOW() - (%s * INTERVAL '1 hour')
            ORDER BY timestamp
            """,
            (machine_id, hours),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise RuntimeError(f"compute_healthy_baseline: no sensor_readings for {machine_id} in last {hours}h")
    arr = np.array(rows, dtype=float)
    arr = np.nan_to_num(arr, nan=np.nan)
    baseline_stats = {}
    for i, name in enumerate(_BASELINE_METRICS):
        col = arr[:, i]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            continue
        mean = float(np.mean(col))
        std = float(np.std(col))
        if np.isnan(std):
            std = 0.0
        p5 = float(np.percentile(col, 5))
        p95 = float(np.percentile(col, 95))
        baseline_stats[name] = {"mean": mean, "std": std, "p5": p5, "p95": p95}

    conn = get_connection()
    try:
        cur = conn.cursor()
        for metric_name, stats in baseline_stats.items():
            cur.execute(
                """
                INSERT INTO machine_baseline_stats (machine_id, metric_name, mean, std, p5, p95)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (machine_id, metric_name) DO UPDATE SET
                    mean = EXCLUDED.mean, std = EXCLUDED.std, p5 = EXCLUDED.p5, p95 = EXCLUDED.p95,
                    computed_at = NOW()
                """,
                (machine_id, metric_name, stats["mean"], stats["std"], stats["p5"], stats["p95"]),
            )
        conn.commit()
        cur.close()
    finally:
        conn.close()
    return baseline_stats


@task
def train_bootstrap_model(machine_id: str, baseline_stats: dict[str, Any]) -> str:
    """Train Isolation Forest on 48h baseline data; register in MLflow as bootstrap for machine_id; return model_id."""
    import pandas as pd
    from services.onboarding_helpers import get_connection
    from anomaly_discovery.detectors.isolation_forest import IsolationForestDetector

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT rotational_speed, temperature, torque, tool_wear
            FROM sensor_readings
            WHERE machine_id = %s AND timestamp >= NOW() - (%s * INTERVAL '1 hour')
            ORDER BY timestamp
            """,
            (machine_id, BASELINE_HOURS),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise RuntimeError(f"train_bootstrap_model: no sensor_readings for {machine_id} in last {BASELINE_HOURS}h")
    df = pd.DataFrame(rows, columns=_BASELINE_METRICS)
    detector = IsolationForestDetector(contamination=0.01)
    detector.fit(df, feature_columns=_BASELINE_METRICS)

    model_id: str
    try:
        import mlflow
        import mlflow.sklearn
        mlflow.set_experiment("onboarding_bootstrap")
        with mlflow.start_run(tags={"machine_id": machine_id}):
            mlflow.sklearn.log_model(detector.model, "model", registered_model_name=f"bootstrap_{machine_id}")
            model_id = mlflow.active_run().info.run_id
    except Exception as e:
        logger.warning("MLflow logging failed for bootstrap model: %s", e)
        import uuid
        model_id = str(uuid.uuid4())
    return model_id


@task
def activate_model(machine_id: str, model_id: str) -> None:
    """Store model_id in onboarding_status so inference/dashboard can resolve model for this machine."""
    _update_onboarding_status(machine_id, model_id=model_id)


@task
def send_onboarding_complete_notification(machine_id: str) -> None:
    """Notify operator and set onboarding_status to COMPLETE."""
    from services.notification_service import notify_critical
    from datetime import datetime, timezone
    notify_critical(
        "onboarding",
        f"Machine {machine_id} is now live. Health monitoring active.",
        {"machine_id": machine_id},
    )
    _update_onboarding_status(
        machine_id,
        current_step="send_onboarding_complete_notification",
        status="COMPLETE",
        error_message="",
        completed_at=datetime.now(timezone.utc),
    )


# -----------------------------------------------------------------------------
# Scheduled retraining
# -----------------------------------------------------------------------------


@flow(name="scheduled-retraining")
def weekly_retraining_flow(machine_id: str) -> dict[str, Any]:
    """If label coverage >= 50, trigger retraining with reason scheduled_weekly."""
    coverage = _get_label_coverage_task(machine_id)
    labeled = coverage.get("labeled") or 0
    if labeled >= 50:
        _trigger_retraining_task(machine_id, "scheduled_weekly")
        return {"machine_id": machine_id, "triggered": True, "labeled": labeled}
    return {"machine_id": machine_id, "triggered": False, "labeled": labeled, "reason": "insufficient labels"}


@flow(name="scheduled-retraining-all")
def weekly_retraining_all_flow() -> list[dict[str, Any]]:
    """Run scheduled retraining for every machine in station_config (used by Sunday 02:00 deployment)."""
    machine_ids = _get_machine_ids()
    results = []
    for machine_id in machine_ids:
        try:
            results.append(weekly_retraining_flow(machine_id))
        except Exception as e:
            results.append({"machine_id": machine_id, "error": str(e)})
    return results


@task
def _get_label_coverage_task(machine_id: str) -> dict[str, Any]:
    from database import get_db_context
    from labeling_engine import label_coverage_report
    async def _run():
        async with get_db_context() as db:
            return await label_coverage_report(db, machine_id)
    return asyncio.run(_run())


# -----------------------------------------------------------------------------
# Shadow evaluation
# -----------------------------------------------------------------------------


@flow(name="model-shadow-evaluation")
def shadow_evaluation_flow(machine_id: str) -> dict[str, Any]:
    """
    After 24h shadow: fetch shadow report; if agreement_rate >= 0.95 and low divergence,
    send promotion notification.
    """
    report = _fetch_shadow_report_task(machine_id)
    total = report.get("total_count") or 0
    if total == 0:
        return {"machine_id": machine_id, "action": "none", "reason": "no shadow data"}

    agreement = report.get("agreement_rate")
    div_staging_crit = report.get("divergence_staging_critical_production_healthy") or 0
    div_prod_crit = report.get("divergence_production_critical_staging_healthy") or 0
    div_pct = (div_staging_crit + div_prod_crit) / total if total else 1.0

    if agreement is not None and agreement >= 0.95 and div_pct < 0.05:
        _notify_promotion_ready_task(machine_id, report)
        return {"machine_id": machine_id, "action": "notified", "agreement_rate": agreement, "div_pct": div_pct}
    return {"machine_id": machine_id, "action": "none", "agreement_rate": agreement, "div_pct": div_pct}


@task
def _fetch_shadow_report_task(machine_id: str) -> dict[str, Any]:
    base = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
    url = f"{base}/api/models/shadow-report/{machine_id}"
    with httpx.Client(timeout=15.0) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.json()


@task
def _notify_promotion_ready_task(machine_id: str, report: dict[str, Any]) -> None:
    from services.notification_service import notify_critical
    notify_critical(
        "shadow_evaluation",
        f"Staging ready for promotion review: machine_id={machine_id}, agreement_rate={report.get('agreement_rate')}",
        {"machine_id": machine_id, "report": report},
    )
