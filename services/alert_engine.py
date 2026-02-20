"""
Gaia Predictive — Alert Engine.

Tiered alerts with confirmation windows and hysteresis.
States: HEALTHY → WATCH → WARNING → CRITICAL → RECOVERING → HEALTHY.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from logger import get_logger

logger = get_logger(__name__)

# Default thresholds (OR: fp above OR health below)
DEFAULT_WATCH_FP = 0.40
DEFAULT_WARNING_FP = 0.65
DEFAULT_CRITICAL_FP = 0.85
DEFAULT_WATCH_HEALTH_MAX = 70
DEFAULT_WARNING_HEALTH_MAX = 55
DEFAULT_CRITICAL_HEALTH_MAX = 35
DEFAULT_CONFIRMATION = {"watch": 10, "warning": 25, "critical": 50}

STATES = ("HEALTHY", "WATCH", "WARNING", "CRITICAL", "RECOVERING")
TIER_ORDER = {"HEALTHY": 0, "WATCH": 1, "WARNING": 2, "CRITICAL": 3, "RECOVERING": 2}  # RECOVERING between WARNING and HEALTHY


class AlertEvent(BaseModel):
    tier: str
    machine_id: str
    state: str
    message: str
    timestamp: str
    prediction_snapshot: Optional[dict] = None


def _load_station_config() -> dict:
    """Load pipeline/station_config.json; return alert_thresholds or empty dict."""
    base = Path(__file__).resolve().parent.parent
    for path in [base / "pipeline" / "station_config.json", base / "station_config.json"]:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                    return data.get("alert_thresholds") or {}
            except Exception as e:
                logger.warning("Could not load station_config.json: %s", e)
    return {}


def _get_thresholds(machine_id: str) -> dict:
    """Per-machine thresholds (merge defaults with station_config overrides)."""
    overrides = _load_station_config().get(machine_id) or {}
    return {
        "watch_fp": float(overrides.get("watch_fp", DEFAULT_WATCH_FP)),
        "warning_fp": float(overrides.get("warning_fp", DEFAULT_WARNING_FP)),
        "critical_fp": float(overrides.get("critical_fp", DEFAULT_CRITICAL_FP)),
        "watch_health_max": int(overrides.get("watch_health_max", DEFAULT_WATCH_HEALTH_MAX)),
        "warning_health_max": int(overrides.get("warning_health_max", DEFAULT_WARNING_HEALTH_MAX)),
        "critical_health_max": int(overrides.get("critical_health_max", DEFAULT_CRITICAL_HEALTH_MAX)),
        "confirmation_cycles": {
            "watch": int(overrides.get("confirmation_cycles", {}).get("watch", DEFAULT_CONFIRMATION["watch"])),
            "warning": int(overrides.get("confirmation_cycles", {}).get("warning", DEFAULT_CONFIRMATION["warning"])),
            "critical": int(overrides.get("confirmation_cycles", {}).get("critical", DEFAULT_CONFIRMATION["critical"])),
        },
    }


def _effective_tier(prediction: dict, thresholds: dict) -> str:
    """Return highest tier satisfied by prediction (OR: fp above threshold OR health below max)."""
    fp = float(prediction.get("failure_probability", 0))
    health = float(prediction.get("health_score", 100))
    if fp > thresholds["critical_fp"] or health < thresholds["critical_health_max"]:
        return "CRITICAL"
    if fp > thresholds["warning_fp"] or health < thresholds["warning_health_max"]:
        return "WARNING"
    if fp > thresholds["watch_fp"] or health < thresholds["watch_health_max"]:
        return "WATCH"
    return "HEALTHY"


def _downgrade_ok(prediction: dict, thresholds: dict, from_state: str) -> bool:
    """True if reading is two tiers below from_state so we can downgrade (hysteresis)."""
    fp = float(prediction.get("failure_probability", 0))
    health = float(prediction.get("health_score", 100))
    if from_state == "CRITICAL":
        return fp < thresholds["warning_fp"] and health > thresholds["warning_health_max"]
    if from_state in ("WARNING", "RECOVERING"):
        return fp < thresholds["watch_fp"] and health > thresholds["watch_health_max"]
    return False


async def _get_current_state(db: AsyncSession, machine_id: str) -> Optional[dict]:
    r = await db.execute(
        text("SELECT machine_id, state, cycle_count, entered_at, acknowledged_at, acknowledged_by, notes FROM alert_current_state WHERE machine_id = :mid"),
        {"mid": machine_id},
    )
    row = r.mappings().first()
    return dict(row) if row else None


async def _upsert_state(
    db: AsyncSession,
    machine_id: str,
    state: str,
    cycle_count: int,
    acknowledged_at: Optional[datetime] = None,
    acknowledged_by: Optional[str] = None,
    notes: Optional[str] = None,
) -> None:
    now = datetime.now(timezone.utc)
    await db.execute(
        text("""
            INSERT INTO alert_current_state (machine_id, state, cycle_count, entered_at, acknowledged_at, acknowledged_by, notes)
            VALUES (:mid, :state, :cycle_count, :entered_at, :ack_at, :ack_by, :notes)
            ON CONFLICT (machine_id) DO UPDATE SET
                state = EXCLUDED.state,
                cycle_count = EXCLUDED.cycle_count,
                entered_at = EXCLUDED.entered_at,
                acknowledged_at = COALESCE(EXCLUDED.acknowledged_at, alert_current_state.acknowledged_at),
                acknowledged_by = COALESCE(EXCLUDED.acknowledged_by, alert_current_state.acknowledged_by),
                notes = COALESCE(EXCLUDED.notes, alert_current_state.notes)
        """),
        {
            "mid": machine_id,
            "state": state,
            "cycle_count": cycle_count,
            "entered_at": now,
            "ack_at": acknowledged_at,
            "ack_by": acknowledged_by,
            "notes": notes,
        },
    )


async def _insert_history(db: AsyncSession, machine_id: str, from_state: str, to_state: str, snapshot: Optional[dict]) -> None:
    await db.execute(
        text("""
            INSERT INTO alert_state_history (machine_id, from_state, to_state, at_timestamp, prediction_snapshot)
            VALUES (:mid, :from_s, :to_s, :ts, :snap)
        """),
        {
            "mid": machine_id,
            "from_s": from_state,
            "to_s": to_state,
            "ts": datetime.now(timezone.utc),
            "snap": json.dumps(snapshot) if snapshot else None,
        },
    )


async def _create_work_order_for_critical(db: AsyncSession, machine_id: str, message: str) -> None:
    """Create a work order when entering CRITICAL; sync to CMMS if configured."""
    try:
        from datetime import date
        year = datetime.now().year
        r = await db.execute(text("SELECT COUNT(*) FROM work_orders WHERE EXTRACT(YEAR FROM created_at) = :y"), {"y": year})
        count = (r.scalar() or 0) or 0
        wo_id = f"WO-{year}-{count + 1:04d}"
        await db.execute(
            text("""
                INSERT INTO work_orders (work_order_id, machine_id, title, description, priority, work_type, scheduled_date, estimated_duration_hours)
                VALUES (:wo_id, :machine_id, :title, :description, 'high', 'corrective', :sched, 4.0)
            """),
            {
                "wo_id": wo_id,
                "machine_id": machine_id,
                "title": f"CRITICAL alert: {machine_id}",
                "description": message,
                "sched": date.today(),
            },
        )
        logger.warning("Work order created for CRITICAL alert", work_order_id=wo_id, machine_id=machine_id)
        try:
            from services.cmms_client import sync_work_order_to_cmms
            status = "synced" if sync_work_order_to_cmms(wo_id) else "failed"
            await db.execute(
                text("UPDATE work_orders SET cmms_sync_status = :status WHERE work_order_id = :wo_id"),
                {"status": status, "wo_id": wo_id},
            )
        except Exception as cmms_e:
            logger.warning("CMMS sync failed for %s: %s", wo_id, cmms_e)
            try:
                await db.execute(
                    text("UPDATE work_orders SET cmms_sync_status = 'failed' WHERE work_order_id = :wo_id"),
                    {"wo_id": wo_id},
                )
            except Exception:
                pass
    except Exception as e:
        logger.warning("Could not create work order for CRITICAL: %s", e)


async def _process_async(machine_id: str, prediction: dict) -> Optional[AlertEvent]:
    from database import get_db_context
    thresholds = _get_thresholds(machine_id)
    effective = _effective_tier(prediction, thresholds)
    cc = thresholds["confirmation_cycles"]
    snapshot = {"failure_probability": prediction.get("failure_probability"), "health_score": prediction.get("health_score")}

    async with get_db_context() as db:
        cur = await _get_current_state(db, machine_id)
        if cur is None:
            cur = {"machine_id": machine_id, "state": "HEALTHY", "cycle_count": 0, "entered_at": None, "acknowledged_at": None, "acknowledged_by": None, "notes": None}

        prev_state = cur["state"]
        cycle_count = int(cur.get("cycle_count") or 0)
        new_state = prev_state
        event: Optional[AlertEvent] = None

        if effective == "HEALTHY":
            cycle_count = 0
            if prev_state == "WATCH" or (prev_state == "RECOVERING" and _downgrade_ok(prediction, thresholds, "WARNING")):
                new_state = "HEALTHY"
            elif _downgrade_ok(prediction, thresholds, prev_state):
                if prev_state == "CRITICAL":
                    new_state = "RECOVERING"
                elif prev_state in ("WARNING", "RECOVERING"):
                    new_state = "HEALTHY"
        else:
            if prev_state == "RECOVERING" and effective != "HEALTHY":
                new_state = effective
                cycle_count = 0
            else:
                tier_order_cur = TIER_ORDER.get(prev_state, 0)
                tier_order_eff = TIER_ORDER.get(effective, 0)
                if tier_order_eff > tier_order_cur:
                    cycle_count += 1
                    target = effective
                    required = cc.get(target, 999)
                    if cycle_count >= required:
                        new_state = target
                        cycle_count = 0
                        ts = datetime.now(timezone.utc).isoformat()
                        event = AlertEvent(
                            tier=target,
                            machine_id=machine_id,
                            state=new_state,
                            message=f"{target}: fp={prediction.get('failure_probability', 0):.2f} health={prediction.get('health_score', 100):.0f}",
                            timestamp=ts,
                            prediction_snapshot=snapshot,
                        )
                        if target == "CRITICAL":
                            await _create_work_order_for_critical(db, machine_id, event.message)
                            try:
                                from services.notification_service import notify_critical
                                notify_critical(
                                    "alert",
                                    event.message,
                                    {"machine_id": machine_id, "state": new_state, "message": event.message, "prediction_snapshot": snapshot},
                                )
                            except Exception as nerr:
                                logger.warning("notify_critical failed: %s", nerr)
                elif tier_order_eff == tier_order_cur and prev_state != "RECOVERING":
                    pass
                elif tier_order_eff < tier_order_cur:
                    cycle_count = 0
                    if _downgrade_ok(prediction, thresholds, prev_state):
                        if prev_state == "CRITICAL":
                            new_state = "RECOVERING"
                        elif prev_state in ("WARNING", "RECOVERING"):
                            new_state = "HEALTHY"

        if new_state != prev_state:
            await _insert_history(db, machine_id, prev_state, new_state, snapshot)
            logger.info("Alert state transition %s -> %s", prev_state, new_state, machine_id=machine_id)

        await _upsert_state(db, machine_id, new_state, cycle_count, cur.get("acknowledged_at"), cur.get("acknowledged_by"), cur.get("notes"))
        await db.commit()

    return event


def process(machine_id: str, prediction: dict) -> Optional[AlertEvent]:
    """Sync entrypoint: evaluate prediction, update state, return AlertEvent if new actionable alert."""
    try:
        return asyncio.run(_process_async(machine_id, prediction))
    except Exception as e:
        logger.exception("Alert engine process failed: %s", e)
        return None


def get_state(machine_id: str) -> Optional[dict]:
    """Return current alert state for machine_id from DB."""
    async def _get():
        from database import get_db_context
        async with get_db_context() as db:
            row = await _get_current_state(db, machine_id)
            if row and row.get("entered_at"):
                row["entered_at"] = row["entered_at"].isoformat() if hasattr(row["entered_at"], "isoformat") else str(row["entered_at"])
            if row and row.get("acknowledged_at"):
                row["acknowledged_at"] = row["acknowledged_at"].isoformat() if hasattr(row["acknowledged_at"], "isoformat") else str(row["acknowledged_at"])
            return row
    try:
        return asyncio.run(_get())
    except Exception as e:
        logger.warning("get_state failed: %s", e)
        return None


def get_active_alerts() -> list[dict]:
    """All machines not in HEALTHY state."""
    async def _get():
        from database import get_db_context
        async with get_db_context() as db:
            r = await db.execute(
                text("SELECT machine_id, state, cycle_count, entered_at, acknowledged_at, acknowledged_by, notes FROM alert_current_state WHERE state != 'HEALTHY' ORDER BY entered_at DESC")
            )
            rows = r.mappings().all()
            out = []
            for row in rows:
                d = dict(row)
                if d.get("entered_at"):
                    d["entered_at"] = d["entered_at"].isoformat() if hasattr(d["entered_at"], "isoformat") else str(d["entered_at"])
                if d.get("acknowledged_at"):
                    d["acknowledged_at"] = d["acknowledged_at"].isoformat() if hasattr(d["acknowledged_at"], "isoformat") else str(d["acknowledged_at"])
                out.append(d)
            return out
    try:
        return asyncio.run(_get())
    except Exception as e:
        logger.warning("get_active_alerts failed: %s", e)
        return []
