#!/usr/bin/env python3
"""
RUL Labeling Engine (Answer Key)
================================
Generates training targets (Remaining Useful Life) for the AI model.

Logic:
1. Load feature data.
2. Identify the last timestamp as "failure" (RUL = 0).
3. Calculate RUL for all preceding rows.
4. Merge RUL back into the dataset.
5. Check correlations to validate physics.

Usage:
    python labeling_engine.py
"""

import sys
import uuid
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "features.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "labeled_features.csv"

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("labeler")

# =============================================================================
# CORE LOGIC
# =============================================================================

def calculate_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Remaining Useful Life (RUL) in minutes.
    Assumes the dataset represents a single run-to-failure trajectory.
    """
    # Ensure timestamps are datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by time
    df = df.sort_values(by='timestamp')
    
    # Identify failure time (last timestamp)
    failure_time = df['timestamp'].iloc[-1]
    logger.info("Identified Failure Time: %s", failure_time)
    
    # Calculate RUL
    # RUL = Failure Time - Current Time
    df['rul_minutes'] = (failure_time - df['timestamp']).dt.total_seconds() / 60
    
    return df

def label_by_threshold(
    df: pd.DataFrame,
    rules: dict[str, float],
    anomaly_col: str = "anomaly",
) -> pd.DataFrame:
    """
    Weak-supervision labeling: set anomaly=1 where any threshold is exceeded.

    Use when confirmed failure labels are not yet available. Example rules:
        {"torque_deviation_zscore": 2.5, "cycle_time_drift_pct": 0.05}
    means: label as anomaly if |torque_deviation_zscore| > 2.5 or
    |cycle_time_drift_pct| > 0.05.

    Args:
        df: Feature DataFrame (must contain rule column names).
        rules: Dict of column_name -> threshold (exceeded if |value| > threshold).
        anomaly_col: Name of the output binary label column.

    Returns:
        DataFrame with anomaly_col added (0 or 1).
    """
    out = df.copy()
    out[anomaly_col] = 0
    for col, threshold in rules.items():
        if col not in out.columns:
            logger.warning("Threshold rule column missing, skipping: %s", col)
            continue
        # Exceeded if absolute value > threshold
        exceeded = out[col].abs() > threshold
        out.loc[exceeded, anomaly_col] = 1
        logger.info("Labeled %d rows as anomaly where |%s| > %s", exceeded.sum(), col, threshold)
    return out


# =============================================================================
# HUMAN-IN-THE-LOOP (DB-backed)
# =============================================================================

async def create_labeling_task(db: AsyncSession, anomaly_event_id: str) -> str:
    """
    Create a pending label request from an anomaly detection event.
    Looks up anomaly_detections by detection_id; inserts into labeling_tasks.
    Returns new task_id (UUID string).
    """
    result = await db.execute(
        text("""
            SELECT detection_id, machine_id, timestamp, anomalous_features
            FROM anomaly_detections
            WHERE detection_id = :aid
        """),
        {"aid": anomaly_event_id},
    )
    row = result.mappings().first()
    if not row:
        raise ValueError(f"Anomaly event not found: {anomaly_event_id}")
    raw_features = row["anomalous_features"]
    feature_snapshot = raw_features if isinstance(raw_features, dict) else {}
    task_id = str(uuid.uuid4())
    await db.execute(
        text("""
            INSERT INTO labeling_tasks (task_id, anomaly_event_id, machine_id, feature_snapshot_json, created_at, status)
            VALUES (:task_id, :anomaly_event_id, :machine_id, :feature_snapshot_json, :created_at, 'pending')
        """),
        {
            "task_id": task_id,
            "anomaly_event_id": anomaly_event_id,
            "machine_id": row["machine_id"],
            "feature_snapshot_json": feature_snapshot,
            "created_at": datetime.now(timezone.utc),
        },
    )
    logger.info("Created labeling task %s for anomaly %s", task_id, anomaly_event_id)
    return task_id


async def create_labeling_task_from_alarm(db: AsyncSession, alarm_id: str) -> str:
    """
    Create (or reuse) a labeling task linked to an alarm. Looks up pdm_alarms for machine_id,
    uses anomaly_event_id = f"alarm:{alarm_id}", inserts into labeling_tasks if not present.
    Returns task_id.
    """
    result = await db.execute(
        text("SELECT machine_id FROM pdm_alarms WHERE alarm_id = :aid"),
        {"aid": alarm_id},
    )
    row = result.mappings().first()
    if not row:
        raise ValueError(f"Alarm not found: {alarm_id}")
    machine_id = row["machine_id"]
    anomaly_event_id = f"alarm:{alarm_id}"
    check = await db.execute(
        text("SELECT task_id FROM labeling_tasks WHERE anomaly_event_id = :aid"),
        {"aid": anomaly_event_id},
    )
    existing = check.mappings().first()
    if existing:
        return existing["task_id"]
    task_id = str(uuid.uuid4())
    await db.execute(
        text("""
            INSERT INTO labeling_tasks (task_id, anomaly_event_id, machine_id, feature_snapshot_json, created_at, status)
            VALUES (:task_id, :anomaly_event_id, :machine_id, :feature_snapshot_json, :created_at, 'pending')
        """),
        {
            "task_id": task_id,
            "anomaly_event_id": anomaly_event_id,
            "machine_id": machine_id,
            "feature_snapshot_json": {},
            "created_at": datetime.now(timezone.utc),
        },
    )
    logger.info("Created labeling task %s from alarm %s", task_id, alarm_id)
    return task_id


async def submit_label(
    db: AsyncSession,
    task_id: str,
    label: int,
    submitted_by: str,
    notes: Optional[str] = None,
) -> dict:
    """
    Record human feedback: insert into labels, set task status=completed.
    Returns {"success": True, "machine_id": str} for milestone check.
    """
    result = await db.execute(
        text("SELECT task_id, machine_id FROM labeling_tasks WHERE task_id = :tid"),
        {"tid": task_id},
    )
    row = result.mappings().first()
    if not row:
        raise ValueError(f"Task not found: {task_id}")
    machine_id = row["machine_id"]
    await db.execute(
        text("""
            INSERT INTO labels (task_id, label, submitted_by, notes, created_at)
            VALUES (:task_id, :label, :submitted_by, :notes, :created_at)
        """),
        {
            "task_id": task_id,
            "label": label,
            "submitted_by": submitted_by,
            "notes": notes or "",
            "created_at": datetime.now(timezone.utc),
        },
    )
    await db.execute(
        text("UPDATE labeling_tasks SET status = 'completed' WHERE task_id = :tid"),
        {"tid": task_id},
    )
    logger.info("Submitted label %s for task %s by %s", label, task_id, submitted_by)
    return {"success": True, "machine_id": machine_id}


async def get_labeled_dataset(
    db: AsyncSession,
    machine_id: str,
    min_labels: int = 50,
) -> Optional[pd.DataFrame]:
    """
    Return labeled training set when enough labels exist for the machine.
    Joins labeling_tasks (status=completed) + labels; flattens feature_snapshot_json into columns.
    Returns None if count < min_labels; else DataFrame with feature columns + 'label'.
    """
    result = await db.execute(
        text("""
            SELECT lt.task_id, lt.feature_snapshot_json, l.label
            FROM labeling_tasks lt
            JOIN labels l ON l.task_id = lt.task_id
            WHERE lt.machine_id = :mid AND lt.status = 'completed'
            ORDER BY lt.created_at
        """),
        {"mid": machine_id},
    )
    rows = result.mappings().all()
    if len(rows) < min_labels:
        return None
    records = []
    for r in rows:
        snap = r["feature_snapshot_json"] or {}
        if not isinstance(snap, dict):
            continue
        rec = dict(snap)
        rec["label"] = int(r["label"])
        records.append(rec)
    if not records:
        return None
    return pd.DataFrame(records)


async def label_coverage_report(db: AsyncSession, machine_id: str) -> dict:
    """
    Return { total_anomalies, labeled, confirmed, rejected, coverage_pct } for the machine.
    """
    # Total tasks for this machine (anomalies that have a labeling task)
    total_result = await db.execute(
        text("SELECT COUNT(*) AS n FROM labeling_tasks WHERE machine_id = :mid"),
        {"mid": machine_id},
    )
    total_anomalies = (total_result.scalar() or 0) or 0
    # Labeled = tasks with at least one label
    labeled_result = await db.execute(
        text("""
            SELECT COUNT(DISTINCT lt.task_id) AS n
            FROM labeling_tasks lt
            JOIN labels l ON l.task_id = lt.task_id
            WHERE lt.machine_id = :mid
        """),
        {"mid": machine_id},
    )
    labeled = (labeled_result.scalar() or 0) or 0
    # Confirmed (label=1) and rejected (label=0)
    conf_result = await db.execute(
        text("""
            SELECT l.label, COUNT(*) AS n
            FROM labeling_tasks lt
            JOIN labels l ON l.task_id = lt.task_id
            WHERE lt.machine_id = :mid
            GROUP BY l.label
        """),
        {"mid": machine_id},
    )
    confirmed = rejected = 0
    for row in conf_result.mappings().all():
        if row["label"] == 1:
            confirmed = row["n"]
        else:
            rejected = row["n"]
    coverage_pct = (labeled / total_anomalies * 100) if total_anomalies else 0.0
    return {
        "total_anomalies": total_anomalies,
        "labeled": labeled,
        "confirmed": confirmed,
        "rejected": rejected,
        "coverage_pct": round(coverage_pct, 2),
    }


def validate_correlations(df: pd.DataFrame):
    """
    Checks if features correlate with RUL.
    Expectation: As RUL decreases, degradation features (RMS, Kurtosis) should INCREASE.
    This implies a NEGATIVE correlation.
    """
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=['number'])
    
    if 'rul_minutes' not in numeric_df.columns:
        logger.warning("RUL column missing, cannot calculate correlations.")
        return

    correlations = numeric_df.corr()['rul_minutes'].sort_values()
    
    logger.info("--- Feature vs RUL Correlations ---")
    logger.info("(Negative values indicate feature rises as RUL drops -> Good predictor)")
    for feature, corr in correlations.items():
        if feature != 'rul_minutes':
            logger.info(f"{feature.ljust(15)}: {corr:.4f}")

# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    if not INPUT_FILE.exists():
        logger.error("Input file not found: %s", INPUT_FILE)
        sys.exit(1)

    logger.info("Loading features from %s...", INPUT_FILE)
    df = pd.read_csv(INPUT_FILE)
    
    if df.empty:
        logger.error("Input file is empty.")
        sys.exit(1)

    # 1. Calculate RUL
    labeled_df = calculate_rul(df)
    
    # 2. Validate
    validate_correlations(labeled_df)
    
    # 3. Save
    labeled_df.to_csv(OUTPUT_FILE, index=False)
    logger.info("Labeled data saved to %s", OUTPUT_FILE)
    logger.info("Rows processed: %d", len(labeled_df))
