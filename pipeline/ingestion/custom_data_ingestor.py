#!/usr/bin/env python3
"""
Custom Data Ingestor — ABB robot & Siemens PLC data into normalized DataFrame.

Accepts:
  1. CSV export from TimescaleDB cwru_features (existing schema).
  2. Simple long-format CSV: timestamp, machine_id, metric_name, value.

Normalizes both to a standard schema for feature extraction:
  timestamp, machine_id, joint_torque_nm, joint_speed_rpm, motor_current_a,
  cycle_time_ms, vibration_rms, temperature_c

Missing columns are filled with NaN and logged.
"""

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

# Standard output columns (robot/PLC telemetry for ABB + Siemens)
STANDARD_COLUMNS = [
    "timestamp",
    "machine_id",
    "joint_torque_nm",
    "joint_speed_rpm",
    "motor_current_a",
    "cycle_time_ms",
    "vibration_rms",
    "temperature_c",
]

# Required for downstream (at least timestamp + machine_id)
REQUIRED_COLUMNS = ["timestamp", "machine_id"]

# cwru_features column -> standard column mapping
CWRU_TO_STANDARD = {
    "timestamp": "timestamp",
    "machine_id": "machine_id",
    "torque": "joint_torque_nm",
    "rotational_speed": "joint_speed_rpm",
    "temperature": "temperature_c",
    # cwru has no motor_current, cycle_time_ms, vibration_rms; leave as NaN
}

# Simple format: metric_name values that map to standard columns
METRIC_TO_STANDARD = {
    "joint_torque_nm": "joint_torque_nm",
    "torque": "joint_torque_nm",
    "joint_speed_rpm": "joint_speed_rpm",
    "rotational_speed": "joint_speed_rpm",
    "speed": "joint_speed_rpm",
    "motor_current_a": "motor_current_a",
    "motor_current": "motor_current_a",
    "current": "motor_current_a",
    "cycle_time_ms": "cycle_time_ms",
    "cycle_timer_ms": "cycle_time_ms",
    "cycle_time": "cycle_time_ms",
    "vibration_rms": "vibration_rms",
    "vibration": "vibration_rms",
    "temperature_c": "temperature_c",
    "temperature": "temperature_c",
}


def _detect_format(df: pd.DataFrame) -> Literal["cwru", "simple", "unknown"]:
    """Detect CSV format from column headers."""
    cols = set(df.columns.str.strip().str.lower())
    # Simple format: exactly timestamp, machine_id, metric_name, value (or similar)
    if cols >= {"timestamp", "machine_id", "metric_name", "value"}:
        return "simple"
    # cwru_features export: has timestamp, machine_id, torque, rotational_speed, temperature, etc.
    if "timestamp" in cols and "machine_id" in cols and ("torque" in cols or "rotational_speed" in cols):
        return "cwru"
    return "unknown"


def _ingest_cwru(df: pd.DataFrame) -> pd.DataFrame:
    """Map cwru_features columns to standard schema. Missing cols -> NaN."""
    out = pd.DataFrame()
    missing = []
    for std_col in STANDARD_COLUMNS:
        if std_col in df.columns:
            out[std_col] = df[std_col].values
            continue
        src = None
        for cwru_col, std_name in CWRU_TO_STANDARD.items():
            if std_name == std_col and cwru_col in df.columns:
                src = cwru_col
                break
        if src is not None:
            out[std_col] = df[src].values
        else:
            out[std_col] = pd.NA
            if std_col not in REQUIRED_COLUMNS:
                missing.append(std_col)
    if missing:
        logger.warning("Missing columns (filled with NaN): %s", missing)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    return out


def _ingest_simple(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format (timestamp, machine_id, metric_name, value) to standard schema."""
    df = df.copy()
    df["metric_name"] = df["metric_name"].astype(str).str.strip().str.lower()
    # Map metric_name to standard column
    df["_std_col"] = df["metric_name"].map(
        lambda m: METRIC_TO_STANDARD.get(m) or METRIC_TO_STANDARD.get(m.replace(" ", "_"))
    )
    df = df.dropna(subset=["_std_col"])
    pivoted = df.pivot_table(
        index=["timestamp", "machine_id"],
        columns="_std_col",
        values="value",
        aggfunc="first",
    ).reset_index()
    pivoted.columns.name = None
    # Ensure all standard columns exist
    out = pd.DataFrame(columns=STANDARD_COLUMNS)
    out["timestamp"] = pivoted.get("timestamp", pd.NA)
    out["machine_id"] = pivoted.get("machine_id", "")
    for std_col in STANDARD_COLUMNS:
        if std_col in ("timestamp", "machine_id"):
            continue
        if std_col in pivoted.columns:
            out[std_col] = pivoted[std_col]
        else:
            out[std_col] = pd.NA
            logger.warning("Missing column after pivot (filled with NaN): %s", std_col)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    return out


def ingest(csv_path: str | Path) -> pd.DataFrame:
    """
    Load CSV, detect format, normalize to standard DataFrame.

    Returns:
        DataFrame with columns: timestamp, machine_id, joint_torque_nm, joint_speed_rpm,
        motor_current_a, cycle_time_ms, vibration_rms, temperature_c.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        logger.warning("CSV is empty: %s", csv_path)
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    fmt = _detect_format(df)
    logger.info("Detected format: %s", fmt)

    if fmt == "cwru":
        normalized = _ingest_cwru(df)
    elif fmt == "simple":
        normalized = _ingest_simple(df)
    else:
        raise ValueError(
            f"Unknown CSV format. Expected cwru_features export or columns: timestamp, machine_id, metric_name, value. Got: {list(df.columns)}"
        )

    # Sort by time for downstream rolling windows
    normalized = normalized.sort_values("timestamp").reset_index(drop=True)
    return normalized
