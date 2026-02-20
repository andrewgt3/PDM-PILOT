#!/usr/bin/env python3
"""
Robot Feature Extractor — rotating machinery and conveyor systems.

Applies the feature library (feature_library.py) to raw readings DataFrames
with configurable windows and toggles from config/feature_config.yaml.
Returns a feature DataFrame with exactly the columns the model expects.
Logs data quality: features with >10% null after extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pipeline.cleansing.feature_library import (
    baseline_zscore,
    cycle_time_drift,
    rate_of_change,
    rolling_max,
    rolling_mean,
    rolling_p95,
    rolling_std,
    peak_to_peak,
    speed_normalized_torque,
    thermal_torque_ratio,
    within_cycle_variance,
)

logger = logging.getLogger(__name__)

# Default config path (project root / config / feature_config.yaml)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "feature_config.yaml"

# Fallback output columns if config is missing (match train_model CUSTOM_FEATURES)
DEFAULT_OUTPUT_COLUMNS = [
    "torque_mean",
    "torque_std",
    "torque_max",
    "torque_p95",
    "torque_deviation_zscore",
    "speed_normalized_torque",
    "cycle_time_drift_pct",
    "temp_rate_of_change",
    "thermal_torque_ratio",
]


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        logger.warning("Feature config not found at %s, using defaults.", path)
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _first_present(df: pd.DataFrame, *candidates: str) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


class RobotFeatureExtractor:
    """
    Applies the full feature library to raw readings.
    Configurable via YAML (windows, baseline periods, feature toggles).
    """

    def __init__(self, config_path: str | Path | None = None):
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._config = _load_config(path)
        self.windows = self._config.get("windows", [50, 200, 500, 2000])
        self.rolling_window = self._config.get("rolling_window") or (self.windows[0] if self.windows else 200)
        self.baseline_n = self._config.get("baseline_n", 1000)
        self.cycle_baseline_n = self._config.get("cycle_baseline_n", 500)
        self.features = self._config.get("features", {})
        self.output_columns = self._config.get("output_columns", DEFAULT_OUTPUT_COLUMNS)
        self.null_warning_threshold = self._config.get("null_warning_threshold", 0.10)

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the feature library to raw readings.
        Returns feature DataFrame with columns from config (only those computed).
        Logs which features had > null_warning_threshold fraction null.
        """
        if df.empty or "timestamp" not in df.columns or "machine_id" not in df.columns:
            return pd.DataFrame(columns=["timestamp", "machine_id"] + self.output_columns)

        df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
        out_list = []

        for machine_id, grp in df.groupby("machine_id", sort=False):
            grp = grp.sort_values("timestamp").reset_index(drop=True)
            torque_col = _first_present(grp, "joint_torque_nm", "torque")
            speed_col = _first_present(grp, "joint_speed_rpm", "rotational_speed")
            cycle_col = _first_present(grp, "cycle_time_ms")
            temp_col = _first_present(grp, "temperature_c", "temperature")
            # cycle_index: optional, for within_cycle_variance (e.g. cycle counter)
            cycle_index_col = _first_present(grp, "cycle_index", "cycle_id")

            torque = grp[torque_col].astype(float) if torque_col else pd.Series(np.nan, index=grp.index)
            speed = grp[speed_col].astype(float) if speed_col else pd.Series(np.nan, index=grp.index)
            cycle = grp[cycle_col].astype(float) if cycle_col else pd.Series(np.nan, index=grp.index)
            temp = grp[temp_col].astype(float) if temp_col else pd.Series(np.nan, index=grp.index)
            cycle_index = grp[cycle_index_col] if cycle_index_col else pd.Series(range(len(grp)), index=grp.index)

            n = len(grp)
            w = min(self.rolling_window, n)
            base_n = min(self.baseline_n, n)
            cycle_base_n = min(self.cycle_baseline_n, n)

            row_dict: dict[str, Any] = {
                "timestamp": grp["timestamp"].values,
                "machine_id": machine_id,
            }

            if torque_col and self.features.get("rolling_mean", True):
                row_dict["torque_mean"] = rolling_mean(torque, w).values
            if torque_col and self.features.get("rolling_std", True):
                row_dict["torque_std"] = rolling_std(torque, w).values
            if torque_col and self.features.get("rolling_max", True):
                row_dict["torque_max"] = rolling_max(torque, w).values
            if torque_col and self.features.get("rolling_p95", True):
                row_dict["torque_p95"] = rolling_p95(torque, w).values
            if torque_col and self.features.get("baseline_zscore", True):
                row_dict["torque_deviation_zscore"] = baseline_zscore(torque, base_n).values
            if torque_col and speed_col and self.features.get("speed_normalized_torque", True):
                row_dict["speed_normalized_torque"] = speed_normalized_torque(torque, speed).values
            if cycle_col and self.features.get("cycle_time_drift", True):
                row_dict["cycle_time_drift_pct"] = cycle_time_drift(cycle, cycle_base_n).values
            if temp_col and self.features.get("rate_of_change", True):
                row_dict["temp_rate_of_change"] = rate_of_change(temp).values
            if torque_col and temp_col and self.features.get("thermal_torque_ratio", True):
                row_dict["thermal_torque_ratio"] = thermal_torque_ratio(torque, temp).values
            if torque_col and self.features.get("peak_to_peak", False):
                row_dict["torque_peak_to_peak"] = peak_to_peak(torque, w).values
            if torque_col and self.features.get("within_cycle_variance", False) and cycle_index_col:
                row_dict["within_cycle_variance"] = within_cycle_variance(torque, cycle_index).values

            out_list.append(pd.DataFrame(row_dict))

        if not out_list:
            return pd.DataFrame(columns=["timestamp", "machine_id"] + self.output_columns)

        result = pd.concat(out_list, ignore_index=True)

        # Restrict to output_columns that exist
        meta = ["timestamp", "machine_id"]
        available = [c for c in self.output_columns if c in result.columns]
        result = result[meta + available]

        # Data quality: log features with > threshold null
        for col in available:
            null_frac = result[col].isna().mean()
            if null_frac > self.null_warning_threshold:
                logger.warning(
                    "Feature %s had %.1f%% null values after extraction (threshold %.0f%%)",
                    col,
                    null_frac * 100,
                    self.null_warning_threshold * 100,
                )

        return result


# -----------------------------------------------------------------------------
# Backward compatibility: function + FEATURE_COLUMNS for train_model
# -----------------------------------------------------------------------------

def extract_features(
    df: pd.DataFrame,
    rolling_cycles: int | None = None,
    baseline_cycles: int | None = None,
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Extract robot/PLC health features (convenience wrapper around RobotFeatureExtractor).
    If rolling_cycles/baseline_cycles are passed, a minimal config is used; else config file.
    """
    if config_path is None and (rolling_cycles is None and baseline_cycles is None):
        extractor = RobotFeatureExtractor(config_path=None)
        return extractor.extract(df)
    # Override config with explicit params if provided
    override = {}
    if rolling_cycles is not None:
        override["rolling_window"] = rolling_cycles
    if baseline_cycles is not None:
        override["baseline_n"] = baseline_cycles
    extractor = RobotFeatureExtractor(config_path=config_path or DEFAULT_CONFIG_PATH)
    if override:
        extractor.rolling_window = override.get("rolling_window", extractor.rolling_window)
        extractor.baseline_n = override.get("baseline_n", extractor.baseline_n)
        extractor.cycle_baseline_n = override.get("cycle_baseline_n", extractor.cycle_baseline_n)
    return extractor.extract(df)


# Columns the model expects (used by train_model when using extract_features)
FEATURE_COLUMNS = DEFAULT_OUTPUT_COLUMNS
