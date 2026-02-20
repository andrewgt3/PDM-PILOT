#!/usr/bin/env python3
"""
Feature library for rotating machinery and conveyor systems.

Each function takes pandas Series (and optional parameters) and returns
a scalar or Series suitable for use in a feature pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Time-domain features
# -----------------------------------------------------------------------------

def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Rolling mean over N samples."""
    return series.rolling(window=window, min_periods=max(1, window // 2)).mean()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation over N samples."""
    return series.rolling(window=window, min_periods=max(1, window // 2)).std()


def rolling_max(series: pd.Series, window: int) -> pd.Series:
    """Rolling maximum over N samples."""
    return series.rolling(window=window, min_periods=max(1, window // 2)).max()


def rolling_p95(series: pd.Series, window: int) -> pd.Series:
    """Rolling 95th percentile over N samples."""
    return series.rolling(window=window, min_periods=max(1, window // 2)).quantile(0.95)


def baseline_zscore(series: pd.Series, baseline_n: int = 1000) -> pd.Series:
    """Z-score vs first N samples (healthy baseline)."""
    n = min(baseline_n, len(series))
    if n == 0:
        return pd.Series(np.nan, index=series.index)
    base_mean = series.iloc[:n].mean()
    base_std = series.iloc[:n].std()
    if base_std == 0 or (base_std != base_std):
        return pd.Series(np.nan, index=series.index)
    return (series - base_mean) / (base_std + 1e-12)


def rate_of_change(series: pd.Series) -> pd.Series:
    """First derivative (difference between consecutive samples)."""
    return series.diff()


def peak_to_peak(series: pd.Series, window: int) -> pd.Series:
    """Max - min in window."""
    roll_max = series.rolling(window=window, min_periods=max(1, window // 2)).max()
    roll_min = series.rolling(window=window, min_periods=max(1, window // 2)).min()
    return roll_max - roll_min


# -----------------------------------------------------------------------------
# Cycle-relative features (require cycle_index column)
# -----------------------------------------------------------------------------

def cycle_time_drift(cycle_times: pd.Series, baseline_n: int = 500) -> pd.Series:
    """
    Current cycle time vs baseline mean.
    Returns (cycle_times - baseline_mean) / baseline_mean as fraction.
    """
    n = min(baseline_n, len(cycle_times))
    if n == 0:
        return pd.Series(np.nan, index=cycle_times.index)
    baseline_mean = cycle_times.iloc[:n].mean()
    if baseline_mean == 0 or (baseline_mean != baseline_mean):
        return pd.Series(np.nan, index=cycle_times.index)
    return (cycle_times - baseline_mean) / (baseline_mean + 1e-12)


def within_cycle_variance(series: pd.Series, cycle_index: pd.Series) -> pd.Series:
    """
    Variance of signal within each cycle.
    Groups by cycle_index, computes variance per group, maps back to original index.
    """
    if len(series) != len(cycle_index):
        return pd.Series(np.nan, index=series.index)
    df = pd.DataFrame({"value": series, "cycle": cycle_index})
    cycle_var = df.groupby("cycle")["value"].transform("var")
    return cycle_var


# -----------------------------------------------------------------------------
# Cross-feature features
# -----------------------------------------------------------------------------

def speed_normalized_torque(torque: pd.Series, speed: pd.Series) -> pd.Series:
    """Torque / speed (removes load correlation). Avoids div by zero."""
    speed_safe = speed.replace(0, np.nan)
    return torque / (speed_safe + 1e-6)


def thermal_torque_ratio(torque: pd.Series, temperature: pd.Series) -> pd.Series:
    """Torque / temp (thermal compensation). Avoids div by zero."""
    temp_safe = temperature.replace(0, np.nan)
    return torque / (temp_safe + 1e-6)
