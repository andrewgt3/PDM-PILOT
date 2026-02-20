"""Data profiling service: computes per-metric quality statistics from sensor_readings."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from logger import get_logger
from schemas.profiling import (
    DataProfile,
    MetricProfile,
    ProfileStatus,
    ValueRange,
)

logger = get_logger(__name__)

# Scalar metrics in sensor_readings to profile (excludes vibration_raw JSONB for v1)
PROFILE_METRICS = ["rotational_speed", "temperature", "torque", "tool_wear"]

# Thresholds
COMPLETENESS_CRITICAL_PCT = 80.0
COMPLETENESS_WARNING_PCT = 95.0
NULL_PCT_CRITICAL = 20.0
NULL_PCT_WARNING = 5.0
FLATLINE_WARNING_COUNT = 100
OUTLIER_STD_THRESHOLD = 3.0


def _status_priority(s: ProfileStatus) -> int:
    """Higher = worse."""
    return {"HEALTHY": 0, "WARNING": 1, "CRITICAL": 2}[s]


def _worst_status(a: ProfileStatus, b: ProfileStatus) -> ProfileStatus:
    return a if _status_priority(a) >= _status_priority(b) else b


def _metric_status(
    completeness_pct: float,
    null_pct: float,
    flatline_count: int,
) -> ProfileStatus:
    """Compute status from thresholds."""
    status: ProfileStatus = "HEALTHY"
    if completeness_pct < COMPLETENESS_CRITICAL_PCT or null_pct > NULL_PCT_CRITICAL:
        return "CRITICAL"
    if completeness_pct < COMPLETENESS_WARNING_PCT or null_pct > NULL_PCT_WARNING:
        status = "WARNING"
    if flatline_count > FLATLINE_WARNING_COUNT:
        status = _worst_status(status, "WARNING")
    return status


def _flatline_count(series: list[float | None]) -> int:
    """Max consecutive identical non-null values."""
    max_run = 0
    run = 0
    prev: float | None = None
    for v in series:
        if v is None:
            run = 0
            prev = None
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            run = 0
            prev = None
            continue
        if prev is not None and x == prev:
            run += 1
        else:
            run = 1
        prev = x
        if run > max_run:
            max_run = run
    return max_run


def _rate_of_change(series: list[float | None]) -> float | None:
    """Average absolute change between consecutive non-null readings."""
    diffs: list[float] = []
    prev: float | None = None
    for v in series:
        if v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if prev is not None:
            diffs.append(abs(x - prev))
        prev = x
    if not diffs:
        return None
    return sum(diffs) / len(diffs)


def _outlier_count(series: list[float | None], mean: float, std: float) -> int:
    """Count values beyond 3 standard deviations from mean."""
    if std <= 0:
        return 0
    count = 0
    for v in series:
        if v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if abs(x - mean) > OUTLIER_STD_THRESHOLD * std:
            count += 1
    return count


def _percentile(sorted_values: list[float], p: float) -> float:
    """Linear interpolation percentile (0-100)."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    idx = (p / 100.0) * (n - 1)
    i = int(idx)
    if i >= n - 1:
        return sorted_values[-1]
    frac = idx - i
    return sorted_values[i] + frac * (sorted_values[i + 1] - sorted_values[i])


class DataProfilerService:
    """Computes data quality profiles from sensor_readings for a machine."""

    def __init__(
        self,
        db: AsyncSession,
        *,
        poll_rate_hz: float = 10.0,
    ) -> None:
        self.db = db
        self.poll_rate_hz = poll_rate_hz

    async def profile(self, machine_id: str, hours: int = 48) -> DataProfile:
        """
        Query last N hours from TimescaleDB for the machine and compute
        per-metric statistics. Returns a DataProfile.
        When hours <= 0, use all available data and treat expected_count as 0 (100% completeness).
        """
        profiled_at = datetime.now(timezone.utc)
        effective_hours = max(0, hours)
        expected_count = int(effective_hours * 3600 * self.poll_rate_hz) if (self.poll_rate_hz > 0 and effective_hours > 0) else 0

        # Fetch rows: when hours <= 0 use all data; otherwise last N hours
        if effective_hours <= 0:
            query = text("""
                SELECT timestamp, rotational_speed, temperature, torque, tool_wear
                FROM sensor_readings
                WHERE machine_id = :machine_id
                ORDER BY timestamp
            """)
            result = await self.db.execute(query, {"machine_id": machine_id})
        else:
            query = text("""
                SELECT timestamp, rotational_speed, temperature, torque, tool_wear
                FROM sensor_readings
                WHERE machine_id = :machine_id
                  AND timestamp >= NOW() - (:hours * INTERVAL '1 hour')
                ORDER BY timestamp
            """)
            result = await self.db.execute(query, {"machine_id": machine_id, "hours": effective_hours})
        rows = result.mappings().all()

        total_rows = len(rows)
        if total_rows == 0:
            logger.warning("No sensor_readings for machine in window", machine_id=machine_id, hours=hours)
            return DataProfile(
                machine_id=machine_id,
                profiled_at=profiled_at,
                hours_analyzed=hours,
                overall_status="WARNING",
                metrics=[],
            )

        # Build column series (same order as rows)
        columns: dict[str, list[float | None]] = {m: [] for m in PROFILE_METRICS}
        for r in rows:
            for m in PROFILE_METRICS:
                val = r.get(m)
                columns[m].append(float(val) if val is not None else None)

        metrics: list[MetricProfile] = []
        overall: ProfileStatus = "HEALTHY"

        for metric_name in PROFILE_METRICS:
            series = columns[metric_name]
            non_null = [v for v in series if v is not None]
            null_count = sum(1 for v in series if v is None)
            n = len(series)
            null_pct = (100.0 * null_count / n) if n else 0.0
            completeness_pct = min(100.0, (100.0 * n / expected_count)) if expected_count else 100.0

            if not non_null:
                value_range = ValueRange(min=0.0, max=0.0, mean=0.0, std=0.0, p5=0.0, p95=0.0)
                outlier_count = 0
                flatline_count = 0
                rate_of_change = None
            else:
                mean = sum(non_null) / len(non_null)
                variance = sum((x - mean) ** 2 for x in non_null) / len(non_null)
                std = variance ** 0.5
                sorted_vals = sorted(non_null)
                value_range = ValueRange(
                    min=min(non_null),
                    max=max(non_null),
                    mean=mean,
                    std=std,
                    p5=_percentile(sorted_vals, 5),
                    p95=_percentile(sorted_vals, 95),
                )
                outlier_count = _outlier_count(series, mean, std)
                flatline_count = _flatline_count(series)
                rate_of_change = _rate_of_change(series)

            status = _metric_status(completeness_pct, null_pct, flatline_count)
            overall = _worst_status(overall, status)

            metrics.append(
                MetricProfile(
                    metric_name=metric_name,
                    status=status,
                    completeness_pct=round(completeness_pct, 2),
                    value_range=value_range,
                    null_count=null_count,
                    null_pct=round(null_pct, 2),
                    outlier_count=outlier_count,
                    flatline_count=flatline_count,
                    rate_of_change=round(rate_of_change, 6) if rate_of_change is not None else None,
                )
            )

        return DataProfile(
            machine_id=machine_id,
            profiled_at=profiled_at,
            hours_analyzed=hours,
            overall_status=overall,
            metrics=metrics,
        )
