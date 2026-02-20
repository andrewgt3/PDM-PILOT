"""Data quality profiling schemas for the data profiler service."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ProfileStatus = Literal["HEALTHY", "WARNING", "CRITICAL"]


class ValueRange(BaseModel):
    """Min, max, mean, std, and percentiles for a metric."""

    min: float = Field(..., description="Minimum value")
    max: float = Field(..., description="Maximum value")
    mean: float = Field(..., description="Mean value")
    std: float = Field(..., description="Standard deviation")
    p5: float = Field(..., description="5th percentile")
    p95: float = Field(..., description="95th percentile")


class MetricProfile(BaseModel):
    """Per-metric data quality profile."""

    metric_name: str = Field(..., description="Name of the metric (e.g. rotational_speed)")
    status: ProfileStatus = Field(..., description="HEALTHY / WARNING / CRITICAL")
    completeness_pct: float = Field(..., ge=0, le=100, description="Percentage of expected readings received")
    value_range: ValueRange = Field(..., description="Min, max, mean, std, p5, p95")
    null_count: int = Field(..., ge=0, description="Number of null/missing readings")
    null_pct: float = Field(..., ge=0, le=100, description="Percentage of null readings")
    outlier_count: int = Field(..., ge=0, description="Readings beyond 3 standard deviations")
    flatline_count: int = Field(..., ge=0, description="Max consecutive identical readings (sensor stuck)")
    rate_of_change: float | None = Field(None, description="Average absolute change between consecutive readings")


class DataProfile(BaseModel):
    """Top-level data quality report for a machine."""

    machine_id: str = Field(..., description="Machine identifier")
    profiled_at: datetime = Field(..., description="When the profile was generated")
    hours_analyzed: int = Field(..., ge=1, description="Number of hours of data analyzed")
    overall_status: ProfileStatus = Field(..., description="Worst status across all metrics")
    metrics: list[MetricProfile] = Field(default_factory=list, description="Per-metric profiles")
