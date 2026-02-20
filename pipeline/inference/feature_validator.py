#!/usr/bin/env python3
"""
Feature validator for inference.

Loads the metadata JSON sidecar for the active model and validates incoming
live data: required features present and values within expected (training)
ranges. Logs warnings for any feature out of training distribution (drift signal).
Runs before every inference call in inference_service.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default path: same dir as gaia_model_latest.pkl
DEFAULT_METADATA_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "gaia_model_latest_metadata.json"


def load_metadata(metadata_path: str | Path | None = None) -> dict[str, Any] | None:
    """Load metadata JSON for the active model. Returns None if file missing."""
    path = Path(metadata_path) if metadata_path else DEFAULT_METADATA_PATH
    if not path.exists():
        logger.debug("No metadata file at %s; skipping feature validation.", path)
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load metadata from %s: %s", path, e)
        return None


def validate_features(
    feature_dict: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    metadata_path: str | Path | None = None,
) -> tuple[bool, list[str]]:
    """
    Validate incoming live feature dict against model metadata.

    Checks:
      - All required features (feature_list) are present.
      - Numeric features are within expected min/max (feature_ranges).
        Values outside range are logged as out-of-distribution (drift signal).

    Returns:
      (valid, list of warning messages).
      valid is False if required features are missing; True otherwise.
      Warnings are always logged and appended for any OOD feature.
    """
    if metadata is None:
        metadata = load_metadata(metadata_path)
    if metadata is None:
        return True, []  # No metadata -> pass without validation

    required = metadata.get("feature_list") or metadata.get("features") or []
    if not required:
        return True, []

    missing = [f for f in required if f not in feature_dict]
    if missing:
        logger.warning(
            "Feature validation: missing required features %s (inference may be unreliable).",
            missing,
        )
        return False, [f"missing features: {missing}"]

    warnings: list[str] = []
    ranges = metadata.get("feature_ranges") or {}
    for col, lim in ranges.items():
        if col not in feature_dict:
            continue
        try:
            val = float(feature_dict[col])
        except (TypeError, ValueError):
            warnings.append(f"{col}: non-numeric value")
            logger.warning(
                "Feature %s out of expected type (potential drift).",
                col,
                extra={"value": feature_dict.get(col)},
            )
            continue
        lo = lim.get("min")
        hi = lim.get("max")
        if lo is not None and val < lo:
            msg = f"{col}={val} below training min {lo}"
            warnings.append(msg)
            logger.warning(
                "Feature %s below training distribution (potential drift): %s",
                col,
                msg,
            )
        if hi is not None and val > hi:
            msg = f"{col}={val} above training max {hi}"
            warnings.append(msg)
            logger.warning(
                "Feature %s above training distribution (potential drift): %s",
                col,
                msg,
            )

    return True, warnings
