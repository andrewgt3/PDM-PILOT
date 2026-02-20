"""
Differential privacy for federated learning (DP-SGD style).

When FL_DP_ENABLED=true: clip parameter updates by L2 norm, then add Gaussian noise.
Controlled by config; off by default for standard deployments.
"""

from __future__ import annotations

import os
from typing import Any, List

import numpy as np

DP_ENABLED = os.getenv("FL_DP_ENABLED", "false").lower() in ("true", "1", "yes")
NOISE_MULTIPLIER = float(os.getenv("FL_DP_NOISE_MULTIPLIER", "0.01"))
CLIPPING_NORM = float(os.getenv("FL_DP_CLIPPING_NORM", "1.0"))


def _l2_norm(params: List[np.ndarray]) -> float:
    total = 0.0
    for p in params:
        total += float(np.sum(np.square(p)))
    return np.sqrt(total)


def _clip_params(params: List[np.ndarray], max_norm: float) -> List[np.ndarray]:
    norm = _l2_norm(params)
    if norm <= 0 or max_norm <= 0:
        return params
    if norm <= max_norm:
        return [p.copy() for p in params]
    scale = max_norm / norm
    return [p * scale for p in params]


def _add_gaussian_noise(params: List[np.ndarray], scale: float) -> List[np.ndarray]:
    return [p + np.random.normal(0, scale, p.shape).astype(p.dtype) for p in params]


def maybe_apply_dp(
    parameters: List[np.ndarray],
    config: Any,
) -> List[np.ndarray]:
    """
    If DP is enabled, clip by L2 norm then add Gaussian noise to the parameter list.
    Uses FL_DP_ENABLED, FL_DP_CLIPPING_NORM, FL_DP_NOISE_MULTIPLIER from env or config.
    """
    enabled = config.get("dp_enabled", DP_ENABLED) if isinstance(config, dict) else DP_ENABLED
    if not enabled:
        return parameters

    clipping_norm = config.get("dp_clipping_norm", CLIPPING_NORM) if isinstance(config, dict) else CLIPPING_NORM
    noise_multiplier = config.get("dp_noise_multiplier", NOISE_MULTIPLIER) if isinstance(config, dict) else NOISE_MULTIPLIER

    clipped = _clip_params(parameters, clipping_norm)
    scale = noise_multiplier * clipping_norm
    return _add_gaussian_noise(clipped, scale)
