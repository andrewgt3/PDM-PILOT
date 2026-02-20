"""
Flower federated learning client for PDM.

Trains on local data from TimescaleDB (labeled feature snapshots for this machine_id).
Data never leaves this process. Uses a numpy-compatible model (MLPClassifier) for FedAvg.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Project root on path for config, database, labeling_engine
_BASE = Path(__file__).resolve().parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from federated.constants import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    MLP_HIDDEN_LAYER_SIZES,
    MLP_MAX_ITER,
    MLP_RANDOM_STATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("federated.client")


def _load_local_data_sync(machine_id: str, min_labels: int = 20) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """Load labeled data from TimescaleDB for this machine_id. Returns (X, y, feature_cols) or (None, None, []). Data never leaves this process."""
    from database import get_db_context
    from labeling_engine import get_labeled_dataset

    async def _fetch():
        async with get_db_context() as db:
            df = await get_labeled_dataset(db, machine_id, min_labels=min_labels)
            return df

    try:
        df = asyncio.run(_fetch())
    except Exception as e:
        logger.warning("Failed to load labeled data for %s: %s", machine_id, e)
        return None, None, []

    if df is None or len(df) < min_labels:
        logger.warning("Insufficient labeled data for %s (need %d)", machine_id, min_labels)
        return None, None, []

    use_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if TARGET_COLUMN not in df.columns and "anomaly" in df.columns:
        df = df.rename(columns={"anomaly": TARGET_COLUMN})
    if TARGET_COLUMN not in df.columns:
        return None, None, []

    df = df.dropna(subset=use_cols + [TARGET_COLUMN])
    if len(df) < min_labels:
        return None, None, []

    # Fixed dimension for FedAvg: always len(FEATURE_COLUMNS), fill missing with 0
    X = np.zeros((len(df), len(FEATURE_COLUMNS)), dtype=np.float64)
    for i, col in enumerate(FEATURE_COLUMNS):
        if col in df.columns:
            X[:, i] = df[col].astype(float).values
    y = df[TARGET_COLUMN].astype(int).values
    return X, y, FEATURE_COLUMNS


def _make_mlp(n_features: int, n_classes: int = 2) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
        max_iter=MLP_MAX_ITER,
        random_state=MLP_RANDOM_STATE,
    )


def _params_from_model(model: MLPClassifier) -> List[np.ndarray]:
    return list(model.coefs_) + list(model.intercepts_)


def _set_model_params(model: MLPClassifier, params: List[np.ndarray]) -> None:
    n_coefs = len(model.coefs_)
    for i, arr in enumerate(model.coefs_):
        if i < len(params):
            arr[:] = params[i]
    for i, arr in enumerate(model.intercepts_):
        if n_coefs + i < len(params):
            arr[:] = params[n_coefs + i]


class PDMFlowerClient(fl.client.NumPyClient):
    """Flower NumPyClient: local training on TimescaleDB labeled data for one machine_id."""

    def __init__(self, machine_id: str, min_labels: int = 20):
        self.machine_id = machine_id
        self.min_labels = min_labels
        self._model: Optional[MLPClassifier] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_cols: List[str] = []
        self._n_features: Optional[int] = None

    def _ensure_model(self, n_features: int) -> None:
        if self._model is not None and self._n_features == n_features:
            return
        self._n_features = n_features
        self._model = _make_mlp(n_features)
        self._scaler = StandardScaler()
        self._feature_cols = FEATURE_COLUMNS[:n_features] if n_features <= len(FEATURE_COLUMNS) else FEATURE_COLUMNS

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        X, y, use_cols = _load_local_data_sync(self.machine_id, self.min_labels)
        if X is None or y is None or len(X) < self.min_labels:
            return []
        self._ensure_model(X.shape[1])
        self._scaler.fit(X)
        X_scaled = self._scaler.transform(X)
        self._model.fit(X_scaled, y)
        return _params_from_model(self._model)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        local_epochs = int(config.get("local_epochs", 5))
        X, y, use_cols = _load_local_data_sync(self.machine_id, self.min_labels)
        if X is None or y is None or len(X) < self.min_labels:
            return parameters, 0, {"loss": 0.0}

        self._ensure_model(X.shape[1])
        self._scaler.fit(X)
        X_scaled = self._scaler.transform(X)

        if len(parameters) == len(self._model.coefs_) + len(self._model.intercepts_):
            _set_model_params(self._model, parameters)

        self._model.max_iter = local_epochs
        self._model.fit(X_scaled, y)

        updated = _params_from_model(self._model)

        try:
            from federated.privacy_config import maybe_apply_dp
            updated = maybe_apply_dp(updated, config)
        except Exception:
            pass

        return updated, int(len(X)), {"train_samples": len(X)}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, float]]:
        X, y, use_cols = _load_local_data_sync(self.machine_id, self.min_labels)
        if X is None or y is None or len(X) < 5:
            return 0.0, 0, {}

        self._ensure_model(X.shape[1])
        if self._scaler is None:
            self._scaler = StandardScaler()
            self._scaler.fit(X)
        X_scaled = self._scaler.transform(X)
        if len(parameters) == len(self._model.coefs_) + len(self._model.intercepts_):
            _set_model_params(self._model, parameters)

        from sklearn.metrics import log_loss
        proba = self._model.predict_proba(X_scaled)
        loss = float(log_loss(y, proba))
        return loss, len(X), {"loss": loss, "num_examples": len(X)}


def main() -> None:
    parser = argparse.ArgumentParser(description="PDM Flower federated client")
    parser.add_argument("--server-address", type=str, default="localhost:8080", help="Central server host:port")
    parser.add_argument("--machine-id", type=str, required=True, help="Machine ID (e.g. WB-001)")
    parser.add_argument("--min-labels", type=int, default=20, help="Minimum labeled examples to participate")
    args = parser.parse_args()

    client = PDMFlowerClient(machine_id=args.machine_id, min_labels=args.min_labels)
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )


if __name__ == "__main__":
    main()
