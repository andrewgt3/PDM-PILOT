"""
Flower aggregation server: FedAvg strategy, MLflow logging after each round.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.neural_network import MLPClassifier

# Project root on path
_BASE = Path(__file__).resolve().parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from federated.constants import (
    FEATURE_COLUMNS,
    MLP_HIDDEN_LAYER_SIZES,
    MLP_MAX_ITER,
    MLP_RANDOM_STATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("federated.server")

N_FEATURES = len(FEATURE_COLUMNS)
N_CLASSES = 2


def _get_initial_parameters() -> list:
    """Initial global model parameters (same architecture as clients)."""
    model = MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
        max_iter=1,
        random_state=MLP_RANDOM_STATE,
    )
    X = np.zeros((2, N_FEATURES), dtype=np.float64)
    y = np.array([0, 1])
    model.fit(X, y)
    return list(model.coefs_) + list(model.intercepts_)


def _parameters_to_ndarrays(parameters) -> list:
    """Convert Flower Parameters to list of ndarrays."""
    import flwr as fl
    return fl.common.parameters_to_ndarrays(parameters)


def _ndarrays_to_parameters(ndarrays: list) -> "fl.common.typing.Parameters":
    import flwr as fl
    return fl.common.ndarrays_to_parameters(ndarrays)


def _save_round_to_mlflow(round_num: int, parameters: list, metrics: Optional[dict] = None) -> None:
    """Save aggregated model to MLflow. No-op if MLflow not configured."""
    try:
        import mlflow
        import mlflow.sklearn
        from sklearn.neural_network import MLPClassifier
    except ImportError:
        logger.warning("MLflow or sklearn not available; skipping model log")
        return

    model = MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
        max_iter=1,
        random_state=MLP_RANDOM_STATE,
    )
    X_dummy = np.zeros((2, N_FEATURES), dtype=np.float64)
    y_dummy = np.array([0, 1])
    model.fit(X_dummy, y_dummy)
    n_coefs = len(model.coefs_)
    for i, arr in enumerate(model.coefs_):
        if i < len(parameters):
            arr[:] = parameters[i]
    for i, arr in enumerate(model.intercepts_):
        if n_coefs + i < len(parameters):
            arr[:] = parameters[n_coefs + i]

    experiment_name = "federated_pdm"
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"fl_round_{round_num}"):
        mlflow.log_param("round", round_num)
        if metrics:
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(model, "model")
    logger.info("Logged round %s to MLflow", round_num)


def main() -> None:
    parser = argparse.ArgumentParser(description="PDM Flower federated server")
    parser.add_argument("--num-rounds", type=int, default=10, help="Number of federation rounds")
    parser.add_argument("--port", type=int, default=8080, help="gRPC port")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum clients before aggregation")
    args = parser.parse_args()

    import flwr as fl

    initial_ndarrays = _get_initial_parameters()
    initial_params = _ndarrays_to_parameters(initial_ndarrays)

    def fit_config(round_num: int):
        from config import get_settings
        try:
            s = get_settings()
            epochs = getattr(s.federated, "fl_local_epochs", 5) if hasattr(s, "federated") else 5
        except Exception:
            epochs = 5
        return {"local_epochs": epochs}

    # Flower 1.11: FedAvg with MLflow logging after each round
    class FedAvgWithMLflow(fl.server.strategy.FedAvg):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._last_round = 0

        def aggregate_fit(self, server_round, results, failures):
            aggregated = super().aggregate_fit(server_round, results, failures)
            if aggregated is not None and server_round is not None:
                try:
                    ndarrays = fl.common.parameters_to_ndarrays(aggregated[0])
                    _save_round_to_mlflow(server_round, ndarrays)
                except Exception as e:
                    logger.warning("MLflow log after round %s failed: %s", server_round, e)
            return aggregated

    strategy = FedAvgWithMLflow(
        min_fit_clients=args.min_clients,
        min_evaluate_clients=1,
        initial_parameters=initial_params,
        on_fit_config_fn=lambda r: fit_config(r),
    )

    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
