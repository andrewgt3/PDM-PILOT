"""Canonical feature list for federated model (must match across all clients and server)."""

# Same as train_model.CUSTOM_FEATURES so FL model uses same inputs as centralized pipeline
FEATURE_COLUMNS = [
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
TARGET_COLUMN = "label"  # labeling_tasks use 'label'; train_model uses 'anomaly'

# MLP architecture for FedAvg (fixed so parameter shapes match)
MLP_HIDDEN_LAYER_SIZES = (64, 32)
MLP_MAX_ITER = 200
MLP_RANDOM_STATE = 42
