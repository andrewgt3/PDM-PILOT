#!/usr/bin/env python3
"""
Prepare historical data for streaming (Azure PdM path).

If the given data dir contains the 5 PdM_*.csv files, runs the Azure PM ingestor
to load sensor_readings (and cwru_features), then prints the command to
start the replay. If the CSVs are missing, prints download instructions.

Usage:
    python scripts/prepare_historical_stream.py
    python scripts/prepare_historical_stream.py --data-dir microsoft_azure_predictive_maintenance
    python scripts/prepare_historical_stream.py --data-dir data/azure_pm --run-ingest-only
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AZURE_PM_DIR = REPO_ROOT / "data" / "azure_pm"

# At least telemetry is required; others are optional for merge
REQUIRED = ["PdM_telemetry.csv"]
OPTIONAL = ["PdM_errors.csv", "PdM_failures.csv", "PdM_machines.csv", "PdM_maint.csv"]
ALL_FILES = REQUIRED + OPTIONAL

KAGGLE_URL = "https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance/data"


def main():
    parser = argparse.ArgumentParser(description="Prepare Azure PdM data for historical stream replay")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Folder containing PdM_*.csv files (default: data/azure_pm). Can be 'microsoft_azure_predictive_maintenance' or any path.)",
    )
    parser.add_argument("--run-ingest-only", action="store_true", help="Only run ingestor; do not print replay command")
    args = parser.parse_args()

    if args.data_dir is not None:
        data_dir = Path(args.data_dir)
        if not data_dir.is_absolute():
            data_dir = (REPO_ROOT / data_dir).resolve()
    else:
        data_dir = DEFAULT_AZURE_PM_DIR

    data_dir.mkdir(parents=True, exist_ok=True)

    present = [f for f in ALL_FILES if (data_dir / f).is_file()]
    missing_required = [f for f in REQUIRED if f not in present]

    if missing_required:
        print(f"Azure PdM CSVs not found in {data_dir}")
        print()
        print("Download the dataset (free with Kaggle account):")
        print(f"  {KAGGLE_URL}")
        print()
        print("Then place these 5 files into your folder (e.g. data/azure_pm/ or microsoft_azure_predictive_maintenance/):")
        for f in ALL_FILES:
            print(f"  - {f}")
        print()
        print("Required at minimum: PdM_telemetry.csv")
        print("Optional: PdM_errors, PdM_failures, PdM_machines, PdM_maint (for full merge).")
        print()
        print("If your folder has a different name, run: python scripts/prepare_historical_stream.py --data-dir YOUR_FOLDER")
        return 1

    print(f"Running Azure PM ingestor on {data_dir} (sensor_readings + cwru_features)...")
    sys.path.insert(0, str(REPO_ROOT))
    from pipeline.ingestion.azure_pm_ingestor import AzurePMIngestor

    ingestor = AzurePMIngestor(data_dir)
    n = ingestor.run(write_cwru=True)
    print(f"Inserted/updated {n} rows in sensor_readings.")
    print()

    if not args.run_ingest_only:
        print("To stream this data through the pipeline (Redis + inference):")
        print()
        print("  python mock_fleet_streamer.py --source azure_pm --speed-multiplier 720")
        print()
        print("  To point at a test server's Redis, use:")
        print("  REDIS_HOST=<server-ip> REDIS_PORT=6379 python mock_fleet_streamer.py --source azure_pm --speed-multiplier 720")
        print("  or: python mock_fleet_streamer.py --source azure_pm --redis-host <ip> --redis-port 6379")
        print()
        print("  (720 = 1 month of data replayed in ~1 hour. Use a smaller value for faster replay.)")
        print()
        print("Keep the API and stream_consumer running on the test server so inference runs on the replayed stream.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
