#!/usr/bin/env python3
"""
Run both Kaggle dataset ingestors (inIT-OWL and Azure PdM) and print a summary.

Optionally copies data from data/downloads/kaggle/ into data/init_owl/ and
data/azure_pm/ so both datasets live under data/ before ingestion.

Usage:
    python scripts/load_training_data.py
    python scripts/load_training_data.py --skip-init-owl
    python scripts/load_training_data.py --skip-azure
"""

import argparse
import shutil
import sys
from pathlib import Path

# Project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Default paths under data/
DATA_ROOT = PROJECT_ROOT / "data"
INIT_OWL_DIR = DATA_ROOT / "init_owl"
INIT_OWL_CSV = DATA_ROOT / "init_owl.csv"
AZURE_PM_DIR = DATA_ROOT / "azure_pm"
KAGGLE_DOWNLOADS = DATA_ROOT / "downloads" / "kaggle"
AZURE_KAGGLE_SOURCE = KAGGLE_DOWNLOADS / "microsoft_azure_predictive_maintenance"
INIT_OWL_KAGGLE_SOURCE = KAGGLE_DOWNLOADS / "init_owl"

# PdM file names we need
AZURE_CSV_NAMES = [
    "PdM_telemetry.csv",
    "PdM_errors.csv",
    "PdM_failures.csv",
    "PdM_machines.csv",
    "PdM_maint.csv",
]


def ensure_data_paths() -> None:
    """
    If data/azure_pm/ or data/init_owl/ are missing but Kaggle downloads exist,
    copy files into data/azure_pm/ and data/init_owl/ so both datasets are under data/.
    """
    # Azure PdM: copy from data/downloads/kaggle/microsoft_azure_predictive_maintenance/ -> data/azure_pm/
    if not AZURE_PM_DIR.exists() or not any((AZURE_PM_DIR / n).exists() for n in AZURE_CSV_NAMES):
        if AZURE_KAGGLE_SOURCE.is_dir():
            AZURE_PM_DIR.mkdir(parents=True, exist_ok=True)
            for name in AZURE_CSV_NAMES:
                src = AZURE_KAGGLE_SOURCE / name
                if src.exists():
                    shutil.copy2(src, AZURE_PM_DIR / name)
                    print(f"Copied {name} -> {AZURE_PM_DIR / name}")

    # inIT-OWL: copy CSVs from data/downloads/kaggle/init_owl/ -> data/init_owl/
    if not INIT_OWL_DIR.exists():
        INIT_OWL_DIR.mkdir(parents=True, exist_ok=True)
    init_owl_has_csv = any(INIT_OWL_DIR.glob("*.csv"))
    if not init_owl_has_csv and not INIT_OWL_CSV.exists() and INIT_OWL_KAGGLE_SOURCE.is_dir():
        for csv_path in INIT_OWL_KAGGLE_SOURCE.rglob("*.csv"):
            dest = INIT_OWL_DIR / csv_path.name
            if not dest.exists() or dest.stat().st_mtime < csv_path.stat().st_mtime:
                shutil.copy2(csv_path, dest)
                print(f"Copied {csv_path.name} -> {dest}")


def get_init_owl_input_path() -> Path | None:
    """Return path to use for inIT-OWL ingestor: init_owl.csv or init_owl/."""
    if INIT_OWL_CSV.exists():
        return INIT_OWL_CSV
    if INIT_OWL_DIR.is_dir() and any(INIT_OWL_DIR.glob("*.csv")):
        return INIT_OWL_DIR
    return None


def run_init_owl_ingestor(write_db: bool = True) -> int:
    """Run inIT-OWL ingestor; return rows written (to DB or CSV count)."""
    from pipeline.ingestion.init_owl_ingestor import InITOWLIngestor

    path = get_init_owl_input_path()
    if not path:
        return -1
    ingestor = InITOWLIngestor(path)
    df = ingestor.ingest()
    if df.empty:
        return 0
    if write_db:
        return ingestor.to_db("db")
    return len(df)


def run_azure_ingestor() -> int:
    """Run Azure PdM ingestor; return rows written to sensor_readings."""
    from pipeline.ingestion.azure_pm_ingestor import AzurePMIngestor

    if not AZURE_PM_DIR.is_dir():
        return -1
    has_telemetry = (AZURE_PM_DIR / "PdM_telemetry.csv").exists() or (AZURE_PM_DIR / "telemetry.csv").exists()
    if not has_telemetry:
        return -1
    ingestor = AzurePMIngestor(AZURE_PM_DIR)
    return ingestor.run(write_cwru=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Load training data from inIT-OWL and Azure PdM")
    parser.add_argument("--skip-init-owl", action="store_true", help="Skip inIT-OWL ingestion")
    parser.add_argument("--skip-azure", action="store_true", help="Skip Azure PdM ingestion")
    parser.add_argument("--no-db", action="store_true", help="Do not write to DB (inIT-OWL: CSV only)")
    args = parser.parse_args()

    ensure_data_paths()

    write_db = not args.no_db
    rows_init_owl = None
    rows_azure = None

    if not args.skip_init_owl:
        path = get_init_owl_input_path()
        if path:
            rows_init_owl = run_init_owl_ingestor(write_db=write_db)
        else:
            print("Skipping inIT-OWL: no data/init_owl.csv or data/init_owl/*.csv found.")
    else:
        print("Skipping inIT-OWL (--skip-init-owl).")

    if not args.skip_azure:
        if AZURE_PM_DIR.is_dir() and (
            (AZURE_PM_DIR / "PdM_telemetry.csv").exists() or (AZURE_PM_DIR / "telemetry.csv").exists()
        ):
            rows_azure = run_azure_ingestor()
        else:
            print("Skipping Azure PdM: data/azure_pm/ missing or no telemetry CSV.")
    else:
        print("Skipping Azure PdM (--skip-azure).")

    # Summary table
    print("\n--- Summary ---")
    print(f"{'Dataset':<20} | {'Rows loaded':<12} | Table(s) written")
    print("-" * 55)
    if rows_init_owl is not None:
        status = str(rows_init_owl) if rows_init_owl >= 0 else "no data"
        tables = "cwru_features" if write_db else "stdout/CSV"
        print(f"{'inIT-OWL':<20} | {status:<12} | {tables}")
    if rows_azure is not None:
        status = str(rows_azure) if rows_azure >= 0 else "no data"
        print(f"{'Azure PdM':<20} | {status:<12} | sensor_readings, cwru_features")
    print("-" * 55)
    return 0


if __name__ == "__main__":
    sys.exit(main())
