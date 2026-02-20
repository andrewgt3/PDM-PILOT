#!/usr/bin/env python3
"""
Download Kaggle training sets for predictive maintenance.

Datasets:
  1. Microsoft Azure Predictive Maintenance
     https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance
  2. inIT-OWL: one slug via --init-owl-dataset, or all via --init-owl-all
     https://www.kaggle.com/organizations/inIT-OWL
     (All: lists org datasets, downloads each, then writes docs/INIT_OWL_TRAINING_RECOMMENDATION.md)

Requires Kaggle API credentials:
  - Create an API token at https://www.kaggle.com/settings (Account -> Create New Token).
  - Place kaggle.json at ~/.kaggle/kaggle.json (or set KAGGLE_USERNAME and KAGGLE_KEY).

Usage:
  pip install kaggle
  python scripts/download_kaggle_data.py
  python scripts/download_kaggle_data.py --azure-only
  python scripts/download_kaggle_data.py --init-owl-dataset init-owl/some-dataset-name
  python scripts/download_kaggle_data.py --init-owl-all   # download all inIT-OWL + recommendation
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = BASE / "data" / "downloads" / "kaggle"

# Dataset slugs (owner/dataset-name)
AZURE_PDM_SLUG = "arnabbiswas1/microsoft-azure-predictive-maintenance"
# inIT-OWL is an organization; specific dataset slug set via env or --init-owl-dataset
INIT_OWL_DATASET_ENV = "KAGGLE_INIT_OWL_DATASET"


def run_kaggle_download(dataset_slug: str, dest_dir: Path) -> bool:
    """Download a Kaggle dataset via 'kaggle datasets download -d <slug>' and unzip into dest_dir."""
    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Kaggle downloads to current working dir as <dataset-name>.zip; we run in dest_dir.parent
    work_dir = dest_dir.parent
    zip_name = dataset_slug.split("/")[-1] + ".zip"
    zip_path = work_dir / zip_name

    try:
        cmd = [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset_slug,
            "-p",
            str(work_dir),
        ]
        print(f"Downloading {dataset_slug}...")
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=600, cwd=str(BASE))
        if result.returncode != 0:
            stderr = result.stderr or ""
            stdout = result.stdout or ""
            print(f"  ERROR: kaggle exit {result.returncode}")
            if stderr:
                print(stderr.strip())
            if stdout and "404" in (stdout + stderr):
                print("  Hint: Check dataset slug and that you have accepted the dataset rules on Kaggle.")
            return False

        # Unzip into dest_dir
        if zip_path.exists():
            print(f"Extracting to {dest_dir}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest_dir)
            zip_path.unlink()
        else:
            # Some datasets download to a folder already
            for f in work_dir.iterdir():
                if f.suffix == ".zip" and dataset_slug.split("/")[-1] in f.stem:
                    with zipfile.ZipFile(f, "r") as zf:
                        zf.extractall(dest_dir)
                    f.unlink()
                    break
        print(f"  Done: {dest_dir}")
        return True
    except subprocess.TimeoutExpired:
        print("  ERROR: Download timed out (10 min)")
        return False
    except FileNotFoundError:
        print("  ERROR: 'kaggle' not found. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Kaggle PdM datasets")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(DOWNLOAD_DIR),
        help=f"Base output directory (default: {DOWNLOAD_DIR})",
    )
    parser.add_argument(
        "--azure-only",
        action="store_true",
        help="Download only Microsoft Azure Predictive Maintenance",
    )
    parser.add_argument(
        "--init-owl-dataset",
        default=os.environ.get(INIT_OWL_DATASET_ENV, ""),
        help="inIT-OWL dataset slug (e.g. init-owl/dataset-name). Or set env KAGGLE_INIT_OWL_DATASET.",
    )
    parser.add_argument(
        "--init-owl-all",
        action="store_true",
        help="Download all inIT-OWL org datasets and generate docs/INIT_OWL_TRAINING_RECOMMENDATION.md.",
    )
    args = parser.parse_args()

    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Kaggle PdM datasets download")
    print("=" * 60)
    print(f"Output base: {base}")
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        print("\nWARNING: ~/.kaggle/kaggle.json not found. Create an API token at https://www.kaggle.com/settings")
    print()

    ok = True

    # 1. Microsoft Azure Predictive Maintenance
    azure_dest = base / "microsoft_azure_predictive_maintenance"
    if not run_kaggle_download(AZURE_PDM_SLUG, azure_dest):
        ok = False

    # 2. inIT-OWL: one slug or all
    if args.init_owl_all:
        if not args.azure_only:
            init_owl_dir = base / "init_owl"
            subprocess.run(
                [sys.executable, str(BASE / "scripts" / "download_init_owl_datasets.py"), "-o", str(init_owl_dir)],
                cwd=str(BASE),
                check=False,
            )
        else:
            print("Use --init-owl-all without --azure-only to download all inIT-OWL datasets.")
    elif not args.azure_only and args.init_owl_dataset.strip():
        slug = args.init_owl_dataset.strip()
        dir_name = slug.replace("/", "_").replace(" ", "_")
        init_dest = base / dir_name
        if not run_kaggle_download(slug, init_dest):
            ok = False
    elif not args.azure_only:
        print("Skipping inIT-OWL: set KAGGLE_INIT_OWL_DATASET, --init-owl-dataset, or --init-owl-all")

    print()
    if ok:
        print("Downloads complete.")
    else:
        print("One or more downloads failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
