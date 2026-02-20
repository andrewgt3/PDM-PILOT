#!/usr/bin/env python3
"""
Copy manually downloaded Kaggle archives from Desktop into project layout.

Source (Desktop):
  archive (1)  -> C*.csv bearing sensor data (Timestamp, L_1, A_1, B_1, C_1, ...)
  archive (2)  -> Industrial control timeseries (pCut::, pSvolFilm::, pSpintor::, ...)
  archive (3)  -> Production/packaging modules (Delivery, Dosing, Filling_*, Production, Storage)

Target (project):
  archive (1)  -> data/downloads/kaggle/init_owl/bearing_condition_runs/
  archive (2)  -> data/downloads/kaggle/init_owl/industrial_control_timeseries/
  archive (3)  -> data/downloads/kaggle/init_owl/production_modules/

Optional: archive (no number) = NASA IMS (1st_test, 2nd_test, 3rd_test) -> data/raw/nasa_ims/
          (only copies if files are missing; project may already have this.)

Usage:
  python scripts/ingest_kaggle_archives_from_desktop.py
  python scripts/ingest_kaggle_archives_from_desktop.py --desktop /path/to/Desktop
"""

import argparse
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DEFAULT_DESKTOP = Path.home() / "Desktop"

MAPPING = [
    {
        "source": "archive (1)",
        "dest": BASE / "data" / "downloads" / "kaggle" / "init_owl" / "bearing_condition_runs",
        "id": "bearing_condition_runs",
        "description": "Bearing condition runs (C7-1, C8, C9, ...); columns Timestamp, L_*, A_*, B_*, C_*.",
    },
    {
        "source": "archive (2)",
        "dest": BASE / "data" / "downloads" / "kaggle" / "init_owl" / "industrial_control_timeseries",
        "id": "industrial_control_timeseries",
        "description": "Industrial control/sensor timeseries (pCut::, pSvolFilm::, pSpintor::); one CSV per run.",
    },
    {
        "source": "archive (3)",
        "dest": BASE / "data" / "downloads" / "kaggle" / "init_owl" / "production_modules",
        "id": "production_modules",
        "description": "Production/packaging modules (Delivery, Dosing, Filling_*, Production, Storage).",
    },
]
NASA_IMS_SOURCE = "archive"
NASA_IMS_DEST = BASE / "data" / "raw" / "nasa_ims"


def copy_tree(src: Path, dest: Path, description: str) -> bool:
    if not src.exists():
        print(f"  Skip (not found): {src}")
        return False
    dest.mkdir(parents=True, exist_ok=True)
    # Copy contents of src into dest (merge)
    for item in src.iterdir():
        if item.name.startswith("."):
            continue
        dest_item = dest / item.name
        if item.is_dir():
            if dest_item.exists():
                shutil.rmtree(dest_item)
            shutil.copytree(item, dest_item)
        else:
            shutil.copy2(item, dest_item)
    print(f"  Copied: {src} -> {dest}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Ingest Kaggle archives from Desktop into project")
    parser.add_argument("--desktop", default=str(DEFAULT_DESKTOP), help="Desktop path containing archive (1), (2), (3)")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be done")
    args = parser.parse_args()
    desktop = Path(args.desktop)

    print("=" * 60)
    print("Ingest Kaggle archives into PDM-PILOT")
    print("=" * 60)
    print(f"Source: {desktop}")
    print()

    for m in MAPPING:
        src = desktop / m["source"]
        dest = m["dest"]
        print(f"{m['id']}:")
        if args.dry_run:
            print(f"  Would copy {src} -> {dest}")
            continue
        if copy_tree(src, dest, m["description"]):
            readme = dest / "README_SOURCE.txt"
            readme.write_text(
                f"Source: Desktop/{m['source']}\n"
                f"Dataset: {m['id']}\n"
                f"Description: {m['description']}\n"
                f"Ingested by scripts/ingest_kaggle_archives_from_desktop.py\n",
                encoding="utf-8",
            )
    print()

    # Optional: NASA IMS from "archive" (no number)
    src_ims = desktop / NASA_IMS_SOURCE
    if src_ims.exists() and not args.dry_run:
        # Only copy if we're missing 1st_test or 2nd_test (project often has 3rd_test)
        dest_ims = NASA_IMS_DEST
        dest_ims.mkdir(parents=True, exist_ok=True)
        for name in ["1st_test", "2nd_test", "3rd_test"]:
            src_sub = src_ims / name
            if src_sub.exists() and not (dest_ims / name).exists():
                shutil.copytree(src_sub, dest_ims / name)
                print(f"  Copied NASA IMS: {name} -> {dest_ims}")
        readme_ims = dest_ims / "README_IMS.txt"
        if not readme_ims.exists():
            readme_ims.write_text(
                "NASA IMS Bearing Data (1st_test, 2nd_test, 3rd_test).\n"
                "Optional source: Desktop/archive (if present).\n",
                encoding="utf-8",
            )

    print("Done.")


if __name__ == "__main__":
    main()
