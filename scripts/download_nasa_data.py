#!/usr/bin/env python3
"""
Download Full NASA Predictive Maintenance Datasets
===================================================
Fetches the official NASA datasets for training. Run this to get the full
datasets instead of the small sample archive.

Datasets:
  1. NASA C-MAPSS - Turbofan engine degradation (FD001-FD004)
  2. NASA IMS - Bearing run-to-failure experiments

Output: archive.zip in project root (or Desktop) ready for "Initialize NASA Data"

Usage:
    python scripts/download_nasa_data.py
    python scripts/download_nasa_data.py --output ~/Desktop/archive.zip
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

try:
    import urllib.request
except ImportError:
    urllib = None

# NASA Open Data Portal direct download URLs
NASA_CMAPSS_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
NASA_IMS_URL = "https://data.nasa.gov/docs/legacy/IMS.zip"

# Approximate sizes for progress
CMAPSS_SIZE_MB = 15
IMS_SIZE_MB = 50


def download_file(url: str, dest: Path, label: str = "") -> bool:
    """Download a file with progress indication."""
    try:
        print(f"Downloading {label or url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "PDM-PILOT/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 8192
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total and total > 0:
                        pct = 100 * downloaded / total
                        print(f"\r  {pct:.1f}% ({downloaded / 1e6:.1f} MB)", end="", flush=True)
            print()
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download NASA PdM datasets")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output path for archive.zip (default: project root or ~/Desktop)",
    )
    parser.add_argument(
        "--cmapss-only",
        action="store_true",
        help="Download only C-MAPSS (turbofan engine data)",
    )
    parser.add_argument(
        "--ims-only",
        action="store_true",
        help="Download only IMS (bearing data)",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Keep datasets separate (don't create combined archive.zip)",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    download_dir = base / "data" / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output) if args.output else (Path.home() / "Desktop" / "archive.zip")

    print("=" * 60)
    print("NASA Predictive Maintenance Dataset Downloader")
    print("=" * 60)
    print()

    success = True

    # 1. C-MAPSS (Turbofan engine degradation)
    if not args.ims_only:
        cmapss_zip = download_dir / "CMAPSSData.zip"
        if not cmapss_zip.exists() or cmapss_zip.stat().st_size < 1000:
            if not download_file(NASA_CMAPSS_URL, cmapss_zip, "C-MAPSS (~15 MB)"):
                success = False
        else:
            print("C-MAPSS already downloaded, skipping.")

    # 2. IMS (Bearing run-to-failure)
    if not args.cmapss_only:
        ims_zip = download_dir / "IMS.zip"
        if not ims_zip.exists() or ims_zip.stat().st_size < 1000:
            if not download_file(NASA_IMS_URL, ims_zip, "IMS Bearings (~50 MB)"):
                success = False
        else:
            print("IMS already downloaded, skipping.")

    if not success:
        print("\nSome downloads failed. Check your internet connection.")
        sys.exit(1)

    # 3. Create combined archive for pipeline bootstrap
    if not args.no_merge:
        print("\nCreating combined archive.zip for pipeline...")
        combined_dir = download_dir / "combined"
        combined_dir.mkdir(exist_ok=True)

        for z in download_dir.glob("*.zip"):
            if z.name == "archive.zip":
                continue
            try:
                with zipfile.ZipFile(z, "r") as zh:
                    zh.extractall(combined_dir)
                print(f"  Extracted {z.name}")
            except Exception as e:
                print(f"  Warning extracting {z.name}: {e}")

        # Repack as archive.zip
        archive_path = output_path
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for root, dirs, files in os.walk(combined_dir):
                for f in files:
                    if f.startswith("."):
                        continue
                    fp = Path(root) / f
                    arcname = fp.relative_to(combined_dir)
                    zout.write(fp, arcname)
        print(f"\nCreated: {archive_path}")
        print(f"Size: {archive_path.stat().st_size / 1e6:.1f} MB")

    print()
    print("=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Place archive.zip on your Desktop, or set NASA_ARCHIVE_PATH in .env")
    if not args.no_merge:
        print(f"   Current location: {output_path}")
    print("2. Click 'Initialize NASA Data' in the Pipeline Operations dashboard")
    print("3. The watcher will process files; refinery extracts features")
    print("4. Train models: python train_rul_model.py (or train_model.py)")
    print()


if __name__ == "__main__":
    main()
