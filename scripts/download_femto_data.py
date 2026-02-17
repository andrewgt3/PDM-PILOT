#!/usr/bin/env python3
"""
Download FEMTO/PRONOSTIA Bearing Dataset
========================================
Fetches the IEEE PHM 2012 Data Challenge dataset (FEMTO-ST PRONOSTIA).

Source: https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset
- Learning_set: 6 run-to-failure bearings (Bearing1_1, 1_2, 2_1, 2_2, 3_1, 3_2)
- Test_set: 11 bearings
- Full_Test_Set: extended test

Output: data/downloads/femto_pronostia/
  Learning_set/Bearing1_1/acc_00001.csv, ...
  Test_set/...
  Full_Test_Set/...

Usage:
    python scripts/download_femto_data.py
    python scripts/download_femto_data.py --learning-only  # smaller, ~50MB
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import urllib.request
except ImportError:
    urllib = None

BASE = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = BASE / "data" / "downloads"
FEMTO_DIR = DOWNLOAD_DIR / "femto_pronostia"
GITHUB_ZIP = "https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset/archive/refs/heads/master.zip"


def download_file(url: str, dest: Path, label: str = "") -> bool:
    """Download with progress. Prefer requests (better SSL handling on macOS)."""
    label = label or url
    if HAS_REQUESTS:
        try:
            print(f"Downloading {label}...")
            resp = requests.get(url, stream=True, timeout=120, headers={"User-Agent": "PDM-PILOT/1.0"})
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 65536
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
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
    if urllib:
        try:
            print(f"Downloading {label}...")
            req = urllib.request.Request(url, headers={"User-Agent": "PDM-PILOT/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 65536
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
    print("  ERROR: Neither requests nor urllib available.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Download FEMTO/PRONOSTIA dataset")
    parser.add_argument(
        "--learning-only",
        action="store_true",
        help="Download only Learning_set (smaller; use git sparse-checkout or manual)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(FEMTO_DIR),
        help="Output directory (default: data/downloads/femto_pronostia)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = DOWNLOAD_DIR / "femto_pronostia.zip"

    print("=" * 60)
    print("FEMTO/PRONOSTIA Dataset Download")
    print("=" * 60)
    print(f"Output: {out_dir}")
    print()

    if not HAS_REQUESTS and urllib is None:
        print("ERROR: Neither requests nor urllib available. pip install requests")
        sys.exit(1)

    # Download full repo zip (~100-150 MB)
    if not zip_path.exists() or zip_path.stat().st_size < 1000:
        if not download_file(GITHUB_ZIP, zip_path, "FEMTO PHM 2012 dataset (~100 MB)"):
            print("\nDownload failed. Ensure you have disk space and network access.")
            sys.exit(1)
    else:
        print(f"Using existing {zip_path}")

    # Extract
    print("Extracting...")
    try:
        import shutil
        extract_root = out_dir.parent
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.startswith("phm-ieee-2012-data-challenge-dataset-master/"):
                    rel = name[len("phm-ieee-2012-data-challenge-dataset-master/"):]
                    if args.learning_only and not rel.startswith("Learning_set"):
                        continue
                    zf.extract(name, extract_root)
        # Move from phm-ieee-...-master/ to femto_pronostia/
        extracted = extract_root / "phm-ieee-2012-data-challenge-dataset-master"
        if extracted.exists():
            for item in extracted.iterdir():
                dest = out_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
            extracted.rmdir()
        print("Done.")
    except Exception as e:
        print(f"Extract failed: {e}")
        sys.exit(1)

    # Verify
    learning = out_dir / "Learning_set"
    if learning.exists():
        bearings = list(learning.iterdir()) if learning.is_dir() else []
        n_files = sum(1 for _ in learning.rglob("*.csv"))
        print(f"\nLearning_set: {len(bearings)} bearings, {n_files} CSV files")
    print(f"\nFEMTO data ready at: {out_dir}")


if __name__ == "__main__":
    main()
