#!/usr/bin/env python3
"""
Bootstrap Script: Local Archive Ingestion
=========================================
Extracts the NASA archive from the Desktop directly into the Drop Zone.
"""

import zipfile
import shutil
from pathlib import Path
import os
import sys

# Paths (NASA_ARCHIVE_PATH env overrides default: ~/Desktop/archive.zip)
ARCHIVE_PATH = Path(
    os.environ.get("NASA_ARCHIVE_PATH", str(Path.home() / "Desktop" / "archive.zip"))
)
INGEST_DIR = Path("data/ingest")

def bootstrap():
    """Extracts or copies the archive to the ingest folder. Supports .zip or folder."""
    if not ARCHIVE_PATH.exists():
        print(f"❌ Error: Archive not found at {ARCHIVE_PATH}")
        print("Set NASA_ARCHIVE_PATH to your archive.zip or archive folder.")
        sys.exit(1)

    INGEST_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if ARCHIVE_PATH.is_dir():
            # Folder: copy contents into ingest
            print(f"🚀 Found archive folder: {ARCHIVE_PATH}")
            print(f"📦 Copying to {INGEST_DIR} ... (This may take a minute)")
            file_count = 0
            for root, dirs, files in os.walk(ARCHIVE_PATH):
                for f in files:
                    if f.startswith("."):
                        continue
                    src = Path(root) / f
                    rel = src.relative_to(ARCHIVE_PATH)
                    dst = INGEST_DIR / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    file_count += 1
                    if file_count % 500 == 0:
                        print(f"  Copied {file_count} files...")
            print(f"✅ Copied {file_count} files.")
        else:
            # Zip file: extract
            size_mb = ARCHIVE_PATH.stat().st_size / 1e6
            print(f"🚀 Found archive: {ARCHIVE_PATH} ({size_mb:.1f} MB)")
            print(f"📦 Extracting to {INGEST_DIR} ... (This may take a minute)")
            with zipfile.ZipFile(ARCHIVE_PATH, 'r') as zip_ref:
                zip_ref.extractall(INGEST_DIR)
            print("✅ Extraction complete.")

        print("-" * 40)
        print("NEXT STEPS:")
        print(f"1. Ensure 'python3 watcher_service.py' is running.")
        print(f"2. The Watcher will process all files in {INGEST_DIR}.")
        print(f"3. Follow progress in 'ingestion_audit.log' or the terminal logs.")
        print("-" * 40)

    except Exception as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    bootstrap()
