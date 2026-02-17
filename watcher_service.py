#!/usr/bin/env python3
"""
Drop Zone Watcher Service
=========================
Monitors `data/ingest/` for new files.
- Moves files to `data/raw/nasa_ims/`
- Generates unique Ingestion ID (UUID)
- Logs event with ISO timestamp
- Triggers `refinery.py` subprocess

Usage:
    python watcher_service.py
"""

import sys
import time
import os
import shutil
import logging
import uuid
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent
INGEST_DIR = BASE_DIR / "data" / "ingest"
RAW_DIR = BASE_DIR / "data" / "raw" / "nasa_ims"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "data" / "models"

# Ensure directory structure exists
for directory in [INGEST_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ingestion_audit.log")
    ]
)
logger = logging.getLogger("watcher")

# =============================================================================
# EVENT HANDLER
# =============================================================================

class DropZoneHandler(FileSystemEventHandler):
    def on_created(self, event):
        """Triggered when a file is created in the watched directory."""
        if event.is_directory:
            return

        filepath = Path(event.src_path)
        
        # Avoid processing hidden files or temporary uploads
        if filepath.name.startswith("."):
            return

        self.process_ingestion(filepath)

    def on_moved(self, event):
        """Triggered when a file is moved into the watched directory."""
        if event.is_directory:
            return

        filepath = Path(event.dest_path)
        self.process_ingestion(filepath)

    def process_ingestion(self, filepath: Path):
        """
        Core Logic:
        1. Generate Ingestion ID
        2. Move to RAW (Preserving subfolder structure)
        3. Audit Log
        4. Trigger Refinery
        """
        # Wait briefly to ensure file write is complete (debounce)
        # Larger debounce for nested uploads
        time.sleep(0.5)
        
        try:
            if not filepath.exists():
                return

            # Calculate relative path to preserve directory structure
            rel_path = filepath.relative_to(INGEST_DIR)
            target_path = RAW_DIR / rel_path
            
            # Ensure target parent directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            ingestion_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # 1. Move File
            shutil.move(str(filepath), str(target_path))
            
            # 2. Audit Log
            log_entry = {
                "event": "INGESTION_COMPLETE",
                "ingestion_id": ingestion_id,
                "timestamp": timestamp,
                "file": str(rel_path),
                "status": "MOVED_TO_RAW"
            }
            logger.info("Ingestion Event: %s", log_entry)

            # 3. Notify Redis (Ingestion Stage visibility)
            try:
                import redis
                import json
                r_direct = redis.Redis(host='localhost', port=6379, decode_responses=True)
                payload = {
                    "machine_id": "Bearing_1",
                    "timestamp": timestamp,
                    "filename": str(rel_path),
                    "event": "FILE_INGESTED"
                }
                r_direct.xadd("raw_sensor_data", {"data": json.dumps(payload)}, maxlen=10000)
            except Exception as re:
                logger.error("Failed to notify Redis: %s", re)
            
            # 4. Trigger Refinery
            logger.info("Triggering refinery for: %s", rel_path)
            subprocess.Popen([sys.executable, "refinery.py", str(target_path)])

        except Exception as e:
            logger.error("Failed to process file %s: %s", filepath, e, exc_info=True)

# =============================================================================
# MAIN LOOP
# =============================================================================

def start_watcher():
    """Starts the watchdog observer."""
    event_handler = DropZoneHandler()
    
    # 1. SCAN FOR EXISTING FILES FIRST
    # This handles files that were extracted BEFORE the watcher started
    logger.info("Performing initial scan of %s", INGEST_DIR)
    for root, dirs, files in os.walk(INGEST_DIR):
        for file in files:
            if file.startswith("."):
                continue
            filepath = Path(root) / file
            logger.info("Processing existing file: %s", file)
            event_handler.process_ingestion(filepath)

    observer = Observer()
    # ENABLE RECURSION
    observer.schedule(event_handler, str(INGEST_DIR), recursive=True)
    
    observer.start()
    logger.info("Recursive Watcher Service started.")
    logger.info("Monitoring Drop Zone: %s", INGEST_DIR)
    logger.info("Target Raw Folder:    %s", RAW_DIR)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Watcher Service stopped by user.")
    
    observer.join()

if __name__ == "__main__":
    start_watcher()
