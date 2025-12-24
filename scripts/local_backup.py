#!/usr/bin/env python3
"""
Local Database Backup Script
============================
Simple disaster recovery backup for PostgreSQL/TimescaleDB in Docker.

Features:
1. pg_dump via Docker container
2. Local file storage with rotation
3. Auto-cleanup of backups older than 7 days

Usage:
    python scripts/local_backup.py

Environment Variables (from .env):
    - DT_POSTGRES_USER: Database username (default: postgres)
    - DT_POSTGRES_PASSWORD: Database password (default: password)  
    - DT_POSTGRES_DB: Database name (default: pdm_timeseries)

Author: PlantAGI Team
"""

import os
import sys
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIGURATION (from environment variables)
# =============================================================================
POSTGRES_USER = os.getenv('DT_POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.getenv('DT_POSTGRES_PASSWORD', 'password')
POSTGRES_DB = os.getenv('DT_POSTGRES_DB', 'pdm_timeseries')
CONTAINER_NAME = os.getenv('DB_CONTAINER_NAME', 'pdm-pilot-timescaledb-1')

# Backup settings
BACKUP_DIR = Path(__file__).parent.parent / 'backups'
RETENTION_DAYS = 7


def ensure_backup_dir():
    """Create backups directory if it doesn't exist."""
    BACKUP_DIR.mkdir(exist_ok=True)
    print(f"✓ Backup directory: {BACKUP_DIR}")


def run_pg_dump() -> tuple[bool, Path | None]:
    """
    Execute pg_dump inside the running database container.
    
    Returns:
        Tuple of (success, backup_file_path)
    """
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M')
    backup_filename = f"backup_{timestamp}.sql"
    backup_path = BACKUP_DIR / backup_filename
    
    print(f"\n📦 Starting backup: {backup_filename}")
    print(f"   Database: {POSTGRES_DB}")
    print(f"   User: {POSTGRES_USER}")
    print(f"   Container: {CONTAINER_NAME}")
    
    # Build docker exec command for pg_dump
    cmd = [
        'docker', 'exec',
        '-e', f'PGPASSWORD={POSTGRES_PASSWORD}',
        CONTAINER_NAME,
        'pg_dump',
        '-U', POSTGRES_USER,
        '-d', POSTGRES_DB,
        '--no-password'
    ]
    
    try:
        # Run pg_dump and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8', errors='replace')
            print(f"❌ pg_dump failed: {error_msg}")
            return False, None
        
        # Write output to backup file
        with open(backup_path, 'wb') as f:
            f.write(result.stdout)
        
        print(f"✓ pg_dump completed successfully")
        return True, backup_path
        
    except subprocess.TimeoutExpired:
        print("❌ pg_dump timed out after 10 minutes")
        return False, None
    except FileNotFoundError:
        print("❌ Docker command not found. Is Docker installed?")
        return False, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, None


def verify_backup(backup_path: Path) -> bool:
    """Verify the backup file was created and is not empty."""
    if not backup_path.exists():
        print("❌ FAILURE: Backup file was not created")
        return False
    
    file_size = backup_path.stat().st_size
    if file_size == 0:
        print("❌ FAILURE: Backup file is empty")
        return False
    
    print(f"✓ Backup verified: {file_size:,} bytes")
    return True


def cleanup_old_backups():
    """Delete backups older than RETENTION_DAYS."""
    cutoff_date = datetime.now() - timedelta(days=RETENTION_DAYS)
    deleted_count = 0
    
    print(f"\n🧹 Cleaning up backups older than {RETENTION_DAYS} days...")
    
    for backup_file in BACKUP_DIR.glob('backup_*.sql'):
        file_mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
        
        if file_mtime < cutoff_date:
            backup_file.unlink()
            print(f"   Deleted: {backup_file.name}")
            deleted_count += 1
    
    if deleted_count > 0:
        print(f"✓ Deleted {deleted_count} old backup(s)")
    else:
        print("✓ No old backups to delete")
    
    return deleted_count


def main():
    """Main backup workflow."""
    print("=" * 60)
    print("LOCAL DATABASE BACKUP")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Step 1: Ensure backup directory exists
    ensure_backup_dir()
    
    # Step 2: Run pg_dump
    success, backup_path = run_pg_dump()
    
    if not success or backup_path is None:
        print("\n❌ FAILURE: No logs found.")
        return 1
    
    # Step 3: Verify backup
    if not verify_backup(backup_path):
        print("\n❌ FAILURE: No logs found.")
        return 1
    
    # Step 4: Cleanup old backups
    cleanup_old_backups()
    
    # Success summary
    print("\n" + "=" * 60)
    print("✅ BACKUP COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"   File: {backup_path}")
    print(f"   Size: {backup_path.stat().st_size:,} bytes")
    print(f"   Retention: {RETENTION_DAYS} days")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
