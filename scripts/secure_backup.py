#!/usr/bin/env python3
"""
Secure Database Backup Script
=============================
Performs encrypted backups of PostgreSQL database to S3.

Features:
1. pg_dump database export
2. Gzip compression
3. Fernet symmetric encryption
4. Upload to private S3 bucket
5. Backup rotation and retention

Usage:
    python scripts/secure_backup.py                    # Full backup
    python scripts/secure_backup.py --dry-run          # Dry run (no upload)
    python scripts/secure_backup.py --generate-key     # Generate encryption key

Environment Variables Required:
    - DATABASE_URL: PostgreSQL connection string
    - AWS_ACCESS_KEY_ID: AWS access key
    - AWS_SECRET_ACCESS_KEY: AWS secret key
    - BACKUP_S3_BUCKET: S3 bucket name
    - BACKUP_ENCRYPTION_KEY: Fernet encryption key (base64)

Author: PlantAGI Security Team
"""

import os
import sys
import gzip
import shutil
import subprocess
import tempfile
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

DATABASE_URL = os.getenv('DATABASE_URL')
BACKUP_S3_BUCKET = os.getenv('BACKUP_S3_BUCKET', 'pdm-pilot-backups')
BACKUP_S3_PREFIX = os.getenv('BACKUP_S3_PREFIX', 'database-backups')
BACKUP_ENCRYPTION_KEY = os.getenv('BACKUP_ENCRYPTION_KEY')
BACKUP_RETENTION_DAYS = int(os.getenv('BACKUP_RETENTION_DAYS', 30))

# Backup naming
BACKUP_TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'


# =============================================================================
# ENCRYPTION UTILITIES
# =============================================================================

def generate_encryption_key() -> str:
    """Generate a new Fernet encryption key."""
    from cryptography.fernet import Fernet
    key = Fernet.generate_key()
    return key.decode('utf-8')


def get_fernet() -> 'Fernet':
    """Get Fernet cipher instance."""
    from cryptography.fernet import Fernet
    
    if not BACKUP_ENCRYPTION_KEY:
        raise ValueError("BACKUP_ENCRYPTION_KEY environment variable is required")
    
    return Fernet(BACKUP_ENCRYPTION_KEY.encode('utf-8'))


def encrypt_file(input_path: Path, output_path: Path) -> None:
    """Encrypt a file using Fernet symmetric encryption."""
    fernet = get_fernet()
    
    with open(input_path, 'rb') as f:
        data = f.read()
    
    encrypted_data = fernet.encrypt(data)
    
    with open(output_path, 'wb') as f:
        f.write(encrypted_data)
    
    logger.info(f"Encrypted: {input_path.name} -> {output_path.name}")


def decrypt_file(input_path: Path, output_path: Path) -> None:
    """Decrypt a Fernet-encrypted file."""
    fernet = get_fernet()
    
    with open(input_path, 'rb') as f:
        encrypted_data = f.read()
    
    decrypted_data = fernet.decrypt(encrypted_data)
    
    with open(output_path, 'wb') as f:
        f.write(decrypted_data)
    
    logger.info(f"Decrypted: {input_path.name} -> {output_path.name}")


# =============================================================================
# DATABASE BACKUP
# =============================================================================

def parse_database_url(url: str) -> dict:
    """Parse DATABASE_URL into components."""
    parsed = urlparse(url)
    
    return {
        'host': parsed.hostname or 'localhost',
        'port': parsed.port or 5432,
        'database': parsed.path.lstrip('/'),
        'user': parsed.username or 'postgres',
        'password': parsed.password or ''
    }


def run_pg_dump(output_path: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Run pg_dump to export database.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    if not DATABASE_URL:
        return False, "DATABASE_URL environment variable is required"
    
    db_config = parse_database_url(DATABASE_URL)
    
    # Build pg_dump command
    cmd = [
        'pg_dump',
        '-h', db_config['host'],
        '-p', str(db_config['port']),
        '-U', db_config['user'],
        '-d', db_config['database'],
        '-F', 'c',  # Custom format (compressed)
        '-b',       # Include large objects
        '-v',       # Verbose
        '-f', str(output_path)
    ]
    
    logger.info(f"Running pg_dump for database: {db_config['database']}")
    logger.info(f"Host: {db_config['host']}:{db_config['port']}")
    
    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        
        # Create a fake dump file for testing
        with open(output_path, 'wb') as f:
            f.write(b'-- Dry run backup placeholder\n')
            f.write(f'-- Database: {db_config["database"]}\n'.encode())
            f.write(f'-- Timestamp: {datetime.now().isoformat()}\n'.encode())
            f.write(b'-- This is a simulated backup for testing purposes.\n')
        
        return True, "Dry run completed - simulated backup created"
    
    # Set password via environment
    env = os.environ.copy()
    env['PGPASSWORD'] = db_config['password']
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            return False, f"pg_dump failed: {result.stderr}"
        
        logger.info(f"pg_dump completed successfully: {output_path}")
        return True, "Backup completed successfully"
        
    except subprocess.TimeoutExpired:
        return False, "pg_dump timed out after 1 hour"
    except FileNotFoundError:
        return False, "pg_dump command not found. Is PostgreSQL client installed?"
    except Exception as e:
        return False, f"pg_dump error: {str(e)}"


# =============================================================================
# COMPRESSION
# =============================================================================

def compress_file(input_path: Path, output_path: Path) -> None:
    """Compress a file using gzip."""
    with open(input_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb', compresslevel=9) as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    original_size = input_path.stat().st_size
    compressed_size = output_path.stat().st_size
    ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
    
    logger.info(f"Compressed: {original_size:,} bytes -> {compressed_size:,} bytes ({ratio:.1f}% reduction)")


def decompress_file(input_path: Path, output_path: Path) -> None:
    """Decompress a gzip file."""
    with gzip.open(input_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    logger.info(f"Decompressed: {input_path.name} -> {output_path.name}")


# =============================================================================
# S3 UPLOAD
# =============================================================================

def upload_to_s3(
    local_path: Path,
    bucket: str,
    s3_key: str,
    dry_run: bool = False
) -> Tuple[bool, str]:
    """
    Upload a file to S3 with server-side encryption.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would upload: {local_path.name} -> s3://{bucket}/{s3_key}")
        return True, f"[DRY RUN] Would upload to s3://{bucket}/{s3_key}"
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        s3_client = boto3.client('s3')
        
        # Upload with server-side encryption
        extra_args = {
            'ServerSideEncryption': 'AES256',  # SSE-S3
            'StorageClass': 'STANDARD_IA',     # Infrequent Access for cost savings
            'Metadata': {
                'backup-timestamp': datetime.now().isoformat(),
                'original-filename': local_path.name
            }
        }
        
        file_size = local_path.stat().st_size
        logger.info(f"Uploading to S3: {local_path.name} ({file_size:,} bytes)")
        
        s3_client.upload_file(
            str(local_path),
            bucket,
            s3_key,
            ExtraArgs=extra_args
        )
        
        s3_uri = f"s3://{bucket}/{s3_key}"
        logger.info(f"Upload complete: {s3_uri}")
        
        return True, s3_uri
        
    except ImportError:
        return False, "boto3 not installed. Run: pip install boto3"
    except ClientError as e:
        return False, f"S3 upload failed: {e}"
    except Exception as e:
        return False, f"Upload error: {str(e)}"


def cleanup_old_backups(
    bucket: str,
    prefix: str,
    retention_days: int,
    dry_run: bool = False
) -> int:
    """
    Delete backups older than retention period.
    
    Returns:
        Number of files deleted
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would cleanup backups older than {retention_days} days")
        return 0
    
    try:
        import boto3
        from datetime import timedelta
        
        s3_client = boto3.client('s3')
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        deleted_count = 0
        
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        for page in pages:
            for obj in page.get('Contents', []):
                if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                    s3_client.delete_object(Bucket=bucket, Key=obj['Key'])
                    logger.info(f"Deleted old backup: {obj['Key']}")
                    deleted_count += 1
        
        return deleted_count
        
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")
        return 0


# =============================================================================
# MAIN BACKUP WORKFLOW
# =============================================================================

def run_backup(dry_run: bool = False) -> dict:
    """
    Execute full backup workflow:
    1. pg_dump database
    2. Compress with gzip
    3. Encrypt with Fernet
    4. Upload to S3
    5. Cleanup old backups
    
    Returns:
        Dictionary with backup results
    """
    result = {
        'success': False,
        'timestamp': datetime.now().isoformat(),
        'dry_run': dry_run,
        'steps': [],
        'errors': []
    }
    
    logger.info("=" * 60)
    logger.info("SECURE DATABASE BACKUP")
    logger.info("=" * 60)
    
    if dry_run:
        logger.info("🔍 DRY RUN MODE - No actual changes will be made")
    
    timestamp = datetime.now().strftime(BACKUP_TIMESTAMP_FORMAT)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # File paths
        dump_file = temp_path / f"pdm_backup_{timestamp}.dump"
        compressed_file = temp_path / f"pdm_backup_{timestamp}.dump.gz"
        encrypted_file = temp_path / f"pdm_backup_{timestamp}.dump.gz.enc"
        
        # Step 1: pg_dump
        logger.info("\n[1/4] Database Export (pg_dump)")
        success, message = run_pg_dump(dump_file, dry_run=dry_run)
        result['steps'].append({'step': 'pg_dump', 'success': success, 'message': message})
        
        if not success:
            result['errors'].append(message)
            logger.error(f"pg_dump failed: {message}")
            return result
        
        # Step 2: Compress
        logger.info("\n[2/4] Compression (gzip)")
        try:
            compress_file(dump_file, compressed_file)
            result['steps'].append({'step': 'compress', 'success': True, 'message': 'Compressed successfully'})
        except Exception as e:
            result['errors'].append(str(e))
            logger.error(f"Compression failed: {e}")
            return result
        
        # Step 3: Encrypt
        logger.info("\n[3/4] Encryption (Fernet)")
        try:
            if not dry_run and not BACKUP_ENCRYPTION_KEY:
                # Generate key for dry run
                logger.warning("No encryption key set - generating temporary key for demo")
                os.environ['BACKUP_ENCRYPTION_KEY'] = generate_encryption_key()
                
            encrypt_file(compressed_file, encrypted_file)
            result['steps'].append({'step': 'encrypt', 'success': True, 'message': 'Encrypted successfully'})
            result['encrypted_size'] = encrypted_file.stat().st_size
        except Exception as e:
            result['errors'].append(str(e))
            logger.error(f"Encryption failed: {e}")
            return result
        
        # Step 4: Upload to S3
        logger.info("\n[4/4] S3 Upload")
        s3_key = f"{BACKUP_S3_PREFIX}/{timestamp}/pdm_backup_{timestamp}.dump.gz.enc"
        success, message = upload_to_s3(encrypted_file, BACKUP_S3_BUCKET, s3_key, dry_run=dry_run)
        result['steps'].append({'step': 's3_upload', 'success': success, 'message': message})
        
        if success:
            result['s3_location'] = message if not dry_run else f"s3://{BACKUP_S3_BUCKET}/{s3_key}"
        else:
            result['errors'].append(message)
            logger.warning(f"S3 upload issue: {message}")
        
        # Cleanup old backups
        if not dry_run:
            deleted = cleanup_old_backups(
                BACKUP_S3_BUCKET,
                BACKUP_S3_PREFIX,
                BACKUP_RETENTION_DAYS,
                dry_run=dry_run
            )
            result['old_backups_deleted'] = deleted
    
    # Final status
    result['success'] = len(result['errors']) == 0
    
    logger.info("\n" + "=" * 60)
    if result['success']:
        logger.info("✅ BACKUP COMPLETED SUCCESSFULLY")
    else:
        logger.error("❌ BACKUP COMPLETED WITH ERRORS")
    logger.info("=" * 60)
    
    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Secure encrypted database backup to S3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/secure_backup.py                    # Full backup
    python scripts/secure_backup.py --dry-run          # Test without uploading
    python scripts/secure_backup.py --generate-key     # Generate encryption key
    python scripts/secure_backup.py --decrypt FILE     # Decrypt a backup file
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate backup without uploading'
    )
    
    parser.add_argument(
        '--generate-key',
        action='store_true',
        help='Generate a new Fernet encryption key'
    )
    
    parser.add_argument(
        '--decrypt',
        metavar='FILE',
        help='Decrypt a backup file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.generate_key:
        key = generate_encryption_key()
        print("\n🔑 Generated Fernet Encryption Key:")
        print("=" * 60)
        print(key)
        print("=" * 60)
        print("\nAdd this to your .env file:")
        print(f"BACKUP_ENCRYPTION_KEY={key}")
        return 0
    
    if args.decrypt:
        input_path = Path(args.decrypt)
        output_path = input_path.with_suffix('')  # Remove .enc
        try:
            decrypt_file(input_path, output_path)
            print(f"✅ Decrypted to: {output_path}")
            return 0
        except Exception as e:
            print(f"❌ Decryption failed: {e}")
            return 1
    
    # Run backup
    result = run_backup(dry_run=args.dry_run)
    
    # Print summary
    print("\n📋 Backup Summary:")
    print(f"   Timestamp: {result['timestamp']}")
    print(f"   Dry Run: {result['dry_run']}")
    print(f"   Success: {result['success']}")
    
    if result.get('s3_location'):
        print(f"   S3 Location: {result['s3_location']}")
    
    if result.get('encrypted_size'):
        print(f"   Encrypted Size: {result['encrypted_size']:,} bytes")
    
    if result['errors']:
        print(f"   Errors: {len(result['errors'])}")
        for error in result['errors']:
            print(f"      - {error}")
    
    return 0 if result['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
