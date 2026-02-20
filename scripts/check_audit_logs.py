#!/usr/bin/env python3
"""
Check Audit Logs Script
=======================
Verifies that the Audit Logger is successfully recording events to PostgreSQL.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    print("FAILURE: DATABASE_URL environment variable is required")
    sys.exit(1)

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError
except ImportError:
    print("FAILURE: sqlalchemy is not installed. Run: pip install sqlalchemy")
    sys.exit(1)

def check_audit_logs():
    """Connect to the database and retrieve the last 5 audit log entries."""
    try:
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as conn:
            # Check if audit_log table exists (append-only audit trail)
            check_table_query = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'audit_log'
                );
            """)
            result = conn.execute(check_table_query)
            table_exists = result.scalar()
            
            if not table_exists:
                print("FAILURE: No logs found. (audit_log table does not exist. Run migrations.)")
                return
            
            # Get the last 5 rows from audit_log
            query = text("""
                SELECT * FROM audit_log
                ORDER BY id DESC
                LIMIT 5;
            """)
            result = conn.execute(query)
            rows = result.fetchall()
            
            if not rows:
                print("FAILURE: No logs found. (audit_log table is empty)")
                return
            
            # Get column names (SQLAlchemy 2.0)
            columns = result.keys()
            
            print(f"SUCCESS: Found {len(rows)} audit log entries\n")
            print("=" * 80)
            
            for i, row in enumerate(rows, 1):
                print(f"\n--- Entry {i} ---")
                for col, val in zip(columns, row):
                    print(f"  {col}: {val}")
            
            print("\n" + "=" * 80)
            print(f"\nTotal entries shown: {len(rows)}")
            
    except SQLAlchemyError as e:
        print(f"FAILURE: Database error - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"FAILURE: Unexpected error - {e}")
        sys.exit(1)


if __name__ == "__main__":
    check_audit_logs()
