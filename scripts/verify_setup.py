#!/usr/bin/env python3
"""
PostgreSQL Production Setup Verification
=========================================
Verifies that all components are properly configured for production deployment.

Usage: python3 verify_setup.py
"""

import os
import sys

def check_env_file():
    """Check if .env file exists and has DB_CONNECTION."""
    print("1️⃣  Checking .env file...")
    
    if not os.path.exists('.env'):
        print("   ❌ .env file not found!")
        print("   📝 Create it with:")
        print("      echo 'DB_CONNECTION=postgresql://gaia_admin:tier1_secure@localhost:5432/pdm_timeseries' > .env")
        return False
    
    with open('.env', 'r') as f:
        content = f.read()
        if 'DB_CONNECTION=' in content:
            print("   ✅ .env file exists with DB_CONNECTION")
            return True
        else:
            print("   ❌ .env file missing DB_CONNECTION variable")
            return False

def check_dotenv_installed():
    """Check if python-dotenv is installed."""
    print("\n2️⃣  Checking python-dotenv installation...")
    try:
        import dotenv
        print("   ✅ python-dotenv is installed")
        return True
    except ImportError:
        print("   ❌ python-dotenv not installed!")
        print("   📝 Install with: pip install python-dotenv")
        return False

def check_sqlalchemy():
    """Check if sqlalchemy and psycopg2 are installed."""
    print("\n3️⃣  Checking database dependencies...")
    try:
        import sqlalchemy
        print("   ✅ sqlalchemy is installed")
    except ImportError:
        print("   ❌ sqlalchemy not installed!")
        print("   📝 Install with: pip install sqlalchemy")
        return False
    
    try:
        import psycopg2
        print("   ✅ psycopg2 is installed")
        return True
    except ImportError:
        print("   ❌ psycopg2 not installed!")
        print("   📝 Install with: pip install psycopg2-binary")
        return False

def check_database_connection():
    """Test connection to PostgreSQL database."""
    print("\n4️⃣  Testing database connection...")
    try:
        from dotenv import load_dotenv
        from sqlalchemy import create_engine, text
        
        load_dotenv()
        db_conn = os.getenv("DB_CONNECTION")
        
        if not db_conn:
            print("   ❌ DB_CONNECTION environment variable not set")
            return False
        
        # Hide password in output
        safe_conn = db_conn.split('@')[1] if '@' in db_conn else db_conn
        print(f"   🔌 Connecting to: {safe_conn}")
        
        engine = create_engine(db_conn)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()")).scalar()
            print(f"   ✅ Connected to PostgreSQL")
            
            # Check if sensors table exists
            table_check = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'sensors'
                )
            """)).scalar()
            
            if table_check:
                print("   ✅ 'sensors' table exists")
                
                # Check recent data
                recent_count = conn.execute(text("""
                    SELECT COUNT(*) FROM sensors 
                    WHERE timestamp > NOW() - INTERVAL '1 hour'
                """)).scalar()
                
                print(f"   📊 Recent data: {recent_count} rows in last hour")
            else:
                print("   ⚠️  'sensors' table not found - create it before running client.py")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")
        return False

def check_files():
    """Check if required files exist."""
    print("\n5️⃣  Checking required files...")
    files = {
        'client.py': 'OPC UA ingest pipeline',
        'dashboard.py': 'Streamlit dashboard',
        'opcua_fleet_server.py': 'OPC UA server',
    }
    
    all_exist = True
    for file, desc in files.items():
        if os.path.exists(file):
            print(f"   ✅ {file} ({desc})")
        else:
            print(f"   ⚠️  {file} not found ({desc})")
            all_exist = False
    
    return all_exist

def main():
    print("=" * 80)
    print("GAIA PREDICTIVE - POSTGRESQL PRODUCTION SETUP VERIFICATION")
    print("=" * 80)
    print()
    
    checks = [
        check_env_file(),
        check_dotenv_installed(),
        check_sqlalchemy(),
        check_files(),
        check_database_connection(),
    ]
    
    print("\n" + "=" * 80)
    if all(checks):
        print("✅ ALL CHECKS PASSED - Ready for production!")
        print("\nNext steps:")
        print("  1. Start OPC UA server: python3 opcua_fleet_server.py")
        print("  2. Start data client: python3 client.py")
        print("  3. Launch dashboard: streamlit run dashboard.py")
    else:
        print("❌ SOME CHECKS FAILED - Review errors above")
        print("\nFix the issues and run this script again.")
    print("=" * 80)
    
    return 0 if all(checks) else 1

if __name__ == "__main__":
    sys.exit(main())
