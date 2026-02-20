#!/usr/bin/env python3
"""
Verify connections to dev infrastructure and optional virtual hardware.

Reads .env (or env) for:
  - DB_* → TimescaleDB
  - REDIS_* → Redis
  - ROBOT_IP, ROBOT_PORT → ABB RWS (HTTP GET to /rw/system)
  - SIEMENS_PLC_IP, SIEMENS_PLC_RACK, SIEMENS_PLC_SLOT → snap7

Prints a pass/fail table. Run from repo root: python scripts/verify_connections.py
"""

import os
import sys
from pathlib import Path

# Load .env from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
env_file = REPO_ROOT / ".env"
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file)

def get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()

def check_timescaledb() -> tuple[bool, str]:
    try:
        import psycopg2
        host = get_env("DB_HOST", "localhost")
        port = int(get_env("DB_PORT", "5432"))
        dbname = get_env("DB_NAME", "gaia_predictive")
        user = get_env("DB_USER", "gaia")
        password = get_env("DB_PASSWORD", "")
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password or None,
            connect_timeout=3,
        )
        conn.close()
        return True, f"{host}:{port}/{dbname}"
    except Exception as e:
        return False, str(e)

def check_redis() -> tuple[bool, str]:
    try:
        import redis
        host = get_env("REDIS_HOST", "localhost")
        port = int(get_env("REDIS_PORT", "6379"))
        password = get_env("REDIS_PASSWORD") or None
        r = redis.Redis(host=host, port=port, password=password, socket_connect_timeout=3)
        r.ping()
        r.close()
        return True, f"{host}:{port}"
    except Exception as e:
        return False, str(e)

def check_abb_rws() -> tuple[bool, str]:
    robot_ip = get_env("ROBOT_IP", get_env("ABB_IP_ADDRESS", "127.0.0.1"))
    robot_port = get_env("ROBOT_PORT", get_env("ABB_PORT", "80"))
    url = f"http://{robot_ip}:{robot_port}/rw/system"
    try:
        import urllib.request
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if 200 <= resp.status < 400:
                return True, f"{robot_ip}:{robot_port}/rw/system"
            return False, f"HTTP {resp.status}"
    except Exception as e:
        return False, str(e)

def check_siemens_s7() -> tuple[bool, str]:
    """Use SiemensS7Adapter.test_read() to verify PLC connection and DB read."""
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from siemens_s7_adapter import SiemensS7Adapter
        adapter = SiemensS7Adapter()
        result = adapter.test_read()
        if result is not None:
            return True, f"{adapter.plc_ip} db={adapter.db_number} (test_read OK)"
        return False, "test_read returned None"
    except ImportError as e:
        return False, f"snap7/adapter not available: {e}"
    except Exception as e:
        return False, str(e)

def main() -> None:
    rows = [
        ("TimescaleDB", check_timescaledb()),
        ("Redis", check_redis()),
        ("ABB RWS", check_abb_rws()),
        ("Siemens S7 (snap7)", check_siemens_s7()),
    ]
    width = 20
    print("Connection verification")
    print("-" * (width + 4 + 8))
    print(f"{'Service':<{width}} {'Status':<8} Detail")
    print("-" * (width + 4 + 8))
    for name, (ok, detail) in rows:
        status = "PASS" if ok else "FAIL"
        print(f"{name:<{width}} {status:<8} {detail}")
    print("-" * (width + 4 + 8))
    failed = sum(1 for _, (ok, _) in rows if not ok)
    if failed > 0:
        sys.exit(1)
    print("All checks passed.")

if __name__ == "__main__":
    main()
