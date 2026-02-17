#!/usr/bin/env python3
"""
Configuration Verification Script

Validates that all environment variables are correctly mapped to Pydantic settings.
Run this script to verify configuration before deployment.

Usage:
    python scripts/verify_config.py
"""

import sys
import os
from pathlib import Path
from pprint import pprint

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def main():
    print("=" * 60)
    print("GAIA PREDICTIVE - CONFIGURATION VERIFICATION")
    print("=" * 60)
    print()
    
    try:
        from config import get_settings
        settings = get_settings()
        print("✓ Settings loaded successfully")
        print()
    except Exception as e:
        print(f"✗ Failed to load settings: {e}")
        sys.exit(1)
    
    # Dump settings (secrets are masked by Pydantic)
    print("=" * 60)
    print("CONFIGURATION DUMP (secrets masked)")
    print("=" * 60)
    
    # Use model_dump with mode='json' for serializable output
    config_dict = settings.model_dump(mode='json')
    
    # Manually mask secret values for display
    def mask_secrets(obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if 'password' in key.lower() or 'secret' in key.lower():
                    obj[key] = "***MASKED***"
                else:
                    mask_secrets(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for item in obj:
                mask_secrets(item, path)
        return obj
    
    masked_config = mask_secrets(config_dict)
    pprint(masked_config, width=80, sort_dicts=False)
    
    print()
    print("=" * 60)
    print("SECTION SUMMARY")
    print("=" * 60)
    print(f"  App:      {settings.app_name} v{settings.app_version}")
    print(f"  Env:      {settings.environment}")
    print(f"  Debug:    {settings.debug}")
    print()
    print(f"  Database: {settings.database.dsn_safe}")
    print(f"  Redis:    {settings.redis.url_safe}")
    print()
    print(f"  Model:    lookback={settings.model.lookback_days}d, window={settings.model.rolling_window_seconds}s")
    print(f"  Edge:     {settings.edge.opc_server_url}")
    print()
    print("=" * 60)
    print("✓ ALL CONFIGURATION VALID")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
