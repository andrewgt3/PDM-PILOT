#!/usr/bin/env python3
"""
Mark Azure PM machines (1..100) as onboarding COMPLETE so they appear as live
in the fleet and machine overview without running the full Prefect onboarding flow.

Run after ingesting Azure PdM data (e.g. after prepare_historical_stream or
run_stream_azure_pm.sh ingest step). Does not start the stream; use
run_stream_azure_pm.sh for full ingest + stream.

Usage:
    python scripts/bootstrap_azure_fleet.py
    python scripts/bootstrap_azure_fleet.py --max 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Mark Azure PM machines as onboarding COMPLETE")
    parser.add_argument("--max", type=int, default=100, help="Max machine ID (1..max inclusive)")
    args = parser.parse_args()

    from services.onboarding_helpers import set_onboarding_complete

    count = 0
    for i in range(1, args.max + 1):
        machine_id = str(i)
        try:
            set_onboarding_complete(machine_id)
            count += 1
        except Exception as e:
            print(f"Warning: failed to set {machine_id} COMPLETE: {e}", file=sys.stderr)
    print(f"Marked {count} machines (1..{args.max}) as onboarding COMPLETE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
