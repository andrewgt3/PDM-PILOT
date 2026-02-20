#!/usr/bin/env python3
"""
Register and serve Prefect deployments for PDM Pilot.

Run from project root. Requires PREFECT_API_URL to point to a running Prefect server.
This script registers the deployments and then blocks, executing scheduled and
event-triggered flow runs (act as the worker).

  python scripts/deploy_flows.py

Or from repo root:
  python -m scripts.deploy_flows
"""

import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flows.deployments import get_scheduled_deployments
from prefect import serve


def main() -> None:
    deployments = get_scheduled_deployments()
    serve(*deployments)


if __name__ == "__main__":
    main()
