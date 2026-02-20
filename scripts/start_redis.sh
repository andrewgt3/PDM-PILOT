#!/usr/bin/env bash
# Start Redis (and optionally TimescaleDB) from the project directory.
# Run from anywhere: ./scripts/start_redis.sh
# This fixes "no configuration file provided" when you run docker compose from the wrong folder.
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
docker compose up -d redis "$@"
echo "Redis started. Optional: docker compose up -d timescaledb"
