#!/usr/bin/env bash
# =============================================================================
# Run the full Azure PM stream A–Z with no human in the loop.
#
# 1. Bootstrap .env if missing (Docker-friendly defaults)
# 2. Start Docker (TimescaleDB + Redis), wait for healthy
# 3. Apply schema (alembic + setup_schema)
# 4. Ingest Azure PdM CSVs into DB
# 5. Start API and stream_consumer in background
# 6. Run replay (foreground, or for --duration N seconds then stop)
#
# Usage:
#   ./scripts/run_stream_azure_pm.sh
#   ./scripts/run_stream_azure_pm.sh --data-dir data/azure_pm
#   ./scripts/run_stream_azure_pm.sh --duration 300 --stop-after-replay
# =============================================================================

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

fail() {
  echo -e "${RED}ERROR: $1${NC}" >&2
  [ -n "$2" ] && echo "$2" >&2
  exit 1
}

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
DATA_DIR_ARG=""
DURATION_SEC=""
STOP_AFTER_REPLAY=false
SPEED_MULTIPLIER=720

while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir)
      DATA_DIR_ARG="$2"
      shift 2
      ;;
    --duration)
      DURATION_SEC="$2"
      shift 2
      ;;
    --stop-after-replay)
      STOP_AFTER_REPLAY=true
      shift
      ;;
    --speed-multiplier)
      SPEED_MULTIPLIER="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--data-dir DIR] [--duration N] [--stop-after-replay] [--speed-multiplier N]" >&2
      exit 1
      ;;
  esac
done

# Resolve data dir: default microsoft_azure_predictive_maintenance then data/azure_pm
if [ -n "$DATA_DIR_ARG" ]; then
  DATA_DIR="$DATA_DIR_ARG"
else
  if [ -d "$REPO_ROOT/microsoft_azure_predictive_maintenance" ] && [ -f "$REPO_ROOT/microsoft_azure_predictive_maintenance/PdM_telemetry.csv" ]; then
    DATA_DIR="microsoft_azure_predictive_maintenance"
  else
    DATA_DIR="data/azure_pm"
  fi
fi
if [ "${DATA_DIR#/}" = "$DATA_DIR" ]; then
  DATA_DIR_ABS="$REPO_ROOT/$DATA_DIR"
else
  DATA_DIR_ABS="$DATA_DIR"
fi

KAGGLE_URL="https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance/data"

# -----------------------------------------------------------------------------
# 1. Env bootstrap (only when .env is missing)
# -----------------------------------------------------------------------------
echo "==> [1/8] Environment"
if [ ! -f .env ]; then
  if [ ! -f .env.example ]; then
    fail ".env missing and no .env.example found."
  fi
  cp .env.example .env
  # Streaming-safe defaults to match Docker Compose (portable: sed to temp then mv)
  sed -e 's/^DB_NAME=.*/DB_NAME=pdm_timeseries/' \
    -e 's/^DB_USER=.*/DB_USER=postgres/' \
    -e 's/^DB_PASSWORD=.*/DB_PASSWORD=password/' \
    -e 's/^DB_SSL_MODE=.*/DB_SSL_MODE=disable/' \
    .env > .env.tmp && mv .env.tmp .env
  # SECURITY_JWT_SECRET required by API (append if missing or empty)
  if ! grep -q '^SECURITY_JWT_SECRET=.\+' .env 2>/dev/null; then
    echo "SECURITY_JWT_SECRET=dev-secret-key-change-in-production-32chars" >> .env
  fi
  echo -e "${GREEN}  Created .env from .env.example with Docker-friendly defaults.${NC}"
else
  echo "  .env exists, using existing values."
fi
set -a
[ -f .env ] && source .env
set +a

# -----------------------------------------------------------------------------
# 2. Venv and Docker
# -----------------------------------------------------------------------------
VENV_DIR="${VENV_DIR:-.venv}"
if [ ! -d "$VENV_DIR" ]; then
  fail "No .venv found. Run ./scripts/setup_dev.sh first, or: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
fi
# shellcheck disable=SC1090,SC1091
source "$VENV_DIR/bin/activate"

echo ""
echo "==> [2/8] Infrastructure (TimescaleDB + Redis)"
if [ "${SKIP_INFRA:-0}" = "1" ]; then
  echo "  SKIP_INFRA=1: skipping Docker start and wait (assume containers already up)."
else
  COMPOSE_CMD="docker compose"
  command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1 && COMPOSE_CMD="docker-compose"
  $COMPOSE_CMD up -d timescaledb redis 2>/dev/null || true

  echo "  Waiting for TimescaleDB (max 60s)..."
  MAX_ATTEMPTS=30
  ATTEMPT=0
  until $COMPOSE_CMD exec -T timescaledb pg_isready -U postgres -q 2>/dev/null; do
    ATTEMPT=$((ATTEMPT + 1))
    [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ] && fail "TimescaleDB did not become ready in time."
    sleep 2
  done
  echo -e "${GREEN}  TimescaleDB ready.${NC}"

  echo "  Waiting for Redis (max 60s)..."
  sleep 3
  ATTEMPT=0
  until $COMPOSE_CMD exec -T redis redis-cli ping 2>/dev/null | grep -q PONG; do
    ATTEMPT=$((ATTEMPT + 1))
    [ "$ATTEMPT" -ge 30 ] && fail "Redis did not become ready in time."
    sleep 2
  done
  echo -e "${GREEN}  Redis ready.${NC}"
fi

# -----------------------------------------------------------------------------
# 3. Schema
# -----------------------------------------------------------------------------
echo ""
echo "==> [3/8] Database schema"
alembic upgrade head 2>/dev/null || fail "alembic upgrade head failed. Run: alembic upgrade head"
python setup_schema.py 2>/dev/null || true
echo -e "${GREEN}  Schema ready.${NC}"

# -----------------------------------------------------------------------------
# 4. Data dir check
# -----------------------------------------------------------------------------
echo ""
echo "==> [4/8] Data directory"
if [ ! -d "$DATA_DIR_ABS" ]; then
  mkdir -p "$DATA_DIR_ABS"
fi
if [ ! -f "$DATA_DIR_ABS/PdM_telemetry.csv" ]; then
  echo "Azure PdM CSVs not found in $DATA_DIR_ABS" >&2
  echo "" >&2
  echo "Download the dataset (free with Kaggle account):" >&2
  echo "  $KAGGLE_URL" >&2
  echo "" >&2
  echo "Then place at least PdM_telemetry.csv into:" >&2
  echo "  $DATA_DIR_ABS" >&2
  echo "" >&2
  echo "Or run with: $0 --data-dir /path/to/folder" >&2
  exit 1
fi
echo "  Using data dir: $DATA_DIR_ABS"

# -----------------------------------------------------------------------------
# 5. Ingest
# -----------------------------------------------------------------------------
echo ""
echo "==> [5/8] Ingest Azure PdM into DB"
python -m pipeline.ingestion.azure_pm_ingestor --data-dir "$DATA_DIR_ABS" || fail "Ingestion failed."
echo -e "${GREEN}  Ingest complete.${NC}"

echo "  Marking Azure machines 1..100 as onboarding COMPLETE (fleet visible in UI)..."
python scripts/bootstrap_azure_fleet.py --max 100 2>/dev/null || true
echo -e "${GREEN}  Fleet bootstrap done.${NC}"

# -----------------------------------------------------------------------------
# 6. API and stream_consumer (background)
# -----------------------------------------------------------------------------
echo ""
echo "==> [6/8] Starting API and stream_consumer (background)"
mkdir -p logs
export REDIS_HOST="${REDIS_HOST:-localhost}"
export REDIS_PORT="${REDIS_PORT:-6379}"
export DB_HOST="${DB_HOST:-localhost}"
export DB_PORT="${DB_PORT:-5432}"

nohup uvicorn api_server:app --host 0.0.0.0 --port 8000 >> logs/api.log 2>&1 &
API_PID=$!
echo "$API_PID" > logs/api.pid
echo "  API PID: $API_PID (logs/api.log)"

nohup python stream_consumer.py >> logs/stream_consumer.log 2>&1 &
CONSUMER_PID=$!
echo "$CONSUMER_PID" > logs/stream_consumer.pid
echo "  stream_consumer PID: $CONSUMER_PID (logs/stream_consumer.log)"

echo "  Waiting 5s for API to bind..."
sleep 5
echo -e "${GREEN}  API and stream_consumer running.${NC}"

# -----------------------------------------------------------------------------
# 7 & 8. Replay (foreground or duration-limited)
# -----------------------------------------------------------------------------
echo ""
echo "==> [7/8] Replay (Azure PM -> Redis)"
if [ -n "$DURATION_SEC" ]; then
  echo "  Running replay for ${DURATION_SEC}s then stopping..."
  python mock_fleet_streamer.py --source azure_pm --speed-multiplier "$SPEED_MULTIPLIER" &
  REPLAY_PID=$!
  sleep "$DURATION_SEC"
  kill "$REPLAY_PID" 2>/dev/null || true
  wait "$REPLAY_PID" 2>/dev/null || true
  echo -e "${GREEN}  Replay stopped after ${DURATION_SEC}s.${NC}"
else
  echo "  Replay running (foreground). Ctrl+C to stop replay; API and consumer keep running."
  echo ""
  echo "  Open the app (e.g. http://localhost:5173 or :8080), log in as Admin / Engineer / Operator to see Machine Overview and Fleet Topology."
  echo ""
  if python mock_fleet_streamer.py --source azure_pm --speed-multiplier "$SPEED_MULTIPLIER"; then
    echo -e "${GREEN}  Replay finished.${NC}"
  else
    echo -e "${YELLOW}  Replay exited with an error.${NC}"
  fi
fi

# -----------------------------------------------------------------------------
# 9. Cleanup (optional)
# -----------------------------------------------------------------------------
echo ""
echo "==> [8/8] Shutdown"
if [ "$STOP_AFTER_REPLAY" = true ] || [ -n "$DURATION_SEC" ]; then
  echo "  Stopping API and stream_consumer..."
  kill "$API_PID" 2>/dev/null || true
  kill "$CONSUMER_PID" 2>/dev/null || true
  wait "$API_PID" 2>/dev/null || true
  wait "$CONSUMER_PID" 2>/dev/null || true
  rm -f logs/api.pid logs/stream_consumer.pid
  echo -e "${GREEN}  Done. Containers still running (docker compose down to stop).${NC}"
else
  echo "  API and stream_consumer still running."
  echo "  To stop them: kill $API_PID $CONSUMER_PID"
  echo "  To stop infra: docker compose down"
fi
