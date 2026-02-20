#!/usr/bin/env bash
# =============================================================================
# Start the PDM "test server" stack on this machine.
#
# Starts: Docker (TimescaleDB + Redis + Mosquitto), then API + stream_consumer
# in the background. Use this when you want to run the Azure PM replay from
# another machine (or same machine) and have data flow into this server.
#
# Usage:
#   ./scripts/start_test_server.sh
#   ./scripts/start_test_server.sh --infra-only   # Only Docker; you run API/consumer yourself
# =============================================================================

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

_print_replay_command() {
  if command -v ipconfig >/dev/null 2>&1; then
    LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)
  else
    LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
  fi
  if [ -z "$LAN_IP" ]; then LAN_IP="127.0.0.1"; fi
  echo "==> Replay from this machine:"
  echo "    cd $REPO_ROOT && source .venv/bin/activate"
  echo "    python3 mock_fleet_streamer.py --source azure_pm --speed-multiplier 720"
  echo ""
  echo "==> Replay from another machine (use this server's IP and port):"
  echo "    python3 mock_fleet_streamer.py --source azure_pm --redis-host $LAN_IP --redis-port 6379 --speed-multiplier 720"
  echo "    (Replace $LAN_IP with this machine's IP if connecting from elsewhere.)"
}

INFRA_ONLY=false
for arg in "$@"; do
  if [ "$arg" = "--infra-only" ]; then INFRA_ONLY=true; fi
done

echo "==> PDM test server (repo: $REPO_ROOT)"
echo ""

# --- Docker infra (no profile = timescaledb, redis, mosquitto only) ---
echo "==> Starting Docker infra (TimescaleDB, Redis, Mosquitto)..."
docker compose up -d
echo ""

# --- Wait for Redis ---
echo "==> Waiting for Redis to be ready..."
for i in 1 2 3 4 5 6 7 8 9 10; do
  if docker compose exec -T redis redis-cli ping 2>/dev/null | grep -q PONG; then
    echo "    Redis is up."
    break
  fi
  if [ "$i" -eq 10 ]; then
    echo "    WARNING: Redis did not become ready in time. Continue anyway."
  fi
  sleep 2
done
echo ""

if [ "$INFRA_ONLY" = true ]; then
  echo "==> Infra only (--infra-only). Start API and stream_consumer yourself, e.g.:"
  echo "    source .venv/bin/activate"
  echo "    uvicorn api_server:app --host 0.0.0.0 --port 8000 &"
  echo "    python stream_consumer.py &"
  echo ""
  _print_replay_command
  exit 0
fi

# --- Optional: run API and stream_consumer in background ---
if [ ! -d ".venv" ]; then
  echo "==> No .venv found. Skipping API and stream_consumer. Run them yourself:"
  echo "    python3 -m venv .venv && source .venv/bin/activate"
  echo "    pip install -r requirements.txt"
  echo "    uvicorn api_server:app --host 0.0.0.0 --port 8000 &"
  echo "    python stream_consumer.py &"
  echo ""
  _print_replay_command
  exit 0
fi

echo "==> Starting API (background)..."
source .venv/bin/activate
export REDIS_HOST=localhost REDIS_PORT=6379
export DB_HOST=localhost DB_PORT=5432
# Use env file if present
if [ -f .env ]; then set -a; source .env; set +a; fi
nohup uvicorn api_server:app --host 0.0.0.0 --port 8000 > logs/api_test_server.log 2>&1 &
API_PID=$!
echo "    API PID: $API_PID (log: logs/api_test_server.log)"

echo "==> Starting stream_consumer (background)..."
mkdir -p logs
nohup python stream_consumer.py > logs/stream_consumer_test_server.log 2>&1 &
CONSUMER_PID=$!
echo "    stream_consumer PID: $CONSUMER_PID (log: logs/stream_consumer_test_server.log)"
echo ""

_print_replay_command
echo ""
echo "==> To stop: kill $API_PID $CONSUMER_PID  && docker compose down"
