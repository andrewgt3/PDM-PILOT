#!/usr/bin/env bash
# =============================================================================
# PDM-PILOT — One-command development environment setup
# =============================================================================
# Run from repository root: ./scripts/setup_dev.sh
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
  echo "$2" >&2
  exit 1
}

# -----------------------------------------------------------------------------
# 1. Check required tools
# -----------------------------------------------------------------------------
echo "Checking required tools..."

command -v python3 >/dev/null 2>&1 || fail "Python 3 not found." \
  "Install: https://www.python.org/downloads/ or your package manager (e.g. brew install python@3.11)."

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 11 ]]; }; then
  fail "Python 3.11+ required (found $PY_VER)." \
    "Install Python 3.11 or newer: https://www.python.org/downloads/"
fi

command -v docker >/dev/null 2>&1 || fail "Docker not found." \
  "Install: https://docs.docker.com/get-docker/"

command -v docker compose >/dev/null 2>&1 || command -v docker-compose >/dev/null 2>&1 || \
  fail "Docker Compose not found." \
  "Install: https://docs.docker.com/compose/install/"

command -v node >/dev/null 2>&1 || fail "Node.js not found." \
  "Install Node 18+: https://nodejs.org/ or your package manager (e.g. brew install node)."

NODE_MAJOR=$(node -e "console.log(process.versions.node.split('.')[0])")
[[ "$NODE_MAJOR" -ge 18 ]] || fail "Node.js 18+ required (found $NODE_MAJOR)." \
  "Install: https://nodejs.org/"

echo -e "${GREEN}✓ All required tools present${NC}"

# -----------------------------------------------------------------------------
# 2. .env from .env.example if missing
# -----------------------------------------------------------------------------
if [[ ! -f .env ]]; then
  if [[ -f .env.example ]]; then
    cp .env.example .env
    echo -e "${GREEN}✓ Created .env from .env.example${NC}"
  else
    echo -e "${YELLOW}⚠ No .env.example found; skipping .env creation${NC}"
  fi
else
  echo "✓ .env already exists"
fi

# -----------------------------------------------------------------------------
# 3. Python venv and dependencies
# -----------------------------------------------------------------------------
VENV_DIR="${VENV_DIR:-.venv}"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment at $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090,SC1091
source "$VENV_DIR/bin/activate"

if [[ -f requirements.txt ]]; then
  echo "Installing Python dependencies..."
  pip install -q -r requirements.txt
  echo -e "${GREEN}✓ Python dependencies installed${NC}"
else
  echo -e "${YELLOW}⚠ requirements.txt not found; skipping pip install${NC}"
fi

# -----------------------------------------------------------------------------
# 4. Start infrastructure (TimescaleDB + Redis)
# -----------------------------------------------------------------------------
COMPOSE_CMD="docker compose"
command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1 && COMPOSE_CMD="docker-compose"

echo "Starting TimescaleDB and Redis..."
$COMPOSE_CMD --profile dev up -d timescaledb redis 2>/dev/null || \
  $COMPOSE_CMD up -d timescaledb redis 2>/dev/null || \
  fail "Failed to start containers. Ensure Docker is running and docker-compose.yml exists."

# -----------------------------------------------------------------------------
# 5. Wait for TimescaleDB to be healthy
# -----------------------------------------------------------------------------
echo "Waiting for TimescaleDB to be ready..."
MAX_ATTEMPTS=30
ATTEMPT=0
until $COMPOSE_CMD --profile dev exec -T timescaledb pg_isready -U postgres -q 2>/dev/null || \
      $COMPOSE_CMD exec -T timescaledb pg_isready -U postgres -q 2>/dev/null; do
  ATTEMPT=$((ATTEMPT + 1))
  [[ $ATTEMPT -ge $MAX_ATTEMPTS ]] && fail "TimescaleDB did not become ready in time."
  echo "  Attempt $ATTEMPT/$MAX_ATTEMPTS..."
  sleep 2
done
echo -e "${GREEN}✓ TimescaleDB is ready${NC}"

# -----------------------------------------------------------------------------
# 6. Run migrations
# -----------------------------------------------------------------------------
if command -v alembic >/dev/null 2>&1; then
  echo "Running database migrations..."
  alembic upgrade head
  echo -e "${GREEN}✓ Migrations applied${NC}"
else
  echo -e "${YELLOW}⚠ alembic not found; run 'pip install alembic' and then 'alembic upgrade head'${NC}"
fi

# -----------------------------------------------------------------------------
# 7. Frontend dependencies
# -----------------------------------------------------------------------------
if [[ -d frontend ]] && [[ -f frontend/package.json ]]; then
  echo "Installing frontend dependencies..."
  (cd frontend && npm install --silent)
  echo -e "${GREEN}✓ Frontend dependencies installed${NC}"
else
  echo -e "${YELLOW}⚠ frontend/package.json not found; skipping npm install${NC}"
fi

# -----------------------------------------------------------------------------
# 8. Success summary
# -----------------------------------------------------------------------------
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Development environment ready${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "  API:      http://localhost:8000"
echo "  Frontend: http://localhost:5173  (run: cd frontend && npm run dev)"
echo ""
echo "  Next steps:"
echo "    1. Activate venv: source $VENV_DIR/bin/activate"
echo "    2. Start API:     uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload"
echo "    3. Start frontend: cd frontend && npm run dev"
echo "    4. Verify connections: python scripts/verify_connections.py"
echo ""
