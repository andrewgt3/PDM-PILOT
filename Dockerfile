# =============================================================================
# Gaia Predictive — Production Dockerfile
# =============================================================================
# TISAX/SOC2 Compliant Container Image
# - Base: python:3.11-slim (Minimal attack surface)
# - User: Non-root 'appuser' (UID 1000)
# - Secrets: Strictly excluded via .dockerignore
# - Multi-stage: Separate builder and runner stages
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system build dependencies
# (required for compiling some python packages like asyncpg/numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies to a temporary location
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt


# -----------------------------------------------------------------------------
# Stage 2: Runner
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runner

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser

# Install runtime system dependencies (e.g., libpq for postgres)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy wheels from builder and install
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

# Copy application code
# (Owned by root but readable by appuser is standard, 
# but for strictness we chown to appuser if they need to write, 
# though code should be immutable. We'll set ownership to appuser just in case.)
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
# Use generic entrypoint that can be overridden
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
