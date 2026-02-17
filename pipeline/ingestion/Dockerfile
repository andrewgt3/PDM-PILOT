# =============================================================================
# Golden Pipeline — Stage 1: Ingestion Service
# =============================================================================
# Publishes raw sensor data to Redis Stream (raw_sensor_data).
# Minimal footprint for edge deployment.
# =============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root user
RUN groupadd --gid 1000 pipeline \
    && useradd --uid 1000 --gid pipeline --shell /bin/bash --create-home pipeline

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY service.py .

# Switch to non-root user
USER pipeline

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import redis; r=redis.Redis(host='${REDIS_HOST:-localhost}'); r.ping(); print('OK')" || exit 1

ENTRYPOINT ["python", "service.py"]
