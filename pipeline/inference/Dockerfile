# =============================================================================
# Golden Pipeline — Stage 5: Inference Service
# =============================================================================
# XGBoost RUL prediction engine. Includes full ML stack.
# Models are copied into the container at build time.
# =============================================================================

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# =============================================================================
FROM python:3.11-slim AS runner

RUN groupadd --gid 1000 pipeline \
    && useradd --uid 1000 --gid pipeline --shell /bin/bash --create-home pipeline

WORKDIR /app

COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

# Copy service code
COPY service.py .

USER pipeline

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import redis; r=redis.Redis(host='${REDIS_HOST:-localhost}'); r.ping(); print('OK')" || exit 1

ENTRYPOINT ["python", "service.py"]
