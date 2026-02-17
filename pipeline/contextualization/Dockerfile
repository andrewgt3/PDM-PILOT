# =============================================================================
# Golden Pipeline — Stage 3: Contextualization Service
# =============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --gid 1000 pipeline \
    && useradd --uid 1000 --gid pipeline --shell /bin/bash --create-home pipeline

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY service.py .

USER pipeline

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import redis; r=redis.Redis(host='${REDIS_HOST:-localhost}'); r.ping(); print('OK')" || exit 1

ENTRYPOINT ["python", "service.py"]
