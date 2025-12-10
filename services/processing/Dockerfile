FROM python:3.11-slim

WORKDIR /app

# Install build dependencies for psycopg2 if needed (slim might miss libpq-dev)
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
