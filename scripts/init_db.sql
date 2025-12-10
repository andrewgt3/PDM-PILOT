-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Sensor Metadata Table
CREATE TABLE IF NOT EXISTS sensors (
    sensor_id TEXT PRIMARY KEY,
    name TEXT,
    location TEXT,
    type TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Sensor Readings Table (Hypertable)
CREATE TABLE IF NOT EXISTS sensor_readings (
    time TIMESTAMPTZ NOT NULL,
    sensor_id TEXT NOT NULL,
    temperature DOUBLE PRECISION,
    vibration DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    rpm DOUBLE PRECISION,
    FOREIGN KEY (sensor_id) REFERENCES sensors (sensor_id)
);

-- Convert to hypertable partitioned by time
SELECT create_hypertable('sensor_readings', 'time', if_not_exists => TRUE);

-- Index for faster queries on sensor_id
CREATE INDEX IF NOT EXISTS idx_sensor_readings_sensor_id ON sensor_readings (sensor_id, time DESC);


-- 3. Maintenance Logs Table (Unstructured + Vector)
CREATE TABLE IF NOT EXISTS maintenance_logs (
    log_id SERIAL PRIMARY KEY,
    machine_id TEXT NOT NULL,
    log_text TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    -- Vector embedding for semantic search (assuming 384 dim for a small model like all-MiniLM-L6-v2)
    embedding vector(384)
);

-- HNSW Index for approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS idx_logs_embedding ON maintenance_logs USING hnsw (embedding vector_cosine_ops);
