# Free Data for Streaming Historical Data

This doc lists **free public datasets** that match the pipeline (vibration/telemetry → sensor_readings → Redis stream → inference) and how to use them for testing.

---

## What to expect when everything is working

- **Pipeline / Redis status:** In the app, pipeline status should show **Redis connected** and stream status **LIVE** (after you log in so the WebSocket sends a token). If Redis was started *after* the API, restart the API once so it connects to Redis.
- **Plant Overview:** You should see the dashboard grid with at least one card (e.g. **Fleet topology** or **Treemap**). If you see “1 asset” but a blank area, ensure Redis and the stream are running and refresh; the topology card now has a minimum height so the diagram has space to render.
- **Machines:** Machine list comes from `GET /api/machines` (TimescaleDB). Run ingest (e.g. Azure PM ingestor) and optionally `scripts/bootstrap_azure_fleet.py` so machines 1..100 are marked onboarding complete and appear in the UI. Log in as **Admin**, **Engineer**, or **Operator** (e.g. `admin` / `secret123` in development) so the list isn’t filtered out.
- **Stream:** Run `stream_consumer` and the replay (e.g. `mock_fleet_streamer.py --source azure_pm`) so events flow: DB → Redis → consumer → inference; the dashboard will update as replay runs.

---

## Troubleshooting

### All 6 pipeline stages show DOWN

The API connects to Redis only at **startup**. If Redis wasn’t running when you started the API, `state.redis` stays `None` and the pipeline status endpoint returns all six stages as DOWN and `redis_connected: false`.

**Fix:**

1. Start Redis (from the **project root**):  
   `./scripts/start_redis.sh` or `docker compose up -d redis`
2. **Restart the API** so it runs startup again and connects to Redis. After that, refresh the app; you should see Redis connected and stage status based on stream lengths.

### Only 1 asset (or very few) in Plant Overview

Possible causes:

1. **Role filter:** The machine list is scoped by role. **Technician** sees only assigned machines (often one). **Admin**, **Engineer**, and **Operator** see all machines. Log in as `admin` / `secret123` (in development) to see the full fleet.
2. **Database has few machines:** The list comes from `cwru_features` in TimescaleDB. If you haven’t run ingest (e.g. Azure PM ingestor) and optionally bootstrap, only machines that already have feature rows will appear. Run ingest, then `scripts/bootstrap_azure_fleet.py` if using Azure PM, so machines 1..100 are present.
3. **Stream not running:** For live updates, run `stream_consumer` and the replay (e.g. `mock_fleet_streamer.py --source azure_pm`). Without the stream, no new data is written to Redis or the DB, so the dashboard won’t reflect new machines or updates.

### Is the stream active?

The UI shows stream status as LIVE when the WebSocket is connected **with a valid token** and the backend is subscribed to Redis. To have an active stream of data:

1. Redis running, API started **after** Redis (so Redis is connected).
2. `stream_consumer` running (subscribes to Redis, runs inference, writes to DB/Redis).
3. Replay running, e.g. `python mock_fleet_streamer.py --source azure_pm --speed-multiplier 720`.
4. Logged in so the WebSocket sends the token; then the app can show LIVE and receive events.

---

## Option 1: Azure Predictive Maintenance (best fit for replay)

**What it is:** Microsoft’s PdM dataset: 100 machines, telemetry (volt, rotate, pressure, vibration) and failure labels. Matches the existing **Azure PM replay** path (DB → Redis with time compression).

**Where to get it (free):**

- **Kaggle** (free with account): [Microsoft Azure Predictive Maintenance](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance/data)  
  Download the dataset, then place the 5 CSVs into `data/azure_pm/`:
  - `PdM_telemetry.csv`
  - `PdM_errors.csv`
  - `PdM_failures.csv`
  - `PdM_machines.csv`
  - `PdM_maint.csv`

**Steps to stream historical data:**

```bash
# 1. Create directory and put the 5 PdM_*.csv files there
mkdir -p data/azure_pm
# (copy or move the downloaded CSVs into data/azure_pm/)

# 2. Load into TimescaleDB (sensor_readings + cwru_features for failure labels)
source .venv/bin/activate
python -m pipeline.ingestion.azure_pm_ingestor --data-dir data/azure_pm/

# 3. Stream from DB to Redis with time compression (e.g. 720x = 1 month in ~1 hour)
python mock_fleet_streamer.py --source azure_pm --speed-multiplier 720
```

Keep the API and `stream_consumer` running so inference runs on the replayed stream. The dashboard will show machines and health as the replay runs.

---

## Option 2: NASA (C-MAPSS + IMS)

**What it is:** NASA turbofan degradation (C-MAPSS) and bearing run-to-failure (IMS). Good for training and for ingestion into the drop zone.

**Where to get it (free, no account):**

- **NASA C-MAPSS:** <https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data>  
- **NASA IMS Bearings:** <https://data.nasa.gov/dataset/ims-bearings>

**Download via script:**

```bash
python scripts/download_nasa_data.py
# Writes archive.zip (or use --output path)
```

**Using it:**

- Set `NASA_ARCHIVE_PATH` in `.env` to your `archive.zip` or extracted folder.
- In the app: **Pipeline Operations** → **Initialize NASA Data** to copy the archive into `data/ingest`. The watcher will process files under `data/ingest`.
- For **streaming** like the Azure path, you’d need a separate loader that writes NASA data into `sensor_readings` (and optionally `cwru_features`); that loader is not included here. The bootstrap + watcher path is for batch ingestion into the pipeline, not for the Azure PM–style replay.

---

## Option 3: FEMTO / PRONOSTIA bearings

**What it is:** IEEE PHM 2012 challenge: run-to-failure bearing data (vibration).

**Where to get it (free):**

- Script in repo: `python scripts/download_femto_data.py`  
- Fetches from GitHub into `data/downloads/femto_pronostia/`.

**Using it:** The pipeline does not yet have a FEMTO → `sensor_readings` or FEMTO → Redis stream script. You can use the raw CSVs for offline analysis or extend the ingestion layer to map FEMTO columns to `sensor_readings` and then use the same replay idea as Azure PM.

---

## Option 4: CWRU Bearing Data

**What it is:** Case Western Reserve University bearing vibration (12/48 kHz), multiple fault types and severities. Very common for bearing-fault and vibration pipelines.

**Where to get it (free):**

- **Official:** <https://engineering.case.edu/bearingdatacenter/download> (select files)
- **Mirror (e.g. GitHub):** search for “CWRU bearing dataset” for CSV or preprocessed mirrors.

**Using it:** The repo has `custom_data_ingestor` that can normalize **CWRU-style CSVs** (timestamp, machine_id, torque, rotational_speed, temperature) to a standard schema for feature extraction. It does **not** write to `sensor_readings`. To stream CWRU through the same flow as Azure PM, you’d add a small script that (1) loads CWRU .mat or CSV, (2) inserts into `sensor_readings` (and optionally `cwru_features`), then (3) run `mock_fleet_streamer.py --source azure_pm`.

---

## Summary

| Dataset        | Free source              | Matches replay? | How to stream historical |
|----------------|--------------------------|------------------|---------------------------|
| **Azure PdM**  | Kaggle (account)         | Yes              | Ingest → `mock_fleet_streamer --source azure_pm` |
| **NASA**       | NASA Open Data / script   | Bootstrap/watcher| Initialize NASA Data; no DB replay yet |
| **FEMTO**      | `download_femto_data.py`  | No loader yet    | Add ingestor → same as Azure |
| **CWRU**       | CWRU / mirrors           | No DB writer yet | Add ingestor → same as Azure |

**Recommended for “stream historical data” with your current flow:** use **Azure PdM** (Option 1): download from Kaggle, run the ingestor, then `mock_fleet_streamer.py --source azure_pm`.

---

## Fully automated run (no human in the loop)

One command runs the full Azure PM stream from scratch: start Docker (TimescaleDB + Redis), apply schema, ingest CSVs, start API and stream_consumer, then run the replay.

**Prerequisites:** Docker (Compose), Python 3.11+ with venv and `requirements.txt` installed (e.g. run `./scripts/setup_dev.sh` once). Azure PdM data in `microsoft_azure_predictive_maintenance/` or `data/azure_pm/` (at least `PdM_telemetry.csv`).

```bash
./scripts/run_stream_azure_pm.sh
```

- If `.env` is missing, the script creates it from `.env.example` with Docker-friendly defaults (`DB_NAME=pdm_timeseries`, `DB_USER=postgres`, `DB_PASSWORD=password`, etc.) so no manual edit is required.
- Replay runs in the foreground until the dataset is done; Ctrl+C stops the replay (API and stream_consumer keep running).

**Options:**

| Option | Description |
|--------|--------------|
| `--data-dir DIR` | Directory containing PdM_*.csv files (default: `microsoft_azure_predictive_maintenance` or `data/azure_pm`). |
| `--duration N` | Run replay for N seconds then stop (e.g. `--duration 300` for a 5-minute demo). |
| `--stop-after-replay` | After replay ends, stop the API and stream_consumer (clean exit). |
| `--speed-multiplier N` | Replay time compression (default 720; 1 month ≈ 1 hour). |

**Examples:**

```bash
# Full run (replay until completion)
./scripts/run_stream_azure_pm.sh

# Use a custom data folder
./scripts/run_stream_azure_pm.sh --data-dir /path/to/my_pdm_csvs

# Run replay for 5 minutes, then stop API and consumer
./scripts/run_stream_azure_pm.sh --duration 300 --stop-after-replay
```

---

## Test server setup (IP + port)

To run the **replay** on your machine but publish to a **remote Redis** (e.g. a test server), use Redis host/port so the stream_consumer and API on the server read the same stream.

**On the test server:**

1. **One-command start (recommended):** From the repo root, run:
   ```bash
   ./scripts/start_test_server.sh
   ```
   This starts Docker (TimescaleDB + Redis + Mosquitto), then the API and stream_consumer in the background, and prints the replay command and this machine's IP for use from another machine.

   To start only Docker infra and run API/stream_consumer yourself:
   ```bash
   ./scripts/start_test_server.sh --infra-only
   ```
2. **Manual alternative:** Run Redis (e.g. Docker: `docker run -d -p 6379:6379 redis`), the API, and the stream_consumer so they use that Redis.
3. Ensure firewall allows TCP to Redis (e.g. 6379) if connecting from another machine.

**On your machine (client):**

1. Ingest into the DB (if the DB is on the server, set `DB_HOST`, `DB_PORT`, etc. in `.env` or run the ingestor on the server):
   ```bash
   python -m pipeline.ingestion.azure_pm_ingestor --data-dir microsoft_azure_predictive_maintenance
   ```
2. Run the replay and point at the server's Redis with **env** or **CLI**:
   ```bash
   # Option A: environment
   REDIS_HOST=192.168.1.100 REDIS_PORT=6379 python mock_fleet_streamer.py --source azure_pm --speed-multiplier 720

   # Option B: CLI
   python mock_fleet_streamer.py --source azure_pm --redis-host 192.168.1.100 --redis-port 6379 --speed-multiplier 720

   # If Redis has a password:
   python mock_fleet_streamer.py --source azure_pm --redis-host 192.168.1.100 --redis-port 6379 --redis-password secret
   # or: REDIS_PASSWORD=secret python mock_fleet_streamer.py --source azure_pm --redis-host 192.168.1.100
   ```
3. Replace `192.168.1.100` with your test server's IP (or hostname). The API and stream_consumer on the server will consume from the same Redis and the dashboard will show the replayed data.
