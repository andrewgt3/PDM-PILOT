# Demos

## Azure PM Full Fleet Demo

This demo replays the Azure Predictive Maintenance (PdM) dataset from TimescaleDB as if it were live telemetry, so you can run the full pipeline and dashboard against a 100-machine fleet with known failure events.

### How to run the full fleet demo (3 commands)

1. **Ingest Azure PM data and populate station config**

   Place the Azure PdM CSV files in `data/azure_pm/` (e.g. `PdM_telemetry.csv`, `PdM_failures.csv`, etc.), then run:

   ```bash
   python -m pipeline.ingestion.azure_pm_ingestor --data-dir data/azure_pm/
   ```

   This merges the CSVs, inserts into `sensor_readings` and `cwru_features`, and updates `pipeline/station_config.json` with 100 machines across 4 areas (MachineShop, Assembly, Stamping, Finishing) and 3 equipment types (CNC Machine, Hydraulic Press, Conveyor).

2. **Start the replay**

   From the project root:

   ```bash
   python mock_fleet_streamer.py --source azure_pm --speed-multiplier 720 --start-from "2023-01-01"
   ```

   Or run the replay module directly:

   ```bash
   python -m demos.azure_pm_replay --speed-multiplier 720 --start-from "2023-01-01"
   ```

   Replay reads from TimescaleDB in timestamp order and publishes each row to Redis `sensor_stream` with time compression. Default `--speed-multiplier 720` means 1 month of history is replayed in 1 hour. Use `--speed-multiplier 1` for real-time.

3. **Start API, frontend, and stream consumer**

   In separate terminals (or your usual runbook):

   - Start Redis (if not already running).
   - Start the API server (e.g. `uvicorn api_server:app` or your normal command).
   - Start the stream consumer so it consumes from Redis and writes to the DB / runs inference.
   - Start the frontend (e.g. `npm run dev` in `frontend/`).

   The dashboard will show live-updating machines as replay progresses. Optionally enable MQTT so UNS topics and `events/failure_ground_truth` are published (set `MQTT_PUBLISH_ENABLED=true` and use `--mqtt` when running `demos.azure_pm_replay` directly).

### What to expect in the dashboard

- **100 machines** in the fleet view, grouped by shop (MachineShop, Assembly, Stamping, Finishing) and line (Line_1 … Line_25).
- **Treemap and topology** show equipment types (CNC Machine, Hydraulic Press, Conveyor). Cell size reflects criticality and health delta from baseline; color goes green → amber → red with worsening health.
- **REPLAY MODE** banner at the top of the fleet treemap when all visible machine IDs are numeric 1–100 (Azure PM IDs).
- **Health and RUL** update as replay progresses; you can select a machine for detail view.

### How to identify upcoming failure events in the replay

- **Ground truth topic**: When a row corresponds to a known failure (from `cwru_features.failure_class = 1`), the replay engine publishes a message to:
  - Redis channel: `events/failure_ground_truth`
  - MQTT topic (if MQTT enabled): `PlantAGI/Demo/events/failure_ground_truth`

  Payload shape: `{ "machine_id", "timestamp", "failure": true }`. Subscribing to this topic (e.g. with a small script or MQTT client) lets you log or display (machine_id, timestamp) of known failures for validation.

- **In the UI**: The dashboard does not currently list “upcoming” ground-truth failures; use the topic above or query `cwru_features` where `failure_class = 1` to see which (machine_id, timestamp) are failure events before or during replay.

- **CLI**: To see how many failure events exist in the dataset, you can query the DB before replay, e.g.:

  ```sql
  SELECT machine_id, timestamp FROM cwru_features WHERE failure_class = 1 ORDER BY timestamp;
  ```
