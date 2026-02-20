# PDM-PILOT Architectural Audit Report

**Purpose:** Structured analysis for implementation roadmap.  
**Target architecture:** Physical → Edge → Middleware → Intelligence → Dashboard (5 layers).

---

## 1. WHAT EXISTS — File & Folder Inventory by Layer

### Layer 1 (Physical) — Devices & Protocol Access

| File / Folder | One-sentence description | Layer |
|---------------|-------------------------|--------|
| `abb_adapter.py` | Connects to ABB robot via **OPC UA** (asyncua), reads speed/torque/joints, publishes JSON to Redis `sensor_stream`. | Physical / Edge |
| `siemens_s7_adapter.py` | Connects to Siemens S7-1500 via **snap7** (native S7), reads DB block (motor_current, cycle_timer_ms, etc.), normalizes to common schema, publishes to Redis. | Physical / Edge |
| `opc_client_adapter.py` | Generic OPC UA edge client; buffers offline, posts to Gaia API when online; used for non-ABB OPC UA sources. | Physical / Edge |

**Note:** Target specifies "ABB robot via **RWS**" — current implementation uses **OPC UA** only. RWS (Robot Web Services) is not present.

---

### Layer 2 (Edge) — Protocol → Normalizer → MQTT

| File / Folder | One-sentence description | Layer |
|---------------|-------------------------|--------|
| `stream_publisher.py` | Reads from Redis or replays `training_data.pkl` at **10 Hz**; optionally publishes to MQTT via UNS mapper; can start ABB + S7 adapters in background. | Edge |
| `mqtt_client.py` | MQTT publisher wrapper using `config.get_settings().mqtt`; connect/publish with optional TLS. | Edge |
| `uns_mapper.py` | Maps machine_id + flat payload to ISA-95 UNS topic hierarchy (`site/area/cell/asset/data_type/metric`); loads `pipeline/station_config.json`. | Edge / Middleware |
| `uns_explorer.py` | CLI to explore UNS topic tree and subscribe to MQTT; uses config for broker. | Edge |
| `refinery.py` | Feature extraction from CSV/vibration; publishes to Redis; used in legacy/sim path. | Edge / Intelligence |
| `watcher_service.py` | File watcher; on new file, generates UUID, pushes JSON to Redis `raw_sensor_data` (maxlen 10000). | Edge |

**Note:** Target specifies "**100 ms**" (10 Hz) publish; `stream_publisher` and pipeline ingestion use 10 Hz; Siemens S7 adapter uses 100 ms poll — aligned. No single "data normalizer" module: normalization is done inside each adapter and in pipeline stages.

---

### Layer 3 (Middleware) — Broker, Streams, Persistence

| File / Folder | One-sentence description | Layer |
|---------------|-------------------------|--------|
| `docker-compose.yml` | Defines TimescaleDB, Redis, MongoDB, **Mosquitto**, pipeline stages (ingestion → cleansing → context → persistence → inference), API, frontend. | Middleware |
| `docker/mosquitto/` | Mosquitto config and certs for MQTT broker. | Middleware |
| `stream_consumer.py` | Consumes Redis stream, runs signal processor + ML inference, batch-inserts into TimescaleDB `cwru_features`; loads `gaia_model.pkl`. | Middleware / Intelligence |
| `pipeline/ingestion/service.py` | Redis stream producer; replays `training_data.pkl` at 10 Hz or uses embedded topology; publishes to `raw_sensor_data`. | Middleware |
| `pipeline/cleansing/service.py` | Consumes `raw_sensor_data`, Z-score outlier filter, publishes to `clean_sensor_data`. | Middleware |
| `pipeline/contextualization/service.py` | Consumes `clean_sensor_data`, applies digital twin from `station_config.json`, publishes to `contextualized_data`. | Middleware |
| `pipeline/persistence/service.py` | Consumes `contextualized_data`, batch-writes to TimescaleDB `sensor_readings`. | Middleware |
| `pipeline/inference/service.py` | Consumes cleansed/contextualized stream, runs vibration features + RUL model, publishes to `inference_results`. | Middleware / Intelligence |
| `database.py` | Async SQLAlchemy engine, session factory, health check; uses `config.get_settings().database`. | Middleware |
| `setup_schema.py` | Creates `sensor_readings` and `cwru_features` tables (TimescaleDB). | Middleware |

---

### Layer 4 (Intelligence) — Features, Training, RUL, Drift

| File / Folder | One-sentence description | Layer |
|---------------|-------------------------|--------|
| `train_model.py` | XGBoost RUL (NASA) or anomaly (custom) trainer; `--data-source nasa|custom`, `--data-path`; saves `xgb_rul_v1.json` / `gaia_model_custom.pkl`. | Intelligence |
| `train_rul_model.py` | Standalone RUL training from sensor features; versioned `rul_model_v*` + metadata; cost-sensitive thresholds. | Intelligence |
| `labeling_engine.py` | RUL labeling from failure time; **label_by_threshold** for weak-supervision anomaly labels. | Intelligence |
| `pipeline/cleansing/robot_feature_extractor.py` | Robot/PLC features from normalized DataFrame (torque stats, z-score, cycle drift, temp rate). | Intelligence |
| `pipeline/ingestion/custom_data_ingestor.py` | Ingest CSV (cwru or simple long format), normalize to standard robot/PLC columns. | Intelligence |
| `analytics_engine.py` | Loads `cwru_features`, anomaly detection (Isolation Forest), RUL correlation, versioned RUL model load; writes `insight_report.json`. | Intelligence |
| `advanced_features.py` | SignalProcessor (FFT, peaks, envelope); used by API, stream_consumer, pipeline inference. | Intelligence |
| `inference_service.py` | Standalone inference: Redis subscriber, load model from `station_config.json`, scaler, RUL model; health score 0–100%. | Intelligence |
| `anomaly_discovery/` | Discovery engine, worker, detectors (Isolation Forest, temporal autoencoder, ensemble), correlation analyzers; confidence scoring. | Intelligence |
| `retrain_gaia_model.py` | Retrain and save `gaia_model.pkl` pipeline. | Intelligence |
| `scripts/collect_baseline.py` | Export last N days from TimescaleDB `cwru_features` to CSV for baseline/custom training. | Intelligence |

**Note:** Target requires "**calibrated RUL with confidence intervals**" and "**drift monitoring**". Current RUL is point estimates only; confidence intervals are not produced. Drift appears in robot_feature_extractor (cycle/time drift) and anomaly_discovery (drift-type anomalies), but there is no dedicated **model/data drift monitoring** service.

---

### Layer 5 (Dashboard) — Fleet, Machine Detail, Alerts, SAP

| File / Folder | One-sentence description | Layer |
|---------------|-------------------------|--------|
| `api_server.py` | FastAPI app: health, auth, /api/features, /api/machines, WebSocket stream, inference latest, analytics trigger, enterprise routes. | Dashboard / Middleware |
| `enterprise_api.py` | REST: alarms, **work orders** (CRUD), MTBF/MTTR, shift schedules; JWT + rate limit. | Dashboard |
| `dashboards/dashboard_streamlit.py` | Streamlit UI: fleet view, asset detail, work order PDF export; uses `DATABASE_URL`, queries `sensors`. | Dashboard |
| `dashboards/_DEPRECATED_dashboard.py` | Legacy dashboard (deprecated). | Dashboard |
| `frontend/` | Frontend container (e.g. React/Vue); serves on 8080 in dev. | Dashboard |

**Note:** Target requires "**SAP work order push**". Current: work orders are internal (DB + API + PDF download). No SAP client or outbound SAP API integration.

---

### Cross-Cutting / Support

| File / Folder | One-sentence description | Layer |
|---------------|-------------------------|--------|
| `config.py` | Pydantic-settings: Database, Redis, Security, Log, Model, Edge, ABB, SiemensS7, MQTT; `get_settings()` cached. | Cross-cutting |
| `logger.py` / `core/logger.py` | Logging setup and structlog integration. | Cross-cutting |
| `dependencies.py` | FastAPI dep: get_current_user, auth headers. | Cross-cutting |
| `auth_utils.py` | JWT decode; uses config. | Cross-cutting |
| `schemas/` | Pydantic models (security, ml, enterprise, machine, response). | Cross-cutting |
| `services/` | auth_service, machine_service, telemetry_service, alarm_service; use DB and config. | Cross-cutting |
| `middleware/` | security_headers, audit_logger. | Cross-cutting |
| `core/exceptions.py` | Custom HTTP exceptions. | Cross-cutting |
| `scripts/` | verify_config, build features, download NASA/FEMTO, validate RUL, migrate, backup, etc. | Cross-cutting |

---

## 2. WHAT'S MISSING vs Target Architecture

| Target | Status | Gap |
|--------|--------|-----|
| **Layer 1: ABB via RWS** | ❌ Missing | ABB adapter uses OPC UA only. Add RWS client for Robot Web Services (or document OPC UA as chosen standard). |
| **Layer 1: Siemens S7-1500 native S7** | ✅ Present | `siemens_s7_adapter.py` uses snap7, configurable DB/rack/slot. |
| **Layer 2: 100 ms** | ✅ Present | S7 poll 100 ms; stream publisher 10 Hz. |
| **Layer 2: Data normalizer** | ⚠️ Partial | Normalization is per-adapter and in pipeline; no single shared “normalizer” contract. |
| **Layer 3: MQTT + UNS** | ✅ Present | Mosquitto in compose; UNS mapper + MQTT client. |
| **Layer 4: Feature extraction** | ✅ Present | Refinery, advanced_features, robot_feature_extractor, pipeline inference. |
| **Layer 4: AutoML training** | ⚠️ Partial | Manual train_model/train_rul_model; no automated hyperparameter or pipeline selection. |
| **Layer 4: Calibrated RUL + confidence intervals** | ❌ Missing | RUL is point estimate only; no quantile/interval output. |
| **Layer 4: Drift monitoring** | ⚠️ Partial | Feature-level drift in robot extractor and anomaly discovery; no dedicated model/data drift service. |
| **Layer 5: Fleet + machine detail** | ✅ Present | API + Streamlit dashboard. |
| **Layer 5: Prescriptive alerts** | ⚠️ Partial | Alarms and recommendations exist; “prescriptive” (actionable) clarity varies. |
| **Layer 5: SAP work order push** | ❌ Missing | Work orders are internal only; no SAP API or outbound push. |

---

## 3. DEPENDENCY MAP

### Key import directions (high level)

- **config** ← Many: api_server, stream_consumer, abb_adapter, siemens_s7_adapter, mqtt_client, uns_explorer, analytics_engine, database, services/auth_service, dependencies (indirect), migrations, tests/conftest, scripts/collect_baseline, opc_client_adapter, enterprise_api, auth_utils.
- **api_server** → config, database, dependencies, schemas, auth_utils, analytics_engine, advanced_features, core.exceptions, enterprise_api (router), joblib (gaia_model.pkl).
- **stream_consumer** → config, logger, advanced_features, redis, psycopg2, joblib (gaia_model.pkl).
- **analytics_engine** → config, sqlalchemy, sklearn, xgboost, pandas; loads RUL model from MODEL settings.
- **enterprise_api** → config, database, dependencies, schemas, services.auth_service.
- **database** → config, logger.
- **dependencies** → logger, schemas.security, services.auth_service.
- **services.auth_service** → config, logger, schemas.security.

### Circular / tight coupling risks

1. **api_server ↔ analytics_engine**  
   API imports analytics_engine; analytics_engine uses config (no direct api_server import). No cycle, but API and analytics share no clear interface (e.g. “run report” contract).

2. **api_server → advanced_features (SignalProcessor)**  
   Global state in lifespan holds processor; tight coupling to one signal implementation.

3. **gaia_model.pkl path**  
   api_server and stream_consumer both `joblib.load('gaia_model.pkl')` with no shared constant; path is CWD-relative. Easy to break when run from different CWDs.

4. **logger vs core.logger**  
   Some files use `logger`, others `core.logger` or `get_logger`; not a cycle but inconsistent.

5. **conftest** imports api_server and config  
   Tests depend on app and config; no circular import observed.

### Files that do not use config (use env or literals)

- stream_publisher (REDIS_HOST/PORT from os.getenv)
- pipeline/*/service.py (REDIS_*, DB_*, STATION_CONFIG_PATH, etc. from os.getenv)
- watcher_service (localhost:6379, raw_sensor_data, maxlen 10000)
- refinery, mock_feature_producer (REDIS_HOST/PORT)
- inference_service (CONFIG_FILE, SCALER_PATH, REDIS_HOST/PORT literals)
- setup_schema (DB_* from os.getenv)
- insert_sample_anomalies (DB_HOST, DB_PORT literals)
- populate_ai_discovery (API_BASE localhost:8000)
- load_fleet_data (dotenv)
- dashboards/dashboard_streamlit (DATABASE_URL only)

---

## 4. CONFIGURATION GAPS (Hardcoded Values)

Values that should be in `config.py` or `.env` but are hardcoded or only in code:

| Location | Hardcoded value | Recommendation |
|----------|------------------|----------------|
| `api_server.py` | `joblib.load('gaia_model.pkl')` | Add `MODEL_PATH` or `GAIA_MODEL_PATH` to config (e.g. ModelSettings). |
| `api_server.py` | `host="0.0.0.0", port=8000` | Already common; can add APP_HOST/APP_PORT to config if desired. |
| `stream_consumer.py` | `joblib.load('gaia_model.pkl')` | Same as above; use shared config. |
| `stream_publisher.py` | REDIS_HOST/PORT via os.getenv only | Use config.redis in this script. |
| `stream_publisher.py` | PUBLISH_RATE_HZ = 10, REDIS_CHANNEL = 'sensor_stream' | Add to config (e.g. Edge or a new StreamPublisher section). |
| `watcher_service.py` | host='localhost', port=6379, maxlen=10000 | Use config.redis; maxlen in config. |
| `refinery.py` | REDIS_HOST, REDIS_PORT | Use config.redis. |
| `mock_feature_producer.py` | REDIS_HOST, REDIS_PORT | Use config.redis. |
| `inference_service.py` | CONFIG_FILE, SCALER_PATH, REDIS_HOST/PORT, MODEL_PATH | Use config (redis, model paths). |
| `pipeline/persistence/service.py` | DB_PASSWORD default 'password' | Prefer config.database; avoid default password in code. |
| `pipeline/ingestion/service.py` | DATA_FILE, PUBLISH_RATE_HZ | Already env; standardize under config. |
| `pipeline/contextualization/service.py` | STATION_CONFIG_PATH, BUFFER_MAX_SIZE | Already env; optional config section. |
| `pipeline/inference/service.py` | MODEL_PATH, RUL_MODEL_PATH, REDIS_* | Use config. |
| `uns_mapper.py` | default config_path "pipeline/station_config.json" | Keep default; allow override via config. |
| `insert_sample_anomalies.py` | DB_HOST, DB_PORT, DB_PASSWORD | Use config.database. |
| `populate_ai_discovery.py` | API_BASE = "http://localhost:8000/api/discovery" | Config (e.g. API_BASE_URL). |
| `enterprise_api.py` | Hardcoded demo user "hashed_password" for admin | Remove or gate behind environment (e.g. seed only in dev). |
| `train_model.py` | Paths: SCALER_PATH, MODEL_PATH, CUSTOM_* under BASE_DIR | Acceptable; optional env override. |
| `scripts/validate_rul_on_nasa.py` | SCALER_PATH, MODEL_PATH under BASE | Acceptable. |
| `dashboards/dashboard_streamlit.py` | DATABASE_URL, REFRESH_RATE_SEC, port check localhost | DB from env is fine; document REFRESH_RATE and port in config if needed. |
| `tests/e2e_full_flow.py` | API_URL, USERNAME, PASSWORD | Use env (e.g. E2E_API_URL, E2E_USER, E2E_PASSWORD). |

---

## 5. TEST COVERAGE

### What is tested

- **tests/test_sanity.py** — Health, /api/health, token factory roles.
- **tests/test_errors.py** — Error handling (HTTP client).
- **tests/test_schema_validation.py** — Pydantic schemas, SQL injection / XSS rejection, work order validation.
- **tests/test_physics_pure.py** — SignalProcessor (FFT, peaks, crest, aliasing); deterministic behavior.
- **tests/test_bad_data.py** — Bad input handling.
- **tests/load_test_async.py** — Async load testing.
- **tests/e2e_full_flow.py** — Health, login, telemetry push, machines, inference (requires running API).
- **tests/conftest.py** — DB engine, session, client, auth fixtures; uses config and api_server.
- **test_mqtt_publish.py**, **test_security_headers.py**, **test_dashboard_query.py** — MQTT, headers, dashboard (root-level).

### What is not tested (or only lightly)

- **ABB adapter** — No unit or integration test; requires robot or stub.
- **Siemens S7 adapter** — No unit test; DB layout and snap7 calls untested in CI.
- **UNS mapper** — No tests for topic layout and payload mapping.
- **stream_consumer** — No test for Redis → DB and inference path.
- **Pipeline stages** (ingestion, cleansing, context, persistence, inference) — No automated tests.
- **train_model.py** (NASA + custom) — No test that training runs and writes expected artifacts.
- **labeling_engine.label_by_threshold** — No unit tests.
- **custom_data_ingestor** — No tests for cwru vs simple format detection and normalization.
- **robot_feature_extractor** — No tests for feature shape and bounds.
- **collect_baseline.py** — No test (DB-dependent).
- **SAP integration** — N/A (not implemented).
- **RUL confidence intervals** — N/A (not implemented).

### Most critical to test and currently missing

1. **Pipeline ingestion → cleansing → context → persistence** (Redis + DB) — One integration test with fake Redis and DB would de-risk deploy.
2. **UNS mapper** — Unit tests for topic strings and payload → topics for a few machines.
3. **custom_data_ingestor** — At least two CSV fixtures (cwru + simple) and assert normalized columns.
4. **label_by_threshold** — One DataFrame + rules, assert anomaly column.
5. **Siemens S7 adapter** — Unit test with mocked snap7 (e.g. byte buffer → SiemensTelemetryData).

---

## 6. PRIORITY ORDER TO CLOSE GAPS

Recommended order with one-line rationale:

| # | Action | Rationale |
|---|--------|-----------|
| 1 | **Move all model paths (gaia_model.pkl, scaler, RUL) and Redis/DB used by stream_consumer and API into config** | Single source of truth; safe CWD and env changes; unblocks Docker and multi-env. |
| 2 | **Add unit tests for custom_data_ingestor (cwru + simple CSV)** | Prevents regressions when swapping in real robot/PLC exports. |
| 3 | **Add unit tests for label_by_threshold and robot_feature_extractor output shape** | Custom training path is critical; tests lock contract. |
| 4 | **Add UNS mapper unit tests (topic + payload → topics)** | MQTT integration bugs are hard to trace without tests. |
| 5 | **One integration test: pipeline stages (ingestion → persistence) with test Redis + DB** | Catches stream/contract breakage before deploy. |
| 6 | **Document or implement ABB RWS vs OPC UA** | Align with target “ABB via RWS” or formally choose OPC UA. |
| 7 | **Add RUL confidence intervals (e.g. quantile regression or posterior)** | Matches target “calibrated RUL with confidence intervals”. |
| 8 | **Introduce explicit data/model drift monitoring (e.g. feature distribution vs baseline, or model performance over time)** | Closes “drift monitoring” in target. |
| 9 | **Replace hardcoded credentials and URLs in insert_sample_anomalies, populate_ai_discovery, e2e tests with config/env** | Security and portability. |
| 10 | **SAP work order push (outbound API or connector)** | Required for “SAP work order push” in Layer 5. |
| 11 | **Optional: AutoML hook (e.g. trigger training or hyperparameter sweep from API/config)** | Improves “AutoML training” alignment. |

---

## Summary Table

| Layer | Exists | Missing / Partial |
|-------|--------|-------------------|
| 1 Physical | Siemens S7 (snap7), ABB (OPC UA) | ABB RWS |
| 2 Edge | 10 Hz / 100 ms, MQTT, UNS | Single normalizer contract |
| 3 Middleware | Redis, Mosquitto, pipeline stages, TimescaleDB | — |
| 4 Intelligence | Features, RUL, anomaly, custom pipeline | Calibrated RUL, dedicated drift monitoring, AutoML |
| 5 Dashboard | Fleet, machine, work orders (internal) | SAP push, stronger prescriptive alerts |

Use this document as the implementation roadmap: address config and tests first, then RUL calibration and drift, then SAP and optional AutoML.
