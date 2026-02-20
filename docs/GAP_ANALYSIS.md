# PDM-PILOT Gap Analysis

Gap-focused view derived from the architectural audit.  
**Target architecture:** Physical → Edge → Middleware → Intelligence → Dashboard (5 layers).  
**Full audit:** `docs/ARCHITECTURAL_AUDIT_REPORT.md`.

> **Reference in prompts:** In every subsequent prompt or planning session, cite this document as: **“as identified in our gap analysis.”**

---

## 1. WHAT'S MISSING vs Target Architecture

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

## 2. CONFIGURATION GAPS (Hardcoded Values)

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

## 3. TEST COVERAGE GAPS

### What is not tested (or only lightly)

- **ABB adapter** — No unit or integration test; requires robot or stub.
- **Siemens S7 adapter** — Unit tests added in `tests/test_siemens_adapter.py` (mocked snap7, byte conversion); CI coverage in place.
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
5. **Siemens S7 adapter** — ✅ Addressed: `tests/test_siemens_adapter.py` mocks snap7 and verifies byte conversion.

---

## 4. PRIORITY ORDER TO CLOSE GAPS

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

## 5. SUMMARY

| Layer | Exists | Missing / Partial |
|-------|--------|-------------------|
| 1 Physical | Siemens S7 (snap7), ABB (OPC UA) | ABB RWS |
| 2 Edge | 10 Hz / 100 ms, MQTT, UNS | Single normalizer contract |
| 3 Middleware | Redis, Mosquitto, pipeline stages, TimescaleDB | — |
| 4 Intelligence | Features, RUL, anomaly, custom pipeline | Calibrated RUL, dedicated drift monitoring, AutoML |
| 5 Dashboard | Fleet, machine, work orders (internal) | SAP push, stronger prescriptive alerts |

Use this document alongside the full audit for prioritised gap closure.
