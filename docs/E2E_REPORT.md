# E2E Test Suite — Comprehensive Audit Report

**Document type:** Architectural Audit (companion to [ARCHITECTURAL_AUDIT_REPORT.md](./ARCHITECTURAL_AUDIT_REPORT.md))  
**Scope:** End-to-end and integration test design, coverage, infrastructure, and risks.  
**Last audit:** 2026-02.

---

## 1. Executive Summary

The PDM-PILOT test suite is split into **unit tests** (pytest, no Docker) and **integration/E2E tests** (marked `@pytest.mark.integration`, require Docker stack and optionally live services). The main E2E pipeline is defined in **`tests/test_e2e_pipeline.py`**: seven ordered tests that validate ingestion → profiling → feature extraction → inference → alert escalation → work order creation → full onboarding. A separate **`tests/e2e_full_flow.py`** provides a standalone async script (health, auth, telemetry push) that is not part of the pytest integration marker and targets a running API with different credentials.

**Findings:**

- **Strengths:** Clear pipeline ordering, WHAT/WHY comments per assertion, shared fixtures (`docker_stack`, `clean_test_data`, `publish_n_readings`), and alignment with the “Redis sensor_stream as the only ingestion path” architecture.
- **Risks:** E2E tests depend on **live services** (API server, `stream_consumer`, and for test_07 Prefect worker); timeouts and polling can be flaky; test_07 depends on Prefect and MLflow; no explicit test isolation/ordering contract beyond fixture usage.
- **Gaps:** No dedicated performance or chaos tests; `e2e_full_flow.py` is not integrated into pytest `-m integration`; telemetry E2E path (bulk) differs from the Redis `sensor_stream` path exercised in test_01.

This report inventories all test assets, maps the seven E2E tests to architecture and run requirements, and lists recommendations.

---

## 2. Test Inventory

### 2.1 Test Files and Roles

| File | Role | Pytest marker / scope | Docker / services |
|------|------|------------------------|-------------------|
| **test_e2e_pipeline.py** | Full pipeline E2E (7 tests) | `integration` | Docker (TimescaleDB, Redis, Mosquitto); test_01/02 need stream_consumer (+ API for 02); test_07 needs API + Prefect worker |
| **conftest.py** | DB rollback, AsyncClient, auth, integration helpers | — | Unit tests use in-memory/override DB; integration uses real DB/Redis |
| **test_sanity.py** | Health, API health, token factory | (none) | Unit; uses conftest `client` + `auth_headers` |
| **test_errors.py** | Bad JSON, 404, auth errors, health | (none) | Unit; uses conftest `client` and `auth_headers` |
| **test_bad_data.py** | Telemetry validation (422 for invalid payloads) | (none) | Unit; uses conftest async `client` + `auth_headers` |
| **test_schema_validation.py** | Passwords, email, SQLi, XSS, ranges, API 422 | (none) | Unit; TestAPIIntegration uses TestClient + DB init |
| **test_physics_pure.py** | Signal processing (sine, RMS, Nyquist, entropy, etc.) | (none) | Unit; no DB/Redis |
| **test_siemens_adapter.py** | Siemens S7 adapter (snap7) | (none) | Unit; skipped if snap7 not installed |
| **e2e_full_flow.py** | Standalone E2E script (health, login, bulk telemetry) | **Not pytest** | Requires running API; uses own credentials (admin/secret123) |

### 2.2 Pytest Configuration

- **pytest.ini:** `asyncio_mode = auto`, `asyncio_default_fixture_loop_scope = session`, `markers = integration: requires full docker stack`, `addopts = --ignore=notebooks`.
- **CI (`.github/workflows/ci.yml`):** PR runs `pytest -m "not integration"`; merge to main runs `pytest -m integration`.

---

## 3. E2E Pipeline Test Suite (test_e2e_pipeline.py)

### 3.1 Architecture Contract

The module docstring states:

- **Architecture:** Redis `sensor_stream` is the **only** ingestion path to TimescaleDB; features API is `GET /api/features?machine_id=<id>`.
- **Run:** `pytest tests/test_e2e_pipeline.py -m integration -v`.

All seven tests are marked `@pytest.mark.integration` and depend on fixtures and/or live services as below.

### 3.2 Test Descriptions and Dependencies

| Test | WHAT it proves | Fixtures | External services | Timeouts / polling |
|------|----------------|----------|-------------------|--------------------|
| **test_01_data_ingestion** | 100 readings published to Redis → ≥100 rows in `cwru_features`; `sensor_readings.uns_topic` matches UNS prefix | `docker_stack`, `test_machine_id`, `clean_test_data` | **stream_consumer** must be running (consumes Redis, writes features) | Poll up to 30s for count ≥ 100 |
| **test_02_data_profiling** | POST profile/run (202) → GET profile returns 200 with overall and per-metric HEALTHY | Same | **API server** (localhost:8000 or API_BASE_URL), **stream_consumer** (to have data) | 2s sleep after publish; poll up to 30s for profile |
| **test_03_feature_extraction** | RobotFeatureExtractor on 2000-row DataFrame produces all DEFAULT_OUTPUT_COLUMNS and >95% non-null | `test_machine_id` only | None (in-process) | — |
| **test_04_model_inference** | Load pipeline model (gaia_model_*.pkl), run SignalProcessor + inference; schema (failure_probability, health_score, RUL interval) | None | None (in-process); skips if no model file | — |
| **test_05_alert_escalation** | Inject inference payloads; state transitions HEALTHY → WATCH (10 cycles) → WARNING (25); no CRITICAL before 50 | `docker_stack`, `test_machine_id`, `clean_test_data` | Redis (inference_results stream); alert_engine in-process | — |
| **test_06_work_order_creation** | Drive to CRITICAL (55 cycles); work order created; `cmms_sync_status == 'synced'` (CMMS mocked) | Same | Redis; DB; CMMS client **mocked** | — |
| **test_07_full_onboarding_flow** | Insert 65 sensor_readings, POST onboarding/start, poll until COMPLETE; MLflow model registered; GET /api/features returns data | `docker_stack`, `test_new_machine_id`, `clean_test_data` | **API server**; **Prefect worker** (onboarding flow); MLflow (optional skip) | Poll up to 300s every 5s |

### 3.3 Data and Cleanup

- **Test machines:** `TEST-001` (test_01, 02, 05, 06), `TEST-NEW-001` (test_07).
- **clean_test_data:** Teardown fixture (autouse=False) that deletes from `cwru_features`, `sensor_readings`, `data_profiles`, `alert_current_state`, `work_orders`, `onboarding_status` for both machine IDs.
- **publish_n_readings:** Publishes to Redis `sensor_stream` (pub/sub) with subsecond timestamps so each message has a unique `(machine_id, timestamp)` for `cwru_features` uniqueness.

### 3.4 Alignment with Architectural Audit Layers

| Layer (from ARCHITECTURAL_AUDIT_REPORT) | E2E coverage |
|----------------------------------------|--------------|
| **Middleware (Redis, TimescaleDB)** | test_01 (Redis → consumer → DB), test_02 (DB read for profiling), test_05/06 (Redis inference_results + DB state/work_orders) |
| **Intelligence (features, model, RUL)** | test_03 (RobotFeatureExtractor), test_04 (pipeline model + RUL), test_07 (onboarding/MLflow) |
| **Dashboard / API** | test_02 (profile/run, GET profile), test_07 (onboarding API, GET /api/features) |
| **Alerts and work orders** | test_05 (escalation), test_06 (work order creation and CMMS sync status) |

---

## 4. Fixtures and Infrastructure (conftest.py)

### 4.1 Unit Test Infrastructure

- **db_engine** (session): Async engine; uses `DB_NAME=gaia_predictive_test` if not set.
- **db_session**: Transaction-scoped session with rollback; overrides `app.dependency_overrides[get_db]` so API tests use the test DB without persisting.
- **client**: AsyncClient (ASGITransport) bound to app, uses overridden `get_db`.
- **user_token_factory**, **auth_headers**, **operator_headers**, **viewer_headers**: JWT tokens for role-based tests.

### 4.2 Integration Fixtures

- **docker_stack** (session): Skips integration tests if TimescaleDB (5432), Redis (6379), or Mosquitto (1883) are not reachable; does **not** start Docker.
- **test_machine_id** / **test_new_machine_id**: `TEST-001` and `TEST-NEW-001`.
- **clean_test_data**: Teardown only; deletes integration data for both machine IDs from the six tables listed above.
- **publish_n_readings(machine_id, n, anomaly=False, …)**: Publishes `n` JSON messages to Redis `sensor_stream` with `vibration_raw` length 128, optional `uns_topic_override`; returns count published.

### 4.3 Configuration and Constants

- **REDIS_CHANNEL** = `"sensor_stream"`.
- **INTEGRATION_N_SAMPLES** = 128 (vibration length).

---

## 5. Dependencies and Run Requirements

### 5.1 For “Unit Only” (CI PR)

- `pytest -m "not integration" --ignore=notebooks`
- DB and Redis can be present for some tests; conftest overrides DB for API tests. Env: `DB_*`, `REDIS_*`, `SECURITY_JWT_SECRET` (min 32 chars).

### 5.2 For Full Integration (test_e2e_pipeline.py)

| Requirement | Used by |
|-------------|--------|
| **Docker stack up** (TimescaleDB, Redis, Mosquitto) | All 7 tests (docker_stack fixture) |
| **stream_consumer.py** running | test_01, test_02 (so data reaches cwru_features / sensor_readings) |
| **API server** (e.g. uvicorn api_server:app --host 127.0.0.1 --port 8000) | test_02, test_07 |
| **Prefect worker** (e.g. `prefect worker start --pool default`) | test_07 (onboarding flow) |
| **MLflow** (optional) | test_07 (assertion on registered model; skips if not configured) |

Recommended run order (sequential): test_01 → … → test_07. Running in parallel or random order can cause state pollution (e.g. shared Redis/DB and alert state).

### 5.3 Environment Variables

- **API_BASE_URL**: Default `http://localhost:8000`; used by test_02 and test_07 for HTTP calls.
- **DB_***, **REDIS_***, **SECURITY_JWT_SECRET**: Required for app and consumer; tests use same config.

---

## 6. Gaps and Risks

### 6.1 Flakiness and Environment

- **Timeouts:** test_01 polls 30s for 100 rows; test_02 polls 30s for profile; test_07 polls 300s. Under load or slow consumer, these can fail.
- **Service dependency:** If stream_consumer or API is not running, test_01/test_02 fail; no auto-start of these services in pytest.
- **Port conflict:** If another process uses 8000, test_02/test_07 may hit the wrong API or fail to connect.

### 6.2 Coverage Gaps

- **e2e_full_flow.py** is not part of `pytest -m integration`; different auth (admin/secret123) and path (bulk telemetry) than test_e2e_pipeline.
- **MQTT path** is not asserted in test_e2e_pipeline; only Redis `sensor_stream` is validated.
- **No performance or chaos tests** (e.g. latency SLOs, consumer lag, DB failover).

### 6.3 Test Data and Isolation

- **clean_test_data** runs at teardown; if a test fails before teardown, leftover data can affect the next run unless the suite is re-run with a clean DB.
- **station_config.json** is read/written by test_05 and test_06 (alert thresholds); shared file can be modified by concurrent runs if any.

### 6.4 test_07 Specifics

- Depends on Prefect deployment/worker; if the onboarding flow is not deployed or worker is down, test_07 times out.
- MLflow assertion is best-effort (skip if not configured); no stub of MLflow in conftest.

---

## 7. Recommendations

1. **Document run order and prerequisites** in README or `tests/README.md`: start Docker → (optional) start API and stream_consumer → (for test_07) start Prefect worker → run `pytest tests/test_e2e_pipeline.py -m integration -v` in order.
2. **Consider a single “integration entrypoint” script** that starts API and stream_consumer in the background (or in CI with services), then runs the seven tests sequentially.
3. **Either integrate e2e_full_flow.py into pytest** (e.g. as an optional integration test with its own marker and credentials) or clearly document it as a manual/smoke script and keep it separate from `-m integration`.
4. **Add a minimal “sanity” integration test** that only checks Docker stack and optionally one Redis write + one DB read, so CI can fail fast without running the full seven tests when the stack is broken.
5. **Optionally increase test_01 deadline** (e.g. 60s) or make it configurable for slower environments, while keeping 30s as default.
6. **Lock or copy station_config.json** for test_05/test_06 so parallel runs or failed runs do not leave the repo config in an unexpected state.

---

## 8. Summary Table — E2E Tests at a Glance

| Test | Pass condition | Needs API | Needs stream_consumer | Needs Prefect | Timeout (s) |
|------|----------------|-----------|------------------------|---------------|-------------|
| test_01_data_ingestion | ≥100 rows cwru_features; uns_topic in sensor_readings | No | Yes | No | 30 |
| test_02_data_profiling | Profile 200, overall_status HEALTHY | Yes | Yes | No | 30 |
| test_03_feature_extraction | All columns, >95% non-null | No | No | No | — |
| test_04_model_inference | Schema + RUL interval | No | No | No | — |
| test_05_alert_escalation | HEALTHY→WATCH→WARNING; no CRITICAL before 50 | No | No | No | — |
| test_06_work_order_creation | Work order exists, cmms_sync_status=synced | No | No | No | — |
| test_07_full_onboarding_flow | COMPLETE, MLflow model, GET /api/features | Yes | No | Yes | 300 |

This report should be updated when new E2E tests are added, when run requirements change, or when CI workflow is modified.
