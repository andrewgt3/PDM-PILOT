# GAIA 6-Stage Pipeline Architecture

> **ISO Golden Standard** — Compartmentalized industrial data pipeline. Each stage is isolated; no cross-stage bleed.
>
> **Standards alignment:** ISO 55001 (Asset Management) Clause 8 — Operation | ISO 13374 (Condition Monitoring) — Data processing, communication, presentation

---

## Overview

The GAIA Predictive Maintenance platform follows a **6-stage pipeline architecture**. Each stage has a single responsibility, defined inputs/outputs, and lives in its own compartment. Stages communicate only via Redis streams—no direct coupling.

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ 1.      │ →  │ 2.      │ →  │ 3.      │ →  │ 4.      │ →  │ 5.      │ →  │ 6.      │
│ INGEST  │    │ CLEANSE │    │ CONTEXT │    │ PERSIST │    │ INFER   │    │ ORCH    │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
     │              │              │              │              │              │
     ▼              ▼              ▼              ▼              ▼              ▼
 raw_sensor    clean_*      contextualized   TimescaleDB   inference_    API + WebSocket
 _data         _data        _data            (cwru_*)      results
```

---

## Stage Mapping (Compartmentalized)

| Stage | Name | Responsibility | Redis Stream In | Redis Stream Out | Component Path |
|-------|------|-----------------|-----------------|------------------|----------------|
| **1** | INGEST | Poll OPC-UA / file watcher; publish raw telemetry | — | `raw_sensor_data` | `pipeline/ingestion/`, `watcher_service.py` |
| **2** | CLEANSE | Outlier removal, feature extraction | `raw_sensor_data` | `clean_features`, `clean_sensor_data` | `pipeline/cleansing/`, `refinery.py` |
| **3** | CONTEXT | Digital twin mapping, enrichment | `clean_*` | `contextualized_data` | `pipeline/contextualization/` |
| **4** | PERSIST | Batch write to TimescaleDB | `contextualized_data` | — (DB) | `pipeline/persistence/`, `stream_consumer.py` |
| **5** | INFER | XGBoost RUL prediction | `clean_features` | `inference_results` | `pipeline/inference/`, `inference_service.py` |
| **6** | ORCH | API Gateway, WebSocket, Frontend | `inference_results` | — (HTTP/WS) | `api_server.py`, `frontend/` |

---

## Compartment Boundaries

- **Stage N** reads only from streams produced by **Stage N-1** (or earlier).
- **Stage N** writes only to its designated output stream(s).
- No stage imports or calls another stage directly.
- Shared config (e.g. `station_config.json`) is read-only.

---

## Directory Structure (Aligned to Stages)

```
PDM-PILOT/
├── pipeline/                    # Stages 1–5 (microservices)
│   ├── ingestion/               # Stage 1
│   ├── cleansing/               # Stage 2
│   ├── contextualization/       # Stage 3
│   ├── persistence/             # Stage 4
│   └── inference/               # Stage 5
├── api_server.py                # Stage 6 (Orchestration)
├── frontend/                    # Stage 6 (UI)
├── watcher_service.py           # Stage 1 (file-based ingest)
├── refinery.py                  # Stage 2 (feature extraction)
├── inference_service.py         # Stage 5 (standalone RUL)
├── stream_consumer.py           # Stage 4 (persistence)
├── scripts/                     # Offline: data download, feature build, train
└── docs/
    └── ARCHITECTURE.md          # This file
```

---

## Data Flow (NASA + FEMTO)

1. **Data** → `download_nasa_data.py`, `download_femto_data.py` (offline)
2. **Ingest** → Upload or Extract Archive → `data/ingest/`
3. **Features** → `build_nasa_features_with_physics.py`, `build_combined_features.py` (offline)
4. **Model** → `retrain_rul_with_physics.py` (offline)
5. **Pipeline** → Stages 1–6 (real-time)

---

## Standards Alignment

| Standard | Scope | Mapping |
|----------|-------|---------|
| **ISO 55001** | Asset management systems | [ISO_55001_MAPPING.md](./ISO_55001_MAPPING.md) |
| **ISO 13372** | Condition monitoring vocabulary | [ISO_13372_MAPPING.md](./ISO_13372_MAPPING.md) |
| **ISO 13374** | Data processing, communication, presentation | [ISO_13372_MAPPING.md](./ISO_13372_MAPPING.md) |

## References

- `docker-compose.yml` — Stage service definitions
- `README.md` — Quick start, API endpoints
- `frontend/src/pages/SetupGuidePage.jsx` — In-app setup guide
