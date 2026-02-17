# ISO 55001 Asset Management — PDM-PILOT Mapping

> **ISO 55001:2024** — Asset management systems. Requirements for establishing, implementing, maintaining, and improving an asset management system.

---

## Clause Mapping

| ISO 55001 Clause | Description | PDM-PILOT Implementation |
|------------------|-------------|---------------------------|
| **4. Context** | Organization context, interested parties | `config.py`, `schemas/`, fleet topology |
| **5. Leadership** | Policy, roles, objectives | Enterprise API auth, RBAC |
| **6. Planning** | SAMP, objectives, risk, opportunities | `services/`, `analytics_engine.py` (risk correlation) |
| **7. Support** | Resources, competence, awareness, communication, **data** | `database.py`, `core/`, `schemas/` |
| **8. Operation** | Operating activities | **6-stage pipeline** (`pipeline/`, `watcher_service.py`, `refinery.py`, `inference_service.py`) |
| **9. Performance** | Monitoring, evaluation, audit | `analytics_engine.py`, MTBF/MTTR, `validate_rul_on_nasa.py` |
| **10. Improvement** | Nonconformity, corrective action | Anomaly discovery, work orders, alarms |

---

## Directory Structure (ISO 55001 Aligned)

```
PDM-PILOT/
├── asset_management/           # ISO 55001 alignment
│   ├── context/                # Clause 4 — config, schemas
│   │   └── (config.py, schemas/, station_config.json)
│   ├── planning/               # Clause 6 — SAMP, risk
│   │   └── (services/, analytics_engine.py)
│   ├── support/                # Clause 7 — data, infrastructure
│   │   └── (database.py, core/, dependencies.py)
│   ├── operations/            # Clause 8 — 6-stage pipeline
│   │   └── (pipeline/, watcher_service.py, refinery.py, inference_service.py)
│   ├── performance/            # Clause 9 — evaluation
│   │   └── (scripts/validate_rul_on_nasa.py, MTBF/MTTR endpoints)
│   └── improvement/           # Clause 10 — corrective action
│       └── (anomaly_discovery/, work_orders, alarms)
│
├── api_server.py               # Orchestration (Clause 8)
├── enterprise_api.py           # Leadership, planning (Clauses 5, 6)
└── frontend/                   # Presentation layer
```

---

## Key Artifacts

| ISO 55001 Artifact | PDM-PILOT Equivalent |
|--------------------|----------------------|
| Strategic Asset Management Plan (SAMP) | Fleet topology, machine metadata, RUL objectives |
| Asset management objectives | RUL targets, failure prediction thresholds |
| Risk assessment | `analytics_engine.py` — correlate anomalies with events |
| Data and information (7.6) | TimescaleDB schema, `cwru_features`, `pdm_alarms` |
| Operating activities (8) | 6-stage pipeline (INGEST → CLEANSE → CONTEXT → PERSIST → INFER → ORCH) |

---

## References

- [ISO 55001:2024](https://www.iso.org/standard/83054.html)
- [ISO 55002](https://www.iso.org/standard/70402.html) — Application guidelines
- [docs/ARCHITECTURE.md](./ARCHITECTURE.md) — 6-stage pipeline detail
