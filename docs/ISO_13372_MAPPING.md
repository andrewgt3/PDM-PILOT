# ISO 13372 / ISO 13374 — Condition Monitoring Mapping

> **ISO 13372:2012** — Condition monitoring and diagnostics of machines. Vocabulary.  
> **ISO 13374** — Data processing, communication and presentation.

---

## ISO 13372 Vocabulary (Key Terms)

| Term | Definition | PDM-PILOT Implementation |
|------|------------|---------------------------|
| **Condition monitoring** | Process of determining the condition of machinery | Vibration analysis, RUL prediction, anomaly detection |
| **Diagnostics** | Process of identifying cause of fault | BPFO/BPFI/BSF analysis, degradation_score |
| **Prognostics** | Process of predicting future condition | XGBoost RUL model, `rul_minutes` |
| **Remaining useful life (RUL)** | Estimated time until failure | `inference_service.py`, `rul_days` |
| **Health indicator** | Parameter indicating machine condition | `degradation_score`, `health_score` |
| **Baseline** | Reference condition for comparison | `rms_ratio_baseline`, first 5 samples |

---

## ISO 13374 Architecture Mapping

| ISO 13374 Part | Title | PDM-PILOT Implementation |
|----------------|-------|---------------------------|
| **Part 1** | General guidelines | 6-stage pipeline design, compartment boundaries |
| **Part 2** | Data processing | Stages 1–5: Ingest → Cleanse → Context → Persist → Infer |
| **Part 3** | Communication | Redis streams, WebSocket, REST API |
| **Part 4** | Presentation | Frontend dashboards, RUL cards, alarms, work orders |

---

## Data Processing (ISO 13374-2) — Stage Mapping

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ISO 13374-2 DATA PROCESSING                           │
├─────────────────────────────────────────────────────────────────────────┤
│  Acquisition    │  Processing      │  Analysis         │  Decision       │
│  (Stage 1)      │  (Stages 2–3)    │  (Stage 5)        │  (Stage 6)      │
├─────────────────┼─────────────────┼───────────────────┼─────────────────┤
│  raw_sensor_    │  clean_features  │  inference_       │  API +          │
│  data           │  contextualized_ │  results          │  Frontend       │
│                 │  data            │  (RUL, health)    │  (alerts, RUL)   │
└─────────────────┴─────────────────┴───────────────────┴─────────────────┘
```

| Processing Block | Component | Output |
|------------------|-----------|--------|
| Data acquisition | `watcher_service.py`, `pipeline/ingestion/` | Raw vibration, telemetry |
| Signal conditioning | `refinery.py`, `pipeline/cleansing/` | RMS, kurtosis, BPFO/BPFI, degradation_score |
| Context enrichment | `pipeline/contextualization/` | Digital twin mapping |
| Prognostic analysis | `inference_service.py`, `pipeline/inference/` | RUL, health_score, imminent_failure |
| Persistence | `stream_consumer.py`, `pipeline/persistence/` | TimescaleDB (cwru_features) |

---

## Communication (ISO 13374-3)

| Mechanism | Purpose |
|-----------|---------|
| Redis Streams | Inter-stage data flow (`raw_sensor_data`, `clean_features`, `inference_results`) |
| REST API | Machine status, alarms, work orders, recommendations |
| WebSocket | Real-time telemetry stream (`/ws/stream`) |

---

## Presentation (ISO 13374-4)

| Element | Component |
|---------|-----------|
| Health information | `RULCard.jsx`, `ReliabilityMetricsCard.jsx` |
| Diagnostic data | `BearingFaultPanel.jsx`, `DegradationTrendChart.jsx` |
| Advisories | `MaintenanceRecommendationCard.jsx`, `ActiveAlarmFeed.jsx` |
| Work orders | `WorkOrderPanel.jsx` |

---

## References

- [ISO 13372:2012](https://www.iso.org/standard/55924.html) — Vocabulary
- [ISO 13374-1](https://www.iso.org/standard/21832.html) — General guidelines
- [ISO 13374-2](https://www.iso.org/standard/36645.html) — Data processing
- [ISO 13374-3](https://www.iso.org/standard/37611.html) — Communication
- [ISO 13374-4](https://www.iso.org/standard/54933.html) — Presentation
- [docs/ARCHITECTURE.md](./ARCHITECTURE.md) — 6-stage pipeline
