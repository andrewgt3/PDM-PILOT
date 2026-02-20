# ISO 13374 Self-Audit — PDM-PILOT

> **Purpose:** Assess alignment with ISO 13374 (Condition monitoring and diagnostics — Data processing, communication and presentation).  
> **References:** [ISO 13374-1](https://www.iso.org/standard/21832.html), [13374-2](https://www.iso.org/standard/36645.html), [13374-3](https://www.iso.org/standard/37611.html), [13374-4](https://www.iso.org/standard/54933.html)

---

## Executive Summary

| Part | Standard focus | Alignment | Gap summary |
|------|----------------|-----------|-------------|
| **13374-1** | General guidelines, software specs | **~75%** | No formal conformance statement; pipeline design is aligned |
| **13374-2** | Data processing, reference models | **~60%** | Processing blocks present; no published reference information model |
| **13374-3** | Communication, interoperability | **~55%** | Redis/REST/WS work; no standard CM&D protocol or published schema |
| **13374-4** | Presentation (diagnostic, prognostic, advisories) | **~80%** | Health, RUL, recommendations in UI; no report format, no advisory taxonomy |

**Overall:** Roughly **65–70%** aligned. Strongest in pipeline structure and presentation; main gaps are formal information model, standardized communication schema, and written reports.

---

## Part 1: General Guidelines (ISO 13374-1)

| Requirement / practice | Status | Evidence / gap |
|------------------------|--------|-----------------|
| Software specs for data processing, communication, presentation | Met | 6-stage pipeline, Redis streams, REST/WS, frontend |
| Clear separation of processing stages | Met | `pipeline/` (ingestion, cleansing, contextualization, persistence, inference); compartment boundaries in [ARCHITECTURE.md](./ARCHITECTURE.md) |
| Documented architecture | Met | [ARCHITECTURE.md](./ARCHITECTURE.md), [ISO_13372_MAPPING.md](./ISO_13372_MAPPING.md) |
| Conformance statement or checklist | Gap | No explicit “ISO 13374 conformance” section or checklist in docs |

**Verdict:** ~75%. Architecture and docs support the standard; add a short conformance statement and optional checklist to close the gap.

---

## Part 2: Data Processing (ISO 13374-2)

| Requirement / practice | Status | Evidence / gap |
|------------------------|--------|-----------------|
| Reference **processing model** (acquisition → conditioning → analysis) | Partially met | Acquisition (Stage 1), conditioning (Stage 2 + refinery), analysis (Stage 5 inference); no single doc that maps to the standard’s block names |
| Reference **information model** | Gap | No published schema defining standard data elements per processing block (e.g. raw, conditioned, health, RUL) |
| Data acquisition block | Met | `watcher_service.py`, `pipeline/ingestion/` → raw sensor/vibration data |
| Signal conditioning block | Met | `refinery.py`, `pipeline/cleansing/` → RMS, kurtosis, BPFO/BPFI, degradation_score |
| Detection / diagnosis block | Partially met | Bearing fault features and degradation; no explicit “detection” vs “diagnosis” block split |
| Prognosis block | Met | `inference_service.py`, `pipeline/inference/` → RUL, health_score, imminent_failure |
| Persistence of CM&D data | Met | `stream_consumer.py`, `pipeline/persistence/` → TimescaleDB |
| Interoperability via standard data model | Gap | Payloads are JSON but not declared against a standard (e.g. MIMOSA, or an internal reference schema) |

**Verdict:** ~60%. Processing flow matches the intent of 13374-2; the main gap is a formal reference information model and explicit mapping of our blocks to the standard’s terminology.

---

## Part 3: Communication (ISO 13374-3)

| Requirement / practice | Status | Evidence / gap |
|------------------------|--------|-----------------|
| Exchange of CM&D information between systems | Met | Redis streams between stages; REST and WebSocket to frontend |
| Defined interfaces between processing blocks | Partially met | Stream names and payload shapes are implicit in code; not documented as a contract |
| Standard or open protocol for CM&D | Gap | Redis/REST/WS are implementation choices; no use of a standard CM&D protocol (e.g. MIMOSA, OPC UA for CM&D) |
| Schema for communicated data | Gap | No single published schema (e.g. JSON Schema or OpenAPI) for `raw_sensor_data`, `clean_features`, `inference_results` |

**Verdict:** ~55%. Communication works and supports interoperability internally; alignment would improve with documented stream/API schemas and, if required, adoption of a standard protocol.

---

## Part 4: Presentation (ISO 13374-4)

| Requirement / practice | Status | Evidence / gap |
|------------------------|--------|-----------------|
| Present **diagnostic** data | Met | `BearingFaultPanel.jsx`, `DegradationTrendChart.jsx`, `FeatureImportancePanel.jsx` |
| Present **prognostic** data | Met | `RULCard.jsx`, `ShiftAwareRUL.jsx`, `ReliabilityMetricsCard.jsx` — RUL, health |
| Present **health** information | Met | Health score, degradation score, status in UI |
| Present **advisories** | Partially met | Alarms and recommendations in UI; no explicit “advisory” vs “recommendation” taxonomy per standard |
| Present **recommendations** | Met | `MaintenanceRecommendationCard.jsx`, API `/api/recommendations` |
| Computer displays | Met | React dashboards, Plant Overview, Machine Detail |
| Written report format | Gap | No defined written report (e.g. PDF/printed) for CM&D results |

**Verdict:** ~80%. Display of diagnostic, prognostic, health, and recommendations is strong; gaps are a clear advisory/recommendation taxonomy and a written report format.

---

## Gap Priority (Remediation)

| Priority | Gap | Effort | Action |
|----------|-----|--------|--------|
| **P1** | Reference information model (13374-2) | Medium | Define and publish a schema (e.g. JSON Schema or doc) for each stream and API: raw, conditioned, health, RUL, advisories. |
| **P2** | Communication schema (13374-3) | Low–Medium | Document Redis stream payloads and key API response shapes; optionally add OpenAPI/JSON Schema. |
| **P3** | Conformance statement (13374-1) | Low | Add a “Conformance to ISO 13374” subsection in README or ARCHITECTURE listing parts and level of alignment. |
| **P4** | Presentation: advisory vs recommendation (13374-4) | Low | In docs and UI, define “advisory” (e.g. alert) vs “recommendation” (e.g. maintenance action) and use consistently. |
| **P5** | Written report (13374-4) | Medium | If required, add export (e.g. PDF) of health/RUL/advisories for a machine or fleet. |

---

## What Is Already Aligned

- **Processing blocks:** Acquisition, conditioning, (implicit) detection/diagnosis, prognosis, and persistence are implemented and separated.
- **Vocabulary (ISO 13372):** RUL, health indicator, baseline, condition monitoring, diagnostics, prognostics are used in code and UI.
- **Presentation:** Diagnostic and prognostic data, health, and recommendations are shown on screen in a structured way.
- **Documentation:** Architecture and ISO mapping docs exist and reference the standards.

---

## Conclusion

We are **about 65–70%** aligned with ISO 13374: the pipeline and presentation are in good shape; the largest gaps are a **formal reference information model**, **documented communication schemas**, and optional **written report** and **advisory taxonomy**. Addressing P1–P3 would raise alignment to roughly **80%+** without changing the core architecture.
