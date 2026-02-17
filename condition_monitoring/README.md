# Condition Monitoring (ISO 13372 / ISO 13374)

This directory represents the **ISO 13372** (vocabulary) and **ISO 13374** (data processing, communication, presentation) alignment of PDM-PILOT.

## ISO 13374 Part → Implementation Mapping

| Part | Implementation Path |
|------|---------------------|
| Part 1 (General) | `../docs/ARCHITECTURE.md`, 6-stage design |
| Part 2 (Data processing) | `../pipeline/` (stages 1–5), `../refinery.py`, `../inference_service.py` |
| Part 3 (Communication) | Redis streams, `../api_server.py`, WebSocket |
| Part 4 (Presentation) | `../frontend/` (RUL cards, alarms, work orders) |

## Key Terms (ISO 13372)

- **Condition monitoring** — Vibration analysis, RUL prediction
- **Prognostics** — XGBoost RUL model
- **Health indicator** — `degradation_score`, `health_score`

See **[../docs/ISO_13372_MAPPING.md](../docs/ISO_13372_MAPPING.md)** for full mapping.
