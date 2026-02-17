# Asset Management (ISO 55001)

This directory represents the **ISO 55001 Asset Management** alignment of PDM-PILOT.

## Clause → Implementation Mapping

| Clause | Implementation Path |
|--------|---------------------|
| 4. Context | `../config.py`, `../schemas/`, `../station_config.json` |
| 5. Leadership | `../enterprise_api.py` (auth, RBAC) |
| 6. Planning | `../services/`, `../analytics_engine.py` |
| 7. Support | `../database.py`, `../core/`, `../dependencies.py` |
| 8. Operation | `../pipeline/`, `../watcher_service.py`, `../refinery.py`, `../inference_service.py` |
| 9. Performance | `../analytics_engine.py`, `../scripts/validate_rul_on_nasa.py` |
| 10. Improvement | `../anomaly_discovery/`, work orders, alarms |

See **[../docs/ISO_55001_MAPPING.md](../docs/ISO_55001_MAPPING.md)** for full mapping.
