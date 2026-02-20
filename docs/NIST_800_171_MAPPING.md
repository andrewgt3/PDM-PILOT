# NIST 800-171 Control Mapping

This document maps each of the 14 NIST SP 800-171 control families to the PDM Pilot implementation. Status: **Implemented**, **Partial**, or **Not Implemented**, with pointers to code or documentation.

---

## 1. AC — Access Control

| Control | Status | Implementation |
|--------|--------|----------------|
| AC-1 through AC-25 | **Implemented** | Role-based access control (RBAC): `viewer`, `operator`, `engineer`, `admin`. Permission checkers in `dependencies.py` (`PermissionChecker`, `require_admin_permission`, `require_engineer_permission`, `require_operator_permission`). All enterprise and pipeline endpoints protected by `Depends(require_*_permission)`. Role definitions in `schemas/security.py` (`UserRole`). |

**Code:** [schemas/security.py](../schemas/security.py), [dependencies.py](../dependencies.py), [enterprise_api.py](../enterprise_api.py), [api_server.py](../api_server.py)

---

## 2. AU — Audit and Accountability

| Control | Status | Implementation |
|--------|--------|----------------|
| AU-1 through AU-16 | **Implemented** | Append-only `audit_log` table (migration `007_audit_log_append_only`). Middleware logs all requests (who, what, when, IP, method, path, status). Auth events: login / login_failed in `/token`. Model promotion/rollback logged with full metadata. GET `/api/admin/audit-log` for filtered query (admin-only). File audit trail: `middleware/audit_logger.py` (security_audit.log). |

**Code:** [middleware/audit_logger.py](../middleware/audit_logger.py), [migrations/versions/007_audit_log_append_only.py](../migrations/versions/007_audit_log_append_only.py), [enterprise_api.py](../enterprise_api.py) (login, promote/rollback, GET audit-log)

---

## 3. AT — Awareness and Training

| Control | Status | Implementation |
|--------|--------|----------------|
| AT-1 through AT-4 | **Not Implemented** | No in-app security awareness or training materials. Addressed by organizational policy and training programs. |

---

## 4. CM — Configuration Management

| Control | Status | Implementation |
|--------|--------|----------------|
| CM-1 through CM-12 | **Partial** | Configuration via environment and `config.py` (Pydantic `Settings`). Docker Compose and Mosquitto config under version control. No formal baseline or change-control automation in code. |

**Code:** [config.py](../config.py), [docker-compose.yml](../docker-compose.yml), [docker/mosquitto/mosquitto.conf](../docker/mosquitto/mosquitto.conf)

---

## 5. IA — Identification and Authentication

| Control | Status | Implementation |
|--------|--------|----------------|
| IA-1 through IA-12 | **Implemented** | JWT access tokens; bcrypt password hashing; login rate limiting (e.g. 5/minute on `/token`). User identity and role in token; verification via `get_current_user` and auth service. |

**Code:** [auth_utils.py](../auth_utils.py), [services/auth_service.py](../services/auth_service.py), [enterprise_api.py](../enterprise_api.py) (login, rate limit), [dependencies.py](../dependencies.py)

---

## 6. IR — Incident Response

| Control | Status | Implementation |
|--------|--------|----------------|
| IR-1 through IR-8 | **Partial** | Alerts and notifications (e.g. critical drift, retraining); webhook for critical events. No formal incident response playbooks or IR procedures in code. |

**Code:** [services/notification_service.py](../services/notification_service.py), [services/alert_engine.py](../services/alert_engine.py), [config.py](../config.py) (webhook URL)

---

## 7. MA — Maintenance

| Control | Status | Implementation |
|--------|--------|----------------|
| MA-1 through MA-6 | **Not Implemented** | Maintenance controls are procedural/organizational; not implemented in application code. |

---

## 8. MP — Media Protection

| Control | Status | Implementation |
|--------|--------|----------------|
| MP-1 through MP-8 | **Not Implemented** | Media protection is handled at the organizational and infrastructure level; not in scope for this application. |

---

## 9. PE — Physical and Environmental Protection

| Control | Status | Implementation |
|--------|--------|----------------|
| PE-1 through PE-18 | **Not Implemented** | Physical and environmental protection is outside application scope. |

---

## 10. PS — Personnel Security

| Control | Status | Implementation |
|--------|--------|----------------|
| PS-1 through PS-8 | **Not Implemented** | Personnel security is addressed by organizational policy; not in application code. |

---

## 11. RA — Risk Assessment

| Control | Status | Implementation |
|--------|--------|----------------|
| RA-1 through RA-5 | **Partial** | Risk-related features: drift monitoring (data/concept drift), anomaly detection, model health. No formal risk assessment or risk register in code. |

**Code:** [services/drift_monitor_service.py](../services/drift_monitor_service.py), [pipeline/inference/feature_validator.py](../pipeline/inference/feature_validator.py)

---

## 12. CA — Security Assessment

| Control | Status | Implementation |
|--------|--------|----------------|
| CA-1 through CA-9 | **Not Implemented** | Security assessments (e.g. scanning, penetration testing) are organizational processes; not automated in this codebase. |

---

## 13. SC — System and Communications Protection

| Control | Status | Implementation |
|--------|--------|----------------|
| SC-1 through SC-43 | **Implemented / Partial** | MQTT TLS on port 8883 with mutual TLS (certs via `scripts/gen_mqtt_certs.sh`; Mosquitto `cafile`/`certfile`/`keyfile`/`require_certificate`). Client TLS in `mqtt_client.py` when `MQTT_TLS_ENABLED` and cert paths set. HTTPS for API is expected at the reverse-proxy/load-balancer layer (document in deployment/network docs). |

**Code:** [config.py](../config.py) (MQTTSettings), [mqtt_client.py](../mqtt_client.py), [docker/mosquitto/mosquitto.conf](../docker/mosquitto/mosquitto.conf), [scripts/gen_mqtt_certs.sh](../scripts/gen_mqtt_certs.sh), [docs/NETWORK_SEGMENTATION.md](NETWORK_SEGMENTATION.md)

---

## 14. SI — System and Information Integrity

| Control | Status | Implementation |
|--------|--------|----------------|
| SI-1 through SI-17 | **Partial** | Drift monitoring and feature validation (out-of-distribution detection). Input validation via Pydantic schemas. No formal integrity monitoring or SIEM integration in code. |

**Code:** [services/drift_monitor_service.py](../services/drift_monitor_service.py), [pipeline/inference/feature_validator.py](../pipeline/inference/feature_validator.py), [schemas/](../schemas/) (validated request bodies)

---

## Summary

| Family | Status |
|--------|--------|
| AC (Access Control) | Implemented |
| AU (Audit) | Implemented |
| AT (Awareness/Training) | Not Implemented |
| CM (Configuration) | Partial |
| IA (Identification/Auth) | Implemented |
| IR (Incident Response) | Partial |
| MA (Maintenance) | Not Implemented |
| MP (Media Protection) | Not Implemented |
| PE (Physical) | Not Implemented |
| PS (Personnel) | Not Implemented |
| RA (Risk Assessment) | Partial |
| CA (Security Assessment) | Not Implemented |
| SC (System/Comms) | Implemented / Partial |
| SI (System/Info Integrity) | Partial |
