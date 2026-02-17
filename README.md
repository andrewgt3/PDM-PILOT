# GAIA Predictive Maintenance Platform (PDM-PILOT)

> **Enterprise-grade Industrial IoT platform for real-time predictive maintenance, anomaly detection, and AI-powered maintenance recommendations.**
>
> **Standards:** ISO 55001 (Asset Management) | ISO 13372/13374 (Condition Monitoring & Diagnostics)

---

## 📋 Project Overview

**GAIA** is a full-stack predictive maintenance system designed for manufacturing environments. It ingests real-time sensor telemetry (vibration, temperature, torque, rotational speed), runs ML inference to predict equipment failures, and provides actionable maintenance recommendations through a modern React dashboard.

### Core Capabilities
- **Real-time Telemetry Streaming** via Redis Pub/Sub + WebSocket
- **ML Failure Prediction** using XGBoost/Scikit-learn models
- **Bearing Fault Detection** (BPFO/BPFI frequency analysis)
- **Remaining Useful Life (RUL)** estimation
- **AI Maintenance Recommendations** (OpenAI/Azure OpenAI integration)
- **Enterprise Features**: Alarms, Work Orders, MTBF/MTTR, Shift Schedules
- **Anomaly Discovery Engine** (Isolation Forest, Temporal Autoencoder)

---

## 🏗️ Architecture

### 6-Stage Pipeline (ISO Golden Standard)

The platform follows a **compartmentalized 6-stage pipeline**. Each stage is isolated; no cross-stage bleed.

| Stage | Name | Responsibility |
|-------|------|-----------------|
| 1 | **INGEST** | OPC-UA / File Watcher → `raw_sensor_data` |
| 2 | **CLEANSE** | Outlier removal, feature extraction → `clean_features` |
| 3 | **CONTEXT** | Digital Twin mapping → `contextualized_data` |
| 4 | **PERSIST** | TimescaleDB batch writer |
| 5 | **INFER** | XGBoost RUL prediction → `inference_results` |
| 6 | **ORCH** | API Gateway + Frontend |

See **[docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)** for full stage mapping and compartment boundaries.

**Standards alignment:**
- [docs/ISO_55001_MAPPING.md](./docs/ISO_55001_MAPPING.md) — Asset management (ISO 55001)
- [docs/ISO_13372_MAPPING.md](./docs/ISO_13372_MAPPING.md) — Condition monitoring (ISO 13372/13374)

---

### System Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                          FRONTEND (React + Vite)                       │
│   FleetTreemap │ MachineDetail │ WorkOrderPanel │ AnomalyDiscovery   │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       API LAYER (FastAPI)                             │
│  api_server.py    │   enterprise_api.py   │   anomaly_discovery/api  │
│  ───────────────────────────────────────────────────────────────────  │
│  /health          │   /api/enterprise/*   │   /api/discovery/*       │
│  /api/machines    │   /api/enterprise/token (Auth)                   │
│  /api/features    │   /api/enterprise/alarms                         │
│  /api/recommendations │ /api/enterprise/work-orders                  │
│  /ws/stream       │   /api/enterprise/reliability                    │
└──────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐         ┌─────────────────┐         ┌─────────────────┐
│    Redis      │         │  PostgreSQL/    │         │  ML Models      │
│  (Pub/Sub)    │         │  TimescaleDB    │         │  (gaia_model.pkl)│
│ sensor_stream │         │  cwru_features  │         │  (rul_model.json)│
└───────────────┘         │  pdm_alarms     │         └─────────────────┘
        ▲                 │  work_orders    │
        │                 │  failure_events │
        │                 └─────────────────┘
        │
┌───────────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                               │
│  stream_publisher.py │ stream_consumer.py │ mock_fleet_streamer.py   │
│  opc_client_adapter.py (OPC-UA) │ high_fidelity_simulator.py         │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure

```
PDM-PILOT/
├── api_server.py           # Main FastAPI application (REST + WebSocket)
├── enterprise_api.py       # Enterprise endpoints (alarms, work orders, auth)
├── config.py               # Centralized settings (Pydantic Settings)
├── database.py             # Async SQLAlchemy engine & session management
├── dependencies.py         # FastAPI dependencies (auth, DB sessions)
├── logger.py               # Structured logging (structlog + JSON)
│
├── core/                   # Core infrastructure
│   ├── exceptions.py       # Custom exception hierarchy
│   ├── logger.py           # Request ID middleware & logging
│   └── monitoring.py       # Prometheus metrics (stub)
│
├── schemas/                # Pydantic models (request/response validation)
│   ├── __init__.py         # Validated schemas (AlarmCreate, WorkOrder, etc.)
│   ├── security.py         # Token, User, UserRole schemas
│   ├── response.py         # APIResponse wrapper, ORJSONResponse
│   ├── enterprise.py       # WorkOrder schema
│   ├── machine.py          # Machine status schemas
│   └── ml.py               # ML prediction schemas
│
├── services/               # Business logic layer (async)
│   ├── base.py             # BaseService with DB session
│   ├── auth_service.py     # JWT creation, password hashing (bcrypt)
│   ├── machine_service.py  # Machine status & RUL calculations
│   ├── alarm_service.py    # Alarm CRUD & threshold checks
│   └── telemetry_service.py # Telemetry ingestion & batch processing
│
├── middleware/             # FastAPI middleware
│   ├── security_headers.py # CSP, HSTS, X-Frame-Options (Helmet.js equiv)
│   └── audit_logger.py     # Audit logging for sensitive operations
│
├── anomaly_discovery/      # Anomaly detection subsystem
│   ├── api.py              # Discovery API routes
│   ├── discovery_engine.py # Main orchestrator
│   ├── detectors/          # IsolationForest, TemporalAutoencoder
│   └── analyzers/          # Correlation analysis
│
├── frontend/               # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx         # Main app with routing
│   │   ├── components/     # UI components
│   │   │   ├── FleetTreemap.jsx      # Fleet health overview
│   │   │   ├── MachineDetail.jsx     # Individual machine view
│   │   │   ├── WorkOrderPanel.jsx    # Maintenance work orders
│   │   │   ├── ActiveAlarmFeed.jsx   # Real-time alarms
│   │   │   ├── RULCard.jsx           # Remaining Useful Life
│   │   │   └── ...
│   │   ├── hooks/
│   │   │   └── useWebSocket.js       # WebSocket hook for /ws/stream
│   │   └── theme.js        # MUI theme configuration
│   └── vite.config.js
│
├── asset_management/      # ISO 55001 alignment
│   └── README.md          # Clause → implementation mapping
├── condition_monitoring/  # ISO 13372/13374 alignment
│   └── README.md          # Part → implementation mapping
├── pipeline/              # Stages 1–5 (compartmentalized)
│   ├── ingestion/         # Stage 1
│   ├── cleansing/         # Stage 2
│   ├── contextualization/ # Stage 3
│   ├── persistence/       # Stage 4
│   └── inference/         # Stage 5
├── docs/
│   ├── ARCHITECTURE.md    # 6-stage pipeline
│   ├── ISO_55001_MAPPING.md
│   └── ISO_13372_MAPPING.md
├── scripts/               # Utility scripts
│   ├── local_backup.py     # Docker pg_dump backup with rotation
│   ├── secure_backup.py    # Encrypted S3 backup
│   ├── migrate_enterprise_features.py  # DB migrations
│   └── check_audit_logs.py # Verify audit logging
│
├── dashboards/             # Legacy/deprecated dashboards
│   └── _DEPRECATED_dashboard.py
│
├── demos/                  # Demo & simulation scripts
│   ├── agent_demo.py       # AI agent demonstration
│   └── start_demo.py       # Demo launcher
│
├── tests/                  # Test suite
│   ├── test_schema_validation.py
│   └── test_bad_data.py
│
├── docker-compose.yml      # Full stack: API, TimescaleDB, Redis
├── Dockerfile              # API container build
├── requirements.txt        # Python dependencies
└── .env.example            # Environment template
```

---

## 🔧 Technology Stack

### Backend
| Component | Technology |
|-----------|------------|
| **Framework** | FastAPI (async) |
| **Database** | PostgreSQL + TimescaleDB (time-series) |
| **ORM** | SQLAlchemy 2.0 (async) + Alembic migrations |
| **Cache/Pub-Sub** | Redis (aioredis) |
| **Auth** | JWT (python-jose) + bcrypt (passlib) |
| **Rate Limiting** | slowapi (100/min global, 5/min on login) |
| **Logging** | structlog (JSON format) |
| **ML** | scikit-learn, XGBoost, joblib |

### Frontend
| Component | Technology |
|-----------|------------|
| **Framework** | React 18 + Vite |
| **UI Library** | Material-UI (MUI) |
| **Charts** | Recharts |
| **State** | React hooks + WebSocket |
| **HTTP** | fetch API |

### Infrastructure
| Component | Technology |
|-----------|------------|
| **Containers** | Docker + docker-compose |
| **Time-Series DB** | TimescaleDB (PostgreSQL extension) |
| **Message Queue** | Redis Pub/Sub |
| **OPC-UA** | asyncua (industrial protocol) |

---

## 🔐 Security Features

1. **Authentication**: JWT tokens with 30-min expiry
2. **Rate Limiting**: 100 req/min global, 5 req/min on `/token`
3. **Security Headers**: CSP, HSTS, X-Frame-Options, X-Content-Type-Options
4. **Input Validation**: Pydantic schemas with strict rules
5. **Audit Logging**: Sensitive operations logged with user context
6. **Global Exception Handler**: No stack traces leaked to clients
7. **RBAC**: `get_current_user` and `get_current_admin_user` dependencies

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- PostgreSQL (or use Docker)

### 1. Environment Setup

```bash
# Clone and enter directory
cd PDM-PILOT

# Copy environment template
cp .env.example .env

# Edit .env with your settings:
# - DATABASE_URL=postgresql://user:pass@localhost:5432/pdm_timeseries
# - SECRET_KEY=your-secret-key
# - ADMIN_PASSWORD=your-admin-password
```

### 2. Start Infrastructure (Docker)

```bash
# Start TimescaleDB + Redis
docker-compose up -d timescaledb redis
```

### 3. Backend Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Start API server
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### 5. Start Data Streaming (Optional)

```bash
# Terminal 1: Start mock sensor data publisher
python stream_publisher.py

# Terminal 2: Start stream consumer (processes & stores)
python stream_consumer.py
```

### 6. Full NASA Dataset (For Training)

To train on the full NASA datasets instead of the sample:

```bash
# Download C-MAPSS (turbofan) + IMS (bearings) from NASA Open Data Portal
python scripts/download_nasa_data.py

# Output: ~/Desktop/archive.zip (or --output /path/to/archive.zip)
# Then click "Initialize NASA Data" in the Pipeline Operations dashboard
```

**Datasets included:**
| Dataset | Size | Description |
|---------|------|-------------|
| [NASA C-MAPSS](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) | ~15 MB | Turbofan engine degradation (FD001–FD004), 100–260 engines each |
| [NASA IMS Bearings](https://data.nasa.gov/dataset/ims-bearings) | ~50 MB | Bearing run-to-failure from IMS/UCincinnati |

**Additional datasets** (for bearing fault diagnosis):
- [CWRU Bearing Data](https://engineering.case.edu/bearingdatacenter/download-data-file) — .mat format; convert to text for refinery

---

## 📡 API Endpoints

### Public Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Load balancer health check |
| GET | `/docs` | Swagger UI documentation |
| GET | `/api/machines` | List all machines with status |
| GET | `/api/features` | Historical feature data |
| GET | `/api/recommendations/{machine_id}` | AI maintenance recommendations |
| WS | `/ws/stream?token=...` | Real-time telemetry stream |

### Authenticated Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/enterprise/token` | Get JWT token (login) |
| GET | `/api/enterprise/alarms` | List alarms |
| POST | `/api/enterprise/alarms` | Create alarm |
| GET | `/api/enterprise/work-orders` | List work orders |
| POST | `/api/enterprise/work-orders` | Create work order |
| GET | `/api/enterprise/reliability/{machine_id}` | MTBF/MTTR metrics |
| GET | `/api/enterprise/schedule` | Shift schedule |
| POST | `/api/analytics/trigger` | Trigger background analytics |

### WebSocket Protocol
```javascript
// Connect with JWT token
const ws = new WebSocket('ws://localhost:8000/ws/stream?token=YOUR_JWT');

// Message types received:
{ "type": "telemetry", "machine_id": "WB-001", "failure_probability": 0.25, ... }
{ "type": "heartbeat", "timestamp": "2024-01-13T10:00:00Z" }
{ "type": "error", "error": "Processing error" }
```

---

## 🗄️ Database Schema (Key Tables)

```sql
-- Time-series sensor features
cwru_features (
    id, timestamp, machine_id,
    peak_freq_1..5, peak_amp_1..5,
    bpfo_amp, bpfi_amp, bsf_amp, ftf_amp,
    degradation_score, failure_prediction, failure_class
)

-- Alarm management
pdm_alarms (
    alarm_id, machine_id, severity, code, message,
    trigger_type, trigger_value, threshold_value,
    active, acknowledged, acknowledged_by, resolved_at
)

-- Work order tracking
work_orders (
    work_order_id, machine_id, title, description,
    priority, status, work_type,
    scheduled_date, estimated_duration_hours,
    assigned_to, actual_duration_hours, notes
)

-- Failure event history
failure_events (
    id, machine_id, timestamp, event_type,
    failure_probability, degradation_score
)
```

---

## 🤖 ML Models

### 1. Failure Prediction Model (`gaia_model.pkl`)
- **Type**: XGBoost Classifier
- **Features**: 28 spectral + bearing fault features
- **Output**: `failure_probability` (0-1), `failure_class`
- **Training**: `train_rul_model.py`

### 2. RUL Model (`rul_model.json`)
- **Type**: Gradient Boosting Regressor
- **Output**: Remaining Useful Life in days
- **Formula**: `RUL = 2000 * (1 - degradation)² / 24`

### 3. Anomaly Detection (`anomaly_discovery/`)
- **Isolation Forest**: Point anomaly detection
- **Temporal Autoencoder**: Sequence anomaly detection
- **Ensemble**: Combined scoring

---

## ⚙️ Configuration

All configuration is centralized in `config.py` using Pydantic Settings:

```python
# Key settings loaded from .env:
settings.database.url          # DATABASE_URL
settings.redis.host            # REDIS_HOST
settings.security.secret_key   # SECRET_KEY
settings.security.algorithm    # HS256
settings.security.token_expire # 30 minutes
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_schema_validation.py -v
```

---

## 📊 Equipment Metadata

The system tracks these machine types:

| ID | Name | Shop | Line | Type |
|----|------|------|------|------|
| WB-001 | 6-Axis Welder #1 | Body Shop | Underbody Weld Cell | Spot Welder |
| WB-002 | 6-Axis Welder #2 | Body Shop | Underbody Weld Cell | Spot Welder |
| HP-200 | Hydraulic Press 2000T | Stamping | Press Line 1 | Hydraulic Press |
| PR-101 | Paint Robot #1 | Paint Shop | Sealer Line | Paint Applicator |
| TS-001 | Torque Station #1 | Final Assembly | Chassis Line | Torque Tool |
| CV-100 | Main Conveyor Drive | Final Assembly | Chassis Line | Conveyor Motor |

---

## 📝 Development Conventions

### Code Style
- **Python**: Black + Ruff (linting)
- **JavaScript**: ESLint + Prettier
- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`)

### File Naming
- Python: `snake_case.py`
- React Components: `PascalCase.jsx`
- Schemas: Pydantic models in `schemas/`
- Services: Business logic in `services/`

### Error Handling
```python
# Use custom exceptions from core/exceptions.py
from core.exceptions import ResourceNotFound, BusinessRuleViolation

# Returns clean JSON, no stack traces
raise ResourceNotFound("machine", machine_id)
```

---

## 🔄 Data Flow

```
1. Sensors → OPC-UA/MQTT → stream_publisher.py
2. stream_publisher.py → Redis (sensor_stream channel)
3. stream_consumer.py ← Redis → PostgreSQL (cwru_features)
4. api_server.py ← PostgreSQL → ML Inference
5. WebSocket → Frontend (real-time updates)
```

---

## 📚 Related Documentation

- [SECURITY.md](./SECURITY.md) - Security policies and practices
- [CONTRIBUTING.md](./CONTRIBUTING.md) - Contribution guidelines
- [docs/AZURE_OPENAI_MIGRATION.md](./docs/AZURE_OPENAI_MIGRATION.md) - AI service migration
- [OPCUA_README.md](./OPCUA_README.md) - OPC-UA integration guide

---

## 📄 Repository

**PDM-PILOT** — Push to: `https://github.com/andrewgt3/PDM-PILOT`

```bash
git remote set-url origin https://github.com/andrewgt3/PDM-PILOT.git
git push origin main
```

---

## 📄 License

Proprietary - PlantAGI / AIJ Engineering Consulting

---

## 👥 Team

**PlantAGI** - Industrial AI Solutions  
Developed by AIJ Engineering Consulting
