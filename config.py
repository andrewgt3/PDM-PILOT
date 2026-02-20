"""Gaia Predictive — Configuration Management.

Strictly-typed configuration system using pydantic-settings.
All settings are loaded from environment variables with validation.

TISAX/SOC2 Compliance:
    - Secrets use SecretStr to prevent accidental logging
    - Missing critical secrets cause immediate startup failure
    - All configuration is immutable after initialization
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root for resolving relative model paths (works regardless of CWD)
_PROJECT_ROOT = Path(__file__).resolve().parent


class DatabaseSettings(BaseSettings):
    """TimescaleDB connection configuration.
    
    Attributes:
        host: Database server hostname.
        port: Database server port.
        name: Database name.
        user: Database username.
        password: Database password (SecretStr for security).
        pool_min_size: Minimum connection pool size.
        pool_max_size: Maximum connection pool size.
        ssl_mode: SSL connection mode.
    """
    
    model_config = SettingsConfigDict(
        env_prefix="DB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    host: str = Field(default="localhost", description="Database server hostname")
    port: int = Field(default=5432, ge=1, le=65535, description="Database server port")
    name: str = Field(default="gaia_predictive", description="Database name")
    user: str = Field(default="gaia", description="Database username")
    password: SecretStr = Field(..., description="Database password (required)")
    pool_size: int = Field(default=20, ge=1, le=100, description="Pool size")
    max_overflow: int = Field(default=10, ge=0, le=100, description="Max overflow connections")
    ssl_mode: Literal["disable", "require", "verify-ca", "verify-full"] = Field(
        default="require",
        description="SSL connection mode",
    )
    
    @field_validator("password", mode="before")
    @classmethod
    def validate_password_not_empty(cls, v: str | None) -> str:
        """Ensure password is provided and not empty."""
        if v is None or v.strip() == "":
            raise ValueError(
                "DB_PASSWORD is required. "
                "Set it in .env or as an environment variable."
            )
        return v
    
    # Removed validate_pool_sizes as explicit pool_size and max_overflow don't strictly require comparison
    # other than being non-negative which is handled by Field validation.
    
    @property
    def async_dsn(self) -> str:
        """Build asyncpg connection string (without exposing password in logs)."""
        return (
            f"postgresql+asyncpg://{self.user}:"
            f"{self.password.get_secret_value()}@"
            f"{self.host}:{self.port}/{self.name}"
            f"?ssl={self.ssl_mode}"
        )
    
    @property
    def dsn_safe(self) -> str:
        """Build connection string safe for logging (password masked)."""
        return (
            f"postgresql+asyncpg://{self.user}:***@"
            f"{self.host}:{self.port}/{self.name}"
        )


class RedisSettings(BaseSettings):
    """Redis cache and pub/sub configuration.
    
    Attributes:
        host: Redis server hostname.
        port: Redis server port.
        password: Redis password (SecretStr, optional for dev).
        db: Redis database number.
        ssl: Enable SSL/TLS connection.
        socket_timeout: Socket timeout in seconds.
        max_connections: Maximum connection pool size.
    """
    
    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    host: str = Field(default="localhost", description="Redis server hostname")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis server port")
    password: SecretStr | None = Field(default=None, description="Redis password")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    ssl: bool = Field(default=False, description="Enable SSL/TLS")
    socket_timeout: float = Field(default=5.0, gt=0, description="Socket timeout (seconds)")
    max_connections: int = Field(default=50, ge=1, le=500, description="Max pool connections")
    redis_channel: str = Field(
        default="sensor_stream",
        description="Pub/sub channel for sensor stream",
        validation_alias="REDIS_CHANNEL",
    )
    publish_rate_hz: float = Field(
        default=10.0,
        gt=0,
        le=100,
        description="Publisher rate in Hz",
        validation_alias="PUBLISH_RATE_HZ",
    )
    stream_maxlen: int = Field(default=10000, ge=1, description="Max length for Redis streams")
    
    @property
    def url(self) -> str:
        """Build Redis URL for connection."""
        protocol = "rediss" if self.ssl else "redis"
        auth = ""
        if self.password:
            auth = f":{self.password.get_secret_value()}@"
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"
    
    @property
    def url_safe(self) -> str:
        """Build Redis URL safe for logging (password masked)."""
        protocol = "rediss" if self.ssl else "redis"
        auth = ":***@" if self.password else ""
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"


class SecuritySettings(BaseSettings):
    """Authentication and security configuration.
    
    Attributes:
        jwt_secret: Secret key for JWT signing (required).
        jwt_algorithm: JWT signing algorithm.
        jwt_expiry_minutes: Access token expiry time.
        jwt_refresh_expiry_days: Refresh token expiry time.
        api_key_header: Header name for API key authentication.
        cors_origins: Allowed CORS origins.
        rate_limit_per_minute: API rate limit per IP.
    """
    
    model_config = SettingsConfigDict(
        env_prefix="SECURITY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    jwt_secret: SecretStr = Field(..., description="JWT signing secret (required)")
    jwt_algorithm: Literal["HS256", "HS384", "HS512", "RS256"] = Field(
        default="HS256",
        description="JWT signing algorithm",
    )
    jwt_expiry_minutes: int = Field(
        default=30,
        ge=1,
        le=1440,
        description="Access token expiry (minutes)",
    )
    jwt_refresh_expiry_days: int = Field(
        default=7,
        ge=1,
        le=90,
        description="Refresh token expiry (days)",
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="API key header name",
    )
    cors_origins: list[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins",
    )
    rate_limit_per_minute: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Rate limit per IP per minute",
    )
    
    @field_validator("jwt_secret", mode="before")
    @classmethod
    def validate_jwt_secret(cls, v: str | None) -> str:
        """Ensure JWT secret is provided and has sufficient entropy."""
        if v is None or v.strip() == "":
            raise ValueError(
                "SECURITY_JWT_SECRET is required. "
                "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
            )
        if len(v) < 32:
            raise ValueError(
                "SECURITY_JWT_SECRET must be at least 32 characters for security. "
                "Generate a secure one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
            )
        return v
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


class LogSettings(BaseSettings):
    """Logging configuration for observability.
    
    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format: Log output format (json for production, text for development).
        include_timestamp: Include ISO timestamp in logs.
        include_correlation_id: Include correlation ID for request tracing.
        output: Log output destination.
        file_path: Log file path (when output includes 'file').
        max_file_size_mb: Maximum log file size before rotation.
        backup_count: Number of rotated log files to keep.
    """
    
    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    format: Literal["json", "text"] = Field(
        default="json",
        description="Log format (json for production)",
    )
    include_timestamp: bool = Field(default=True, description="Include ISO timestamp")
    include_correlation_id: bool = Field(default=True, description="Include correlation ID")
    output: Literal["stdout", "file", "both"] = Field(
        default="stdout",
        description="Log output destination",
    )
    file_path: str = Field(
        default="/var/log/gaia/app.log",
        description="Log file path",
    )
    max_file_size_mb: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Max log file size (MB)",
    )
    backup_count: int = Field(
        default=5,
        ge=0,
        le=50,
        description="Rotated files to keep",
    )


class ModelSettings(BaseSettings):
    """ML model and analytics engine configuration.
    
    Attributes:
        rolling_window_seconds: Rolling window for feature engineering.
        lookback_days: Days of historical data to analyze.
        contamination_rate: Anomaly detection contamination rate.
        event_lookback_minutes: Minutes to look back for event correlation.
        rul_warning_threshold: Hours before RUL warning.
        rul_critical_threshold: Hours before RUL critical alert.
    """
    
    model_config = SettingsConfigDict(
        env_prefix="MODEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    rolling_window_seconds: int = Field(default=60, ge=1, le=3600, description="Rolling window (seconds)")
    lookback_days: int = Field(default=7, ge=1, le=90, description="Historical lookback (days)")
    contamination_rate: float = Field(default=0.01, ge=0.001, le=0.5, description="Anomaly contamination rate")
    event_lookback_minutes: int = Field(default=30, ge=1, le=120, description="Event correlation lookback (minutes)")
    rul_warning_threshold: int = Field(default=72, ge=1, description="RUL warning threshold (hours)")
    rul_critical_threshold: int = Field(default=24, ge=1, description="RUL critical threshold (hours)")
    output_file: str = Field(default="insight_report.json", description="Analytics output file")
    model_dir: str = Field(default="models", description="Model directory")

    # Resolved relative to project root (works in Docker / any CWD)
    gaia_model_path: str = Field(
        default="models/gaia_model.pkl",
        description="Path to Gaia classifier pipeline (GAIA_MODEL_PATH)",
        validation_alias="GAIA_MODEL_PATH",
    )
    rul_model_path: str = Field(
        default="models/rul_model.pkl",
        description="Path to RUL regressor (RUL_MODEL_PATH)",
        validation_alias="RUL_MODEL_PATH",
    )
    scaler_path: str = Field(
        default="models/scaler.pkl",
        description="Path to feature scaler (SCALER_PATH)",
        validation_alias="SCALER_PATH",
    )

    def _resolved_path(self, path_str: str) -> Path:
        """Resolve path relative to project root; return as-is if already absolute."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        return (_PROJECT_ROOT / path_str).resolve()

    @property
    def resolved_gaia_model_path(self) -> Path:
        return self._resolved_path(self.gaia_model_path)

    @property
    def resolved_rul_model_path(self) -> Path:
        return self._resolved_path(self.rul_model_path)

    @property
    def resolved_scaler_path(self) -> Path:
        return self._resolved_path(self.scaler_path)


class EdgeSettings(BaseSettings):
    """OPC UA edge client configuration.
    
    Attributes:
        opc_server_url: OPC UA server URL.
        api_base_url: Gaia API base URL.
        api_username: API username for authentication.
        api_password: API password for authentication.
        buffer_max_size: Maximum offline buffer size.
        buffer_batch_size: Batch size for buffer flush.
    """
    
    model_config = SettingsConfigDict(
        env_prefix="EDGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    opc_server_url: str = Field(
        default="opc.tcp://localhost:4840/freeopcua/server/",
        description="OPC UA server URL"
    )
    api_base_url: str = Field(default="http://localhost:8000", description="Gaia API base URL")
    api_username: str | None = Field(default=None, description="API username")
    api_password: SecretStr | None = Field(default=None, description="API password")
    buffer_max_size: int = Field(default=10000, ge=100, le=100000, description="Offline buffer max size")
    buffer_batch_size: int = Field(default=50, ge=1, le=1000, description="Buffer flush batch size")
    namespace_uri: str = Field(default="http://gaia.predictive.maintenance", description="OPC UA namespace")
    # Hardened offline buffer and replay (SQLite + replay loop)
    offline_buffer_path: str = Field(
        default="data/offline_buffer.db",
        description="Path to SQLite buffer DB",
        validation_alias="OFFLINE_BUFFER_PATH",
    )
    offline_buffer_enabled: bool = Field(default=True, description="Enable offline buffer (OFFLINE_BUFFER_ENABLED)")
    offline_replay_interval_seconds: float = Field(default=30, ge=1, le=300, description="Replay loop interval (OFFLINE_REPLAY_INTERVAL_SECONDS)")
    offline_buffer_purge_days: int = Field(default=7, ge=1, le=90, description="Purge replayed rows older than N days (OFFLINE_BUFFER_PURGE_DAYS)")


class ABBRobotSettings(BaseSettings):
    """ABB Robot specific configuration.

    Attributes:
        ip_address: Robot IP address.
        port: OPC UA port (usually 4840).
        node_ids: Mapping of logical names to NodeIDs.
    """

    model_config = SettingsConfigDict(
        env_prefix="ABB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ip_address: str = Field(default="127.0.0.1", description="ABB Robot IP Address")
    port: int = Field(default=4840, description="OPC UA Port")

    # NodeID configuration (can be overridden by env vars like ABB_NODE_SPEED="ns=4;s=Speed")
    node_speed: str = Field(default="ns=4;s=Speed", description="NodeID for Speed")
    node_torque: str = Field(default="ns=4;s=Torque", description="NodeID for Torque")
    node_joints: str = Field(default="ns=4;s=Joints", description="NodeID for Joint Angles")

    @property
    def url(self) -> str:
        """Build OPC UA URL."""
        return f"opc.tcp://{self.ip_address}:{self.port}"


class SiemensS7Settings(BaseSettings):
    """Siemens S7 PLC configuration (Profinet/S7, python-snap7).

    Attributes:
        plc_ip: PLC IP address.
        rack: Rack number (default 0).
        slot: Slot number (default 1).
        db_number: Data block number to read from.
    """

    model_config = SettingsConfigDict(
        env_prefix="SIEMENS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    plc_ip: str = Field(default="127.0.0.1", description="Siemens PLC IP (SIEMENS_PLC_IP)")
    plc_rack: int = Field(default=0, ge=0, le=7, description="PLC rack (SIEMENS_PLC_RACK)")
    plc_slot: int = Field(default=1, ge=0, le=31, description="PLC slot (SIEMENS_PLC_SLOT)")
    db_number: int = Field(default=1, ge=1, le=65535, description="Data block number (SIEMENS_DB_NUMBER)")


class DriftSettings(BaseSettings):
    """Drift monitoring thresholds and lookback."""

    model_config = SettingsConfigDict(
        env_prefix="DRIFT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    drift_warning_psi: float = Field(default=0.1, ge=0.0, description="PSI above this = WARNING")
    drift_critical_psi: float = Field(default=0.25, ge=0.0, description="PSI above this = CRITICAL")
    drift_check_lookback_days: int = Field(default=7, ge=1, le=90, description="Days of data for drift check")


class FederatedLearningSettings(BaseSettings):
    """Federated learning (Flower) configuration. When enabled, flows use FL client instead of direct training."""

    model_config = SettingsConfigDict(
        env_prefix="FL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    fl_enabled: bool = Field(default=False, description="Use federated learning client instead of centralized training")
    fl_server_address: str = Field(default="localhost:8080", description="Flower server host:port")
    fl_local_epochs: int = Field(default=5, ge=1, le=50, description="Local training epochs per round")
    fl_dp_enabled: bool = Field(default=False, description="Enable differential privacy (clip + noise on updates)")
    fl_dp_noise_multiplier: float = Field(default=0.01, ge=0.0, description="DP Gaussian noise scale (when fl_dp_enabled)")
    fl_dp_clipping_norm: float = Field(default=1.0, gt=0, description="DP L2 clipping norm for updates (when fl_dp_enabled)")


class MQTTSettings(BaseSettings):
    """MQTT broker configuration (Eclipse Mosquitto).

    Attributes:
        broker_host: Broker hostname (MQTT_BROKER_HOST).
        broker_port: MQTT port, default 1883 (MQTT_BROKER_PORT).
        ws_port: WebSocket port, default 9001 (MQTT_WS_PORT).
        client_id: Client identifier (MQTT_CLIENT_ID).
        username: Optional username (empty for anonymous).
        password: Optional password (SecretStr).
        publish_enabled: Enable MQTT publishing.
        qos: Default QoS (0, 1, or 2).
        tls_enabled: Use TLS for connection.
        ca_cert_path: Path to CA cert for TLS (MQTT_CA_CERT_PATH).
        client_cert_path: Path to client cert for mutual TLS (MQTT_CLIENT_CERT_PATH).
        client_key_path: Path to client key for mutual TLS (MQTT_CLIENT_KEY_PATH).
    """

    model_config = SettingsConfigDict(
        env_prefix="MQTT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    broker_host: str = Field(default="localhost", description="MQTT broker hostname (MQTT_BROKER_HOST)")
    broker_port: int = Field(default=1883, ge=1, le=65535, description="MQTT port (MQTT_BROKER_PORT)")
    ws_port: int = Field(default=9001, ge=1, le=65535, description="WebSocket port (MQTT_WS_PORT)")
    client_id: str = Field(default="pdm-pilot", description="MQTT client ID (MQTT_CLIENT_ID)")
    username: str | None = Field(default=None, description="MQTT username (empty for anonymous)")
    password: SecretStr | None = Field(default=None, description="MQTT password")
    publish_enabled: bool = Field(default=True, description="Enable MQTT publishing (MQTT_PUBLISH_ENABLED)")
    qos: int = Field(default=1, ge=0, le=2, description="Default QoS 0/1/2 (MQTT_QOS)")
    tls_enabled: bool = Field(default=False, description="Use TLS for MQTT connection (MQTT_TLS_ENABLED)")
    ca_cert_path: str | None = Field(default=None, description="Path to CA cert for TLS (MQTT_CA_CERT_PATH)")
    client_cert_path: str | None = Field(default=None, description="Path to client cert for mutual TLS (MQTT_CLIENT_CERT_PATH)")
    client_key_path: str | None = Field(default=None, description="Path to client key for mutual TLS (MQTT_CLIENT_KEY_PATH)")


class Settings(BaseSettings):
    """Root application settings aggregating all configuration sections.
    
    This is the main entry point for configuration. Use get_settings()
    to obtain a cached singleton instance.
    
    Attributes:
        app_name: Application name for identification.
        app_version: Application version string.
        environment: Deployment environment.
        debug: Enable debug mode (never in production!).
        database: Database configuration.
        redis: Redis configuration.
        security: Security configuration.
        log: Logging configuration.
    
    Example:
        >>> settings = get_settings()
        >>> print(settings.database.dsn_safe)
        postgresql+asyncpg://gaia:***@localhost:5432/gaia_predictive
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Application metadata
    app_name: str = Field(default="Gaia Predictive", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment",
    )
    debug: bool = Field(default=False, description="Debug mode (disable in production!)")
    
    # Optional notification webhook (alerts + drift CRITICAL)
    notification_webhook_url: str | None = Field(
        default=None,
        description="Webhook URL for critical alerts and drift notifications (NOTIFICATION_WEBHOOK_URL)",
    )

    # CMMS integration (work order sync to external system)
    CMMS_ENABLED: bool = Field(default=False, description="Enable CMMS webhook sync (CMMS_ENABLED)")
    CMMS_WEBHOOK_URL: str | None = Field(default=None, description="CMMS webhook URL (CMMS_WEBHOOK_URL)")

    # Multi-tenancy (row-level)
    DEFAULT_TENANT_ID: str = Field(default="default", description="Default tenant for backfill and new rows (DEFAULT_TENANT_ID)")
    TENANT_ID: str = Field(default="default", description="Tenant for this deployment; set per customer (TENANT_ID)")

    # Nested configuration sections
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    log: LogSettings = Field(default_factory=LogSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    drift: DriftSettings = Field(default_factory=DriftSettings)
    edge: EdgeSettings = Field(default_factory=EdgeSettings)
    abb: ABBRobotSettings = Field(default_factory=ABBRobotSettings)
    siemens_s7: SiemensS7Settings = Field(default_factory=SiemensS7Settings)
    mqtt: MQTTSettings = Field(default_factory=MQTTSettings)
    federated: FederatedLearningSettings = Field(default_factory=FederatedLearningSettings)

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """Enforce strict security settings in production."""
        if self.environment == "production":
            if self.debug:
                raise ValueError("Debug mode must be disabled in production")
            if self.log.level == "DEBUG":
                raise ValueError("DEBUG log level is not allowed in production")
        return self


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings singleton.
    
    Returns:
        Settings: Validated application configuration.
    
    Raises:
        ValidationError: If required settings are missing or invalid.
            This will cause immediate application startup failure.
    
    Example:
        >>> from config import get_settings
        >>> settings = get_settings()
        >>> print(settings.environment)
        'development'
    """
    return Settings()


def validate_startup() -> None:
    """Validate all required configuration at application startup.
    
    Call this in your FastAPI lifespan or startup event to ensure
    all configuration is valid before accepting requests.
    
    Raises:
        SystemExit: If configuration validation fails.
    """
    try:
        settings = get_settings()
        print(f"✓ Configuration loaded for environment: {settings.environment}")
        print(f"✓ Database: {settings.database.dsn_safe}")
        print(f"✓ Redis: {settings.redis.url_safe}")
        print(f"✓ Log level: {settings.log.level}")
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        raise SystemExit(1) from e
