"""Gaia Predictive — Structured Logging System.

Provides structured JSON logging for production (Splunk/Datadog) and
colored text output for local development. Implements PII filtering
and automatic request context injection.

TISAX/SOC2 Compliance:
    - No PII (emails, names, IPs) logged
    - No secrets (passwords, tokens, API keys) logged
    - All logs include correlation IDs for traceability
    - JSON format enables SIEM ingestion

Usage:
    from logger import get_logger, configure_logging
    
    # Initialize at startup
    configure_logging(environment="production")
    
    # Get a logger
    logger = get_logger(__name__)
    logger.info("Processing request", machine_id="abc-123")
"""

from __future__ import annotations

import logging
import re
import sys
from contextvars import ContextVar
from typing import Any
from collections.abc import Callable, MutableMapping

import structlog
from structlog.types import EventDict, Processor, WrappedLogger

# =============================================================================
# Context Variables for Request Tracking
# =============================================================================
# These are set by middleware and automatically included in all log entries

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)


# =============================================================================
# PII and Secret Filtering
# =============================================================================

# Patterns that indicate sensitive field names
SENSITIVE_FIELD_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"passwd", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"token", re.IGNORECASE),
    re.compile(r"api[_-]?key", re.IGNORECASE),
    re.compile(r"auth", re.IGNORECASE),
    re.compile(r"credential", re.IGNORECASE),
    re.compile(r"private[_-]?key", re.IGNORECASE),
    re.compile(r"access[_-]?key", re.IGNORECASE),
    re.compile(r"bearer", re.IGNORECASE),
    re.compile(r"jwt", re.IGNORECASE),
    re.compile(r"session[_-]?id", re.IGNORECASE),
    re.compile(r"cookie", re.IGNORECASE),
)

# Patterns that match PII in values
PII_VALUE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Email addresses
    (re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "[EMAIL_REDACTED]"),
    # Phone numbers (various formats)
    (re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"), "[PHONE_REDACTED]"),
    # SSN
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN_REDACTED]"),
    # Credit card numbers (basic pattern)
    (re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"), "[CC_REDACTED]"),
    # IPv4 addresses
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "[IP_REDACTED]"),
    # IPv6 addresses (simplified)
    (re.compile(r"([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}"), "[IP_REDACTED]"),
)

# Fields that should never be logged (exact match)
BLOCKLIST_FIELDS: frozenset[str] = frozenset({
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "private_key",
    "secret_key",
    "authorization",
    "cookie",
    "session_id",
    "credit_card",
    "ssn",
    "social_security",
})

REDACTED = "[REDACTED]"


def _is_sensitive_field(field_name: str) -> bool:
    """Check if a field name indicates sensitive data."""
    field_lower = field_name.lower()
    if field_lower in BLOCKLIST_FIELDS:
        return True
    return any(pattern.search(field_name) for pattern in SENSITIVE_FIELD_PATTERNS)


def _redact_pii_in_value(value: str) -> str:
    """Redact PII patterns found in a string value."""
    result = value
    for pattern, replacement in PII_VALUE_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def _sanitize_value(value: Any, field_name: str = "") -> Any:
    """Recursively sanitize a value, redacting sensitive data."""
    if _is_sensitive_field(field_name):
        return REDACTED
    
    if isinstance(value, str):
        return _redact_pii_in_value(value)
    
    if isinstance(value, dict):
        return {k: _sanitize_value(v, k) for k, v in value.items()}
    
    if isinstance(value, (list, tuple)):
        return type(value)(_sanitize_value(item, field_name) for item in value)
    
    return value


# =============================================================================
# Structlog Processors
# =============================================================================

def add_request_context(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Inject request context (request_id, user_id) into log entries."""
    request_id = request_id_var.get()
    if request_id:
        event_dict["request_id"] = request_id
    
    user_id = user_id_var.get()
    if user_id:
        event_dict["user_id"] = user_id
    
    correlation_id = correlation_id_var.get()
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    
    return event_dict


def sanitize_sensitive_data(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Remove PII and secrets from log entries."""
    return {k: _sanitize_value(v, k) for k, v in event_dict.items()}


def add_service_context(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add service metadata for log aggregation."""
    event_dict["service"] = "gaia-predictive"
    event_dict["version"] = "1.0.0"
    return event_dict


def drop_color_message_key(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Remove color_message key added by structlog (not needed in JSON)."""
    event_dict.pop("color_message", None)
    return event_dict


# =============================================================================
# Logging Configuration
# =============================================================================

def configure_logging(
    environment: str = "development",
    log_level: str = "INFO",
    json_format: bool | None = None,
) -> None:
    """Configure structured logging for the application.
    
    Args:
        environment: Deployment environment (development, staging, production).
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: Force JSON output. If None, auto-detect based on environment.
    
    Example:
        >>> configure_logging(environment="production", log_level="INFO")
    """
    # Auto-detect JSON format based on environment
    use_json = json_format if json_format is not None else (environment != "development")
    
    # Shared processors for all outputs
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        add_service_context,
        add_request_context,
        sanitize_sensitive_data,
    ]
    
    if use_json:
        # Production: JSON output for SIEM ingestion
        shared_processors.extend([
            drop_color_message_key,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ])
    else:
        # Development: Colored console output
        shared_processors.extend([
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.rich_traceback,
            ),
        ])
    
    # Configure structlog
    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure stdlib logging to work with structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
        force=True,
    )
    
    # Suppress noisy third-party loggers
    for noisy_logger in ("uvicorn.access", "httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a configured structlog logger.
    
    Args:
        name: Logger name (typically __name__).
    
    Returns:
        Configured structlog logger with automatic context injection.
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("User authenticated", user_id="user-123")
    """
    return structlog.stdlib.get_logger(name)


# =============================================================================
# FastAPI Middleware for Request Context
# =============================================================================

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
import uuid
import time


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware to inject request context into all log entries.
    
    Automatically extracts or generates request_id and user_id,
    making them available in all log entries for the request duration.
    
    Usage:
        app = FastAPI()
        app.add_middleware(RequestContextMiddleware)
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process request and inject logging context."""
        # Extract or generate request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request_id_var.set(request_id)
        
        # Extract correlation ID for distributed tracing
        correlation_id = request.headers.get("X-Correlation-ID") or request_id
        correlation_id_var.set(correlation_id)
        
        # Extract user ID from request state (set by auth middleware)
        user_id: str | None = None
        if hasattr(request.state, "user") and request.state.user:
            user_id = str(getattr(request.state.user, "id", None) or 
                         getattr(request.state.user, "user_id", None) or
                         request.state.user)
            user_id_var.set(user_id)
        
        # Get logger for request logging
        logger = get_logger("gaia.http")
        
        # Log request start
        start_time = time.perf_counter()
        logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
            query=str(request.query_params) if request.query_params else None,
        )
        
        try:
            response = await call_next(request)
            
            # Log request completion
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "Request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )
            
            # Add request ID to response headers for client correlation
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
            
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(
                "Request failed",
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration_ms, 2),
                error_type=type(exc).__name__,
            )
            raise
        
        finally:
            # Clear context variables
            request_id_var.set(None)
            user_id_var.set(None)
            correlation_id_var.set(None)


# =============================================================================
# Dependency Injection for FastAPI Routes
# =============================================================================

def get_request_logger(request: Request) -> structlog.stdlib.BoundLogger:
    """FastAPI dependency to get a logger with request context.
    
    Usage:
        @app.get("/machines")
        async def get_machines(logger: BoundLogger = Depends(get_request_logger)):
            logger.info("Fetching machines")
            ...
    """
    logger = get_logger("gaia.api")
    return logger.bind(
        path=request.url.path,
        method=request.method,
    )


# =============================================================================
# Utility Functions
# =============================================================================

def log_database_query(
    query: str,
    params: dict[str, Any] | None = None,
    duration_ms: float | None = None,
) -> None:
    """Log a database query with sanitized parameters.
    
    Args:
        query: SQL query string (will be truncated if too long).
        params: Query parameters (will be sanitized).
        duration_ms: Query duration in milliseconds.
    """
    logger = get_logger("gaia.database")
    
    # Truncate very long queries
    truncated_query = query[:500] + "..." if len(query) > 500 else query
    
    logger.debug(
        "Database query executed",
        query=truncated_query,
        params=_sanitize_value(params or {}, "") if params else None,
        duration_ms=duration_ms,
    )


def log_external_call(
    service: str,
    method: str,
    url: str,
    status_code: int | None = None,
    duration_ms: float | None = None,
    error: str | None = None,
) -> None:
    """Log an external API call.
    
    Args:
        service: Name of the external service.
        method: HTTP method.
        url: Request URL (will have query params sanitized).
        status_code: Response status code.
        duration_ms: Call duration in milliseconds.
        error: Error message if call failed.
    """
    logger = get_logger("gaia.external")
    
    # Remove query params from URL (may contain secrets)
    safe_url = url.split("?")[0] if "?" in url else url
    
    if error:
        logger.error(
            "External call failed",
            service=service,
            method=method,
            url=safe_url,
            duration_ms=duration_ms,
            error=error,
        )
    else:
        logger.info(
            "External call completed",
            service=service,
            method=method,
            url=safe_url,
            status_code=status_code,
            duration_ms=duration_ms,
        )
