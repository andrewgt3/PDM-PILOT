"""Gaia Predictive — Core Logging Module.

Structured JSON logging with timestamp and log_level for production
observability (Splunk, Datadog, CloudWatch). Exports get_logger from
the main logger module for convenience.

Configuration:
    LOG_FORMAT=json   → JSON output with timestamp, log_level, service
    LOG_FORMAT=console → Colored console output for development
    LOG_LEVEL=INFO    → Logging level (DEBUG, INFO, WARNING, ERROR)

Usage:
    from core.logger import get_logger, configure_json_logging
    
    configure_json_logging()  # Call once at startup
    logger = get_logger(__name__)
    logger.info("Processing request", machine_id="WB-001")

Output (JSON mode):
    {"timestamp": "2025-12-25T03:56:57Z", "log_level": "info", 
     "event": "Processing request", "machine_id": "WB-001", "service": "gaia-predictive"}
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

# =============================================================================
# JSON Processors
# =============================================================================

def add_timestamp(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add ISO 8601 timestamp to log entry."""
    from datetime import datetime, timezone
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def add_log_level(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add log_level field to log entry."""
    event_dict["log_level"] = method_name
    return event_dict


def add_service_info(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add service metadata for log aggregation."""
    event_dict["service"] = os.getenv("SERVICE_NAME", "gaia-predictive")
    event_dict["environment"] = os.getenv("ENVIRONMENT", "development")
    return event_dict


def sanitize_exceptions(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Format exceptions as strings for JSON serialization."""
    exc_info = event_dict.pop("exc_info", None)
    if exc_info:
        import traceback
        event_dict["exception"] = "".join(traceback.format_exception(*exc_info))
    return event_dict


# =============================================================================
# Configuration
# =============================================================================

_configured = False


def configure_json_logging(
    log_level: str = "INFO",
    force_json: bool = False,
) -> None:
    """Configure structlog to output JSON with timestamp and log_level.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        force_json: Force JSON output even in development
    
    Also respects environment variables:
        LOG_FORMAT=json → Force JSON output
        LOG_LEVEL=DEBUG → Override log level
    """
    global _configured
    if _configured:
        return
    
    # Environment overrides
    level = os.getenv("LOG_LEVEL", log_level).upper()
    use_json = force_json or os.getenv("LOG_FORMAT", "").lower() == "json"
    
    # Build processor chain
    processors: list[Any] = [
        structlog.stdlib.filter_by_level,
        add_timestamp,
        add_log_level,
        add_service_info,
        structlog.stdlib.add_logger_name,
        structlog.processors.StackInfoRenderer(),
    ]
    
    if use_json:
        # Production: JSON output
        processors.extend([
            sanitize_exceptions,
            structlog.processors.JSONRenderer(sort_keys=True),
        ])
    else:
        # Development: Colored console output
        processors.extend([
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=True),
        ])
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level),
        force=True,
    )
    
    # Suppress noisy loggers
    for noisy in ("uvicorn.access", "httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a configured structlog logger.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        BoundLogger with JSON output and automatic context
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Machine status", machine_id="WB-001", status="healthy")
        {"timestamp": "2025-12-25T04:00:00Z", "log_level": "info", 
         "event": "Machine status", "machine_id": "WB-001", "status": "healthy"}
    """
    # Auto-configure if not already done
    if not _configured:
        configure_json_logging()
    
    return structlog.stdlib.get_logger(name)


# =============================================================================
# Request ID Middleware
# =============================================================================

from contextvars import ContextVar
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
import uuid
import time

# Context variables for request tracking
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def add_request_context(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Inject request_id into log entries from contextvars."""
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


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware that generates a UUID for every request.
    
    Sets the request_id in structlog.contextvars so every log line
    for that request automatically includes the ID.
    
    Usage:
        from core.logger import RequestIdMiddleware
        app.add_middleware(RequestIdMiddleware)
    
    All subsequent log calls will include:
        {"request_id": "550e8400-e29b-...", ...}
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Generate request ID and inject into logging context."""
        # Use incoming header or generate new UUID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        correlation_id = request.headers.get("X-Correlation-ID") or request_id
        
        # Set in contextvars for structlog
        request_id_var.set(request_id)
        correlation_id_var.set(correlation_id)
        
        # Also bind to structlog contextvars for automatic inclusion
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            correlation_id=correlation_id,
        )
        
        logger = get_logger("http")
        start_time = time.perf_counter()
        status_code = 500  # Default in case of exception
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
            
        except Exception as exc:
            logger.exception(
                "http_request",
                path=request.url.path,
                method=request.method,
                status=500,
                latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
                error_type=type(exc).__name__,
            )
            raise
        
        else:
            # Log successful request as http_request event
            # Filter out /health 200 OK to avoid log spam
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            
            is_health_check = request.url.path in ("/health", "/api/health")
            is_success = status_code == 200
            
            # Skip logging healthy health checks (reduces noise)
            if not (is_health_check and is_success):
                logger.info(
                    "http_request",
                    path=request.url.path,
                    method=request.method,
                    status=status_code,
                    latency_ms=latency_ms,
                )
        
        finally:
            # Clear context
            request_id_var.set(None)
            correlation_id_var.set(None)
            structlog.contextvars.clear_contextvars()


# Alias for backward compatibility with logger.py
RequestContextMiddleware = RequestIdMiddleware
