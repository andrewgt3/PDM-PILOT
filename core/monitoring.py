"""Gaia Predictive — Error Monitoring Stub.

Provides a capture_exception function that can be wired throughout
the codebase. When Sentry is configured, this will forward exceptions.
Until then, it logs to structlog.

Usage:
    from core.monitoring import capture_exception
    
    try:
        risky_operation()
    except Exception as e:
        capture_exception(e)
        raise

Configuration (future):
    SENTRY_DSN=https://xxx@sentry.io/yyy
"""

from __future__ import annotations

import os
from typing import Any

from logger import get_logger

logger = get_logger("monitoring")

# =============================================================================
# Sentry Configuration (Stub)
# =============================================================================

_sentry_initialized = False
_sentry_dsn = os.getenv("SENTRY_DSN")


def init_sentry() -> None:
    """Initialize Sentry SDK if DSN is configured.
    
    Call this during application startup.
    """
    global _sentry_initialized
    
    if _sentry_initialized:
        return
    
    if _sentry_dsn:
        try:
            import sentry_sdk
            sentry_sdk.init(
                dsn=_sentry_dsn,
                traces_sample_rate=0.1,
                environment=os.getenv("ENVIRONMENT", "development"),
            )
            _sentry_initialized = True
            logger.info("sentry_initialized", dsn=_sentry_dsn[:20] + "...")
        except ImportError:
            logger.warning("sentry_sdk_not_installed", action="using_stub")
        except Exception as e:
            logger.error("sentry_init_failed", error=str(e))
    else:
        logger.debug("sentry_not_configured", action="using_stub")


def capture_exception(
    exception: BaseException,
    context: dict[str, Any] | None = None,
) -> None:
    """Capture an exception for error monitoring.
    
    When Sentry is configured, forwards to sentry_sdk.capture_exception().
    Otherwise, logs the exception with structlog.
    
    Args:
        exception: The exception to capture.
        context: Optional additional context to attach.
    
    Example:
        try:
            process_request()
        except Exception as e:
            capture_exception(e, context={"user_id": user.id})
            raise
    """
    # Always log locally first
    logger.error(
        "exception_captured",
        error_type=type(exception).__name__,
        error=str(exception),
        context=context,
    )
    
    # Forward to Sentry if configured
    if _sentry_initialized and _sentry_dsn:
        try:
            import sentry_sdk
            with sentry_sdk.push_scope() as scope:
                if context:
                    for key, value in context.items():
                        scope.set_extra(key, value)
                sentry_sdk.capture_exception(exception)
        except Exception:
            # Don't let monitoring errors crash the app
            pass


def capture_message(
    message: str,
    level: str = "info",
    context: dict[str, Any] | None = None,
) -> None:
    """Capture a message for error monitoring.
    
    Args:
        message: The message to capture.
        level: Message level (info, warning, error).
        context: Optional additional context.
    """
    logger.log(level, message, **(context or {}))
    
    if _sentry_initialized and _sentry_dsn:
        try:
            import sentry_sdk
            sentry_sdk.capture_message(message, level=level)
        except Exception:
            pass
