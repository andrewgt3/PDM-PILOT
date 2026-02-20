"""
Gaia Predictive — Shared notification for critical alerts and drift.

Logs and optionally POSTs to a webhook (NOTIFICATION_WEBHOOK_URL).
Used by drift_monitor_service and alert_engine.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def notify_critical(source: str, message: str, payload: dict[str, Any]) -> None:
    """
    Log at WARNING level and POST to webhook if NOTIFICATION_WEBHOOK_URL is set.

    Args:
        source: e.g. "drift" or "alert"
        message: Short human-readable message
        payload: Arbitrary dict (e.g. DriftReport model_dump() or alert context)
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    logger.warning(
        "Critical notification | source=%s message=%s payload_keys=%s",
        source,
        message,
        list(payload.keys()) if isinstance(payload, dict) else "n/a",
    )
    url = None
    try:
        from config import get_settings
        settings = get_settings()
        url = getattr(settings, "notification_webhook_url", None) or (
            getattr(settings, "alert_webhook_url", None) if hasattr(settings, "alert_webhook_url") else None
        )
    except Exception as e:
        logger.debug("Could not load settings for webhook: %s", e)
    if not url or not str(url).strip():
        return
    try:
        import urllib.request
        import json
        body = json.dumps({
            "source": source,
            "message": message,
            "payload": payload,
            "timestamp": timestamp,
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status >= 400:
                logger.warning("Webhook returned status %s for %s", resp.status, url)
    except Exception as e:
        logger.warning("Webhook POST failed: %s", e)
