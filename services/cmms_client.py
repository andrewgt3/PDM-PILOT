"""CMMS/SAP integration client for syncing work orders to external systems.

Designed to be monkeypatched in tests. When disabled, logs and returns True.
When enabled, POSTs to CMMS_WEBHOOK_URL and returns success bool.
"""

from __future__ import annotations

from typing import Union

from config import get_settings
from logger import get_logger

logger = get_logger(__name__)


def sync_work_order_to_cmms(work_order_id: Union[int, str]) -> bool:
    """
    Sync a work order to the configured CMMS webhook.
    Returns True if sync succeeded or CMMS is disabled; False on failure.
    """
    settings = get_settings()
    if not getattr(settings, "CMMS_ENABLED", False):
        logger.info("CMMS sync skipped (disabled)")
        return True
    url = getattr(settings, "CMMS_WEBHOOK_URL", None)
    if not url:
        logger.info("CMMS sync skipped (disabled)")
        return True
    try:
        import httpx
        payload = {"work_order_id": str(work_order_id)}
        r = httpx.post(url, json=payload, timeout=10)
        if r.status_code in (200, 201, 202):
            return True
        logger.warning("CMMS sync failed for %s: %s %s", work_order_id, r.status_code, r.text)
        return False
    except Exception as e:
        logger.warning("CMMS sync error for %s: %s", work_order_id, e)
        return False
