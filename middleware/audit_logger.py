#!/usr/bin/env python3
"""
Audit Logger Middleware for FastAPI
====================================
Captures security-relevant information for all API requests.

Logged fields:
- Timestamp (UTC ISO format)
- User ID (from JWT token or 'anonymous')
- IP Address (client IP, X-Forwarded-For aware)
- Action (HTTP method + path)
- Response Status Code
- Request Duration (ms)

Author: PlantAGI Security Team
"""

import os
import time
import logging
from datetime import datetime
from typing import Callable, Optional
from functools import wraps
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from dotenv import load_dotenv

load_dotenv()

# Configure audit logger
AUDIT_LOG_FILE = os.getenv('AUDIT_LOG_FILE', 'security_audit.log')
AUDIT_LOG_LEVEL = os.getenv('AUDIT_LOG_LEVEL', 'INFO')

# Create dedicated security audit logger
audit_logger = logging.getLogger('security.audit')
audit_logger.setLevel(getattr(logging, AUDIT_LOG_LEVEL))

# File handler for persistent audit trail
file_handler = logging.FileHandler(AUDIT_LOG_FILE)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S%z'
))
audit_logger.addHandler(file_handler)

# Console handler for development visibility
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '🔒 AUDIT | %(message)s'
))
audit_logger.addHandler(console_handler)

# Prevent propagation to root logger
audit_logger.propagate = False


# =============================================================================
# SENSITIVE ROUTES (From Broken Access Control Report)
# =============================================================================
SENSITIVE_ROUTES = {
    # api_service.py - Stream control
    'POST /api/v1/stream/control',
    
    # enterprise_api.py - Alarms
    'POST /api/enterprise/alarms',
    'POST /api/enterprise/alarms/{alarm_id}/acknowledge',
    'POST /api/enterprise/alarms/{alarm_id}/resolve',
    
    # enterprise_api.py - Work orders
    'POST /api/enterprise/work-orders',
    'PATCH /api/enterprise/work-orders/{work_order_id}',
    
    # enterprise_api.py - Schedule
    'POST /api/enterprise/schedule/mode',
    
    # anomaly_discovery/api.py - ML operations
    'POST /api/discovery/train',
    'POST /api/discovery/detect',
    'POST /api/discovery/anomalies/{detection_id}/review',
    'POST /api/discovery/correlations/analyze',
    'POST /api/discovery/insights/generate',
    'POST /api/discovery/insights/{insight_id}/acknowledge',
}


def is_sensitive_route(method: str, path: str) -> bool:
    """
    Check if a route is marked as sensitive.
    Handles path parameters by normalizing patterns.
    """
    route_key = f"{method} {path}"
    
    # Exact match
    if route_key in SENSITIVE_ROUTES:
        return True
    
    # Pattern matching for path parameters
    for sensitive in SENSITIVE_ROUTES:
        parts = sensitive.split()
        if len(parts) == 2:
            s_method, s_pattern = parts
            if method == s_method:
                # Convert pattern to regex-like matching
                if '{' in s_pattern:
                    # Simple pattern matching: replace {param} with wildcard
                    import re
                    pattern = re.sub(r'\{[^}]+\}', r'[^/]+', s_pattern)
                    if re.fullmatch(pattern, path):
                        return True
    
    return False


def get_client_ip(request: Request) -> str:
    """
    Extract client IP address, handling proxies.
    Checks X-Forwarded-For header first.
    """
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        # X-Forwarded-For can contain multiple IPs: client, proxy1, proxy2
        return forwarded_for.split(',')[0].strip()
    
    # Fallback to direct client
    return request.client.host if request.client else 'unknown'


def get_user_id(request: Request) -> str:
    """
    Extract user ID from request.
    Checks for JWT token in Authorization header.
    Returns 'anonymous' if no auth present.
    """
    auth_header = request.headers.get('Authorization', '')
    
    if auth_header.startswith('Bearer '):
        token = auth_header[7:]
        # In production: decode JWT and extract user_id claim
        # For now, return token hash as identifier
        return f"token:{token[:8]}..." if len(token) > 8 else "token:short"
    
    # Check for API key
    api_key = request.headers.get('X-API-Key', '')
    if api_key:
        return f"apikey:{api_key[:8]}..." if len(api_key) > 8 else "apikey:short"
    
    return 'anonymous'


class AuditLoggerMiddleware(BaseHTTPMiddleware):
    """
    FastAPI/Starlette middleware for comprehensive audit logging.
    
    Captures all requests with special attention to sensitive routes.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        log_all_requests: bool = False,
        log_request_body: bool = False
    ):
        super().__init__(app)
        self.log_all_requests = log_all_requests
        self.log_request_body = log_request_body
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Capture request metadata
        start_time = time.time()
        method = request.method
        path = request.url.path
        ip_address = get_client_ip(request)
        user_id = get_user_id(request)
        timestamp = datetime.utcnow().isoformat() + 'Z'
        
        # Determine if this is a sensitive route
        is_sensitive = is_sensitive_route(method, path)
        
        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
            error = None
        except Exception as e:
            status_code = 500
            error = str(e)
            raise
        finally:
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log sensitive routes always, others based on config
            if is_sensitive or self.log_all_requests:
                log_entry = (
                    f"timestamp={timestamp} | "
                    f"user_id={user_id} | "
                    f"ip={ip_address} | "
                    f"action={method} {path} | "
                    f"status={status_code} | "
                    f"duration_ms={duration_ms:.2f}"
                )
                
                if is_sensitive:
                    log_entry = f"[SENSITIVE] {log_entry}"
                
                if error:
                    log_entry += f" | error={error}"
                
                # Log level based on status
                if status_code >= 500:
                    audit_logger.error(log_entry)
                elif status_code >= 400:
                    audit_logger.warning(log_entry)
                else:
                    audit_logger.info(log_entry)
        
        return response


def audit_log(action_description: str = None):
    """
    Decorator for adding explicit audit logging to specific endpoints.
    Use when you need more detailed action descriptions.
    
    Usage:
        @app.post("/api/admin/delete-all")
        @audit_log("DELETE ALL RECORDS")
        async def delete_all():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from kwargs if available
            request = kwargs.get('request')
            
            if request:
                user_id = get_user_id(request)
                ip_address = get_client_ip(request)
                action = action_description or func.__name__
                
                audit_logger.info(
                    f"[ACTION] user_id={user_id} | "
                    f"ip={ip_address} | "
                    f"action={action}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Export for use in API modules
__all__ = [
    'AuditLoggerMiddleware',
    'audit_log',
    'audit_logger',
    'is_sensitive_route',
    'SENSITIVE_ROUTES'
]
