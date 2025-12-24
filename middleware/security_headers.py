#!/usr/bin/env python3
"""
Security Headers Middleware for FastAPI
========================================
Python equivalent of Helmet.js for Express.

Implements critical security headers:
- Content-Security-Policy (CSP)
- X-Frame-Options
- Strict-Transport-Security (HSTS)
- X-Content-Type-Options
- X-XSS-Protection
- Referrer-Policy
- Permissions-Policy

Author: PlantAGI Security Team
"""

import os
from typing import Callable, Dict, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# SECURITY HEADER CONFIGURATIONS
# =============================================================================

# Content-Security-Policy directives
# Customize based on your application's needs
DEFAULT_CSP = {
    "default-src": "'self'",
    "script-src": "'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net",  # Needed for React dev + Swagger UI
    "style-src": "'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net",
    "font-src": "'self' https://fonts.gstatic.com",
    "img-src": "'self' data: blob: https:",
    "connect-src": "'self' ws: wss: http://localhost:* https://api.openai.com https://api.anthropic.com",
    "frame-ancestors": "'none'",
    "form-action": "'self'",
    "base-uri": "'self'",
    "object-src": "'none'",
}

# HSTS Configuration
HSTS_MAX_AGE = int(os.getenv('HSTS_MAX_AGE', 31536000))  # 1 year default
HSTS_INCLUDE_SUBDOMAINS = os.getenv('HSTS_INCLUDE_SUBDOMAINS', 'true').lower() == 'true'
HSTS_PRELOAD = os.getenv('HSTS_PRELOAD', 'false').lower() == 'true'

# Environment check
IS_PRODUCTION = os.getenv('ENVIRONMENT', 'development').lower() == 'production'


def build_csp_header(directives: Dict[str, str]) -> str:
    """Build CSP header string from directives dictionary."""
    return "; ".join(f"{key} {value}" for key, value in directives.items())


def build_hsts_header(
    max_age: int = HSTS_MAX_AGE,
    include_subdomains: bool = HSTS_INCLUDE_SUBDOMAINS,
    preload: bool = HSTS_PRELOAD
) -> str:
    """Build HSTS header string."""
    header = f"max-age={max_age}"
    if include_subdomains:
        header += "; includeSubDomains"
    if preload:
        header += "; preload"
    return header


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    FastAPI/Starlette middleware for adding security headers.
    
    Equivalent to Helmet.js for Express applications.
    
    Headers added:
    - Content-Security-Policy
    - X-Frame-Options
    - Strict-Transport-Security
    - X-Content-Type-Options
    - X-XSS-Protection (legacy browsers)
    - Referrer-Policy
    - Permissions-Policy
    - Cache-Control (for sensitive endpoints)
    """
    
    def __init__(
        self,
        app: ASGIApp,
        csp_directives: Optional[Dict[str, str]] = None,
        enable_hsts: bool = True,
        hsts_max_age: int = HSTS_MAX_AGE,
        frame_options: str = "DENY",
        content_type_options: bool = True,
        xss_protection: bool = True,
        referrer_policy: str = "strict-origin-when-cross-origin",
        permissions_policy: Optional[str] = None
    ):
        super().__init__(app)
        
        # CSP Configuration
        self.csp_directives = csp_directives or DEFAULT_CSP
        self.csp_header = build_csp_header(self.csp_directives)
        
        # HSTS Configuration
        self.enable_hsts = enable_hsts
        self.hsts_header = build_hsts_header(max_age=hsts_max_age)
        
        # Other security headers
        self.frame_options = frame_options
        self.content_type_options = content_type_options
        self.xss_protection = xss_protection
        self.referrer_policy = referrer_policy
        
        # Permissions Policy (formerly Feature Policy)
        self.permissions_policy = permissions_policy or (
            "accelerometer=(), "
            "camera=(), "
            "geolocation=(), "
            "gyroscope=(), "
            "magnetometer=(), "
            "microphone=(), "
            "payment=(), "
            "usb=()"
        )
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Content-Security-Policy
        response.headers["Content-Security-Policy"] = self.csp_header
        
        # X-Frame-Options (prevents clickjacking)
        response.headers["X-Frame-Options"] = self.frame_options
        
        # Strict-Transport-Security (HSTS)
        # Only enable in production or when explicitly configured
        if self.enable_hsts:
            response.headers["Strict-Transport-Security"] = self.hsts_header
        
        # X-Content-Type-Options (prevents MIME sniffing)
        if self.content_type_options:
            response.headers["X-Content-Type-Options"] = "nosniff"
        
        # X-XSS-Protection (legacy browser protection)
        if self.xss_protection:
            response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer-Policy
        response.headers["Referrer-Policy"] = self.referrer_policy
        
        # Permissions-Policy (restricts browser features)
        response.headers["Permissions-Policy"] = self.permissions_policy
        
        # Remove potentially dangerous headers (use del with check)
        if "Server" in response.headers:
            del response.headers["Server"]
        if "X-Powered-By" in response.headers:
            del response.headers["X-Powered-By"]
        
        # Cache-Control for API responses (prevent caching of sensitive data)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        return response


# Convenience function for quick setup
def get_security_middleware(
    strict_mode: bool = False,
    custom_csp: Optional[Dict[str, str]] = None
) -> SecurityHeadersMiddleware:
    """
    Factory function to create SecurityHeadersMiddleware with common configurations.
    
    Args:
        strict_mode: If True, uses stricter CSP without 'unsafe-inline/eval'
        custom_csp: Custom CSP directives to merge with defaults
    
    Returns:
        Configured SecurityHeadersMiddleware instance
    """
    csp = DEFAULT_CSP.copy()
    
    if strict_mode:
        csp["script-src"] = "'self'"
        csp["style-src"] = "'self' https://fonts.googleapis.com"
    
    if custom_csp:
        csp.update(custom_csp)
    
    return SecurityHeadersMiddleware


# Export for use in API modules
__all__ = [
    'SecurityHeadersMiddleware',
    'get_security_middleware',
    'DEFAULT_CSP',
    'build_csp_header',
    'build_hsts_header'
]
