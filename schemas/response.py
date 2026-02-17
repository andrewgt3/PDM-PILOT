"""Gaia Predictive — Standardized API Response Schemas.

Provides a consistent response format for all API endpoints and a
high-performance ORJSONResponse class for faster serialization.

Response Format:
    {
        "data": T,              // The actual response payload
        "meta": {...},          // Pagination, timing, version info
        "error": null | string  // Error message if applicable
    }

Usage:
    from schemas.response import APIResponse, ORJSONResponse
    
    @app.get("/machines", response_class=ORJSONResponse)
    async def get_machines() -> APIResponse[list[Machine]]:
        machines = await service.get_all()
        return APIResponse(data=machines, meta={"count": len(machines)})
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

import orjson
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Generic Type Variable
# =============================================================================

T = TypeVar("T")


# =============================================================================
# Standard Response Components
# =============================================================================

class ResponseMeta(BaseModel):
    """Metadata included in every API response.
    
    Attributes:
        timestamp: ISO 8601 timestamp of response generation.
        request_id: Correlation ID for tracing.
        version: API version string.
        count: Number of items in data (for collections).
        page: Current page number (for paginated responses).
        page_size: Items per page.
        total_pages: Total number of pages.
        total_count: Total number of items across all pages.
    """
    
    model_config = ConfigDict(extra="allow")  # Allow additional metadata
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response generation timestamp",
    )
    request_id: str | None = Field(None, description="Request correlation ID")
    version: str = Field(default="1.0.0", description="API version")
    
    # Collection metadata
    count: int | None = Field(None, description="Number of items in response")
    
    # Pagination metadata
    page: int | None = Field(None, ge=1, description="Current page number")
    page_size: int | None = Field(None, ge=1, le=100, description="Items per page")
    total_pages: int | None = Field(None, ge=0, description="Total pages")
    total_count: int | None = Field(None, ge=0, description="Total items")


class APIResponse(BaseModel, Generic[T]):
    """Standardized API response wrapper.
    
    All API responses should use this format for consistency.
    The frontend can always expect this structure.
    
    Attributes:
        data: The actual response payload (generic type T).
        meta: Response metadata (timing, pagination, etc.).
        error: Error message if the request failed.
    
    Example Success:
        {
            "data": [{"id": "123", "name": "CNC-001"}],
            "meta": {"count": 1, "timestamp": "2024-12-24T20:00:00Z"},
            "error": null
        }
    
    Example Error:
        {
            "data": null,
            "meta": {"timestamp": "2024-12-24T20:00:00Z"},
            "error": "Machine not found"
        }
    """
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "data": {"id": "123", "name": "Example"},
                "meta": {
                    "timestamp": "2024-12-24T20:00:00Z",
                    "request_id": "abc-123",
                    "version": "1.0.0",
                },
                "error": None,
            }
        }
    )
    
    data: T | None = Field(None, description="Response payload")
    meta: ResponseMeta = Field(
        default_factory=ResponseMeta,
        description="Response metadata",
    )
    error: str | None = Field(None, description="Error message if failed")
    
    @classmethod
    def success(
        cls,
        data: T,
        request_id: str | None = None,
        **extra_meta: Any,
    ) -> "APIResponse[T]":
        """Create a successful response.
        
        Args:
            data: The response payload.
            request_id: Optional correlation ID.
            **extra_meta: Additional metadata fields.
        
        Returns:
            APIResponse with data and metadata.
        """
        meta = ResponseMeta(request_id=request_id, **extra_meta)
        
        # Auto-populate count for lists
        if isinstance(data, list):
            meta.count = len(data)
        
        return cls(data=data, meta=meta, error=None)
    
    @classmethod
    def error(
        cls,
        message: str,
        request_id: str | None = None,
    ) -> "APIResponse[None]":
        """Create an error response.
        
        Args:
            message: Error description.
            request_id: Optional correlation ID.
        
        Returns:
            APIResponse with error message.
        """
        return cls(
            data=None,
            meta=ResponseMeta(request_id=request_id),
            error=message,
        )
    
    @classmethod
    def paginated(
        cls,
        data: list[T],
        page: int,
        page_size: int,
        total_count: int,
        request_id: str | None = None,
    ) -> "APIResponse[list[T]]":
        """Create a paginated response.
        
        Args:
            data: Items for the current page.
            page: Current page number (1-indexed).
            page_size: Items per page.
            total_count: Total items across all pages.
            request_id: Optional correlation ID.
        
        Returns:
            APIResponse with pagination metadata.
        """
        total_pages = (total_count + page_size - 1) // page_size if page_size > 0 else 0
        
        meta = ResponseMeta(
            request_id=request_id,
            count=len(data),
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            total_count=total_count,
        )
        
        return cls(data=data, meta=meta, error=None)


# =============================================================================
# High-Performance ORJSON Response
# =============================================================================

def _orjson_serializer(obj: Any) -> bytes:
    """Serialize object to JSON bytes using orjson.
    
    Handles special types like datetime and Pydantic models.
    """
    return orjson.dumps(
        obj,
        option=(
            orjson.OPT_SERIALIZE_NUMPY |      # Handle numpy arrays
            orjson.OPT_UTC_Z |                 # Use Z suffix for UTC
            orjson.OPT_NAIVE_UTC |             # Treat naive datetimes as UTC
            orjson.OPT_NON_STR_KEYS            # Allow non-string dict keys
        ),
    )


class ORJSONResponse(JSONResponse):
    """High-performance JSON response using orjson.
    
    orjson is 3-10x faster than the standard json library and
    natively handles datetime, UUID, and numpy arrays.
    
    Usage:
        from schemas.response import ORJSONResponse
        
        app = FastAPI(default_response_class=ORJSONResponse)
        
        # Or per-route:
        @app.get("/fast", response_class=ORJSONResponse)
        async def get_fast():
            return {"data": "fast!"}
    """
    
    media_type = "application/json"
    
    def render(self, content: Any) -> bytes:
        """Render content to JSON bytes using orjson.
        
        Args:
            content: The content to serialize.
        
        Returns:
            JSON bytes.
        """
        # Handle Pydantic models
        if hasattr(content, "model_dump"):
            content = content.model_dump(mode="json")
        elif hasattr(content, "dict"):  # Pydantic v1 compatibility
            content = content.dict()
        
        return _orjson_serializer(content)


# =============================================================================
# Error Response Models
# =============================================================================

class ErrorDetail(BaseModel):
    """Detailed error information for debugging.
    
    Attributes:
        code: Machine-readable error code.
        message: Human-readable error message.
        field: Field name if validation error.
        details: Additional error context.
    """
    
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    field: str | None = Field(None, description="Related field name")
    details: dict[str, Any] | None = Field(None, description="Additional context")


class ErrorResponse(BaseModel):
    """Standard error response for 4xx/5xx status codes.
    
    Example:
        {
            "data": null,
            "meta": {"timestamp": "2024-12-24T20:00:00Z"},
            "error": "Validation failed",
            "errors": [
                {"code": "invalid_format", "message": "Invalid UUID", "field": "machine_id"}
            ]
        }
    """
    
    data: None = Field(None, description="Always null for errors")
    meta: ResponseMeta = Field(default_factory=ResponseMeta)
    error: str = Field(..., description="Primary error message")
    errors: list[ErrorDetail] | None = Field(
        None,
        description="Detailed error list (for validation errors)",
    )
    
    @classmethod
    def from_exception(
        cls,
        error: str,
        details: list[ErrorDetail] | None = None,
        request_id: str | None = None,
    ) -> "ErrorResponse":
        """Create error response from exception details."""
        return cls(
            data=None,
            meta=ResponseMeta(request_id=request_id),
            error=error,
            errors=details,
        )


# =============================================================================
# Pagination Request Schema
# =============================================================================

class PaginationParams(BaseModel):
    """Standard pagination query parameters.
    
    Usage:
        @app.get("/machines")
        async def list_machines(pagination: PaginationParams = Depends()):
            return service.get_paginated(pagination.page, pagination.page_size)
    """
    
    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")
    
    @property
    def offset(self) -> int:
        """Calculate SQL offset for pagination."""
        return (self.page - 1) * self.page_size
    
    @property
    def limit(self) -> int:
        """Alias for page_size, for SQL LIMIT clause."""
        return self.page_size
