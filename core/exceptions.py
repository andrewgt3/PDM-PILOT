"""Gaia Predictive — Core Exceptions.

Domain-specific exceptions for the service layer.
These exceptions are caught by API routes and converted to HTTP responses.

Usage:
    from core.exceptions import ResourceNotFound, BusinessRuleViolation
    
    class MachineService:
        async def get_or_raise(self, machine_id: str):
            machine = await self.get(machine_id)
            if not machine:
                raise ResourceNotFound("Machine", machine_id)
            return machine
"""

from __future__ import annotations

from typing import Any


class GaiaBaseException(Exception):
    """Base exception for all Gaia Predictive domain errors."""
    
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class ResourceNotFound(GaiaBaseException):
    """Raised when a requested resource does not exist.
    
    Maps to HTTP 404 Not Found.
    
    Attributes:
        resource_type: Type of resource (e.g., "Machine", "Alarm").
        resource_id: Identifier of the missing resource.
    """
    
    def __init__(self, resource_type: str, resource_id: Any):
        self.resource_type = resource_type
        self.resource_id = resource_id
        message = f"{resource_type} with id '{resource_id}' not found"
        super().__init__(message, {"resource_type": resource_type, "resource_id": str(resource_id)})


class PermissionDenied(GaiaBaseException):
    """Raised when user lacks permission for an action.
    
    Maps to HTTP 403 Forbidden.
    
    Attributes:
        action: The attempted action (e.g., "delete", "update").
        resource: The target resource (optional).
        required_role: The role required for this action (optional).
    """
    
    def __init__(
        self, 
        action: str, 
        resource: str | None = None, 
        required_role: str | None = None
    ):
        self.action = action
        self.resource = resource
        self.required_role = required_role
        
        if resource:
            message = f"Permission denied: cannot {action} {resource}"
        else:
            message = f"Permission denied: cannot {action}"
        
        if required_role:
            message += f" (requires {required_role} role)"
        
        details = {"action": action}
        if resource:
            details["resource"] = resource
        if required_role:
            details["required_role"] = required_role
            
        super().__init__(message, details)


class BusinessRuleViolation(GaiaBaseException):
    """Raised when a business rule is violated.
    
    Maps to HTTP 422 Unprocessable Entity or 400 Bad Request.
    
    Examples:
        - Cannot schedule maintenance on inactive machine
        - Cannot acknowledge already-acknowledged alarm
        - Degradation score exceeds valid range
    
    Attributes:
        rule: Name or description of the violated rule.
        context: Additional context about the violation.
    """
    
    def __init__(self, rule: str, context: dict[str, Any] | None = None):
        self.rule = rule
        self.context = context or {}
        super().__init__(f"Business rule violation: {rule}", {"rule": rule, **self.context})


class ValidationError(GaiaBaseException):
    """Raised when input validation fails beyond Pydantic's scope.
    
    Maps to HTTP 400 Bad Request.
    """
    
    def __init__(self, field: str, message: str):
        self.field = field
        super().__init__(f"Validation error on '{field}': {message}", {"field": field})


class ExternalServiceError(GaiaBaseException):
    """Raised when an external service call fails.
    
    Maps to HTTP 502 Bad Gateway or 503 Service Unavailable.
    
    Attributes:
        service_name: Name of the external service.
        original_error: The underlying error message.
    """
    
    def __init__(self, service_name: str, original_error: str):
        self.service_name = service_name
        self.original_error = original_error
        super().__init__(
            f"External service '{service_name}' failed: {original_error}",
            {"service": service_name, "error": original_error}
        )


class RateLimitExceeded(GaiaBaseException):
    """Raised when rate limit is exceeded.
    
    Maps to HTTP 429 Too Many Requests.
    """
    
    def __init__(self, limit: str, retry_after: int | None = None):
        self.limit = limit
        self.retry_after = retry_after
        details = {"limit": limit}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        super().__init__(f"Rate limit exceeded: {limit}", details)
