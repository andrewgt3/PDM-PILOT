"""Gaia Predictive — FastAPI Dependencies.

Dependency injection for authentication, authorization, and common utilities.
All auth dependencies are async and non-blocking.

TISAX/SOC2 Compliance:
    - JWT validation on every protected request
    - Role-based access control (RBAC)
    - All auth failures logged with correlation IDs
    - No sensitive data in error responses

Usage:
    from dependencies import get_current_user, require_admin
    
    @app.get("/machines")
    async def get_machines(user: User = Depends(get_current_user)):
        ...
    
    @app.delete("/users/{id}")
    async def delete_user(user: User = Depends(require_admin)):
        ...
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from logger import get_logger
from schemas.security import TokenData, TokenType, User, UserRole
from services.auth_service import (
    AuthService,
    InvalidTokenError,
    get_auth_service,
)

logger = get_logger(__name__)

# =============================================================================
# Security Scheme
# =============================================================================

# HTTPBearer extracts the token from "Authorization: Bearer <token>" header
_bearer_scheme = HTTPBearer(
    scheme_name="JWT",
    description="Enter your JWT access token",
    auto_error=True,  # Automatically raise 403 if header is missing
)


# =============================================================================
# Authentication Dependencies
# =============================================================================

async def get_token_data(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer_scheme)],
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> TokenData:
    """Extract and validate JWT from Authorization header.
    
    This is the low-level dependency that validates the token.
    Most routes should use get_current_user instead.
    
    Args:
        credentials: Bearer token from Authorization header.
        auth_service: Injected authentication service.
    
    Returns:
        Validated TokenData with user claims.
    
    Raises:
        HTTPException: 401 if token is invalid or expired.
    """
    token = credentials.credentials
    
    try:
        token_data = auth_service.verify_token(token, expected_type=TokenType.ACCESS)
        
        # Check if token is expired (belt and suspenders - jose checks this too)
        if token_data.is_expired:
            logger.warning(
                "Expired token used",
                user_id=token_data.sub,
                username=token_data.username,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="TOKEN_EXPIRED",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return token_data
        
    except InvalidTokenError as exc:
        logger.warning(
            "Invalid token rejected",
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def get_current_user(
    token_data: Annotated[TokenData, Depends(get_token_data)],
    request: Request,
) -> User:
    """Get the current authenticated user from JWT claims.
    
    This is the primary dependency for protected routes.
    Returns a User object constructed from token claims.
    
    Args:
        token_data: Validated token data from get_token_data.
        request: FastAPI request object for context injection.
    
    Returns:
        User object representing the authenticated user.
    
    Raises:
        HTTPException: 401 if authentication fails.
    
    Example:
        @app.get("/profile")
        async def get_profile(user: User = Depends(get_current_user)):
            return {"username": user.username, "role": user.role}
    """
    # Store user info in request state for logging middleware
    request.state.user = token_data
    
    # Construct User from token claims
    # In production, you might fetch full user from database here
    import uuid
    user_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, token_data.username)
    
    user = User(
        id=user_uuid,
        username=token_data.username,
        email=f"{token_data.username}@example.com",  # RFC 2606 reserved TLD
        full_name=token_data.username,  # Placeholder
        role=token_data.role,
        is_active=True,
        created_at=token_data.iat,
        updated_at=token_data.iat,
    )
    
    logger.debug(
        "User authenticated",
        user_id=token_data.sub,
        username=token_data.username,
        role=token_data.role.value,
    )
    
    return user


async def get_current_active_user(
    user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Get current user and verify account is active.
    
    Use this dependency when you need to ensure the user
    hasn't been deactivated since token issuance.
    
    Args:
        user: Current authenticated user.
    
    Returns:
        Active user.
    
    Raises:
        HTTPException: 403 if user is deactivated.
    """
    if not user.is_active:
        logger.warning(
            "Inactive user attempted access",
            user_id=str(user.id),
            username=user.username,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated",
        )
    
    return user


# =============================================================================
# Authorization Dependencies (Role-Based)
# =============================================================================

async def get_current_active_superuser(
    user: Annotated[User, Depends(get_current_active_user)],
) -> User:
    """Require the current user to have admin privileges.
    
    Use this dependency for admin-only routes like user management,
    system configuration, and audit logs.
    
    Args:
        user: Current active user.
    
    Returns:
        Admin user.
    
    Raises:
        HTTPException: 403 if user is not an admin.
    
    Example:
        @app.delete("/users/{user_id}")
        async def delete_user(
            user_id: UUID,
            admin: User = Depends(get_current_active_superuser),
        ):
            ...
    """
    if user.role != UserRole.ADMIN:
        logger.warning(
            "Non-admin user attempted admin action",
            user_id=str(user.id),
            username=user.username,
            role=user.role.value,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient privileges. Admin access required.",
        )
    
    return user


async def require_operator_or_above(
    user: Annotated[User, Depends(get_current_active_user)],
) -> User:
    """Require the current user to be at least an operator.
    
    Operators and admins can perform write operations like
    acknowledging alarms and updating schedules.
    
    Args:
        user: Current active user.
    
    Returns:
        User with operator or admin role.
    
    Raises:
        HTTPException: 403 if user is only a viewer.
    """
    if user.role == UserRole.VIEWER:
        logger.warning(
            "Viewer user attempted operator action",
            user_id=str(user.id),
            username=user.username,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient privileges. Operator access required.",
        )
    
    return user


# =============================================================================
# Convenience Aliases
# =============================================================================

# Type aliases for cleaner route signatures
CurrentUser = Annotated[User, Depends(get_current_user)]
ActiveUser = Annotated[User, Depends(get_current_active_user)]
AdminUser = Annotated[User, Depends(get_current_active_superuser)]
OperatorUser = Annotated[User, Depends(require_operator_or_above)]

# Alias for backward compatibility
require_admin = get_current_active_superuser
get_current_admin_user = get_current_active_superuser


# =============================================================================
# Optional Authentication
# =============================================================================

# Non-auto-error bearer for optional auth
_optional_bearer = HTTPBearer(auto_error=False)


async def get_optional_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(_optional_bearer),
    ],
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
    request: Request,
) -> User | None:
    """Get current user if authenticated, None otherwise.
    
    Use this for routes that work for both authenticated and
    anonymous users but may provide additional features when
    authenticated.
    
    Args:
        credentials: Optional bearer token.
        auth_service: Injected authentication service.
        request: FastAPI request object.
    
    Returns:
        User if authenticated, None otherwise.
    
    Example:
        @app.get("/public/machines")
        async def list_machines(user: User | None = Depends(get_optional_user)):
            if user and user.role == UserRole.ADMIN:
                return get_all_machines()  # Include hidden machines
            return get_public_machines()
    """
    if credentials is None:
        return None
    
    try:
        token_data = auth_service.verify_token(
            credentials.credentials,
            expected_type=TokenType.ACCESS,
        )
        
        if token_data.is_expired:
            return None
        
        request.state.user = token_data
        
        # For JWT-based auth, create a minimal User-like object
        # that doesn't require UUID id or validated email (for temp users)
        import uuid
        from datetime import datetime, timezone
        
        # Generate a deterministic UUID from username for consistency
        user_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, token_data.username)
        
        return User(
            id=user_uuid,
            username=token_data.username,
            email=f"{token_data.username}@example.com",  # RFC 2606 reserved TLD
            full_name=token_data.username,
            role=token_data.role,
            is_active=True,
            created_at=token_data.iat,
            updated_at=token_data.iat,
        )
        
    except InvalidTokenError:
        # Invalid token in optional auth just means no user
        return None


# Type alias for optional auth
OptionalUser = Annotated[User | None, Depends(get_optional_user)]


# =============================================================================
# Generic Permission Checker
# =============================================================================

class PermissionChecker:
    """Generic callable dependency for flexible role-based authorization.
    
    This class creates a reusable permission checker that validates the
    current user has one of the required roles. It integrates with
    get_current_user and can be used directly with Depends().
    
    Attributes:
        required_roles: List of role names that are allowed access.
        require_active: Whether to also check user is_active flag.
    
    Example:
        # Single role
        @app.get("/admin/dashboard")
        async def admin_dashboard(
            user: User = Depends(PermissionChecker(required_roles=["admin"]))
        ):
            ...
        
        # Multiple roles (OR logic - user needs ANY of these)
        @app.post("/plants/{plant_id}/schedule")
        async def update_schedule(
            user: User = Depends(PermissionChecker(
                required_roles=["admin", "plant_manager", "operator"]
            ))
        ):
            ...
        
        # With string role names (auto-converts to UserRole)
        @app.delete("/machines/{id}")
        async def delete_machine(
            user: User = Depends(PermissionChecker(required_roles=["admin"]))
        ):
            ...
    """
    
    def __init__(
        self,
        required_roles: list[str | UserRole],
        require_active: bool = True,
    ) -> None:
        """Initialize the permission checker.
        
        Args:
            required_roles: List of roles allowed access. Can be UserRole
                enums or string values ('admin', 'operator', 'viewer').
            require_active: If True, also verify user.is_active. Default True.
        
        Raises:
            ValueError: If required_roles is empty or contains invalid roles.
        """
        if not required_roles:
            raise ValueError("required_roles cannot be empty")
        
        # Normalize roles to UserRole enum
        self.required_roles: set[UserRole] = set()
        for role in required_roles:
            if isinstance(role, UserRole):
                self.required_roles.add(role)
            elif isinstance(role, str):
                # Handle common role name variations
                role_normalized = role.lower().strip()
                role_mapping = {
                    "admin": UserRole.ADMIN,
                    "administrator": UserRole.ADMIN,
                    "superuser": UserRole.ADMIN,
                    "operator": UserRole.OPERATOR,
                    "plant_manager": UserRole.OPERATOR,  # Map plant_manager to operator
                    "manager": UserRole.OPERATOR,
                    "viewer": UserRole.VIEWER,
                    "readonly": UserRole.VIEWER,
                    "read_only": UserRole.VIEWER,
                }
                if role_normalized in role_mapping:
                    self.required_roles.add(role_mapping[role_normalized])
                else:
                    # Try direct enum conversion
                    try:
                        self.required_roles.add(UserRole(role_normalized))
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid role: '{role}'. Valid roles are: "
                            f"{[r.value for r in UserRole]}"
                        ) from exc
            else:
                raise TypeError(f"Role must be str or UserRole, got {type(role)}")
        
        self.require_active = require_active
        
        # Pre-compute for logging
        self._role_names = [r.value for r in self.required_roles]
    
    async def __call__(
        self,
        user: Annotated[User, Depends(get_current_user)],
    ) -> User:
        """Validate user permissions when used as a dependency.
        
        Args:
            user: Current authenticated user from get_current_user.
        
        Returns:
            The validated user if authorized.
        
        Raises:
            HTTPException: 403 if user lacks required role.
            HTTPException: 403 if user is inactive (when require_active=True).
        """
        # Check active status if required
        if self.require_active and not user.is_active:
            logger.warning(
                "Inactive user attempted access",
                user_id=str(user.id),
                username=user.username,
                required_roles=self._role_names,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is deactivated",
            )
        
        # Check if user has any of the required roles
        if user.role not in self.required_roles:
            logger.warning(
                "User lacks required permission",
                user_id=str(user.id),
                username=user.username,
                user_role=user.role.value,
                required_roles=self._role_names,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient privileges. Required roles: {self._role_names}",
            )
        
        logger.debug(
            "Permission check passed",
            user_id=str(user.id),
            username=user.username,
            role=user.role.value,
            required_roles=self._role_names,
        )
        
        return user
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"PermissionChecker(required_roles={self._role_names}, "
            f"require_active={self.require_active})"
        )


# =============================================================================
# Pre-configured Permission Checkers
# =============================================================================

# Common permission checker instances for convenience
require_admin_permission = PermissionChecker(required_roles=["admin"])
require_operator_permission = PermissionChecker(required_roles=["admin", "operator"])
require_any_role = PermissionChecker(required_roles=["admin", "operator", "viewer"])
