"""Gaia Predictive — Security & Authentication Schemas.

Pydantic models for authentication, authorization, and user management.

TISAX/SOC2 Compliance:
    - User.hashed_password is NEVER serialized to JSON
    - All sensitive fields use SecretStr
    - Token expiry is strictly enforced
    - All models are immutable after creation
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated
import uuid

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    SecretStr,
    field_serializer,
    model_validator,
)


# =============================================================================
# Enums
# =============================================================================

class UserRole(str, Enum):
    """User authorization roles.

    Hierarchy: ADMIN > ENGINEER > PLANT_MANAGER/RELIABILITY_ENGINEER > OPERATOR > TECHNICIAN > VIEWER.
    - viewer: Read-only (dashboard, reports).
    - operator: Viewer + acknowledge alerts, submit labels.
    - technician: Assigned machines only, work orders only, no fleet view.
    - plant_manager: All machines at one site, all work orders, no cross-site.
    - reliability_engineer: Aggregated metrics across all sites, no raw sensor readings, no individual machine detail.
    - engineer: Operator + create work orders, run profiling, drift/shadow.
    - admin: Engineer + model promotion/rollback, user management, system config, audit log.
    """
    VIEWER = "viewer"                     # Read-only access
    OPERATOR = "operator"                 # View dashboard, acknowledge alerts, submit labels
    TECHNICIAN = "technician"             # Assigned machines + work orders only, no fleet view
    PLANT_MANAGER = "plant_manager"       # All machines at one site, no cross-site
    RELIABILITY_ENGINEER = "reliability_engineer"  # Aggregates only, no raw data / machine detail
    ENGINEER = "engineer"                 # Operator + work orders, profiling, drift
    ADMIN = "admin"                       # Engineer + model promotion, user mgmt, config, audit


# Used for permission checks (higher = more privileged).
ROLE_HIERARCHY = {
    "admin": 5,
    "engineer": 4,
    "plant_manager": 3,
    "reliability_engineer": 3,
    "operator": 2,
    "technician": 1,
    "viewer": 0,
}


class TokenType(str, Enum):
    """JWT token types."""
    ACCESS = "access"
    REFRESH = "refresh"


# =============================================================================
# Authentication Requests
# =============================================================================

class LoginRequest(BaseModel):
    """User login credentials.
    
    Attributes:
        username: User's username or email.
        password: User's password (SecretStr to prevent logging).
    
    Example:
        {
            "username": "operator@ford.com",
            "password": "secure_password_123"
        }
    """
    
    model_config = ConfigDict(
        strict=True,
        str_strip_whitespace=True,
    )
    
    username: Annotated[
        str,
        Field(
            min_length=3,
            max_length=255,
            description="Username or email address",
            examples=["operator@ford.com"],
        ),
    ]
    password: Annotated[
        SecretStr,
        Field(
            min_length=8,
            max_length=128,
            description="User password",
        ),
    ]


class RefreshTokenRequest(BaseModel):
    """Request to refresh an access token.
    
    Attributes:
        refresh_token: Valid refresh token.
    """
    
    model_config = ConfigDict(strict=True)
    
    refresh_token: Annotated[
        str,
        Field(
            description="Refresh token from login response",
        ),
    ]


class PasswordChangeRequest(BaseModel):
    """Request to change user password.
    
    Attributes:
        current_password: Current password for verification.
        new_password: New password to set.
    """
    
    model_config = ConfigDict(strict=True)
    
    current_password: SecretStr = Field(
        ...,
        min_length=8,
        description="Current password for verification",
    )
    new_password: SecretStr = Field(
        ...,
        min_length=12,  # Enforce stronger password for changes
        max_length=128,
        description="New password (minimum 12 characters)",
    )
    
    @model_validator(mode="after")
    def passwords_must_differ(self) -> "PasswordChangeRequest":
        """Ensure new password differs from current."""
        if self.current_password.get_secret_value() == self.new_password.get_secret_value():
            raise ValueError("New password must differ from current password")
        return self


# =============================================================================
# Token Models
# =============================================================================

class Token(BaseModel):
    """OAuth2-compatible token response.
    
    Returned after successful authentication.
    
    Attributes:
        access_token: JWT access token for API authorization.
        refresh_token: JWT refresh token for obtaining new access tokens.
        token_type: Token type (always "bearer").
        expires_in: Access token validity in seconds.
    
    Example:
        {
            "access_token": "eyJhbGciOiJIUzI1NiIs...",
            "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
            "token_type": "bearer",
            "expires_in": 1800
        }
    """
    
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 1800,
            }
        },
    )
    
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str | None = Field(None, description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., gt=0, description="Token validity in seconds")


class TokenData(BaseModel):
    """Decoded JWT token payload.
    
    This is the internal representation of the claims within a JWT.
    Used by authentication middleware to validate and extract user info.
    
    Attributes:
        sub: Subject (user ID).
        username: User's username.
        role: User's authorization role.
        token_type: Type of token (access or refresh).
        exp: Token expiration timestamp.
        iat: Token issued-at timestamp.
        jti: Unique token identifier for revocation.
    """
    
    model_config = ConfigDict(frozen=True)  # Immutable after creation
    
    sub: Annotated[
        str,
        Field(description="Subject (user ID)"),
    ]
    username: Annotated[
        str,
        Field(description="Username"),
    ]
    role: UserRole = Field(default=UserRole.VIEWER, description="User role")
    token_type: TokenType = Field(default=TokenType.ACCESS, description="Token type")
    exp: datetime = Field(..., description="Expiration time")
    iat: datetime = Field(..., description="Issued at time")
    jti: Annotated[
        str,
        Field(default_factory=lambda: str(uuid.uuid4()), description="Token ID"),
    ]
    site_id: str | None = Field(None, description="Site for plant_manager scoping")
    assigned_machine_ids: list[str] = Field(
        default_factory=list,
        description="Pre-loaded machine IDs for technician",
    )

    @property
    def user_id(self) -> str:
        """Alias for sub (subject) claim."""
        return self.sub
    
    @property
    def is_expired(self) -> bool:
        """Check if token has expired."""
        from datetime import timezone
        return datetime.now(timezone.utc) > self.exp
    
    @property
    def is_admin(self) -> bool:
        """Check if token belongs to an admin user."""
        return self.role == UserRole.ADMIN


# =============================================================================
# User Models
# =============================================================================

class UserBase(BaseModel):
    """Base user fields shared across models.
    
    Attributes:
        username: Unique username.
        email: User's email address.
        full_name: User's display name.
        role: Authorization role.
        is_active: Whether user account is active.
    """
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        from_attributes=True,  # Allow ORM model conversion
    )
    
    username: Annotated[
        str,
        Field(
            min_length=3,
            max_length=50,
            pattern=r"^[a-zA-Z0-9_-]+$",
            description="Unique username (alphanumeric, underscore, hyphen)",
            examples=["john_doe"],
        ),
    ]
    email: EmailStr = Field(..., description="Email address")
    full_name: Annotated[
        str,
        Field(
            min_length=1,
            max_length=255,
            description="Full display name",
            examples=["John Doe"],
        ),
    ]
    role: UserRole = Field(default=UserRole.VIEWER, description="User role")
    is_active: bool = Field(default=True, description="Account active status")
    site_id: str | None = Field(None, description="Site for plant_manager scoping")
    assigned_machine_ids: list[str] = Field(default_factory=list, description="Assigned machine IDs for technician")


class UserCreate(UserBase):
    """Schema for creating a new user.
    
    Includes password field which is only used during creation.
    
    Attributes:
        password: Initial password (must meet complexity requirements).
    """
    
    password: SecretStr = Field(
        ...,
        min_length=12,
        max_length=128,
        description="Password (minimum 12 characters)",
    )


class UserUpdate(BaseModel):
    """Schema for updating user fields.
    
    All fields are optional. Only provided fields are updated.
    """
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    email: EmailStr | None = Field(None, description="New email address")
    full_name: str | None = Field(None, min_length=1, max_length=255)
    role: UserRole | None = Field(None, description="New role")
    is_active: bool | None = Field(None, description="Active status")


class User(UserBase):
    """User model for API responses.
    
    CRITICAL: This model explicitly EXCLUDES hashed_password.
    The field is defined but configured to never serialize to JSON.
    
    Attributes:
        id: Unique user identifier.
        created_at: Account creation timestamp.
        updated_at: Last update timestamp.
        last_login: Last successful login timestamp.
        hashed_password: Password hash (NEVER serialized).
    """
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "username": "john_doe",
                "email": "john@example.com",
                "full_name": "John Doe",
                "role": "operator",
                "is_active": True,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-06-20T14:45:00Z",
                "last_login": "2024-12-24T08:00:00Z",
            }
        },
    )
    
    id: uuid.UUID = Field(..., description="User ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    last_login: datetime | None = Field(None, description="Last login timestamp")
    
    # CRITICAL: This field exists for ORM compatibility but is NEVER serialized
    hashed_password: str | None = Field(
        default=None,
        exclude=True,  # Exclude from serialization
        repr=False,    # Exclude from __repr__
        description="Password hash (never serialized)",
    )
    
    @field_serializer("hashed_password", when_used="always")
    @classmethod
    def _never_serialize_password(cls, v: str | None) -> None:
        """Ensure hashed_password is never serialized, even if exclude fails."""
        return None


class UserInDB(User):
    """Internal user model including hashed password.
    
    This model is for INTERNAL USE ONLY. Never return this from an API.
    Used for password verification during authentication.
    
    Attributes:
        hashed_password: The bcrypt password hash.
    """
    
    model_config = ConfigDict(from_attributes=True)
    
    # Override to include password for internal operations
    hashed_password: str = Field(
        ...,
        exclude=True,  # Still excluded if accidentally serialized
        repr=False,
        description="Bcrypt password hash (internal only)",
    )


# =============================================================================
# API Response Models
# =============================================================================

class AuthResponse(BaseModel):
    """Complete authentication response.
    
    Returned after successful login, includes user details and tokens.
    """
    
    user: User = Field(..., description="Authenticated user details")
    token: Token = Field(..., description="Access and refresh tokens")


class UserList(BaseModel):
    """Paginated list of users."""
    
    users: list[User] = Field(..., description="List of users")
    total: int = Field(..., ge=0, description="Total number of users")
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, le=100, description="Items per page")
