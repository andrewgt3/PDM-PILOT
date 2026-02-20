"""Gaia Predictive — Authentication Service.

Core authentication logic for password hashing and JWT token management.
Uses passlib[bcrypt] for secure password hashing and python-jose for JWT.

TISAX/SOC2 Compliance:
    - Passwords hashed with bcrypt (work factor 12)
    - JWTs have mandatory expiration
    - Tokens include JTI for revocation support
    - All auth events are logged

Usage:
    from services.auth_service import AuthService
    
    auth = AuthService()
    
    # Hash a password for storage
    hashed = auth.hash_password("user_password")
    
    # Verify password during login
    if auth.verify_password("user_password", hashed):
        token = auth.create_access_token(user_id="123", username="john")
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
import asyncio
import uuid

import bcrypt
from jose import JWTError, jwt

from config import get_settings
from logger import get_logger
from schemas.security import TokenData, TokenType, UserRole

logger = get_logger(__name__)


# =============================================================================
# Password Hashing Configuration
# =============================================================================

# No configuration needed for direct bcrypt usage
# Work factor (rounds) is handled in the calls (default 12 for robust security)


# =============================================================================
# Authentication Service
# =============================================================================

class AuthService:
    """Core authentication service for password and token operations.
    
    This service handles:
        - Secure password hashing with bcrypt
        - Password verification with timing-attack resistance
        - JWT access token creation and validation
        - JWT refresh token creation and validation
    
    Attributes:
        secret_key: JWT signing secret (from config).
        algorithm: JWT signing algorithm (from config).
        access_expiry_minutes: Access token lifetime.
        refresh_expiry_days: Refresh token lifetime.
    
    Example:
        >>> auth = AuthService()
        >>> hashed = auth.hash_password("my_secure_password")
        >>> auth.verify_password("my_secure_password", hashed)
        True
    """
    
    def __init__(self) -> None:
        """Initialize the authentication service with config settings."""
        settings = get_settings()
        security = settings.security
        
        self._secret_key = security.jwt_secret.get_secret_value()
        self._algorithm = security.jwt_algorithm
        self._access_expiry_minutes = security.jwt_expiry_minutes
        self._refresh_expiry_days = security.jwt_refresh_expiry_days
        
        logger.debug(
            "AuthService initialized",
            algorithm=self._algorithm,
            access_expiry_minutes=self._access_expiry_minutes,
            refresh_expiry_days=self._refresh_expiry_days,
        )
    
    # =========================================================================
    # Password Operations
    # =========================================================================
    
    async def hash_password(self, plain_password: str) -> str:
        """Hash a plain-text password using bcrypt (non-blocking).
        
        Args:
            plain_password: The plain-text password to hash.
        
        Returns:
            The bcrypt hash string (includes salt and work factor).
        
        Example:
            >>> auth = AuthService()
            >>> hashed = await auth.hash_password("secure_password_123")
            >>> hashed.startswith("$2b$")
            True
        """
        def _hash():
            salt = bcrypt.gensalt(rounds=12)
            return bcrypt.hashpw(plain_password.encode('utf-8'), salt).decode('utf-8')
            
        return await asyncio.to_thread(_hash)
    
    async def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a plain-text password against a bcrypt hash (non-blocking).
        
        This method is timing-attack resistant due to bcrypt's design.
        
        Args:
            plain_password: The plain-text password to verify.
            hashed_password: The bcrypt hash to verify against.
        
        Returns:
            True if password matches, False otherwise.
        
        Example:
            >>> auth = AuthService()
            >>> hashed = await auth.hash_password("my_password")
            >>> await auth.verify_password("my_password", hashed)
            True
        Example:
            >>> auth = AuthService()
            >>> hashed = await auth.hash_password("my_password")
            >>> await auth.verify_password("my_password", hashed)
            True
            >>> await auth.verify_password("wrong_password", hashed)
            False
        """
        def _verify():
            try:
                return bcrypt.checkpw(
                    plain_password.encode('utf-8'), 
                    hashed_password.encode('utf-8')
                )
            except ValueError:
                return False

        try:
            return await asyncio.to_thread(_verify)
        except Exception as exc:
            # Log but don't expose verification errors
            logger.warning(
                "Password verification error",
                error_type=type(exc).__name__,
            )
            return False
    
    
    # Needs rehash check is complex with direct bcrypt, skipping for now as strict rounds=12 is enforced above.
    # If needed, we can inspect the hash prefix/rounds manually.
    def check_needs_rehash(self, hashed_password: str) -> bool:
        """Check if a password hash needs to be upgraded.
        
        For now, simply returns False as we enforce bcrypt 12 rounds.
        """
        return False
    
    # =========================================================================
    # Login Authentication with Event Logging
    # =========================================================================
    
    async def authenticate_user(
        self,
        username: str,
        password: str,
        hashed_password: str,
        user_id: str,
        ip_address: str | None = None,
    ) -> bool:
        """Authenticate user credentials and log appropriate events.
        
        SECURITY: Never logs passwords. Only logs user_id on success,
        ip_address on failure for security monitoring.
        
        Args:
            username: Username attempting login.
            password: Plain-text password (NEVER logged).
            hashed_password: Stored password hash to verify against.
            user_id: User identifier (logged on success).
            ip_address: Client IP address (logged on failure).
        
        Returns:
            True if authentication successful, False otherwise.
        
        Example:
            >>> success = await auth.authenticate_user(
            ...     username="john",
            ...     password=request_password,  # Not logged
            ...     hashed_password=user.password_hash,
            ...     user_id=user.id,
            ...     ip_address=request.client.host,
            ... )
        """
        is_valid = await self.verify_password(password, hashed_password)
        
        if is_valid:
            self._log_login_success(user_id=user_id, username=username)
            return True
        else:
            self._log_login_failed(username=username, ip_address=ip_address)
            return False
    
    def _log_login_success(self, user_id: str, username: str) -> None:
        """Log successful login event.
        
        SECURITY: Logs user_id for audit trail. Never logs password.
        """
        logger.info(
            "login_success",
            user_id=user_id,
            username=username,
        )
    
    def _log_login_failed(
        self,
        username: str,
        ip_address: str | None = None,
    ) -> None:
        """Log failed login attempt.
        
        SECURITY: Logs ip_address for security monitoring.
        Never logs password or reveals if username exists.
        """
        logger.warning(
            "login_failed",
            username=username,
            ip_address=ip_address or "unknown",
        )
    
    # =========================================================================
    # JWT Token Operations
    # =========================================================================
    
    def create_access_token(
        self,
        user_id: str,
        username: str,
        role: UserRole = UserRole.VIEWER,
        expires_delta: timedelta | None = None,
        site_id: str | None = None,
        assigned_machine_ids: list[str] | None = None,
    ) -> str:
        """Create a JWT access token.
        
        Args:
            user_id: Unique user identifier (stored in 'sub' claim).
            username: User's username.
            role: User's authorization role.
            expires_delta: Custom expiration time. Defaults to config value.
            site_id: Site for plant_manager scoping (optional).
            assigned_machine_ids: Pre-loaded machine IDs for technician (optional).
        
        Returns:
            Encoded JWT access token string.
        """
        if expires_delta is None:
            expires_delta = timedelta(minutes=self._access_expiry_minutes)
        
        return self._create_token(
            user_id=user_id,
            username=username,
            role=role,
            token_type=TokenType.ACCESS,
            expires_delta=expires_delta,
            site_id=site_id,
            assigned_machine_ids=assigned_machine_ids,
        )
    
    def create_refresh_token(
        self,
        user_id: str,
        username: str,
        role: UserRole = UserRole.VIEWER,
        expires_delta: timedelta | None = None,
        site_id: str | None = None,
        assigned_machine_ids: list[str] | None = None,
    ) -> str:
        """Create a JWT refresh token.
        
        Args:
            user_id: Unique user identifier.
            username: User's username.
            role: User's authorization role.
            expires_delta: Custom expiration time. Defaults to config value.
            site_id: Site for plant_manager scoping (optional).
            assigned_machine_ids: Pre-loaded machine IDs for technician (optional).
        
        Returns:
            Encoded JWT refresh token string.
        """
        if expires_delta is None:
            expires_delta = timedelta(days=self._refresh_expiry_days)
        
        return self._create_token(
            user_id=user_id,
            username=username,
            role=role,
            token_type=TokenType.REFRESH,
            expires_delta=expires_delta,
            site_id=site_id,
            assigned_machine_ids=assigned_machine_ids,
        )
    
    def _create_token(
        self,
        user_id: str,
        username: str,
        role: UserRole,
        token_type: TokenType,
        expires_delta: timedelta,
        site_id: str | None = None,
        assigned_machine_ids: list[str] | None = None,
    ) -> str:
        """Internal method to create a JWT with specified claims."""
        now = datetime.now(timezone.utc)
        expire = now + expires_delta
        jti = str(uuid.uuid4())
        
        payload: dict[str, Any] = {
            "sub": user_id,
            "username": username,
            "role": role.value,
            "token_type": token_type.value,
            "exp": expire,
            "iat": now,
            "jti": jti,
        }
        if site_id is not None:
            payload["site_id"] = site_id
        if assigned_machine_ids is not None:
            payload["assigned_machine_ids"] = assigned_machine_ids
        
        token = jwt.encode(payload, self._secret_key, algorithm=self._algorithm)
        
        logger.info(
            "Token created",
            token_type=token_type.value,
            user_id=user_id,
            username=username,
            expires_at=expire.isoformat(),
            jti=jti,
        )
        
        return token
    
    def verify_token(self, token: str, expected_type: TokenType | None = None) -> TokenData:
        """Verify and decode a JWT token.
        
        Args:
            token: The JWT token string to verify.
            expected_type: If provided, validates the token type matches.
        
        Returns:
            TokenData with decoded claims.
        
        Raises:
            InvalidTokenError: If token is invalid, expired, or wrong type.
        
        Example:
            >>> auth = AuthService()
            >>> token = auth.create_access_token(user_id="123", username="john")
            >>> data = auth.verify_token(token, expected_type=TokenType.ACCESS)
            >>> data.username
            'john'
        """
        try:
            payload = jwt.decode(
                token,
                self._secret_key,
                algorithms=[self._algorithm],
            )
            
            # Extract and validate claims
            token_type_str = payload.get("token_type", TokenType.ACCESS.value)
            token_type = TokenType(token_type_str)
            
            if expected_type and token_type != expected_type:
                logger.warning(
                    "Token type mismatch",
                    expected=expected_type.value,
                    actual=token_type.value,
                )
                raise InvalidTokenError(
                    f"Expected {expected_type.value} token, got {token_type.value}"
                )
            
            # Parse timestamps
            exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
            iat = datetime.fromtimestamp(payload["iat"], tz=timezone.utc)
            
            token_data = TokenData(
                sub=payload["sub"],
                username=payload["username"],
                role=UserRole(payload.get("role", UserRole.VIEWER.value)),
                token_type=token_type,
                exp=exp,
                iat=iat,
                jti=payload.get("jti", str(uuid.uuid4())),
                site_id=payload.get("site_id"),
                assigned_machine_ids=payload.get("assigned_machine_ids") or [],
            )
            
            logger.debug(
                "Token verified",
                user_id=token_data.sub,
                username=token_data.username,
                token_type=token_type.value,
            )
            
            return token_data
            
        except JWTError as exc:
            logger.warning(
                "Token verification failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise InvalidTokenError(f"Invalid token: {exc}") from exc
    
    def get_token_expiry_seconds(self, token_type: TokenType = TokenType.ACCESS) -> int:
        """Get the expiry duration in seconds for a token type.
        
        Args:
            token_type: ACCESS or REFRESH token.
        
        Returns:
            Expiry duration in seconds.
        """
        if token_type == TokenType.ACCESS:
            return self._access_expiry_minutes * 60
        else:
            return self._refresh_expiry_days * 24 * 60 * 60


# =============================================================================
# Custom Exceptions
# =============================================================================

class AuthenticationError(Exception):
    """Base exception for authentication errors."""
    pass


class InvalidTokenError(AuthenticationError):
    """Raised when a JWT token is invalid or expired."""
    pass


class InvalidCredentialsError(AuthenticationError):
    """Raised when login credentials are incorrect."""
    pass


# =============================================================================
# Module-Level Convenience Functions
# =============================================================================

# Singleton instance for simple usage
_auth_service: AuthService | None = None


def get_auth_service() -> AuthService:
    """Get the singleton AuthService instance.
    
    Returns:
        Configured AuthService instance.
    
    Example:
        >>> from services.auth_service import get_auth_service
        >>> auth = get_auth_service()
        >>> auth.verify_password("password", hashed)
    """
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service


async def hash_password(plain_password: str) -> str:
    """Convenience function to hash a password (async).
    
    Args:
        plain_password: Plain-text password to hash.
    
    Returns:
        Bcrypt hash string.
    """
    return await get_auth_service().hash_password(plain_password)


async def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Convenience function to verify a password (async).
    
    Args:
        plain_password: Plain-text password to verify.
        hashed_password: Bcrypt hash to compare against.
    
    Returns:
        True if password matches, False otherwise.
    """
    return await get_auth_service().verify_password(plain_password, hashed_password)


def create_access_token(
    user_id: str,
    username: str,
    role: UserRole = UserRole.VIEWER,
    site_id: str | None = None,
    assigned_machine_ids: list[str] | None = None,
) -> str:
    """Convenience function to create an access token.
    
    Args:
        user_id: User identifier.
        username: Username.
        role: User role.
        site_id: Site for plant_manager scoping (optional).
        assigned_machine_ids: Pre-loaded machine IDs for technician (optional).
    
    Returns:
        JWT access token string.
    """
    return get_auth_service().create_access_token(
        user_id, username, role,
        site_id=site_id,
        assigned_machine_ids=assigned_machine_ids,
    )


def verify_token(token: str, expected_type: TokenType | None = None) -> TokenData:
    """Convenience function to verify a token.
    
    Args:
        token: JWT token string.
        expected_type: Expected token type (optional).
    
    Returns:
        Decoded TokenData.
    
    Raises:
        InvalidTokenError: If token is invalid.
    """
    return get_auth_service().verify_token(token, expected_type)
