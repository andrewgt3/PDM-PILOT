#!/usr/bin/env python3
"""
Authentication Utilities for PDM-PILOT API
==========================================
Provides secure password hashing and JWT token management.

Security Features:
- Bcrypt password hashing (industry standard)
- JWT tokens with configurable expiration
- Secret key loaded from environment variables

Usage:
    from auth_utils import get_password_hash, verify_password, create_access_token
    
    # Hash a password for storage
    hashed = get_password_hash("user_password")
    
    # Verify a password
    is_valid = verify_password("user_password", hashed)
    
    # Create a JWT token
    token = create_access_token({"sub": "user@example.com"})

Author: PlantAGI Security Team
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import bcrypt
from jose import jwt, JWTError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Secret key for JWT signing - MUST be set in production!
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    # Generate a warning but use a default for development
    import warnings
    warnings.warn(
        "JWT_SECRET_KEY not set in environment! Using insecure default. "
        "Set JWT_SECRET_KEY in your .env file for production.",
        UserWarning
    )
    SECRET_KEY = "INSECURE_DEFAULT_KEY_CHANGE_ME_IN_PRODUCTION"

# JWT Configuration
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))


# =============================================================================
# PASSWORD FUNCTIONS
# =============================================================================

def get_password_hash(password: str) -> str:
    """
    Hash a plain-text password using bcrypt.
    
    Args:
        password: The plain-text password to hash
        
    Returns:
        The bcrypt hashed password string
        
    Example:
        >>> hashed = get_password_hash("my_secure_password")
        >>> print(hashed)
        '$2b$12$...'
    """
    # Bcrypt has a 72-byte limit, truncate if necessary
    password_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain-text password against a bcrypt hash.
    
    Args:
        plain_password: The plain-text password to verify
        hashed_password: The bcrypt hash to compare against
        
    Returns:
        True if the password matches, False otherwise
        
    Example:
        >>> hashed = get_password_hash("my_password")
        >>> verify_password("my_password", hashed)
        True
        >>> verify_password("wrong_password", hashed)
        False
    """
    try:
        # Bcrypt has a 72-byte limit
        password_bytes = plain_password.encode('utf-8')[:72]
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception:
        return False


# =============================================================================
# JWT TOKEN FUNCTIONS
# =============================================================================

def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Dictionary of claims to include in the token.
              Should include "sub" (subject) for user identification.
        expires_delta: Optional custom expiration time.
                      Defaults to ACCESS_TOKEN_EXPIRE_MINUTES.
    
    Returns:
        Encoded JWT token string
        
    Example:
        >>> token = create_access_token({"sub": "user@example.com", "role": "admin"})
        >>> print(token)
        'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...'
    """
    to_encode = data.copy()
    
    # Set expiration time
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # Add standard JWT claims
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),  # Issued at
        "type": "access"
    })
    
    # Encode and sign the token
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and validate a JWT access token.
    
    Args:
        token: The JWT token string to decode
        
    Returns:
        Dictionary of token claims if valid, None if invalid/expired
        
    Example:
        >>> token = create_access_token({"sub": "user@example.com"})
        >>> payload = decode_access_token(token)
        >>> print(payload["sub"])
        'user@example.com'
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def create_refresh_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT refresh token (longer-lived than access token).
    
    Args:
        data: Dictionary of claims to include in the token
        expires_delta: Optional custom expiration (default: 7 days)
    
    Returns:
        Encoded JWT refresh token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(days=7)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "refresh"
    })
    
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_token_expired(token: str) -> bool:
    """
    Check if a token is expired without raising an exception.
    
    Args:
        token: JWT token to check
        
    Returns:
        True if expired or invalid, False if still valid
    """
    payload = decode_access_token(token)
    return payload is None


# =============================================================================
# TESTING / DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AUTH UTILITIES - DEMO")
    print("=" * 60)
    
    # Test password hashing
    print("\n[1] Password Hashing Test")
    test_password = "SecurePassword123!"
    hashed = get_password_hash(test_password)
    print(f"  Original: {test_password}")
    print(f"  Hashed:   {hashed[:50]}...")
    print(f"  Verify (correct):  {verify_password(test_password, hashed)}")
    print(f"  Verify (wrong):    {verify_password('WrongPassword', hashed)}")
    
    # Test JWT token
    print("\n[2] JWT Token Test")
    token_data = {"sub": "admin@plantagi.com", "role": "admin"}
    token = create_access_token(token_data)
    print(f"  Token: {token[:50]}...")
    
    decoded = decode_access_token(token)
    print(f"  Decoded subject: {decoded.get('sub')}")
    print(f"  Decoded role: {decoded.get('role')}")
    print(f"  Expires: {decoded.get('exp')}")
    
    print("\n✅ All auth utilities working correctly!")
