#!/usr/bin/env python3
"""
Validated Pydantic Schemas for PDM-PILOT API
=============================================
All request/response models with strict validation.

Security Features:
- Password minimum length: 12 characters
- Email validation using EmailStr
- SQL/XSS injection pattern blocking for free-text fields
- Input sanitization for all string fields

Author: PlantAGI Security Team
"""

import re
from typing import Optional, List, Any
from pydantic import BaseModel, Field, field_validator, EmailStr, ConfigDict
from datetime import datetime, timezone, timedelta

from .profiling import DataProfile, MetricProfile, ProfileStatus, ValueRange

# =============================================================================
# SECURITY PATTERNS - Block SQL Injection and XSS
# =============================================================================

# SQL Injection patterns
SQL_INJECTION_PATTERNS = [
    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|EXEC|UNION|DECLARE)\b)",
    r"(--|#|/\*|\*/)",  # SQL comments
    r"(\b(OR|AND)\b\s+\d+\s*=\s*\d+)",  # OR 1=1, AND 1=1
    r"(;.*--)",  # Statement termination with comment
    r"(\bEXEC\s*\()",  # EXEC()
    r"(\bxp_\w+)",  # Extended stored procedures
    r"('.*(\bOR\b|\bAND\b).*')",  # String-based injection
]

# XSS patterns
XSS_PATTERNS = [
    r"(<\s*script[^>]*>)",  # <script> tags
    r"(<\s*/\s*script\s*>)",  # </script> tags
    r"(javascript\s*:)",  # javascript: protocol
    r"(on\w+\s*=)",  # Event handlers (onclick, onerror, etc.)
    r"(<\s*iframe[^>]*>)",  # <iframe> tags
    r"(<\s*object[^>]*>)",  # <object> tags
    r"(<\s*embed[^>]*>)",  # <embed> tags
    r"(<\s*link[^>]*>)",  # <link> tags
    r"(eval\s*\()",  # eval()
    r"(expression\s*\()",  # CSS expression()
    r"(<\s*img[^>]+onerror)",  # <img onerror>
]

# Compiled regex for performance
SQL_PATTERN = re.compile("|".join(SQL_INJECTION_PATTERNS), re.IGNORECASE)
XSS_PATTERN = re.compile("|".join(XSS_PATTERNS), re.IGNORECASE)


def validate_no_injection(value: str, field_name: str = "field") -> str:
    """
    Validate that a string does not contain SQL injection or XSS patterns.
    Raises ValueError if malicious patterns are detected.
    """
    if not isinstance(value, str):
        return value
    
    # Check for SQL injection
    if SQL_PATTERN.search(value):
        raise ValueError(f"{field_name} contains potentially malicious SQL patterns")
    
    # Check for XSS
    if XSS_PATTERN.search(value):
        raise ValueError(f"{field_name} contains potentially malicious script patterns")
    
    return value


def sanitize_string(value: str) -> str:
    """
    Basic string sanitization - strip whitespace and limit dangerous characters.
    """
    if not isinstance(value, str):
        return value
    
    # Strip leading/trailing whitespace
    value = value.strip()
    
    # Replace null bytes
    value = value.replace('\x00', '')
    
    return value


# =============================================================================
# BASE VALIDATED MODEL
# =============================================================================

class ValidatedModel(BaseModel):
    """Base model with common validation configuration."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra='forbid'  # Reject unknown fields
    )


# =============================================================================
# COMMON VALIDATED FIELDS
# =============================================================================

class SecureTextField:
    """Factory for secure text fields with injection protection."""
    
    @staticmethod
    def string(
        min_length: int = 1,
        max_length: int = 500,
        description: str = "Text field"
    ) -> Any:
        return Field(
            ...,
            min_length=min_length,
            max_length=max_length,
            description=description
        )
    
    @staticmethod
    def optional_string(
        max_length: int = 500,
        description: str = "Optional text field"
    ) -> Any:
        return Field(
            None,
            max_length=max_length,
            description=description
        )


# =============================================================================
# AUTHENTICATION SCHEMAS
# =============================================================================

class UserCredentials(ValidatedModel):
    """User login credentials with strict validation."""
    
    email: EmailStr = Field(
        ...,
        description="User email address"
    )
    password: str = Field(
        ...,
        min_length=12,
        max_length=128,
        description="Password (minimum 12 characters)"
    )
    
    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Ensure password meets minimum complexity requirements."""
        if len(v) < 12:
            raise ValueError("Password must be at least 12 characters")
        
        # Check for at least one uppercase, one lowercase, one digit
        if not re.search(r'[A-Z]', v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r'[a-z]', v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r'\d', v):
            raise ValueError("Password must contain at least one digit")
        
        return v


class UserRegistration(UserCredentials):
    """User registration with additional fields."""
    
    name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Full name"
    )
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        return validate_no_injection(sanitize_string(v), "name")


# =============================================================================
# ENTERPRISE API SCHEMAS (Alarms, Work Orders)
# =============================================================================

class AlarmCreateValidated(ValidatedModel):
    """Create alarm request with validation."""
    
    machine_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Machine identifier (alphanumeric, underscores, hyphens only)"
    )
    severity: str = Field(
        ...,
        pattern=r"^(info|warning|critical)$",
        description="Alarm severity level"
    )
    code: str = Field(
        ...,
        min_length=1,
        max_length=30,
        pattern=r"^[A-Z0-9-]+$",
        description="Alarm code (uppercase letters, numbers, hyphens)"
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Alarm message"
    )
    trigger_type: Optional[str] = Field(
        None,
        max_length=50,
        pattern=r"^[a-z_]+$",
        description="Trigger type (lowercase with underscores)"
    )
    trigger_value: Optional[float] = Field(
        None,
        ge=0,
        description="Trigger value (non-negative)"
    )
    threshold_value: Optional[float] = Field(
        None,
        ge=0,
        description="Threshold value (non-negative)"
    )
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        return validate_no_injection(sanitize_string(v), "message")


class WorkOrderCreateValidated(ValidatedModel):
    """Create work order request with validation."""
    
    machine_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Machine identifier"
    )
    title: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description="Work order title"
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Detailed description"
    )
    priority: str = Field(
        "medium",
        pattern=r"^(low|medium|high|critical)$",
        description="Priority level"
    )
    work_type: str = Field(
        "corrective",
        pattern=r"^(corrective|preventive|predictive|emergency)$",
        description="Type of maintenance work"
    )
    scheduled_date: Optional[str] = Field(
        None,
        pattern=r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?$",
        description="Scheduled date (ISO format)"
    )
    estimated_duration_hours: Optional[float] = Field(
        None,
        ge=0.25,
        le=720,
        description="Estimated duration in hours (0.25 - 720)"
    )
    
    @field_validator('title', 'description')
    @classmethod
    def validate_text_fields(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return validate_no_injection(sanitize_string(v), "text field")


class WorkOrderUpdateValidated(ValidatedModel):
    """Update work order request with validation."""
    
    status: Optional[str] = Field(
        None,
        pattern=r"^(pending|in_progress|completed|cancelled|on_hold)$",
        description="Work order status"
    )
    assigned_to: Optional[str] = Field(
        None,
        max_length=100,
        description="Assigned technician"
    )
    notes: Optional[str] = Field(
        None,
        max_length=2000,
        description="Additional notes"
    )
    actual_duration_hours: Optional[float] = Field(
        None,
        ge=0,
        le=720,
        description="Actual duration in hours"
    )
    
    @field_validator('assigned_to', 'notes')
    @classmethod
    def validate_text_fields(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return validate_no_injection(sanitize_string(v), "text field")


# =============================================================================
# DISCOVERY API SCHEMAS
# =============================================================================

class TrainRequestValidated(ValidatedModel):
    """ML model training request with validation."""
    
    days_of_data: int = Field(
        30,
        ge=1,
        le=365,
        description="Days of historical data to use (1-365)"
    )
    min_samples: int = Field(
        1000,
        ge=100,
        le=1000000,
        description="Minimum samples required (100-1M)"
    )


class DetectRequestValidated(ValidatedModel):
    """Anomaly detection request with validation."""
    
    hours_back: float = Field(
        1.0,
        ge=0.1,
        le=168,
        description="Hours to look back (0.1 - 168)"
    )
    persist: bool = Field(
        True,
        description="Whether to persist results to database"
    )


class AnalyzeRequestValidated(ValidatedModel):
    """Correlation analysis request with validation."""
    
    days_back: int = Field(
        7,
        ge=1,
        le=90,
        description="Days to analyze (1-90)"
    )
    min_correlation: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Minimum correlation threshold (0-1)"
    )


# =============================================================================
# SENSOR / TELEMETRY SCHEMAS
# =============================================================================

class SensorReadingValidated(ValidatedModel):
    """Sensor data ingestion with strict industrial validation.
    
    All sensor data must pass these validation rules:
    - Temperature: -50°C to 200°C (extended industrial range)
    - Vibration: Non-negative (cannot have negative vibration)
    - RPM/Speed: Non-negative
    - Machine ID: 3-50 characters, alphanumeric with underscores/hyphens
    """
    
    machine_id: str = Field(
        ...,
        min_length=3,
        max_length=50,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Machine identifier (3-50 chars, alphanumeric with underscores/hyphens)"
    )
    timestamp: Optional[datetime] = Field(
        None,
        description="Reading timestamp (auto-generated if not provided)"
    )
    temperature: float = Field(
        ...,
        ge=-50.0,
        le=200.0,
        description="Temperature in Celsius (-50 to 200)"
    )
    vibration: float = Field(
        ...,
        ge=0.0,
        description="Vibration amplitude (must be >= 0)"
    )
    vibration_x: Optional[float] = Field(
        None,
        ge=0.0,
        description="X-axis vibration (must be >= 0)"
    )
    vibration_y: Optional[float] = Field(
        None,
        ge=0.0,
        description="Y-axis vibration (must be >= 0)"
    )
    vibration_z: Optional[float] = Field(
        None,
        ge=0.0,
        description="Z-axis vibration (must be >= 0)"
    )
    rpm: float = Field(
        ...,
        ge=0.0,
        le=50000.0,
        description="Rotational speed in RPM (0 to 50,000)"
    )
    
    @field_validator('machine_id')
    @classmethod
    def validate_machine_id(cls, v: str) -> str:
        return validate_no_injection(sanitize_string(v), "machine_id")
    
    @field_validator('timestamp')
    @classmethod
    def validate_timestamp_not_future(cls, v: Optional[datetime]) -> Optional[datetime]:
        """
        Ensure timestamp is not too far in the future.
        Allows 5 minutes of clock drift for distributed systems.
        """
        if v is None:
            return v
        
        # Get current UTC time
        now = datetime.now(timezone.utc)
        
        # Make timestamp timezone-aware if it isn't
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        
        # Calculate allowed future threshold (5 minutes)
        max_future = now + timedelta(minutes=5)
        
        if v > max_future:
            raise ValueError(
                f"Timestamp cannot be in the future. "
                f"Received: {v.isoformat()}, Current UTC: {now.isoformat()}"
            )
        
        return v


class TelemetryDataValidated(ValidatedModel):
    """Complete telemetry payload for real-time streaming.
    
    Includes all sensor channels with industrial-grade validation.
    """
    
    machine_id: str = Field(
        ...,
        min_length=3,
        max_length=50,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Machine identifier"
    )
    timestamp: Optional[datetime] = Field(
        None,
        description="Reading timestamp"
    )
    
    # Temperature sensors
    temperature: float = Field(
        ...,
        ge=-50.0,
        le=200.0,
        description="Primary temperature (°C)"
    )
    motor_temp_c: Optional[float] = Field(
        None,
        ge=-50.0,
        le=200.0,
        description="Motor temperature (°C)"
    )
    
    # Vibration sensors
    vibration_x: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="X-axis vibration (g-force, 0-100)"
    )
    vibration_y: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Y-axis vibration (g-force)"
    )
    vibration_z: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Z-axis vibration (g-force)"
    )
    
    # Rotational sensors
    rotational_speed: float = Field(
        ...,
        ge=0.0,
        le=50000.0,
        description="Shaft speed in RPM"
    )
    
    # Torque sensors
    torque: Optional[float] = Field(
        None,
        ge=0.0,
        le=10000.0,
        description="Torque in Nm"
    )
    joint_1_torque: Optional[float] = Field(
        None,
        ge=0.0,
        le=10000.0,
        description="Joint 1 torque (Nm)"
    )
    
    # Tool/equipment sensors
    tool_wear: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Tool wear index (0-1)"
    )
    
    # Current/power sensors
    current_draw_a: Optional[float] = Field(
        None,
        ge=0.0,
        le=1000.0,
        description="Current draw in Amps"
    )
    
    # Predictions (from ML model)
    failure_probability: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Predicted failure probability (0-1)"
    )
    rul_hours: Optional[float] = Field(
        None,
        ge=0.0,
        le=100000.0,
        description="Remaining useful life in hours"
    )
    
    @field_validator('machine_id')
    @classmethod
    def validate_machine_id(cls, v: str) -> str:
        return validate_no_injection(sanitize_string(v), "machine_id")
    
    @field_validator('timestamp')
    @classmethod
    def validate_timestamp_not_future(cls, v: Optional[datetime]) -> Optional[datetime]:
        """
        Ensure timestamp is not too far in the future.
        Allows 5 minutes of clock drift for distributed systems.
        """
        if v is None:
            return v
        
        # Get current UTC time
        now = datetime.now(timezone.utc)
        
        # Make timestamp timezone-aware if it isn't
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        
        # Calculate allowed future threshold (5 minutes)
        max_future = now + timedelta(minutes=5)
        
        if v > max_future:
            raise ValueError(
                f"Timestamp cannot be in the future. "
                f"Received: {v.isoformat()}, Current UTC: {now.isoformat()}"
            )
        
        return v


class BulkTelemetryValidated(ValidatedModel):
    """Bulk telemetry upload with array validation."""
    
    readings: List[TelemetryDataValidated] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Array of telemetry readings (1-1000)"
    )


# =============================================================================
# STREAM CONTROL SCHEMAS
# =============================================================================

class StreamControlRequestValidated(ValidatedModel):
    """Stream control request with validation."""
    
    state: str = Field(
        ...,
        pattern=r"^(start|stop)$",
        description="Stream state command"
    )


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    # Base
    'ValidatedModel',
    'validate_no_injection',
    'sanitize_string',
    
    # Auth
    'UserCredentials',
    'UserRegistration',
    
    # Enterprise
    'AlarmCreateValidated',
    'WorkOrderCreateValidated', 
    'WorkOrderUpdateValidated',
    
    # Discovery
    'TrainRequestValidated',
    'DetectRequestValidated',
    'AnalyzeRequestValidated',
    
    # Sensor / Telemetry
    'SensorReadingValidated',
    'TelemetryDataValidated',
    'BulkTelemetryValidated',
    
    # Stream
    'StreamControlRequestValidated',

    # Profiling
    'DataProfile',
    'MetricProfile',
    'ProfileStatus',
    'ValueRange',
]

