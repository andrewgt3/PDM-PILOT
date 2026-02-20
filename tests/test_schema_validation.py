#!/usr/bin/env python3
"""
Validation Test Suite for Pydantic Schemas
==========================================
Tests that malicious payloads are correctly rejected with 422 Unprocessable Entity.

Run with: pytest tests/test_schema_validation.py -v

Author: PlantAGI Security Team
"""

import pytest
from pydantic import ValidationError

# Import schemas
from schemas import (
    UserCredentials,
    UserRegistration,
    AlarmCreateValidated,
    WorkOrderCreateValidated,
    WorkOrderUpdateValidated,
    TrainRequestValidated,
    DetectRequestValidated,
    AnalyzeRequestValidated,
    StreamControlRequestValidated,
    validate_no_injection,
)


# =============================================================================
# SQL INJECTION TEST PAYLOADS
# =============================================================================

SQL_INJECTION_PAYLOADS = [
    "'; DROP TABLE users; --",
    "1; SELECT * FROM users",
    "' OR '1'='1",
    "' OR 1=1 --",
    "admin'--",
    "1' AND '1'='1",
    "1 UNION SELECT * FROM passwords",
    "'; EXEC xp_cmdshell('dir'); --",
    "1; UPDATE users SET admin=1 WHERE id=1",
    "'; DELETE FROM alarms; --",
    "/* comment */ SELECT * FROM users",
    "1'; TRUNCATE TABLE data; --",
    "admin' OR '1'='1'/*",
    "1 AND 1=1",
    "1 OR 1=1",
]


# =============================================================================
# XSS TEST PAYLOADS
# =============================================================================

XSS_PAYLOADS = [
    "<script>alert('XSS')</script>",
    "<img src=x onerror=alert('XSS')>",
    "<svg onload=alert('XSS')>",
    "javascript:alert('XSS')",
    "<iframe src='evil.com'></iframe>",
    "<body onload=alert('XSS')>",
    "<input onfocus=alert('XSS') autofocus>",
    "<object data='javascript:alert(1)'>",
    "<embed src='javascript:alert(1)'>",
    "onclick=alert('XSS')",
    "<link rel='stylesheet' href='evil.css'>",
    "<script src='evil.js'></script>",
    "eval('alert(1)')",
    "expression(alert('XSS'))",
    "<img src=x onerror='alert(1)'>",
]


# =============================================================================
# PASSWORD VALIDATION TESTS
# =============================================================================

class TestPasswordValidation:
    """Test password length and complexity requirements."""
    
    def test_password_too_short_rejected(self):
        """Passwords under 12 characters should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            UserCredentials(
                email="test@example.com",
                password="Short1"  # Only 6 characters
            )
        
        assert "at least 12 characters" in str(exc_info.value).lower() or \
               "min_length" in str(exc_info.value).lower()
    
    def test_password_11_chars_rejected(self):
        """Password with exactly 11 characters should be rejected."""
        with pytest.raises(ValidationError):
            UserCredentials(
                email="test@example.com",
                password="Abcdefgh123"  # 11 characters
            )
    
    def test_password_12_chars_accepted(self):
        """Password with exactly 12 characters should be accepted."""
        creds = UserCredentials(
            email="test@example.com",
            password="Abcdefgh1234"  # 12 characters with complexity
        )
        assert len(creds.password) == 12
    
    def test_password_missing_uppercase_rejected(self):
        """Password without uppercase should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            UserCredentials(
                email="test@example.com",
                password="abcdefgh1234"  # No uppercase
            )
        assert "uppercase" in str(exc_info.value).lower()
    
    def test_password_missing_lowercase_rejected(self):
        """Password without lowercase should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            UserCredentials(
                email="test@example.com",
                password="ABCDEFGH1234"  # No lowercase
            )
        assert "lowercase" in str(exc_info.value).lower()
    
    def test_password_missing_digit_rejected(self):
        """Password without digit should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            UserCredentials(
                email="test@example.com",
                password="Abcdefghijkl"  # No digit
            )
        assert "digit" in str(exc_info.value).lower()
    
    def test_valid_password_accepted(self):
        """Valid password meeting all requirements should be accepted."""
        creds = UserCredentials(
            email="test@example.com",
            password="SecurePass123!"
        )
        assert creds.password == "SecurePass123!"


# =============================================================================
# EMAIL VALIDATION TESTS
# =============================================================================

class TestEmailValidation:
    """Test email format validation."""
    
    def test_invalid_email_rejected(self):
        """Invalid email format should be rejected."""
        with pytest.raises(ValidationError):
            UserCredentials(
                email="not-an-email",
                password="SecurePass123!"
            )
    
    def test_email_without_at_rejected(self):
        """Email without @ should be rejected."""
        with pytest.raises(ValidationError):
            UserCredentials(
                email="testexample.com",
                password="SecurePass123!"
            )
    
    def test_email_without_domain_rejected(self):
        """Email without domain should be rejected."""
        with pytest.raises(ValidationError):
            UserCredentials(
                email="test@",
                password="SecurePass123!"
            )
    
    def test_valid_email_accepted(self):
        """Valid email should be accepted."""
        creds = UserCredentials(
            email="valid@example.com",
            password="SecurePass123!"
        )
        assert creds.email == "valid@example.com"


# =============================================================================
# SQL INJECTION TESTS
# =============================================================================

class TestSQLInjectionPrevention:
    """Test that SQL injection payloads are blocked."""
    
    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_sql_injection_in_alarm_message_rejected(self, payload):
        """SQL injection in alarm message should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            AlarmCreateValidated(
                machine_id="WB-001",
                severity="warning",
                code="ALM-001",
                message=payload
            )
        
        # Check that it was caught by our validation
        error_str = str(exc_info.value).lower()
        assert "malicious" in error_str or "pattern" in error_str or "sql" in error_str
    
    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_sql_injection_in_work_order_title_rejected(self, payload):
        """SQL injection in work order title should be rejected."""
        with pytest.raises(ValidationError):
            WorkOrderCreateValidated(
                machine_id="WB-001",
                title=payload
            )
    
    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_sql_injection_in_work_order_notes_rejected(self, payload):
        """SQL injection in work order notes should be rejected."""
        with pytest.raises(ValidationError):
            WorkOrderUpdateValidated(
                notes=payload
            )
    
    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_sql_injection_in_user_name_rejected(self, payload):
        """SQL injection in user registration name should be rejected."""
        with pytest.raises(ValidationError):
            UserRegistration(
                email="test@example.com",
                password="SecurePass123!",
                name=payload
            )


# =============================================================================
# XSS PREVENTION TESTS
# =============================================================================

class TestXSSPrevention:
    """Test that XSS payloads are blocked."""
    
    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_xss_in_alarm_message_rejected(self, payload):
        """XSS in alarm message should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            AlarmCreateValidated(
                machine_id="WB-001",
                severity="warning",
                code="ALM-001",
                message=payload
            )
        
        error_str = str(exc_info.value).lower()
        assert "malicious" in error_str or "script" in error_str or "pattern" in error_str
    
    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_xss_in_work_order_description_rejected(self, payload):
        """XSS in work order description should be rejected."""
        with pytest.raises(ValidationError):
            WorkOrderCreateValidated(
                machine_id="WB-001",
                title="Valid Title Here",
                description=payload
            )
    
    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_xss_in_user_name_rejected(self, payload):
        """XSS in user registration name should be rejected."""
        with pytest.raises(ValidationError):
            UserRegistration(
                email="test@example.com",
                password="SecurePass123!",
                name=payload
            )


# =============================================================================
# FIELD PATTERN VALIDATION TESTS
# =============================================================================

class TestFieldPatternValidation:
    """Test that field patterns are enforced."""
    
    def test_machine_id_invalid_chars_rejected(self):
        """Machine ID with invalid characters should be rejected."""
        with pytest.raises(ValidationError):
            AlarmCreateValidated(
                machine_id="WB@001!",  # Invalid chars
                severity="warning",
                code="ALM-001",
                message="Test message"
            )
    
    def test_machine_id_valid_accepted(self):
        """Valid machine ID should be accepted."""
        alarm = AlarmCreateValidated(
            machine_id="WB-001_test",
            severity="warning",
            code="ALM-001",
            message="Test alarm message"
        )
        assert alarm.machine_id == "WB-001_test"
    
    def test_severity_invalid_value_rejected(self):
        """Invalid severity value should be rejected."""
        with pytest.raises(ValidationError):
            AlarmCreateValidated(
                machine_id="WB-001",
                severity="extreme",  # Not in allowed values
                code="ALM-001",
                message="Test message"
            )
    
    def test_severity_valid_values_accepted(self):
        """Valid severity values should be accepted."""
        for severity in ["info", "warning", "critical"]:
            alarm = AlarmCreateValidated(
                machine_id="WB-001",
                severity=severity,
                code="ALM-001",
                message="Test message"
            )
            assert alarm.severity == severity
    
    def test_priority_invalid_value_rejected(self):
        """Invalid priority value should be rejected."""
        with pytest.raises(ValidationError):
            WorkOrderCreateValidated(
                machine_id="WB-001",
                title="Valid Work Order Title",
                priority="urgent"  # Not in allowed values
            )
    
    def test_stream_control_invalid_state_rejected(self):
        """Invalid stream state should be rejected."""
        with pytest.raises(ValidationError):
            StreamControlRequestValidated(state="pause")  # Not start/stop


# =============================================================================
# RANGE VALIDATION TESTS
# =============================================================================

class TestRangeValidation:
    """Test that numeric ranges are enforced."""
    
    def test_train_days_too_large_rejected(self):
        """Days of data over 365 should be rejected."""
        with pytest.raises(ValidationError):
            TrainRequestValidated(days_of_data=500)
    
    def test_train_days_negative_rejected(self):
        """Negative days should be rejected."""
        with pytest.raises(ValidationError):
            TrainRequestValidated(days_of_data=-5)
    
    def test_detect_hours_too_large_rejected(self):
        """Hours back over 168 should be rejected."""
        with pytest.raises(ValidationError):
            DetectRequestValidated(hours_back=200)
    
    def test_correlation_threshold_over_1_rejected(self):
        """Correlation threshold over 1.0 should be rejected."""
        with pytest.raises(ValidationError):
            AnalyzeRequestValidated(min_correlation=1.5)
    
    def test_estimated_duration_negative_rejected(self):
        """Negative duration should be rejected."""
        with pytest.raises(ValidationError):
            WorkOrderCreateValidated(
                machine_id="WB-001",
                title="Valid Title",
                estimated_duration_hours=-1
            )


# =============================================================================
# VALID INPUT TESTS (Ensure we don't over-block)
# =============================================================================

class TestValidInputAccepted:
    """Test that valid inputs are accepted (no false positives)."""
    
    def test_valid_alarm_accepted(self):
        """Valid alarm creation should be accepted."""
        alarm = AlarmCreateValidated(
            machine_id="WB-001",
            severity="warning",
            code="ALM-TEMP-HIGH",
            message="Temperature exceeded threshold at 85°C"
        )
        assert alarm.machine_id == "WB-001"
    
    def test_valid_work_order_accepted(self):
        """Valid work order creation should be accepted."""
        wo = WorkOrderCreateValidated(
            machine_id="HP-200",
            title="Scheduled Bearing Replacement",
            description="Replace main drive bearings as part of preventive maintenance.",
            priority="high",
            work_type="preventive",
            estimated_duration_hours=4.5
        )
        assert wo.title == "Scheduled Bearing Replacement"
    
    def test_valid_train_request_accepted(self):
        """Valid training request should be accepted."""
        req = TrainRequestValidated(
            days_of_data=30,
            min_samples=5000
        )
        assert req.days_of_data == 30
    
    def test_text_with_normal_punctuation_accepted(self):
        """Normal punctuation should not trigger false positives."""
        alarm = AlarmCreateValidated(
            machine_id="WB-001",
            severity="critical",
            code="ALM-001",
            message="Motor failure detected! Please check bearings (ID: 12345) - urgent action required."
        )
        assert "!" in alarm.message
        assert "(" in alarm.message


# =============================================================================
# API INTEGRATION TESTS (422 Response)
# =============================================================================

class TestAPIIntegration:
    """
    Integration tests that verify the API returns 422 for invalid payloads.
    These tests require a running API server.
    """
    
    @pytest.fixture
    def client(self):
        """Create test client with DB initialized so get_db works. Skip if FastAPI not available."""
        try:
            import asyncio
            from fastapi.testclient import TestClient
            from api_server import app
            from database import init_database
            # Initialize DB so get_db() does not raise "Database not initialized"
            try:
                asyncio.get_event_loop().run_until_complete(init_database())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(init_database())
            return TestClient(app)
        except ImportError:
            pytest.skip("FastAPI test client not available")
    
    def test_invalid_alarm_returns_422(self, client, auth_headers):
        """Invalid alarm payload should return 422."""
        response = client.post(
            "/api/enterprise/alarms",
            json={
                "machine_id": "WB@001!",  # Invalid chars
                "severity": "warning",
                "code": "ALM-001",
                "message": "Test"
            },
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_sql_injection_returns_422(self, client, auth_headers):
        """SQL injection attempt should return 422."""
        response = client.post(
            "/api/enterprise/alarms",
            json={
                "machine_id": "WB-001",
                "severity": "warning",
                "code": "ALM-001",
                "message": "'; DROP TABLE alarms; --"
            },
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_xss_returns_422(self, client, auth_headers):
        """XSS attempt should return 422."""
        response = client.post(
            "/api/enterprise/work-orders",
            json={
                "machine_id": "WB-001",
                "title": "<script>alert('XSS')</script>"
            },
            headers=auth_headers,
        )
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
