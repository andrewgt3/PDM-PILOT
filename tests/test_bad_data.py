#!/usr/bin/env python3
"""
Data Validation Test Suite
===========================
Tests that our Pydantic validation rules correctly reject bad data.

All tests should return HTTP 422 Unprocessable Entity.
If any test returns 200 OK, the validation is broken!

Usage:
    python tests/test_bad_data.py
    
    # Or with pytest:
    pytest tests/test_bad_data.py -v

Requires:
    - API server running on localhost:8000
    - httpx installed (pip install httpx)

Author: PlantAGI QA Team
"""

import sys
from datetime import datetime, timezone

import httpx
import pytest

# Configuration
API_BASE_URL = "http://localhost:8000"
TELEMETRY_ENDPOINT = f"{API_BASE_URL}/api/enterprise/telemetry"

# Test payloads - ALL of these should be REJECTED with 422
BAD_DATA_TESTS = [
    {
        "name": "🌋 The Magma Test",
        "description": "Temperature way too high (10,000°C vs max 200°C)",
        "payload": {
            "machine_id": "ROBOT_001",
            "temperature": 10000.0,  # ❌ Invalid: max is 200.0
            "vibration_x": 0.5,
            "rotational_speed": 1800.0
        },
        "expected_error": "temperature"
    },
    {
        "name": "⏰ The Time Traveler",
        "description": "Timestamp from the year 2099",
        "payload": {
            "machine_id": "ROBOT_001",
            "timestamp": "2099-01-01T00:00:00Z",  # ❌ Invalid: too far in future
            "temperature": 65.0,
            "vibration_x": 0.5,
            "rotational_speed": 1800.0
        },
        "expected_error": "timestamp"
    },
    {
        "name": "🔮 The Negative Physics",
        "description": "Negative vibration (physically impossible)",
        "payload": {
            "machine_id": "ROBOT_001",
            "temperature": 65.0,
            "vibration_x": -5.0,  # ❌ Invalid: must be >= 0
            "rotational_speed": 1800.0
        },
        "expected_error": "vibration_x"
    },
    {
        "name": "📭 The Empty Packet",
        "description": "Missing all required fields",
        "payload": {},  # ❌ Invalid: missing machine_id, temperature, vibration_x, rotational_speed
        "expected_error": "machine_id"
    },
    {
        "name": "🔤 The Invalid ID",
        "description": "Machine ID too short (min 3 chars)",
        "payload": {
            "machine_id": "AB",  # ❌ Invalid: min length is 3
            "temperature": 65.0,
            "vibration_x": 0.5,
            "rotational_speed": 1800.0
        },
        "expected_error": "machine_id"
    },
    {
        "name": "💉 The SQL Injection",
        "description": "Attempt SQL injection in machine_id",
        "payload": {
            "machine_id": "ROBOT'; DROP TABLE sensors;--",  # ❌ Invalid: contains SQL patterns
            "temperature": 65.0,
            "vibration_x": 0.5,
            "rotational_speed": 1800.0
        },
        "expected_error": "machine_id"
    },
    {
        "name": "❄️ The Absolute Zero Breach",
        "description": "Temperature below -50°C minimum",
        "payload": {
            "machine_id": "ROBOT_001",
            "temperature": -100.0,  # ❌ Invalid: min is -50.0
            "vibration_x": 0.5,
            "rotational_speed": 1800.0
        },
        "expected_error": "temperature"
    },
    {
        "name": "🚀 The Hyperspeed",
        "description": "RPM beyond maximum (999,999 vs max 50,000)",
        "payload": {
            "machine_id": "ROBOT_001",
            "temperature": 65.0,
            "vibration_x": 0.5,
            "rotational_speed": 999999.0  # ❌ Invalid: max is 50,000
        },
        "expected_error": "rotational_speed"
    },
]


def run_validation_tests():
    """Run all bad data validation tests."""
    print("=" * 70)
    print("DATA VALIDATION TEST SUITE")
    print("=" * 70)
    print(f"Target: {TELEMETRY_ENDPOINT}")
    print(f"Expected Result: All tests should return HTTP 422")
    print("=" * 70)
    print()
    
    passed = 0
    failed = 0
    errors = 0
    
    with httpx.Client(timeout=10.0) as client:
        for i, test in enumerate(BAD_DATA_TESTS, 1):
            print(f"[{i}/{len(BAD_DATA_TESTS)}] {test['name']}")
            print(f"    Description: {test['description']}")
            
            try:
                response = client.post(
                    TELEMETRY_ENDPOINT,
                    json=test['payload'],
                    headers={"Content-Type": "application/json"}
                )
                
                status = response.status_code
                
                if status == 422:
                    print(f"    ✅ Status: {status} (Validation rejected bad data)")
                    
                    # Check if the expected field was flagged
                    try:
                        detail = response.json().get('detail', [])
                        fields_flagged = [err.get('loc', ['', ''])[-1] for err in detail]
                        if test['expected_error'] in fields_flagged:
                            print(f"    ✅ Correct field flagged: {test['expected_error']}")
                        else:
                            print(f"    ⚠️  Expected '{test['expected_error']}', got: {fields_flagged}")
                    except:
                        pass
                    
                    passed += 1
                    
                elif status == 200:
                    print(f"    ❌ TEST FAILED: Bad data was accepted!")
                    print(f"       Status: {status}")
                    print(f"       Response: {response.text[:200]}")
                    failed += 1
                    
                else:
                    print(f"    ⚠️  Unexpected status: {status}")
                    print(f"       Response: {response.text[:200]}")
                    failed += 1
                    
            except httpx.ConnectError:
                print(f"    ⚠️  CONNECTION ERROR: Is the API server running?")
                errors += 1
            except Exception as e:
                print(f"    ⚠️  Error: {e}")
                errors += 1
            
            print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Tests Passed (422):  {passed}")
    print(f"  Tests Failed (200):  {failed}")
    print(f"  Connection Errors:   {errors}")
    print()
    
    if failed > 0:
        print("❌ VALIDATION IS BROKEN! Bad data was accepted.")
        return 1
    elif errors > 0:
        print("⚠️  Could not complete all tests (connection errors)")
        print("   Ensure API is running: uvicorn api_server:app --reload")
        return 2
    else:
        print("✅ ALL VALIDATION RULES ARE WORKING CORRECTLY!")
        return 0


# Pytest-compatible test functions (use conftest client so no live server required)
TELEMETRY_PATH = "/api/enterprise/telemetry"


@pytest.mark.anyio
async def test_magma_temperature(client, auth_headers):
    """Test that extremely high temperature is rejected."""
    response = await client.post(TELEMETRY_PATH, json={
        "machine_id": "ROBOT_001",
        "temperature": 10000.0,
        "vibration_x": 0.5,
        "rotational_speed": 1800.0
    }, headers=auth_headers)
    assert response.status_code == 422


@pytest.mark.anyio
async def test_future_timestamp(client, auth_headers):
    """Test that far-future timestamp is rejected."""
    response = await client.post(TELEMETRY_PATH, json={
        "machine_id": "ROBOT_001",
        "timestamp": "2099-01-01T00:00:00Z",
        "temperature": 65.0,
        "vibration_x": 0.5,
        "rotational_speed": 1800.0
    }, headers=auth_headers)
    assert response.status_code == 422


@pytest.mark.anyio
async def test_negative_vibration(client, auth_headers):
    """Test that negative vibration is rejected."""
    response = await client.post(TELEMETRY_PATH, json={
        "machine_id": "ROBOT_001",
        "temperature": 65.0,
        "vibration_x": -5.0,
        "rotational_speed": 1800.0
    }, headers=auth_headers)
    assert response.status_code == 422


@pytest.mark.anyio
async def test_empty_payload(client, auth_headers):
    """Test that empty payload is rejected."""
    response = await client.post(TELEMETRY_PATH, json={}, headers=auth_headers)
    assert response.status_code == 422


@pytest.mark.anyio
async def test_short_machine_id(client, auth_headers):
    """Test that too-short machine_id is rejected."""
    response = await client.post(TELEMETRY_PATH, json={
        "machine_id": "AB",
        "temperature": 65.0,
        "vibration_x": 0.5,
        "rotational_speed": 1800.0
    }, headers=auth_headers)
    assert response.status_code == 422


if __name__ == "__main__":
    exit_code = run_validation_tests()
    sys.exit(exit_code)
