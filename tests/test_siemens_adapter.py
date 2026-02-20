"""
Unit tests for Siemens S7 adapter: byte conversion logic for each DB data type.

Mocks the snap7 client; tests _parse_db_bytes and build_payload with hand-built
big-endian bytearrays (S7 byte order).
"""

import struct
from unittest.mock import patch

import pytest

# Module under test (parse and build helpers + class). Skip entire module if snap7/pkg_resources unavailable.
try:
    from siemens_s7_adapter import (
        DB_READ_SIZE,
        SiemensS7Adapter,
        _parse_db_bytes,
        build_payload,
    )
except ImportError as e:
    pytest.skip(
        f"Cannot import siemens_s7_adapter (missing dependency: {e}). Install setuptools and python-snap7.",
        allow_module_level=True,
    )


def _pack_real(value: float) -> bytes:
    """S7 REAL = 4 bytes big-endian IEEE 754."""
    return struct.pack(">f", value)


def _pack_dword(value: int) -> bytes:
    """S7 DWORD = 4 bytes big-endian unsigned."""
    return struct.pack(">I", value & 0xFFFFFFFF)


def _pack_byte(bits: int) -> bytes:
    """Single byte with bit 0 = photo_eye_trigger, bit 1 = fault_active (LSB = bit 0)."""
    return bytes([bits & 0xFF])


def test_parse_db_bytes_motor_current():
    """REAL at DBD[0]: motor_current (Amps)."""
    # motor_current=12.5, rest zeros
    data = bytearray(_pack_real(12.5) + _pack_dword(0) + _pack_byte(0) + _pack_byte(0) + _pack_real(0.0) + _pack_real(0.0) + _pack_real(0.0))
    data.extend(bytearray(22 - len(data)))  # pad to 22
    result = _parse_db_bytes(data)
    assert result["motor_current"] == pytest.approx(12.5)


def test_parse_db_bytes_cycle_timer_ms():
    """DWORD at DBD[4]: cycle_timer_ms."""
    data = bytearray(22)
    data[0:4] = _pack_real(0.0)
    data[4:8] = _pack_dword(12345)
    data[8:9] = _pack_byte(0)
    data[10:14] = _pack_real(0.0)
    data[14:18] = _pack_real(0.0)
    data[18:22] = _pack_real(0.0)
    result = _parse_db_bytes(data)
    assert result["cycle_timer_ms"] == 12345


def test_parse_db_bytes_photo_eye_and_fault():
    """DBX[8.0] photo_eye_trigger, DBX[8.1] fault_active."""
    data = bytearray(22)
    data[0:4] = _pack_real(0.0)
    data[4:8] = _pack_dword(0)
    # Byte 8: bit 0 = 1 (photo_eye), bit 1 = 1 (fault)
    data[8] = 0x03
    data[10:14] = _pack_real(0.0)
    data[14:18] = _pack_real(0.0)
    data[18:22] = _pack_real(0.0)
    result = _parse_db_bytes(data)
    assert result["photo_eye_trigger"] is True
    assert result["fault_active"] is True

    data[8] = 0x00
    result = _parse_db_bytes(data)
    assert result["photo_eye_trigger"] is False
    assert result["fault_active"] is False

    data[8] = 0x01
    result = _parse_db_bytes(data)
    assert result["photo_eye_trigger"] is True
    assert result["fault_active"] is False


def test_parse_db_bytes_conveyor_speed_and_temperature_and_vibration():
    """REAL at DBD[10], DBD[14], DBD[18]."""
    data = bytearray(22)
    data[0:4] = _pack_real(0.0)
    data[4:8] = _pack_dword(0)
    data[8:10] = bytes([0, 0])
    data[10:14] = _pack_real(0.5)   # conveyor_speed m/s
    data[14:18] = _pack_real(55.25) # motor_temperature_c
    data[18:22] = _pack_real(1.25)  # vibration_rms mm/s
    result = _parse_db_bytes(data)
    assert result["conveyor_speed"] == pytest.approx(0.5)
    assert result["motor_temperature_c"] == pytest.approx(55.25)
    assert result["vibration_rms"] == pytest.approx(1.25)


def test_parse_db_bytes_full_roundtrip():
    """All fields set; verify full parse."""
    data = bytearray(22)
    data[0:4] = _pack_real(2.5)
    data[4:8] = _pack_dword(1000)
    data[8] = 0x01  # photo_eye True, fault False
    data[10:14] = _pack_real(0.3)
    data[14:18] = _pack_real(42.0)
    data[18:22] = _pack_real(0.8)
    result = _parse_db_bytes(data)
    assert result["motor_current"] == pytest.approx(2.5)
    assert result["cycle_timer_ms"] == 1000
    assert result["photo_eye_trigger"] is True
    assert result["fault_active"] is False
    assert result["conveyor_speed"] == pytest.approx(0.3)
    assert result["motor_temperature_c"] == pytest.approx(42.0)
    assert result["vibration_rms"] == pytest.approx(0.8)


def test_parse_db_bytes_insufficient_data_raises():
    """Too few bytes raises ValueError."""
    with pytest.raises(ValueError, match="Need at least"):
        _parse_db_bytes(bytearray(10))
    with pytest.raises(ValueError, match="Need at least"):
        _parse_db_bytes(None)


def test_build_payload_structure():
    """Payload has machine_id, timestamp, source, telemetry nested."""
    telemetry = {
        "motor_current": 1.0,
        "cycle_timer_ms": 500,
        "photo_eye_trigger": False,
        "fault_active": False,
        "conveyor_speed": 0.2,
        "motor_temperature_c": 40.0,
        "vibration_rms": 0.1,
    }
    payload = build_payload(telemetry, machine_id="SIEMENS-PLC-001")
    assert payload["machine_id"] == "SIEMENS-PLC-001"
    assert "timestamp" in payload
    assert payload["source"] == "siemens_s7"
    assert payload["telemetry"] == telemetry


def test_adapter_test_read_returns_none_when_connect_fails():
    """test_read() returns None when PLC connection fails."""
    with patch.object(SiemensS7Adapter, "_connect", return_value=False):
        adapter = SiemensS7Adapter(plc_ip="192.0.2.1")
        assert adapter.test_read() is None


def test_adapter_test_read_returns_payload_when_mock_read_succeeds():
    """test_read() returns payload dict when connect and read succeed (mocked)."""
    raw = bytearray(22)
    raw[0:4] = _pack_real(3.0)
    raw[4:8] = _pack_dword(2000)
    raw[8] = 0
    raw[10:14] = _pack_real(0.4)
    raw[14:18] = _pack_real(50.0)
    raw[18:22] = _pack_real(0.5)

    with patch.object(SiemensS7Adapter, "_connect", return_value=True), \
         patch.object(SiemensS7Adapter, "_read_db", return_value=raw), \
         patch.object(SiemensS7Adapter, "_disconnect"):
        adapter = SiemensS7Adapter(plc_ip="127.0.0.1")
        result = adapter.test_read()
    assert result is not None
    assert result["machine_id"] == "SIEMENS-PLC-001"
    assert result["source"] == "siemens_s7"
    assert result["telemetry"]["motor_current"] == pytest.approx(3.0)
    assert result["telemetry"]["cycle_timer_ms"] == 2000
    assert result["telemetry"]["motor_temperature_c"] == pytest.approx(50.0)
