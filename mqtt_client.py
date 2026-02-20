#!/usr/bin/env python3
"""
MQTT publisher client for GAIA stream publisher.
Uses paho-mqtt with config from config.py mqtt group.
Thread-safe: uses loop_start() for background network thread.
Auto-reconnect: on_disconnect schedules connect() after 5s delay.
"""

import json
import logging
import ssl
import threading
from typing import Any

import paho.mqtt.client as mqtt

from config import get_settings
from logger import get_logger

logger = get_logger("mqtt_client")
_paho_logger = logging.getLogger("mqtt_client")

RECONNECT_DELAY_S = 5


class MQTTPublisher:
    """
    MQTT publisher with auto-reconnect. Uses paho-mqtt; network loop runs
    in a background thread (loop_start()). connect/publish/disconnect are thread-safe.
    """

    def __init__(self) -> None:
        self._cfg = get_settings().mqtt
        self._client: mqtt.Client | None = None
        self._connected = False
        self._lock = threading.Lock()
        self._loop_started = False
        self._reconnect_timer: threading.Timer | None = None
        self._shutdown = False

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    def _on_connect(self, client: mqtt.Client, userdata: Any, flags: Any, reason_code: Any, *args: Any) -> None:
        rc = getattr(reason_code, "value", reason_code) if reason_code is not None else 0
        if rc == 0:
            with self._lock:
                self._connected = True
            logger.info(
                "MQTT CONNACK",
                event="CONNACK",
                host=self._cfg.broker_host,
                port=self._cfg.broker_port,
                reason_code=rc,
                session_present=getattr(flags, "session_present", None),
            )
        else:
            with self._lock:
                self._connected = False
            logger.warning(
                "MQTT CONNACK failed",
                event="CONNACK",
                host=self._cfg.broker_host,
                port=self._cfg.broker_port,
                reason_code=reason_code,
            )

    def _on_disconnect(self, client: mqtt.Client, userdata: Any, disconnect_flags: int, reason_code: Any = None, *args: Any) -> None:
        with self._lock:
            self._connected = False
        logger.info(
            "MQTT DISCONNECT",
            event="DISCONNECT",
            disconnect_flags=disconnect_flags,
            reason_code=reason_code,
            unexpected=disconnect_flags != 0,
        )
        # Auto-reconnect: re-queue connection attempt after 5s (unless shutdown)
        if not self._shutdown and self._reconnect_timer is None:
            self._reconnect_timer = threading.Timer(RECONNECT_DELAY_S, self._reconnect)
            self._reconnect_timer.daemon = True
            self._reconnect_timer.start()

    def _reconnect(self) -> None:
        self._reconnect_timer = None
        if self._shutdown or self._client is None:
            return
        logger.info("MQTT auto-reconnect in %s seconds", RECONNECT_DELAY_S)
        self.connect()

    def _on_publish(self, client: mqtt.Client, userdata: Any, mid: int, *args: Any) -> None:
        reason_code = args[0] if args else None
        rc = getattr(reason_code, "value", reason_code) if reason_code is not None else None
        logger.debug(
            "MQTT PUBACK",
            event="PUBACK",
            mid=mid,
            reason_code=rc,
        )

    def connect(self) -> bool:
        """Connect to broker using config from config.py. Sets on_connect, on_disconnect, on_publish. Returns True if connected."""
        if self._client is not None:
            self.disconnect()
        self._shutdown = False

        self._client = mqtt.Client(
            client_id=self._cfg.client_id,
            protocol=mqtt.MQTTv5,
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_publish = self._on_publish
        self._client.enable_logger(_paho_logger)

        if self._cfg.username:
            password = self._cfg.password.get_secret_value() if self._cfg.password else None
            self._client.username_pw_set(self._cfg.username, password)

        if self._cfg.tls_enabled:
            ca_certs = self._cfg.ca_cert_path
            certfile = self._cfg.client_cert_path
            keyfile = self._cfg.client_key_path
            if ca_certs or certfile:
                self._client.tls_set(
                    ca_certs=ca_certs,
                    certfile=certfile,
                    keyfile=keyfile,
                    cert_reqs=ssl.CERT_REQUIRED,
                    tls_version=ssl.PROTOCOL_TLS_CLIENT,
                )
            else:
                self._client.tls_set()

        try:
            self._client.connect(self._cfg.broker_host, self._cfg.broker_port, keepalive=60)
        except Exception as e:
            logger.error("MQTT connect error: %s", e)
            return False

        self._client.loop_start()
        self._loop_started = True

        for _ in range(50):
            if self.is_connected:
                return True
            threading.Event().wait(0.1)
        logger.warning("MQTT connect timeout")
        return self.is_connected

    def publish(self, topic: str, payload_dict: Any, qos: int = 1, retain: bool = False) -> bool:
        """Serialize payload to JSON, publish to topic. Returns True on success, False if not connected or on error."""
        if self._client is None or not self.is_connected:
            return False
        try:
            body = json.dumps(payload_dict) if not isinstance(payload_dict, str) else payload_dict
            self._client.publish(topic, body, qos=qos, retain=retain)
            return True
        except Exception as e:
            logger.warning("MQTT publish failed: %s", e)
            return False

    def disconnect(self) -> None:
        """Clean shutdown: cancel reconnect timer, stop loop, disconnect."""
        self._shutdown = True
        if self._reconnect_timer is not None:
            self._reconnect_timer.cancel()
            self._reconnect_timer = None
        if self._client is None:
            return
        if self._loop_started:
            self._client.loop_stop()
            self._loop_started = False
        try:
            self._client.disconnect()
        except Exception as e:
            logger.debug("MQTT disconnect: %s", e)
        self._client = None
        with self._lock:
            self._connected = False
