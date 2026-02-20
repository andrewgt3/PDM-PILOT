#!/usr/bin/env python3
"""
Subscribe to machines/# on the Mosquitto broker and print each message with a timestamp.
Verifies that telemetry from stream_publisher (or adapters) is arriving at the broker.

Usage:
    python test_mqtt_publish.py

Requires: Mosquitto broker running (e.g. docker-compose mosquitto), MQTT_BROKER_HOST/PORT in .env or env.
"""

import json
import sys
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

from config import get_settings


def main() -> None:
    cfg = get_settings().mqtt
    print(f"Connecting to MQTT broker {cfg.broker_host}:{cfg.broker_port} ...")
    print(f"Subscribing to machines/# (Ctrl+C to stop)\n")

    client = mqtt.Client(client_id="pdm-test-subscriber", protocol=mqtt.MQTTv5)
    if cfg.username:
        password = cfg.password.get_secret_value() if cfg.password else None
        client.username_pw_set(cfg.username, password)
    if cfg.tls_enabled:
        client.tls_set()

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if getattr(reason_code, "value", reason_code) != 0:
            print(f"Connection failed: {reason_code}", file=sys.stderr)
            return
        client.subscribe("machines/#", qos=1)
        print("Subscribed to machines/#\n")

    def on_message(client, userdata, msg):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        try:
            payload = json.loads(msg.payload.decode())
            machine_id = payload.get("machine_id", "?")
            print(f"[{ts}] {msg.topic} | machine_id={machine_id} | seq={payload.get('sequence_id')}")
        except Exception:
            print(f"[{ts}] {msg.topic} | (raw) {msg.payload[:80]!r}")

    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(cfg.broker_host, cfg.broker_port, keepalive=60)
    except Exception as e:
        print(f"Connect error: {e}", file=sys.stderr)
        sys.exit(1)

    client.loop_forever()


if __name__ == "__main__":
    main()
