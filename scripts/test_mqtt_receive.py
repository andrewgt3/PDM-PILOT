#!/usr/bin/env python3
"""
Subscribe to MQTT topic machines/# and print each message for 30 seconds.
Used to manually verify MQTT is flowing from stream_publisher.

Usage:
  python scripts/test_mqtt_receive.py
"""

import sys
import time
from datetime import datetime, timezone

# Run from repo root so config and env are found
sys.path.insert(0, ".")

import paho.mqtt.client as mqtt

from config import get_settings

RUN_SECONDS = 30
TOPIC = "machines/#"
message_count = 0


def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        print(f"Connected to {userdata['host']}:{userdata['port']}; subscribing to {TOPIC}")
        client.subscribe(TOPIC, qos=1)
    else:
        print(f"Connect failed: reason_code={reason_code}")


def on_message(client, userdata, msg):
    global message_count
    message_count += 1
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    payload = msg.payload.decode("utf-8", errors="replace")
    print(f"[{ts}] topic: {msg.topic} | payload: {payload}")


def main():
    cfg = get_settings().mqtt
    client = mqtt.Client(
        client_id=cfg.client_id + "-receive-test",
        protocol=mqtt.MQTTv5,
    )
    client.user_data_set({"host": cfg.broker_host, "port": cfg.broker_port})
    client.on_connect = on_connect
    client.on_message = on_message

    if cfg.username:
        password = cfg.password.get_secret_value() if cfg.password else None
        client.username_pw_set(cfg.username, password)
    if cfg.tls_enabled:
        client.tls_set()

    try:
        client.connect(cfg.broker_host, cfg.broker_port, keepalive=60)
    except Exception as e:
        print(f"Connection error: {e}")
        return 1

    client.loop_start()
    time.sleep(RUN_SECONDS)
    client.loop_stop()
    client.disconnect()

    print(f"\nReceived {message_count} message(s) in {RUN_SECONDS} seconds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
