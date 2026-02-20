#!/usr/bin/env python3
"""
UNS Explorer — live tree view of active MQTT topics.

Subscribes to # on the broker and builds a tree of all topics seen,
refreshing the terminal every 5 seconds.

Usage:
    python uns_explorer.py

Requires: Mosquitto broker running, MQTT_BROKER_HOST/PORT in .env or env.
"""

import sys
import threading
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

from config import get_settings


def _tree_from_topics(topics: set[str]) -> dict:
    """Build nested dict from topic strings: 'a/b/c' -> {'a': {'b': {'c': None}}}."""
    root: dict = {}
    for topic in sorted(topics):
        if not topic.strip():
            continue
        parts = topic.split("/")
        d = root
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                d[part] = None  # leaf
            else:
                if d.get(part) is None or not isinstance(d.get(part), dict):
                    d[part] = {}
                d = d[part]
    return root


def _print_tree(node: dict, indent: int = 0, prefix: str = "") -> None:
    """Print nested dict as tree with branch characters."""
    if node is None:
        return
    items = sorted(node.items())
    for i, (key, child) in enumerate(items):
        is_last = i == len(items) - 1
        branch = "└── " if is_last else "├── "
        print(f"{prefix}{branch}{key}")
        if child and isinstance(child, dict):
            ext = "    " if is_last else "│   "
            _print_tree(child, indent + 1, prefix + ext)


def _render(topics: set[str]) -> None:
    """Clear and draw tree (Unix clear; fallback for Windows)."""
    try:
        print("\033[2J\033[H", end="")  # clear screen, cursor home
    except Exception:
        pass
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"UNS Explorer — live topic tree (refresh 5s) — {ts}")
    print(f"Topics seen: {len(topics)}")
    print()
    if not topics:
        print("  (no topics yet — waiting for messages on #)")
        return
    tree = _tree_from_topics(topics)
    _print_tree(tree)
    print()


def main() -> None:
    cfg = get_settings().mqtt
    topics: set[str] = set()
    lock = threading.Lock()

    client = mqtt.Client(client_id="pdm-uns-explorer", protocol=mqtt.MQTTv5)
    if cfg.username:
        password = cfg.password.get_secret_value() if cfg.password else None
        client.username_pw_set(cfg.username, password)
    if cfg.tls_enabled:
        client.tls_set()

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if getattr(reason_code, "value", reason_code) != 0:
            print(f"Connection failed: {reason_code}", file=sys.stderr)
            return
        client.subscribe("#", qos=1)

    def on_message(client, userdata, msg):
        with lock:
            topics.add(msg.topic)

    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(cfg.broker_host, cfg.broker_port, keepalive=60)
    except Exception as e:
        print(f"Connect error: {e}", file=sys.stderr)
        sys.exit(1)

    client.loop_start()

    try:
        while True:
            time.sleep(5)
            with lock:
                snapshot = set(topics)
            _render(snapshot)
    except KeyboardInterrupt:
        client.loop_stop()
        client.disconnect()
        print("\nStopped.")


if __name__ == "__main__":
    main()
