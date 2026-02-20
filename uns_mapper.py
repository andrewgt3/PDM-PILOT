#!/usr/bin/env python3
"""
Unified Namespace (UNS) topic mapper — ISA-95 aligned MQTT topic hierarchy.

Transforms flat sensor payloads into
  {enterprise}/{site}/{area}/{cell}/{asset}/{category}/{metric}
and returns (topic, value, retain) tuples for MQTT publish.

Categories: telemetry (raw readings), health (model outputs), status (boolean state), events (alarms).
retain=True for health/ and status/; retain=False for telemetry/; retain=True for events/.

Usage:
  From code: UNSMapper(config_path).explode(machine_id, payload) -> [(topic, value, retain), ...]
  CLI:       python uns_mapper.py --preview   # Print complete topic tree (no broker connection)
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from logger import get_logger

logger = get_logger("uns_mapper")

# UNS schema: enterprise/site/area/cell/asset/category/metric = 7 levels
UNS_MAX_LEVELS = 9
UNS_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

# Default prefix when uns_path missing (ISA-95: enterprise/site/area/cell/asset)
DEFAULT_PREFIX = "PlantAGI/Factory1/Unknown/Unknown/Unknown"

# Payload keys -> (topic_metric_name). telemetry = raw readings, retain=False
TELEMETRY_KEYS = [
    ("rotational_speed", "rotational_speed"),
    ("temperature", "temperature"),
    ("torque", "torque"),
    ("tool_wear", "tool_wear"),
    ("vibration_raw", "vibration_raw"),
    ("timestamp", "timestamp"),
    ("motor_current", "motor_current"),
    ("cycle_timer_ms", "cycle_timer_ms"),
    ("conveyor_speed", "conveyor_speed"),
]
# health = model outputs, retain=True
HEALTH_METRICS = [
    ("machine_failure", "failure_flag"),
    ("tool_wear", "degradation_score"),
]
# status = boolean state, retain=True
STATUS_METRICS = [
    ("machine_failure", "fault_active"),
    ("timestamp", "last_seen"),
]


def _normalize_uns(s: str) -> str:
    """Replace spaces/special chars with nothing or underscore for UNS path segments."""
    s = re.sub(r"[^\w\-]", "_", s)
    return re.sub(r"_+", "_", s).strip("_") or "Unknown"


def validate_uns_topic(topic: str) -> bool:
    """
    Validate UNS topic: no spaces, only alphanumeric/underscore/hyphen per segment, max 8 levels.
    Returns True if valid.
    """
    if " " in topic:
        return False
    levels = topic.split("/")
    if len(levels) > UNS_MAX_LEVELS:
        return False
    for seg in levels:
        if not seg or not UNS_SEGMENT_PATTERN.match(seg):
            return False
    return True


class UNSMapper:
    """
    Maps machine_id + flat payload to ISA-95 UNS topic list.
    Loads site/area/cell/asset from station_config.json (uns_path per machine).
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._config_path = Path(config_path or "pipeline/station_config.json")
        self._prefix_by_machine: dict[str, str] = {}
        self._load_config()

    def _load_config(self) -> None:
        if not self._config_path.exists():
            return
        try:
            with open(self._config_path) as f:
                data = json.load(f)
        except Exception:
            return
        node_mappings = data.get("node_mappings") or {}
        for machine_id, entry in node_mappings.items():
            uns_path = entry.get("uns_path")
            if uns_path:
                self._prefix_by_machine[machine_id] = uns_path.rstrip("/")
            else:
                # Build from shop/line/asset_name (enterprise/site/area/cell/asset)
                base = "/".join(DEFAULT_PREFIX.split("/")[:2])  # PlantAGI/Factory1
                shop = _normalize_uns(entry.get("shop") or "Unknown")
                line = _normalize_uns(entry.get("line") or "Unknown")
                asset_name = entry.get("asset_name") or machine_id
                asset = _normalize_uns(asset_name)
                self._prefix_by_machine[machine_id] = f"{base}/{shop}/{line}/{asset}"

    def _get_prefix(self, machine_id: str) -> str:
        if machine_id in self._prefix_by_machine:
            return self._prefix_by_machine[machine_id]
        safe_id = _normalize_uns(machine_id)
        return f"{DEFAULT_PREFIX.rsplit('/', 1)[0]}/{safe_id}"

    def _value_for_metric(self, payload: dict, key: str, metric: str) -> Any:
        v = payload.get(key)
        if metric == "fault_active":
            return bool(int(v)) if v is not None else False
        if metric == "degradation_score":
            return float(v) if v is not None else 0.0
        return v

    def explode(self, machine_id: str, payload_dict: dict) -> list[tuple[str, Any, bool]]:
        """
        Explode flat payload into (topic, value, retain) tuples.
        Topic structure: {enterprise}/{site}/{area}/{cell}/{asset}/{category}/{metric}
        - telemetry/* -> retain=False
        - health/*, status/*, events/* -> retain=True
        """
        prefix = self._get_prefix(machine_id)
        out: list[tuple[str, Any, bool]] = []

        def add(topic: str, value: Any, retain: bool) -> None:
            if validate_uns_topic(topic):
                out.append((topic, value, retain))
            else:
                logger.warning("UNS schema validation failed, skipping publish", topic=topic, reason="invalid_chars_or_depth")

        # Telemetry (raw readings) — retain=False
        for key, metric in TELEMETRY_KEYS:
            if key not in payload_dict:
                continue
            add(f"{prefix}/telemetry/{metric}", payload_dict[key], False)

        # Health (model outputs) — retain=True
        for key, metric in HEALTH_METRICS:
            if key not in payload_dict:
                continue
            add(f"{prefix}/health/{metric}", self._value_for_metric(payload_dict, key, metric), True)

        # Status (boolean state) — retain=True
        for key, metric in STATUS_METRICS:
            if key not in payload_dict:
                continue
            add(f"{prefix}/status/{metric}", self._value_for_metric(payload_dict, key, metric), True)

        # Events (alarms) — retain=True; emit when fault
        fault = payload_dict.get("machine_failure")
        if fault is not None and int(fault) == 1:
            add(f"{prefix}/events/fault", {"fault": True, "machine_failure": 1}, True)

        return out

    def map(self, machine_id: str, payload: dict) -> list[tuple[str, Any, bool]]:
        """Alias for explode() for backward compatibility."""
        return self.explode(machine_id, payload)

    def all_machine_ids(self) -> list[str]:
        """Return sorted list of machine IDs known in config (for --preview)."""
        return sorted(self._prefix_by_machine.keys())

    def get_prefix(self, machine_id: str) -> str:
        """Return UNS path prefix for a machine (for preview)."""
        return self._get_prefix(machine_id)

    def preview_topics(self, machine_id: str, sample_payload: dict | None = None) -> list[tuple[str, Any, bool]]:
        """Return topic list for one machine; if sample_payload None, use minimal placeholder."""
        if sample_payload is None:
            sample_payload = {
                "timestamp": "2026-01-01T12:00:00Z",
                "rotational_speed": 1800.0,
                "temperature": 70.0,
                "torque": 40.0,
                "tool_wear": 0.1,
                "vibration_raw": [],
                "machine_failure": 0,
            }
        return self.explode(machine_id, sample_payload)

    def get_topic_tree(self) -> None:
        """Print full UNS topic tree for all registered machines (no broker connection)."""
        machine_ids = self.all_machine_ids()
        if not machine_ids:
            print("No machines in station_config (node_mappings empty or missing uns_path).", file=sys.stderr)
            return
        print("UNS topic tree: {enterprise}/{site}/{area}/{cell}/{asset}/{category}/{metric}")
        print("=" * 70)
        for mid in machine_ids:
            prefix = self._get_prefix(mid)
            topics = self.preview_topics(mid)
            print(f"\n{mid} -> {prefix}")
            for topic, value, retain in topics:
                r = "R" if retain else "-"
                val_str = str(value)
                val_preview = (val_str[:40] + "...") if len(val_str) > 40 else val_str
                print(f"  [{r}] {topic} = {val_preview}")
        print()


def _preview_cli() -> None:
    parser = argparse.ArgumentParser(description="UNS topic mapper — preview topic tree")
    parser.add_argument("--preview", action="store_true", help="Print complete topic tree for all machines (no broker)")
    parser.add_argument("--config", default="pipeline/station_config.json", help="Path to station_config.json")
    args = parser.parse_args()
    if not args.preview:
        parser.print_help()
        return
    mapper = UNSMapper(config_path=args.config)
    mapper.get_topic_tree()


if __name__ == "__main__":
    _preview_cli()
