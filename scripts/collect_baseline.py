#!/usr/bin/env python3
"""
Collect baseline export from TimescaleDB for a given machine.

Queries the last N days from cwru_features, writes a CSV in the format
expected by custom_data_ingestor (cwru-style columns). Use this when
connecting a new machine to establish its healthy baseline before
training the custom anomaly model.

Usage:
    python scripts/collect_baseline.py --machine-id ROBOT-01 --days 7
    python scripts/collect_baseline.py --machine-id ROBOT-01 --days 7 --output data/baseline_robot01.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import psycopg2

# Project root for config
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings


# Columns to export (cwru_features schema → CSV for custom_data_ingestor)
CWRU_EXPORT_COLUMNS = [
    "timestamp",
    "machine_id",
    "torque",
    "rotational_speed",
    "temperature",
    "tool_wear",
]


def get_db_connection():
    """TimescaleDB connection using app config."""
    s = get_settings().database
    return psycopg2.connect(
        host=s.host,
        port=s.port,
        database=s.name,
        user=s.user,
        password=s.password.get_secret_value(),
        connect_timeout=10,
    )


def collect_baseline(
    machine_id: str | None,
    days: int,
    output_path: Path,
) -> pd.DataFrame:
    """
    Query last N days from cwru_features, optionally filter by machine_id.
    Returns DataFrame and writes CSV to output_path.
    """
    conn = get_db_connection()
    cols = ", ".join(CWRU_EXPORT_COLUMNS)
    # Use parameterized query; interval built from trusted int
    interval_sql = f"NOW() - INTERVAL '{days} days'"
    try:
        if machine_id:
            query = f"""
                SELECT {cols}
                FROM cwru_features
                WHERE timestamp >= {interval_sql}
                  AND machine_id = %s
                ORDER BY timestamp
            """
            with conn.cursor() as cur:
                cur.execute(query, (machine_id,))
                colnames = [d[0] for d in cur.description]
                rows = cur.fetchall()
        else:
            query = f"""
                SELECT {cols}
                FROM cwru_features
                WHERE timestamp >= {interval_sql}
                ORDER BY machine_id, timestamp
            """
            with conn.cursor() as cur:
                cur.execute(query)
                colnames = [d[0] for d in cur.description]
                rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows, columns=colnames)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def print_stats(df: pd.DataFrame) -> None:
    """Print data completeness %, value ranges, and cycle count."""
    if df.empty:
        print("No rows in export.")
        return
    n = len(df)
    print("--- Baseline export summary ---")
    print(f"Cycle count (rows): {n}")
    print("Data completeness (% non-null):")
    for col in df.columns:
        if col in ("timestamp", "machine_id"):
            continue
        pct = (df[col].notna().sum() / n) * 100
        print(f"  {col}: {pct:.1f}%")
    print("Value ranges (numeric columns):")
    for col in df.columns:
        if col in ("timestamp", "machine_id"):
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any():
            print(f"  {col}: min={s.min():.4f}, max={s.max():.4f}, mean={s.mean():.4f}")
    print("--------------------------------")


def main():
    parser = argparse.ArgumentParser(description="Export baseline data from TimescaleDB for custom ingestor.")
    parser.add_argument("--machine-id", type=str, default=None, help="Filter by machine_id (optional).")
    parser.add_argument("--days", type=int, default=7, help="Last N days of data.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: data/baseline_export_<machine>_<date>.csv",
    )
    args = parser.parse_args()

    out = args.output
    if out is None:
        suffix = (args.machine_id or "all").replace(" ", "_")
        out = PROJECT_ROOT / "data" / f"baseline_export_{suffix}.csv"

    df = collect_baseline(args.machine_id, args.days, out)
    print(f"Exported {len(df)} rows to {out}")
    print_stats(df)


if __name__ == "__main__":
    main()
