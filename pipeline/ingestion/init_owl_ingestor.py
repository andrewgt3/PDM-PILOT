#!/usr/bin/env python3
"""
inIT-OWL / Genesis CSV Ingestor.

Reads one or more CSVs from a file or directory, auto-detects schema,
normalizes to long form (timestamp, machine_id, sensor_type, value, label),
and optionally inserts into TimescaleDB cwru_features.

Usage:
    python -m pipeline.ingestion.init_owl_ingestor --input data/init_owl.csv --output csv
    python -m pipeline.ingestion.init_owl_ingestor --input data/init_owl/ --output db
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Schema detection: keyword lists (case-insensitive substring match)
# -----------------------------------------------------------------------------
VIBRATION_KEYWORDS = ("vibration", "l_1", "a_1", "b_1", "c_1", "vib", "accel")
TEMPERATURE_KEYWORDS = ("temperature", "temp", "t_")
CURRENT_KEYWORDS = ("current", "motor_current", "i_", "amp")
CONDITION_KEYWORDS = ("condition", "label", "fault", "failure", "health", "state", "class")
TIMESTAMP_KEYWORDS = ("timestamp", "datetime", "time", "date", "t ")
MACHINE_KEYWORDS = ("machine_id", "machineid", "asset_id", "machine", "asset", "id")


def _col_lower(s: str) -> str:
    return (s or "").strip().lower()


def _matches_any(col: str, keywords: tuple[str, ...]) -> bool:
    c = _col_lower(col)
    return any(k in c for k in keywords)


def _detect_column_role(col: str) -> str | None:
    """Return one of: timestamp, machine_id, vibration, temperature, current, condition, other."""
    if _matches_any(col, TIMESTAMP_KEYWORDS):
        return "timestamp"
    if _matches_any(col, MACHINE_KEYWORDS):
        return "machine_id"
    if _matches_any(col, VIBRATION_KEYWORDS):
        return "vibration"
    if _matches_any(col, TEMPERATURE_KEYWORDS):
        return "temperature"
    if _matches_any(col, CURRENT_KEYWORDS):
        return "current"
    if _matches_any(col, CONDITION_KEYWORDS):
        return "condition"
    return "other"


# Normal → 0, anomaly → 1 (configurable)
NORMAL_VALUES = ("normal", "healthy", "ok", "good", "0", "false", "no")
ANOMALY_VALUES = ("degraded", "fault", "failure", "warning", "anomaly", "bad", "1", "true", "yes")


def _condition_to_label(val) -> int | None:
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in NORMAL_VALUES:
        return 0
    if s in ANOMALY_VALUES:
        return 1
    # numeric
    try:
        n = int(float(val))
        return 0 if n == 0 else 1
    except (TypeError, ValueError):
        return 1  # unknown → treat as anomaly


# Default gap limit for forward-fill (consecutive NaNs per (machine_id, sensor_type))
DEFAULT_GAP_LIMIT = 10

# Output long-form columns
OUTPUT_COLUMNS = ["timestamp", "machine_id", "sensor_type", "value", "label"]


class InITOWLIngestor:
    """
    Ingest inIT-OWL or Genesis-style CSV(s): auto-detect schema, normalize to
    long form, optional DB insert into cwru_features.
    """

    def __init__(
        self,
        input_path: str | Path,
        *,
        gap_limit: int = DEFAULT_GAP_LIMIT,
        normal_keywords: tuple[str, ...] = NORMAL_VALUES,
        anomaly_keywords: tuple[str, ...] = ANOMALY_VALUES,
    ):
        self.input_path = Path(input_path)
        self.gap_limit = gap_limit
        self.normal_keywords = normal_keywords
        self.anomaly_keywords = anomaly_keywords

    def _collect_dataframes(self) -> list[pd.DataFrame]:
        if self.input_path.is_file():
            if self.input_path.suffix.lower() != ".csv":
                logger.warning("Non-CSV file: %s", self.input_path)
                return []
            return [pd.read_csv(self.input_path)]
        if self.input_path.is_dir():
            csvs = sorted(self.input_path.glob("**/*.csv"))
            if not csvs:
                return []
            return [pd.read_csv(p) for p in csvs]
        return []

    def _detect_schema(self, df: pd.DataFrame) -> dict[str, str]:
        """Map original column name -> role (timestamp, machine_id, vibration, etc.)."""
        role_map: dict[str, str] = {}
        for col in df.columns:
            role = _detect_column_role(col)
            if role != "other":
                role_map[col] = role
        # If multiple columns map to same role, keep first (or we could concatenate later)
        return role_map

    def _normalize_single(self, df: pd.DataFrame) -> pd.DataFrame:
        role_map = self._detect_schema(df)
        if not role_map:
            logger.warning("No recognizable columns in DataFrame with columns: %s", list(df.columns))
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        ts_col = next((c for c, r in role_map.items() if r == "timestamp"), None)
        machine_col = next((c for c, r in role_map.items() if r == "machine_id"), None)
        condition_col = next((c for c, r in role_map.items() if r == "condition"), None)

        if not ts_col:
            logger.warning("No timestamp column detected; cannot normalize.")
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        # Build long-form rows: (timestamp, machine_id, sensor_type, value, label)
        rows: list[dict] = []
        sensor_cols = [c for c, r in role_map.items() if r in ("vibration", "temperature", "current")]

        df = df.copy()
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col])
        machine_id = "default"
        if machine_col and machine_col in df.columns:
            df[machine_col] = df[machine_col].astype(str)

        for _, r in df.iterrows():
            ts = r[ts_col]
            if machine_col and machine_col in r:
                machine_id = str(r[machine_col])
            label_val = None
            if condition_col and condition_col in r:
                label_val = _condition_to_label(r[condition_col])
            for col in sensor_cols:
                val = r.get(col)
                if pd.isna(val):
                    continue
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                sensor_type = role_map[col]
                rows.append({
                    "timestamp": ts,
                    "machine_id": machine_id,
                    "sensor_type": sensor_type,
                    "value": v,
                    "label": label_val,
                })

        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        # Forward-fill value with gap limit per (machine_id, sensor_type)
        out = out.sort_values(["machine_id", "sensor_type", "timestamp"]).reset_index(drop=True)
        out["value"] = (
            out.groupby(["machine_id", "sensor_type"])["value"]
            .transform(lambda x: x.ffill(limit=self.gap_limit))
        )
        out = out.reindex(columns=OUTPUT_COLUMNS)
        return out

    def ingest(self) -> pd.DataFrame:
        """Load CSV(s), normalize to long form; return DataFrame with timestamp, machine_id, sensor_type, value, label."""
        dfs = self._collect_dataframes()
        if not dfs:
            logger.warning("No CSV data found at %s", self.input_path)
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        normalized = [self._normalize_single(df) for df in dfs]
        combined = pd.concat([n for n in normalized if not n.empty], ignore_index=True)
        if combined.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        combined = combined.sort_values(["timestamp", "machine_id"]).reset_index(drop=True)
        return combined

    def to_db(self, output: Literal["db"]) -> int:
        """
        Insert normalized data into TimescaleDB cwru_features.
        Maps to (machine_id, timestamp, failure_class, rotational_speed, temperature, torque, tool_wear);
        FFT/envelope columns left NULL.
        Returns number of rows inserted/updated.
        """
        if output != "db":
            raise ValueError('to_db only supports output="db"')
        df = self.ingest()
        if df.empty:
            return 0

        # Pivot so we have one row per (machine_id, timestamp) with telemetry columns
        wide = df.pivot_table(
            index=["timestamp", "machine_id"],
            columns="sensor_type",
            values="value",
            aggfunc="first",
        ).reset_index()
        wide.columns.name = None
        # failure_class from label: take first non-null label per (timestamp, machine_id)
        label_agg = df.groupby(["timestamp", "machine_id"])["label"].first().reset_index()
        label_agg = label_agg.rename(columns={"label": "failure_class"})
        wide = wide.merge(label_agg, on=["timestamp", "machine_id"], how="left")
        wide["failure_class"] = wide["failure_class"].fillna(0).astype(int)
        # Map sensor_type columns to cwru_features telemetry (only where names match)
        for telemetry_col in ("rotational_speed", "temperature", "torque", "tool_wear"):
            if telemetry_col not in wide.columns:
                wide[telemetry_col] = None

        # Select columns for cwru_features insert
        cols = [
            "machine_id", "timestamp", "failure_class",
            "rotational_speed", "temperature", "torque", "tool_wear",
        ]
        for c in cols:
            if c not in wide.columns:
                wide[c] = None
        insert_df = wide[cols].dropna(subset=["machine_id", "timestamp"]).copy()
        insert_df["timestamp"] = pd.to_datetime(insert_df["timestamp"])
        insert_df["machine_id"] = insert_df["machine_id"].astype(str)

        try:
            from config import get_settings
            s = get_settings().database
            import psycopg2
            conn = psycopg2.connect(
                host=s.host,
                port=s.port,
                database=s.name,
                user=s.user,
                password=s.password.get_secret_value(),
                connect_timeout=10,
            )
        except Exception as e:
            logger.error("Database connection failed: %s", e)
            raise

        insert_sql = """
            INSERT INTO cwru_features (machine_id, timestamp, failure_class, rotational_speed, temperature, torque, tool_wear)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (machine_id, timestamp) DO UPDATE SET
                failure_class = EXCLUDED.failure_class,
                rotational_speed = COALESCE(EXCLUDED.rotational_speed, cwru_features.rotational_speed),
                temperature = COALESCE(EXCLUDED.temperature, cwru_features.temperature),
                torque = COALESCE(EXCLUDED.torque, cwru_features.torque),
                tool_wear = COALESCE(EXCLUDED.tool_wear, cwru_features.tool_wear)
        """
        count = 0
        with conn.cursor() as cur:
            for _, row in insert_df.iterrows():
                cur.execute(insert_sql, (
                    str(row["machine_id"]),
                    row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"],
                    int(row["failure_class"]),
                    float(row["rotational_speed"]) if pd.notna(row["rotational_speed"]) else None,
                    float(row["temperature"]) if pd.notna(row["temperature"]) else None,
                    float(row["torque"]) if pd.notna(row["torque"]) else None,
                    float(row["tool_wear"]) if pd.notna(row["tool_wear"]) else None,
                ))
                count += 1
        conn.commit()
        conn.close()
        return count


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="inIT-OWL / Genesis CSV ingestor")
    parser.add_argument("--input", required=True, help="Path to CSV file or directory of CSVs (e.g. data/init_owl.csv or data/init_owl/)")
    parser.add_argument("--output", choices=("csv", "db"), default="csv", help="Output: csv (stdout or path) or db (TimescaleDB cwru_features)")
    parser.add_argument("--out-path", default=None, help="If output=csv, write to this file instead of stdout")
    parser.add_argument("--gap-limit", type=int, default=DEFAULT_GAP_LIMIT, help="Max consecutive NaNs to forward-fill per (machine_id, sensor_type)")
    args = parser.parse_args()

    ingestor = InITOWLIngestor(args.input, gap_limit=args.gap_limit)
    if args.output == "db":
        n = ingestor.to_db("db")
        print(f"Inserted/updated {n} rows in cwru_features.")
        return 0
    df = ingestor.ingest()
    if df.empty:
        print("No data produced.", file=sys.stderr)
        return 1
    if args.out_path:
        df.to_csv(args.out_path, index=False)
        print(f"Wrote {len(df)} rows to {args.out_path}")
    else:
        df.to_csv(sys.stdout, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
