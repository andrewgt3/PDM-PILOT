#!/usr/bin/env python3
"""
Azure Predictive Maintenance (PdM) Dataset Ingestor.

Loads the five PdM_*.csv files from a directory, merges on machineID + datetime,
adds 100 virtual machines to pipeline/station_config.json, and writes merged
data to TimescaleDB sensor_readings (and optionally failure labels to cwru_features).

Usage:
    python -m pipeline.ingestion.azure_pm_ingestor --data-dir data/azure_pm/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Accept both PdM_* and bare names
FILE_ALIASES = {
    "PdM_telemetry.csv": "telemetry",
    "telemetry.csv": "telemetry",
    "PdM_errors.csv": "errors",
    "errors.csv": "errors",
    "PdM_failures.csv": "failures",
    "failures.csv": "failures",
    "PdM_machines.csv": "machines",
    "machines.csv": "machines",
    "PdM_maint.csv": "maint",
    "maint.csv": "maint",
}

STATION_CONFIG_PATH = Path(__file__).resolve().parent.parent / "station_config.json"


def _find_files(data_dir: Path) -> dict[str, Path]:
    """Return dict role -> path for telemetry, errors, failures, machines, maint."""
    found: dict[str, Path] = {}
    for f in data_dir.iterdir():
        if not f.is_file() or f.suffix.lower() != ".csv":
            continue
        name = f.name
        if name in FILE_ALIASES:
            role = FILE_ALIASES[name]
            if role not in found:
                found[role] = f
    return found


class AzurePMIngestor:
    """
    Ingest Azure PdM dataset: merge five CSVs, update station_config.json
    with 100 machines, write to sensor_readings (and optionally cwru_features).
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def _load_files(self) -> dict[str, pd.DataFrame]:
        files = _find_files(self.data_dir)
        if "telemetry" not in files:
            raise FileNotFoundError(
                f"telemetry CSV not found in {self.data_dir}. "
                "Expected PdM_telemetry.csv or telemetry.csv."
            )
        out = {}
        for role, path in files.items():
            df = pd.read_csv(path)
            # Normalize machine ID column to machine_id for merge
            if "machineID" in df.columns and "machine_id" not in df.columns:
                df["machine_id"] = df["machineID"].astype(str)
            if "datetime" in df.columns and "timestamp" not in df.columns:
                df["timestamp"] = pd.to_datetime(df["datetime"], errors="coerce")
            out[role] = df
        return out

    def merge(self) -> pd.DataFrame:
        """Load all five CSVs and merge on machine_id + timestamp/datetime."""
        data = self._load_files()
        # Dedupe column names so merges don't fail (e.g. CSV with duplicate headers)
        for role in data:
            data[role] = data[role].loc[:, ~data[role].columns.duplicated()].copy()
        tele = data["telemetry"]
        if "timestamp" not in tele.columns and "datetime" in tele.columns:
            tele = tele.copy()
            tele["timestamp"] = pd.to_datetime(tele["datetime"], errors="coerce")
        tele = tele.rename(columns={"machineID": "machine_id"}) if "machineID" in tele.columns else tele
        merged = tele.copy()
        merged["machine_id"] = merged["machine_id"].astype(str)
        merged = merged.loc[:, ~merged.columns.duplicated()]

        # Left-merge others on (machine_id, timestamp)
        for role in ("errors", "failures", "maint"):
            if role not in data:
                continue
            df = data[role]
            if "timestamp" not in df.columns and "datetime" in df.columns:
                df = df.copy()
                df["timestamp"] = pd.to_datetime(df["datetime"], errors="coerce")
            if "machineID" in df.columns:
                df = df.copy()
                df["machine_id"] = df["machineID"].astype(str)
            df = df.loc[:, ~df.columns.duplicated()]
            key = ["machine_id", "timestamp"]
            if role == "errors":
                # Keep errorID; we might use as flag
                merge_cols = key + [c for c in df.columns if c not in key]
                merged = merged.merge(
                    df[merge_cols].drop_duplicates(key),
                    on=key,
                    how="left",
                    suffixes=("", f"_{role}"),
                )
            elif role == "failures":
                merged = merged.merge(
                    df[["machine_id", "timestamp", "failure"]].drop_duplicates(key),
                    on=key,
                    how="left",
                )
            elif role == "maint":
                merged = merged.merge(
                    df[["machine_id", "timestamp", "comp"]].drop_duplicates(key),
                    on=key,
                    how="left",
                    suffixes=("", "_maint"),
                )
            merged = merged.loc[:, ~merged.columns.duplicated()]

        if "machines" in data:
            mach = data["machines"]
            mach = mach.copy()
            mach["machine_id"] = mach["machineID"].astype(str) if "machineID" in mach.columns else mach["machine_id"]
            merged = merged.merge(
                mach[["machine_id", "model", "age"]].drop_duplicates("machine_id"),
                on="machine_id",
                how="left",
            )
        return merged

    def update_station_config(self) -> None:
        """Add 100 Azure machines to pipeline/station_config.json with 4 areas and 3 equipment types."""
        data = self._load_files()
        tele = data["telemetry"]
        machine_col = "machineID" if "machineID" in tele.columns else "machine_id"
        ids = sorted(tele[machine_col].unique().astype(int))
        if not ids:
            logger.warning("No machine IDs found in telemetry; skipping station_config update.")
            return

        # 4 virtual factory areas; ~25 machines per area (Line_1 .. Line_25)
        shops = ["MachineShop", "Assembly", "Stamping", "Finishing"]
        # 3 equipment types: ~33% each
        equipment_types = ["CNC Machine", "Hydraulic Press", "Conveyor"]

        path = STATION_CONFIG_PATH
        if not path.exists():
            config = {"node_mappings": {}, "metadata": {"version": "1.0.0"}}
        else:
            with open(path) as f:
                config = json.load(f)
        node_mappings = config.setdefault("node_mappings", {})

        for idx, i in enumerate(ids):
            sid = str(i)
            if sid in node_mappings:
                continue
            shop = shops[idx % len(shops)]
            line_num = (idx % 25) + 1  # Line_1 .. Line_25 per area
            line = f"Line_{line_num}"
            equipment_type = equipment_types[idx % len(equipment_types)]
            uns_path = f"PlantAGI/{shop}/{line}/Machine_{i}"
            node_mappings[sid] = {
                "asset_name": f"Azure Machine {i}",
                "shop": shop,
                "line": line,
                "equipment_type": equipment_type,
                "opc_node_id": f"ns=2;s=Azure_{i}",
                "criticality": "medium",
                "uns_path": uns_path,
            }

        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info("Updated %s with %d Azure machine(s).", path, len(ids))

    def run(self, *, write_cwru: bool = True) -> int:
        """
        Merge CSVs, update station_config.json, insert into sensor_readings
        (and optionally cwru_features). Returns number of sensor_readings rows written.
        """
        merged = self.merge()
        self.update_station_config()

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

        # Map to sensor_readings: machine_id, timestamp, rotational_speed <- rotate,
        # temperature <- null, vibration_raw <- JSON array or scalar, torque/tool_wear null
        insert_sql = """
            INSERT INTO sensor_readings (machine_id, timestamp, rotational_speed, temperature, torque, tool_wear, vibration_raw)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (machine_id, timestamp) DO UPDATE SET
                rotational_speed = COALESCE(EXCLUDED.rotational_speed, sensor_readings.rotational_speed),
                temperature = COALESCE(EXCLUDED.temperature, sensor_readings.temperature),
                torque = COALESCE(EXCLUDED.torque, sensor_readings.torque),
                tool_wear = COALESCE(EXCLUDED.tool_wear, sensor_readings.tool_wear),
                vibration_raw = COALESCE(EXCLUDED.vibration_raw, sensor_readings.vibration_raw)
        """
        count = 0
        with conn.cursor() as cur:
            for _, row in merged.iterrows():
                ts = row["timestamp"]
                if hasattr(ts, "to_pydatetime"):
                    ts = ts.to_pydatetime()
                vib = row.get("vibration")
                if pd.notna(vib):
                    try:
                        v = float(vib)
                        vibration_raw = json.dumps([v])
                    except (TypeError, ValueError):
                        vibration_raw = json.dumps([str(vib)])
                else:
                    vibration_raw = None
                cur.execute(insert_sql, (
                    str(row["machine_id"]),
                    ts,
                    float(row["rotate"]) if pd.notna(row.get("rotate")) else None,
                    None,  # temperature
                    None,  # torque
                    None,  # tool_wear
                    vibration_raw,
                ))
                count += 1
        conn.commit()

        if write_cwru:
            # failure_class: 1 if failure at that (machine_id, timestamp), else 0
            fail_df = merged.copy()
            fail_df["failure_class"] = fail_df["failure"].notna().astype(int)
            cwru_sql = """
                INSERT INTO cwru_features (machine_id, timestamp, failure_class, rotational_speed, temperature, torque, tool_wear)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (machine_id, timestamp) DO UPDATE SET
                    failure_class = EXCLUDED.failure_class,
                    rotational_speed = COALESCE(EXCLUDED.rotational_speed, cwru_features.rotational_speed),
                    temperature = COALESCE(EXCLUDED.temperature, cwru_features.temperature),
                    torque = COALESCE(EXCLUDED.torque, cwru_features.torque),
                    tool_wear = COALESCE(EXCLUDED.tool_wear, cwru_features.tool_wear)
            """
            with conn.cursor() as cur:
                for _, row in fail_df.iterrows():
                    ts = row["timestamp"]
                    if hasattr(ts, "to_pydatetime"):
                        ts = ts.to_pydatetime()
                    cur.execute(cwru_sql, (
                        str(row["machine_id"]),
                        ts,
                        int(row["failure_class"]),
                        float(row["rotate"]) if pd.notna(row.get("rotate")) else None,
                        None,
                        None,
                        None,
                    ))
            conn.commit()

        conn.close()
        return count


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Azure PdM dataset ingestor")
    parser.add_argument("--data-dir", default="data/azure_pm/", help="Directory containing PdM_*.csv files")
    parser.add_argument("--no-cwru", action="store_true", help="Do not write failure labels to cwru_features")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Error: not a directory: {data_dir}", file=sys.stderr)
        return 1
    ingestor = AzurePMIngestor(data_dir)
    try:
        n = ingestor.run(write_cwru=not args.no_cwru)
        print(f"Inserted/updated {n} rows in sensor_readings.")
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception("Ingestion failed")
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
