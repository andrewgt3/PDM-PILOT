#!/usr/bin/env python3
"""
Build Combined RUL Features (NASA IMS + FEMTO/PRONOSTIA)
========================================================
Processes both NASA IMS and FEMTO bearing data into a unified feature CSV
with identical schema for joint RUL model training.

Output: data/processed/combined_features_physics.csv
- dataset_source: 'nasa' | 'femto'
- Same 21 features + rul_minutes, time_pct, run_id, etc.

Usage:
    1. Run download_femto_data.py (requires disk space)
    2. Ensure NASA data in data/ingest/
    3. python scripts/build_combined_features.py
    4. python scripts/retrain_rul_with_physics.py  # use COMBINED_CSV
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

BASE = Path(__file__).resolve().parent.parent
INGEST = BASE / "data" / "ingest"
FEMTO_DIR = BASE / "data" / "downloads" / "femto_pronostia"
OUT_PATH = BASE / "data" / "processed" / "combined_features_physics.csv"

# Dataset specs
NASA_SAMPLE_RATE = 20000
NASA_SAMPLES = 20480
NASA_RPM = 2000
NASA_INTERVAL_MIN = 10

FEMTO_SAMPLE_RATE = 25600  # Hz
FEMTO_SAMPLES = 2560
FEMTO_RPM = 1800  # Condition 1; 2=4000, 3=4200 - use 1800 as default
FEMTO_INTERVAL_MIN = 10  # 10s between acquisitions

sys.path.insert(0, str(BASE / "scripts"))
from feature_extractor import extract_all


def parse_nasa_filename(fname: str) -> pd.Timestamp:
    parts = str(fname).split(".")
    if len(parts) == 6:
        return pd.Timestamp(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]))
    return pd.NaT


def load_nasa_file(fp: Path) -> np.ndarray:
    data = np.loadtxt(fp)
    if data.ndim == 2:
        data = data[:, 0]
    return data.flatten()


def load_femto_file(fp: Path) -> np.ndarray:
    """FEMTO CSV: cols 0-3 metadata, 4=horizontal, 5=vertical. Use horizontal."""
    df = pd.read_csv(fp, header=None)
    if df.shape[1] >= 6:
        data = df.iloc[:, 4].values  # horizontal
    else:
        data = df.iloc[:, -1].values
    return data.astype(np.float64)


def process_nasa(ingest_dir: Path) -> pd.DataFrame:
    """Process NASA IMS files."""
    files = list(ingest_dir.rglob("*"))
    files = [f for f in files if f.is_file() and not f.name.startswith(".")]
    files = [f for f in files if parse_nasa_filename(str(f.name).replace(".txt", "").split("/")[-1]) is not pd.NaT]

    if not files:
        return pd.DataFrame()

    rows = []
    for fp in tqdm(files, desc="NASA"):
        try:
            data = load_nasa_file(fp)
            if len(data) < 2048:
                continue
            if len(data) > NASA_SAMPLES:
                data = data[:NASA_SAMPLES]
            elif len(data) < NASA_SAMPLES:
                data = np.pad(data, (0, NASA_SAMPLES - len(data)))

            feats = extract_all(data, NASA_SAMPLE_RATE, NASA_RPM)
            rows.append({
                "filename": fp.name,
                "dataset_source": "nasa",
                **feats,
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["filedate"] = df["filename"].apply(lambda x: parse_nasa_filename(str(x).split("/")[-1].replace(".txt", "")))
    df = df.dropna(subset=["filedate"])
    df = df.sort_values("filedate").reset_index(drop=True)
    df["gap_hours"] = df["filedate"].diff().dt.total_seconds() / 3600
    df["run_id"] = "nasa_" + (df["gap_hours"] > 24).cumsum().astype(str)
    df["interval_min"] = NASA_INTERVAL_MIN
    return df


def process_femto(femto_dir: Path) -> pd.DataFrame:
    """Process FEMTO/PRONOSTIA Learning_set."""
    learning = femto_dir / "Learning_set"
    if not learning.exists():
        return pd.DataFrame()

    rows = []
    for bearing_dir in sorted(learning.iterdir()):
        if not bearing_dir.is_dir():
            continue
        csv_files = sorted(bearing_dir.glob("acc_*.csv"))
        for i, fp in enumerate(tqdm(csv_files, desc=f"FEMTO {bearing_dir.name}", leave=False)):
            try:
                data = load_femto_file(fp)
                if len(data) < 256:
                    continue
                if len(data) > FEMTO_SAMPLES:
                    data = data[:FEMTO_SAMPLES]
                elif len(data) < FEMTO_SAMPLES:
                    data = np.pad(data, (0, FEMTO_SAMPLES - len(data)))

                feats = extract_all(data, FEMTO_SAMPLE_RATE, FEMTO_RPM)
                rows.append({
                    "filename": f"{bearing_dir.name}/{fp.name}",
                    "dataset_source": "femto",
                    "bearing_id": bearing_dir.name,
                    **feats,
                })
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["filedate"] = pd.NaT  # FEMTO has no datetime in filename
    df["run_id"] = "femto_" + df["bearing_id"]
    df["interval_min"] = FEMTO_INTERVAL_MIN
    return df


def add_rul_and_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Add RUL, time_pct, rolling features per run."""
    def _rolling_slope(series: pd.Series, window: int = 5) -> pd.Series:
        out = pd.Series(np.nan, index=series.index)
        for i in range(window - 1, len(series)):
            window_vals = series.iloc[i - window + 1 : i + 1].values
            if np.isfinite(window_vals).all():
                x = np.arange(window)
                slope = np.polyfit(x, window_vals, 1)[0]
                out.iloc[i] = slope
        return out

    def add_rul(g):
        n = len(g)
        g = g.copy()
        interval = g["interval_min"].iloc[0]
        run_len = (n - 1) * interval
        g["rul_minutes"] = (n - 1 - np.arange(n)) * interval
        g["time_pct"] = np.arange(n) / max(n - 1, 1)
        g["run_length_minutes"] = run_len
        rms_baseline = g["rms"].iloc[:5].mean()
        g["rms_ratio_baseline"] = g["rms"] / (rms_baseline + 1e-10)
        g["rms_slope_5"] = _rolling_slope(g["rms"], 5)
        g["kurtosis_slope_5"] = _rolling_slope(g["kurtosis"], 5)
        g["rms_slope_5"] = g["rms_slope_5"].fillna(0)
        g["kurtosis_slope_5"] = g["kurtosis_slope_5"].fillna(0)
        g["bpfo_to_rms"] = g["bpfo_amp"] / (g["rms"] + 1e-10)
        g["bpfi_to_rms"] = g["bpfi_amp"] / (g["rms"] + 1e-10)
        g["bsf_to_rms"] = g["bsf_amp"] / (g["rms"] + 1e-10)
        return g

    df = df.groupby("run_id", group_keys=False).apply(add_rul, include_groups=False)
    return df


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Combined Feature Builder (NASA + FEMTO)")
    print("=" * 60)

    dfs = []

    if INGEST.exists():
        df_nasa = process_nasa(INGEST)
        if len(df_nasa) > 0:
            dfs.append(df_nasa)
            print(f"NASA: {len(df_nasa)} rows, {df_nasa['run_id'].nunique()} runs")
    else:
        print("NASA: data/ingest not found, skipping")

    if FEMTO_DIR.exists():
        df_femto = process_femto(FEMTO_DIR)
        if len(df_femto) > 0:
            dfs.append(df_femto)
            print(f"FEMTO: {len(df_femto)} rows, {df_femto['run_id'].nunique()} runs")
    else:
        print("FEMTO: data/downloads/femto_pronostia not found. Run download_femto_data.py first.")

    if not dfs:
        print("No data found. Ensure NASA data in data/ingest and/or run download_femto_data.py")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    # Assign global run_idx for train/test split (run_id is string like nasa_0, femto_Bearing1_1)
    run_ids = df["run_id"].unique()
    run_map = {r: i for i, r in enumerate(run_ids)}
    df["run_idx"] = df["run_id"].map(run_map)
    df = add_rul_and_rolling(df)

    # Drop bearing_id if present (FEMTO only)
    if "bearing_id" in df.columns:
        df = df.drop(columns=["bearing_id"])

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {OUT_PATH}")
    print(f"Runs: {df['run_id'].nunique()}")
    print(f"Sources: {df['dataset_source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
