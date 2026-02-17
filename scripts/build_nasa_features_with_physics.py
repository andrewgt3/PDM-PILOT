#!/usr/bin/env python3
"""
Build NASA IMS Features with Physics (Refinery + Bearing Fault Frequencies)
===========================================================================
Processes raw NASA IMS vibration files and extracts:
- 6 time-domain features (refinery: rms, kurtosis, p2p, skewness, crest, shape)
- 5 physics features (BPFO, BPFI, BSF, FTF amplitudes + sideband_strength)
- 1 degradation score
- 3 spectral features (low/mid/high band power)

Total: 15 features for RUL model. Output: data/processed/nasa_features_physics.csv

NASA IMS: 20 kHz sampling, 20480 samples/file, 8 channels. Uses channel 0.
Bearing: Rexnord ZA-2115 (NASA IMS test rig) - similar to 6205, use generic ratios.
RPM: 2000 (typical IMS test rig speed).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt, hilbert
from tqdm import tqdm

BASE = Path(__file__).resolve().parent.parent
# NASA IMS specs
NASA_SAMPLE_RATE = 20000  # Hz
NASA_SAMPLES = 20480
NASA_RPM = 2000
# Bearing ratios (Rexnord/IMS similar to 6205)
BPFO_R, BPFI_R, BSF_R, FTF_R = 3.58, 5.42, 2.36, 0.40


def load_nasa_file(filepath: Path) -> np.ndarray:
    """Load NASA IMS file. Returns 1D array (channel 0)."""
    data = np.loadtxt(filepath)
    if data.ndim == 2:
        data = data[:, 0]  # First channel
    return data.flatten()


def extract_refinery_features(data: np.ndarray) -> dict:
    """Refinery time-domain features."""
    data = data.flatten()
    rms = np.sqrt(np.mean(data**2))
    p2p = np.ptp(data)
    kurt = kurtosis(data, fisher=False)
    skew_val = skew(data)
    peak_abs = np.max(np.abs(data))
    crest = peak_abs / rms if rms > 0 else 0
    mean_abs = np.mean(np.abs(data))
    shape = rms / mean_abs if mean_abs > 0 else 0
    return {"rms": rms, "p2p": p2p, "kurtosis": kurt, "skewness": skew_val, "crest": crest, "shape": shape}


def extract_physics_features(data: np.ndarray, sample_rate: int = NASA_SAMPLE_RATE, rpm: float = NASA_RPM) -> dict:
    """Envelope analysis: BPFO, BPFI, BSF, FTF amplitudes + sideband."""
    data = data.flatten()
    n = len(data)
    freq_bins = rfftfreq(n, d=1.0/sample_rate)
    shaft_hz = rpm / 60.0
    fault_freqs = {
        "bpfo": BPFO_R * shaft_hz,
        "bpfi": BPFI_R * shaft_hz,
        "bsf": BSF_R * shaft_hz,
        "ftf": FTF_R * shaft_hz,
    }
    # High-pass filter (2 kHz) then envelope
    nyq = sample_rate / 2
    b, a = butter(4, min(2000/nyq, 0.99), btype="high")
    try:
        filtered = filtfilt(b, a, data)
    except ValueError:
        filtered = data
    envelope = np.abs(hilbert(filtered))
    env_fft = rfft(envelope)
    env_psd = np.abs(env_fft) ** 2 / (len(envelope) ** 2)
    fault_amps = {}
    for name, f in fault_freqs.items():
        tol = f * 0.05
        mask = (freq_bins >= f - tol) & (freq_bins <= f + tol)
        fault_amps[f"{name}_amp"] = float(np.sqrt(np.max(env_psd[mask]))) if np.any(mask) else 0.0
    amps = list(fault_amps.values())
    sideband = float(np.std(amps) / (np.mean(amps) + 1e-10))
    return {**fault_amps, "sideband_strength": sideband}


def extract_spectral_features(data: np.ndarray, sample_rate: int = NASA_SAMPLE_RATE) -> dict:
    """Band powers: low (0-500), mid (500-2000), high (2000-6000) Hz."""
    data = data.flatten()
    n = len(data)
    fft_vals = rfft(data * np.hanning(n))
    psd = np.abs(fft_vals) ** 2 / (n ** 2)
    freqs = rfftfreq(n, d=1.0/sample_rate)
    low = np.sum(psd[(freqs >= 0) & (freqs < 500)])
    mid = np.sum(psd[(freqs >= 500) & (freqs < 2000)])
    high = np.sum(psd[(freqs >= 2000) & (freqs < 6000)])
    return {"low_band_power": low, "mid_band_power": mid, "high_band_power": high}


def extract_degradation(data: np.ndarray, baseline_rms: float = 0.07) -> dict:
    """Degradation score 0-1."""
    rms = np.sqrt(np.mean(data.flatten() ** 2))
    d = (rms - baseline_rms) / (10 * baseline_rms)
    return {"degradation_score": float(np.clip(d, 0.0, 1.0))}


def parse_filename(fname: str) -> pd.Timestamp:
    """Parse YYYY.MM.DD.HH.MM.SS."""
    parts = str(fname).split(".")
    if len(parts) == 6:
        return pd.Timestamp(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]))
    return pd.NaT


def main():
    ingest = BASE / "data" / "ingest"
    out_path = BASE / "data" / "processed" / "nasa_features_physics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = list(ingest.rglob("*"))
    files = [f for f in files if f.is_file() and not f.name.startswith(".")]
    # Filter to NASA-style filenames (YYYY.MM.DD.HH.MM.SS) - use full name, Path.stem truncates
    def is_nasa(f):
        name = f.name
        if name.endswith((".txt", ".pdf", ".csv")):
            name = str(Path(name).stem)
        return parse_filename(name) is not pd.NaT
    files = [f for f in files if is_nasa(f)]


    if not files:
        print("No NASA IMS files found in data/ingest.")
        sys.exit(1)

    print(f"Processing {len(files)} NASA IMS files...")
    rows = []
    for fp in tqdm(files, desc="Extracting"):
        try:
            data = load_nasa_file(fp)
            if len(data) < 2048:
                continue
            if len(data) > NASA_SAMPLES:
                data = data[:NASA_SAMPLES]
            elif len(data) < NASA_SAMPLES:
                data = np.pad(data, (0, NASA_SAMPLES - len(data)))

            ref = extract_refinery_features(data)
            phys = extract_physics_features(data)
            spec = extract_spectral_features(data)
            deg = extract_degradation(data)
            row = {
                "filename": fp.name if hasattr(fp, 'name') else str(fp).split("/")[-1],
                **ref,
                **phys,
                **spec,
                **deg,
            }
            rows.append(row)
        except Exception as e:
            continue

    df = pd.DataFrame(rows)
    # Use full filename for parsing - Path.stem truncates "2004.03.10.09.42.46" to "2004.03.10.09.42"
    df["filedate"] = df["filename"].apply(lambda x: parse_filename(str(x).split("/")[-1].replace(".txt", "")))
    df = df.dropna(subset=["filedate"])
    df = df.sort_values("filedate").reset_index(drop=True)

    # Assign run_id (gap > 24h = new run)
    df["gap_hours"] = df["filedate"].diff().dt.total_seconds() / 3600
    df["run_id"] = (df["gap_hours"] > 24).cumsum()

    # Compute RUL, time_pct, and rolling/baseline features per run (10 min between files)
    def _rolling_slope(series: pd.Series, window: int = 5) -> pd.Series:
        """Linear slope over rolling window. NaN for first window-1 samples."""
        out = pd.Series(np.nan, index=series.index)
        for i in range(window - 1, len(series)):
            window_vals = series.iloc[i - window + 1 : i + 1].values
            if np.isfinite(window_vals).all():
                x = np.arange(window)
                slope = np.polyfit(x, window_vals, 1)[0]
                out.iloc[i] = slope
        return out

    def add_rul_and_time(g):
        n = len(g)
        g = g.copy()
        run_len = (n - 1) * 10  # minutes (10 min between files)
        g["rul_minutes"] = (n - 1 - np.arange(n)) * 10
        g["time_pct"] = np.arange(n) / max(n - 1, 1)  # 0=start, 1=end of run
        g["run_length_minutes"] = run_len  # scale cue for RUL (avoids negative R² on long runs)
        # Baseline: mean of first 5 samples (healthy state)
        rms_baseline = g["rms"].iloc[:5].mean()
        g["rms_ratio_baseline"] = g["rms"] / (rms_baseline + 1e-10)
        # Rolling slopes (degradation trend)
        g["rms_slope_5"] = _rolling_slope(g["rms"], 5)
        g["kurtosis_slope_5"] = _rolling_slope(g["kurtosis"], 5)
        # Fault-to-RMS ratio (physics: fault energy vs overall level)
        g["bpfo_to_rms"] = g["bpfo_amp"] / (g["rms"] + 1e-10)
        g["bpfi_to_rms"] = g["bpfi_amp"] / (g["rms"] + 1e-10)
        g["bsf_to_rms"] = g["bsf_amp"] / (g["rms"] + 1e-10)
        # Fill first few NaNs with 0 (no trend yet)
        g["rms_slope_5"] = g["rms_slope_5"].fillna(0)
        g["kurtosis_slope_5"] = g["kurtosis_slope_5"].fillna(0)
        return g

    df = df.groupby("run_id", group_keys=False).apply(add_rul_and_time, include_groups=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(f"Runs: {df['run_id'].nunique()}")
    print(f"Features: {[c for c in df.columns if c not in ('filename','filedate','gap_hours','run_id')]}")


if __name__ == "__main__":
    main()
