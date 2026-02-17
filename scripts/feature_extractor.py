"""
Unified Feature Extractor for Bearing Vibration Data
====================================================
Shared extraction logic for NASA IMS and FEMTO/PRONOSTIA.
Produces identical feature set for model compatibility.
"""

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt, hilbert

# Bearing ratios (6205 / Rexnord ZA-2115 - similar)
BPFO_R, BPFI_R, BSF_R, FTF_R = 3.58, 5.42, 2.36, 0.40


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


def extract_physics_features(
    data: np.ndarray, sample_rate: int, rpm: float
) -> dict:
    """Envelope analysis: BPFO, BPFI, BSF, FTF amplitudes + sideband."""
    data = data.flatten()
    n = len(data)
    freq_bins = rfftfreq(n, d=1.0 / sample_rate)
    shaft_hz = rpm / 60.0
    fault_freqs = {
        "bpfo": BPFO_R * shaft_hz,
        "bpfi": BPFI_R * shaft_hz,
        "bsf": BSF_R * shaft_hz,
        "ftf": FTF_R * shaft_hz,
    }
    nyq = sample_rate / 2
    b, a = butter(4, min(2000 / nyq, 0.99), btype="high")
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


def extract_spectral_features(data: np.ndarray, sample_rate: int) -> dict:
    """Band powers: low (0-500), mid (500-2000), high (2000-6000) Hz."""
    data = data.flatten()
    n = len(data)
    fft_vals = rfft(data * np.hanning(n))
    psd = np.abs(fft_vals) ** 2 / (n ** 2)
    freqs = rfftfreq(n, d=1.0 / sample_rate)
    low = np.sum(psd[(freqs >= 0) & (freqs < 500)])
    mid = np.sum(psd[(freqs >= 500) & (freqs < 2000)])
    high = np.sum(psd[(freqs >= 2000) & (freqs < 6000)])
    return {"low_band_power": low, "mid_band_power": mid, "high_band_power": high}


def extract_degradation(data: np.ndarray, baseline_rms: float = 0.07) -> dict:
    """Degradation score 0-1."""
    rms = np.sqrt(np.mean(data.flatten() ** 2))
    d = (rms - baseline_rms) / (10 * baseline_rms)
    return {"degradation_score": float(np.clip(d, 0.0, 1.0))}


def extract_all(
    data: np.ndarray, sample_rate: int, rpm: float
) -> dict:
    """Extract all features from 1D vibration array."""
    ref = extract_refinery_features(data)
    phys = extract_physics_features(data, sample_rate, rpm)
    spec = extract_spectral_features(data, sample_rate)
    deg = extract_degradation(data)
    return {**ref, **phys, **spec, **deg}
