#!/usr/bin/env python3
"""
Advanced Signal Processing for Predictive Maintenance

Complete feature extraction pipeline implementing:
- Sub-Task 2.1: FFT Features (16 features)
- Sub-Task 2.2: Envelope Analysis (5 features)
- Sub-Task 2.3: Degradation Score (1 feature)
- Telemetry Passthrough (4 features)

Total: 26 features matching CWRUFeature schema

Author: Senior Signal Processing Engineer
"""

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import entropy, kurtosis
from typing import Dict, Optional

# Import real bearing specifications database
try:
    from bearing_database import BEARING_DATABASE, get_fault_frequencies
    BEARING_DB_AVAILABLE = True
except ImportError:
    BEARING_DB_AVAILABLE = False
    print("[WARNING] bearing_database.py not found - using generic bearing ratios")


# =============================================================================
# SIGNAL PROCESSING CONSTANTS
# =============================================================================
SAMPLE_RATE = 12000  # Hz - Sampling frequency
N_SAMPLES = 1024     # Points per signal window

# =============================================================================
# BEARING GEOMETRY CONSTANTS (Default Fault Frequency Ratios)
# =============================================================================
# These are RATIOS (multiply by shaft frequency Hz to get actual fault frequency)
# Default: CWRU 6205-2RS bearing (matches training data)
DEFAULT_BEARING_MODEL = "CWRU 6205-2RS (Drive End)"
DEFAULT_BPFO_RATIO = 3.5848   # Ball Pass Frequency Outer race ratio
DEFAULT_BPFI_RATIO = 5.4152   # Ball Pass Frequency Inner race ratio
DEFAULT_BSF_RATIO = 2.357     # Ball Spin Frequency ratio
DEFAULT_FTF_RATIO = 0.3983    # Fundamental Train Frequency ratio
TOLERANCE_RATIO = 0.05        # Frequency tolerance window (±5% of fault freq)

# =============================================================================
# FILTER PARAMETERS
# =============================================================================
HIGHPASS_CUTOFF = 2000  # Hz - High-pass filter cutoff for envelope analysis
FILTER_ORDER = 4        # Butterworth filter order


class SignalProcessor:
    """
    Complete signal processing pipeline for vibration-based fault detection.
    
    Extracts 26 features from raw vibration waveforms:
    
    FFT Features (16):
        - 5 peak frequencies (dominant spectral components)
        - 5 peak amplitudes (magnitude at dominant frequencies)
        - 3 band powers (low/mid/high frequency energy)
        - 2 spectral shape metrics (entropy, kurtosis)
        - 1 total spectral power
    
    Envelope Features (5):
        - 4 fault frequency amplitudes (BPFO, BPFI, BSF, FTF)
        - 1 sideband strength (modulation indicator)
    
    Degradation Features (1):
        - 1 degradation score (health index 0-1)
    
    Telemetry Features (4):
        - rotational_speed, temperature, torque, tool_wear
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, n_samples: int = N_SAMPLES, 
                 bearing_model: str = None):
        """
        Initialize the signal processor with optional bearing specification.
        
        Args:
            sample_rate: Sampling frequency in Hz
            n_samples: Number of samples per signal window
            bearing_model: Bearing model name from bearing_database (e.g., "SKF 6205")
                          If None, uses CWRU 6205-2RS (matches training data)
        """
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        
        # Pre-compute frequency bins for rfft
        self.freq_bins = rfftfreq(n_samples, d=1.0/sample_rate)
        
        # Frequency band definitions (Hz)
        self.bands = {
            'low': (0, 50),
            'mid': (50, 300),
            'high': (300, 500)
        }
        
        # Load bearing fault frequency ratios from database
        self.bearing_model = bearing_model or DEFAULT_BEARING_MODEL
        self._load_bearing_ratios(self.bearing_model)
        
        print(f"[SignalProcessor] Using bearing: {self.bearing_model}")
        print(f"  Ratios: BPFO={self.fault_ratios['BPFO']:.4f}, BPFI={self.fault_ratios['BPFI']:.4f}, "
              f"BSF={self.fault_ratios['BSF']:.4f}, FTF={self.fault_ratios['FTF']:.4f}")
        
        # Design high-pass Butterworth filter for envelope analysis
        nyquist = sample_rate / 2
        normalized_cutoff = HIGHPASS_CUTOFF / nyquist
        # Clamp to valid range (0, 1)
        normalized_cutoff = min(normalized_cutoff, 0.99)
        self.hp_b, self.hp_a = butter(
            FILTER_ORDER,
            normalized_cutoff,
            btype='high',
            analog=False
        )
        
        # Baseline RMS for degradation calculation (typical healthy bearing)
        self.baseline_rms = 0.1
    
    def _load_bearing_ratios(self, bearing_model: str):
        """
        Load bearing fault frequency ratios from database.
        
        Args:
            bearing_model: Model name to look up (e.g., "SKF 6205")
        """
        if BEARING_DB_AVAILABLE and bearing_model in BEARING_DATABASE:
            bearing = BEARING_DATABASE[bearing_model]
            self.fault_ratios = bearing["fault_ratios"].copy()
        else:
            # Use default CWRU bearing ratios (matches training data)
            self.fault_ratios = {
                'BPFO': DEFAULT_BPFO_RATIO,
                'BPFI': DEFAULT_BPFI_RATIO,
                'BSF': DEFAULT_BSF_RATIO,
                'FTF': DEFAULT_FTF_RATIO
            }
    
    def get_fault_frequencies(self, shaft_rpm: float) -> Dict[str, float]:
        """
        Calculate actual fault frequencies for current bearing at given RPM.
        
        Args:
            shaft_rpm: Shaft rotational speed in RPM
            
        Returns:
            Dictionary with fault frequencies in Hz
        """
        shaft_freq_hz = shaft_rpm / 60.0
        return {
            'bpfo': self.fault_ratios['BPFO'] * shaft_freq_hz,
            'bpfi': self.fault_ratios['BPFI'] * shaft_freq_hz,
            'bsf': self.fault_ratios['BSF'] * shaft_freq_hz,
            'ftf': self.fault_ratios['FTF'] * shaft_freq_hz
        }

    
    # =========================================================================
    # SUB-TASK 2.1: FFT FEATURES (16 features)
    # =========================================================================
    def compute_fft_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Compute 16 FFT-based features from a vibration signal.
        
        Features:
            - peak_freq_1 to peak_freq_5: Top 5 dominant frequencies (Hz)
            - peak_amp_1 to peak_amp_5: Amplitudes at those frequencies
            - low_band_power: Power in 0-50 Hz band
            - mid_band_power: Power in 50-300 Hz band
            - high_band_power: Power in 300-500 Hz band
            - spectral_entropy: Shannon entropy of normalized PSD
            - spectral_kurtosis: Kurtosis of PSD distribution
            - total_power: Sum of all spectral power
        
        Args:
            signal: 1D numpy array of vibration samples
            
        Returns:
            Dictionary with 16 named features
        """
        # Compute Real FFT
        fft_coeffs = rfft(signal)
        
        # Power Spectral Density (PSD) = |FFT|²
        psd = np.abs(fft_coeffs) ** 2
        psd = psd / (len(signal) ** 2)
        psd[1:-1] *= 2  # Double non-DC/Nyquist components
        
        # --- Peak Frequencies and Amplitudes ---
        psd_no_dc = psd.copy()
        psd_no_dc[0] = 0
        peak_indices = np.argsort(psd_no_dc)[::-1][:5]
        peak_freqs = self.freq_bins[peak_indices]
        peak_amps = psd[peak_indices]
        
        # --- Band Powers ---
        band_powers = {}
        for band_name, (f_low, f_high) in self.bands.items():
            mask = (self.freq_bins >= f_low) & (self.freq_bins < f_high)
            band_powers[f'{band_name}_band_power'] = float(np.sum(psd[mask]))
        
        # --- Spectral Entropy ---
        psd_sum = np.sum(psd)
        if psd_sum > 0:
            psd_norm = psd / psd_sum
            spectral_ent = float(entropy(psd_norm + 1e-12))
        else:
            spectral_ent = 0.0
        
        # --- Spectral Kurtosis ---
        spectral_kurt = float(kurtosis(psd, fisher=True))
        
        # --- Total Power ---
        total_power = float(np.sum(psd))
        
        # Assemble features
        features = {
            'peak_freq_1': float(peak_freqs[0]),
            'peak_freq_2': float(peak_freqs[1]),
            'peak_freq_3': float(peak_freqs[2]),
            'peak_freq_4': float(peak_freqs[3]),
            'peak_freq_5': float(peak_freqs[4]),
            'peak_amp_1': float(peak_amps[0]),
            'peak_amp_2': float(peak_amps[1]),
            'peak_amp_3': float(peak_amps[2]),
            'peak_amp_4': float(peak_amps[3]),
            'peak_amp_5': float(peak_amps[4]),
            'low_band_power': band_powers['low_band_power'],
            'mid_band_power': band_powers['mid_band_power'],
            'high_band_power': band_powers['high_band_power'],
            'spectral_entropy': spectral_ent,
            'spectral_kurtosis': spectral_kurt,
            'total_power': total_power
        }
        
        return features
    
    # =========================================================================
    # SUB-TASK 2.2: ENVELOPE FEATURES (5 features)
    # =========================================================================
    def compute_envelope_features(self, signal: np.ndarray, shaft_rpm: float = 1800.0) -> Dict[str, float]:
        """
        Compute 5 envelope-based features for bearing fault detection.
        
        Processing Pipeline:
            1. High-pass filter (Butterworth, Order=4, Cutoff=2000Hz)
            2. Hilbert transform → Analytic signal
            3. Envelope = |analytic signal|
            4. FFT of envelope
            5. Extract amplitudes at fault frequencies (calculated from RPM)
        
        Mathematical Definition:
            BPFO_amp = max(|E(f)|) for f ∈ [BPFO-TOL, BPFO+TOL]
        
        Features:
            - bpfo_amp: Amplitude at outer race fault frequency
            - bpfi_amp: Amplitude at inner race fault frequency
            - bsf_amp: Amplitude at ball spin frequency
            - ftf_amp: Amplitude at cage/train frequency
            - sideband_strength: Modulation indicator
        
        Args:
            signal: 1D numpy array of vibration samples
            shaft_rpm: Shaft rotational speed in RPM (used to calculate fault frequencies)
            
        Returns:
            Dictionary with 5 envelope features
        """
        # Calculate actual fault frequencies from bearing ratios and shaft speed
        fault_freqs = self.get_fault_frequencies(shaft_rpm)
        
        # Step 1: Apply High-Pass Butterworth Filter
        # Removes low-frequency machine vibration, isolates bearing resonances
        try:
            filtered = filtfilt(self.hp_b, self.hp_a, signal)
        except ValueError:
            # Filter failed (signal too short), use original
            filtered = signal
        
        # Step 2: Compute Analytic Signal via Hilbert Transform
        # H(x) = x + j*hilbert(x) → complex analytic signal
        analytic_signal = hilbert(filtered)
        
        # Step 3: Calculate Envelope (magnitude of analytic signal)
        # Envelope reveals amplitude modulation from bearing impacts
        envelope = np.abs(analytic_signal)
        
        # Step 4: Compute FFT of Envelope
        env_fft = rfft(envelope)
        env_psd = np.abs(env_fft) ** 2
        env_psd = env_psd / (len(envelope) ** 2)
        
        # Step 5: Extract amplitudes at fault frequencies (±5% tolerance)
        fault_amps = {}
        for fault_name, fault_freq in fault_freqs.items():
            # Dynamic tolerance based on frequency (±5%)
            tolerance = fault_freq * TOLERANCE_RATIO
            
            # Find frequency indices within tolerance window
            mask = (self.freq_bins >= fault_freq - tolerance) & \
                   (self.freq_bins <= fault_freq + tolerance)
            
            if np.any(mask):
                # Max amplitude in window (convert power to amplitude)
                fault_amps[f'{fault_name}_amp'] = float(np.sqrt(np.max(env_psd[mask])))
            else:
                fault_amps[f'{fault_name}_amp'] = 0.0
        
        # Sideband Strength: Ratio of sidebands to carrier
        # Indicates amplitude modulation from rotating faults
        all_fault_amps = list(fault_amps.values())
        mean_fault_amp = np.mean(all_fault_amps)
        sideband_strength = float(np.std(all_fault_amps) / (mean_fault_amp + 1e-10))
        
        features = {
            'bpfo_amp': fault_amps['bpfo_amp'],
            'bpfi_amp': fault_amps['bpfi_amp'],
            'bsf_amp': fault_amps['bsf_amp'],
            'ftf_amp': fault_amps['ftf_amp'],
            'sideband_strength': sideband_strength
        }
        
        return features
    
    # =========================================================================
    # SUB-TASK 2.3: DEGRADATION SCORE (1 feature)
    # =========================================================================
    def compute_degradation(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Compute health degradation score from signal energy.
        
        The degradation score D_t is a continuous health index:
            D_t = 0: Healthy (RMS at or below baseline)
            D_t = 1: Severely degraded (RMS at 10× baseline)
        
        Formula:
            D_t = clip((RMS - baseline) / (10 × baseline), 0, 1)
        
        Args:
            signal: 1D numpy array of vibration samples
            
        Returns:
            Dictionary with degradation score
        """
        # Calculate RMS (Root Mean Square) - measure of signal energy
        rms = np.sqrt(np.mean(signal ** 2))
        
        # Compute degradation score
        # Normalized to [0, 1] range based on deviation from baseline
        degradation = (rms - self.baseline_rms) / (10 * self.baseline_rms)
        degradation = float(np.clip(degradation, 0.0, 1.0))
        
        return {'degradation_score': degradation}
    
    # =========================================================================
    # MAIN PIPELINE: PROCESS SIGNAL (26 features)
    # =========================================================================
    def process_signal(
        self,
        raw_signal: np.ndarray,
        telemetry_metadata: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Complete feature extraction pipeline.
        
        Combines all feature groups into a single 26-feature vector:
            - 16 FFT features
            - 5 Envelope features
            - 1 Degradation score
            - 4 Telemetry features
        
        Args:
            raw_signal: 1D numpy array of vibration samples (1024 points)
            telemetry_metadata: Dictionary with keys:
                - rotational_speed (RPM)
                - temperature (°C)
                - torque (Nm)
                - tool_wear (mm)
                If None, defaults to zeros.
        
        Returns:
            Flattened dictionary with 26 named features
        """
        # Validate signal length
        if len(raw_signal) != self.n_samples:
            # Pad or truncate to expected length
            if len(raw_signal) < self.n_samples:
                raw_signal = np.pad(raw_signal, (0, self.n_samples - len(raw_signal)))
            else:
                raw_signal = raw_signal[:self.n_samples]
        
        # Default telemetry if not provided
        if telemetry_metadata is None:
            telemetry_metadata = {
                'rotational_speed': 1800.0,  # Default to typical motor speed
                'temperature': 0.0,
                'torque': 0.0,
                'tool_wear': 0.0
            }
        
        # Get shaft RPM for dynamic fault frequency calculation
        shaft_rpm = float(telemetry_metadata.get('rotational_speed', 1800.0))
        if shaft_rpm <= 0:
            shaft_rpm = 1800.0  # Fallback to typical speed
        
        # Extract all feature groups
        fft_features = self.compute_fft_features(raw_signal)          # 16 features
        envelope_features = self.compute_envelope_features(raw_signal, shaft_rpm)  # 5 features (uses real bearing specs!)
        degradation_features = self.compute_degradation(raw_signal)    # 1 feature
        
        # Telemetry passthrough (4 features)
        telemetry_features = {
            'rotational_speed': float(telemetry_metadata.get('rotational_speed', 0.0)),
            'temperature': float(telemetry_metadata.get('temperature', 0.0)),
            'torque': float(telemetry_metadata.get('torque', 0.0)),
            'tool_wear': float(telemetry_metadata.get('tool_wear', 0.0))
        }
        
        # Combine all features into single dictionary
        all_features = {}
        all_features.update(fft_features)        # 16
        all_features.update(envelope_features)   # 5
        all_features.update(degradation_features)  # 1
        all_features.update(telemetry_features)   # 4
        
        return all_features  # Total: 26 features


# =============================================================================
# VALIDATION / DEMO
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SIGNAL PROCESSOR - Complete Feature Extraction Pipeline")
    print("=" * 70)
    
    # Initialize processor
    processor = SignalProcessor(sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES)
    
    # Generate test signal
    np.random.seed(42)
    t = np.arange(N_SAMPLES) / SAMPLE_RATE
    test_signal = (
        1.0 * np.sin(2 * np.pi * 30 * t) +   # Shaft frequency
        0.5 * np.sin(2 * np.pi * 100 * t) +  # BPFO
        0.3 * np.sin(2 * np.pi * 160 * t) +  # BPFI
        0.2 * np.random.randn(N_SAMPLES)     # Noise
    )
    
    # Test telemetry metadata
    telemetry = {
        'rotational_speed': 1800.0,
        'temperature': 72.5,
        'torque': 45.2,
        'tool_wear': 0.15
    }
    
    # Extract all features
    features = processor.process_signal(test_signal, telemetry)
    
    # Validate output
    print(f"\nTest Signal: Multi-component with noise")
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Signal Length: {N_SAMPLES} points")
    
    print("\n" + "-" * 70)
    print(f"EXTRACTED FEATURES: {len(features)} total")
    print("-" * 70)
    
    # Group and display features
    print("\n[FFT FEATURES - 16]")
    fft_keys = ['peak_freq_1', 'peak_freq_2', 'peak_freq_3', 'peak_freq_4', 'peak_freq_5',
                'peak_amp_1', 'peak_amp_2', 'peak_amp_3', 'peak_amp_4', 'peak_amp_5',
                'low_band_power', 'mid_band_power', 'high_band_power',
                'spectral_entropy', 'spectral_kurtosis', 'total_power']
    for key in fft_keys:
        print(f"  {key}: {features[key]:.6f}")
    
    print("\n[ENVELOPE FEATURES - 5]")
    env_keys = ['bpfo_amp', 'bpfi_amp', 'bsf_amp', 'ftf_amp', 'sideband_strength']
    for key in env_keys:
        print(f"  {key}: {features[key]:.6f}")
    
    print("\n[DEGRADATION FEATURES - 1]")
    print(f"  degradation_score: {features['degradation_score']:.6f}")
    
    print("\n[TELEMETRY FEATURES - 4]")
    tel_keys = ['rotational_speed', 'temperature', 'torque', 'tool_wear']
    for key in tel_keys:
        print(f"  {key}: {features[key]:.6f}")
    
    # Final validation
    print("\n" + "=" * 70)
    expected_count = 26
    actual_count = len(features)
    status = "✓ PASS" if actual_count == expected_count else "✗ FAIL"
    print(f"VALIDATION: {status} - Expected {expected_count}, Got {actual_count} features")
    print("=" * 70)
    
    # Print all feature keys for schema verification
    print("\nFeature Keys (for schema verification):")
    for i, key in enumerate(sorted(features.keys()), 1):
        print(f"  {i:2d}. {key}")
