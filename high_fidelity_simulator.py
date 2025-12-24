#!/usr/bin/env python3
"""
High-Fidelity Virtual Smart Sensor: Rolling Element Bearing Vibration Simulator

This module simulates realistic vibration waveforms from a rolling element bearing,
incorporating physics-based fault models for predictive maintenance research.

Physics Background:
-------------------
Rolling element bearings generate characteristic vibration signatures based on their
geometry and rotational speed. When faults develop on bearing surfaces, they produce
repetitive impulses at specific frequencies:

    BPFO (Ball Pass Frequency Outer): f_bpfo = (n/2) * f_shaft * (1 - d/D * cos(α))
    BPFI (Ball Pass Frequency Inner): f_bpfi = (n/2) * f_shaft * (1 + d/D * cos(α))

Where:
    n = number of rolling elements
    d = ball diameter
    D = pitch diameter
    α = contact angle
    f_shaft = shaft rotation frequency

For this simulation, we use simplified ratios:
    BPFO ≈ 5.4 × f_shaft
    BPFI ≈ 7.2 × f_shaft

Author: VibrationSimulator Physics Engine
"""

import json
import time
import numpy as np
from scipy import signal
from datetime import datetime, timezone
from typing import Literal


class VibrationSimulator:
    """
    Simulates realistic bearing vibration waveforms with physics-based fault injection.
    
    The simulator generates time-domain vibration signals that can be analyzed using
    FFT to extract characteristic fault frequencies - exactly as would be done with
    real accelerometer data from industrial machinery.
    
    Attributes:
        fs (int): Sampling frequency in Hz (default: 2048 Hz per Nyquist requirements)
        duration (float): Duration of each waveform snapshot in seconds
        machine_id (str): Identifier for the simulated machine
        
    Physics Constants:
        BPFO_RATIO: Ball Pass Frequency Outer race ratio (5.4× shaft frequency)
        BPFI_RATIO: Ball Pass Frequency Inner race ratio (7.2× shaft frequency)
    """
    
    # Characteristic fault frequency ratios for a typical deep-groove ball bearing
    # These ratios are derived from bearing geometry (e.g., SKF 6205 equivalent)
    BPFO_RATIO = 5.4  # Outer race fault frequency multiplier
    BPFI_RATIO = 7.2  # Inner race fault frequency multiplier
    
    def __init__(
        self,
        fs: int = 2048,
        duration: float = 1.0,
        machine_id: str = "CNC-001"
    ):
        """
        Initialize the vibration simulator.
        
        Args:
            fs: Sampling frequency in Hz. Default 2048 Hz provides adequate bandwidth
                for capturing bearing fault frequencies up to ~1 kHz (Nyquist limit).
            duration: Duration of each waveform snapshot in seconds.
            machine_id: Identifier string for the simulated machine.
        """
        self.fs = fs
        self.duration = duration
        self.machine_id = machine_id
        self.n_samples = int(fs * duration)
        
        # Current operating state
        self._current_rpm = 1800.0  # Default nominal speed
        self._fault_amplitude = 0.0  # For progressive fault simulation
        
    def _generate_time_vector(self) -> np.ndarray:
        """Generate the time vector for the waveform."""
        return np.linspace(0, self.duration, self.n_samples, endpoint=False)
    
    def _generate_noise_floor(self, t: np.ndarray) -> np.ndarray:
        """
        Generate Gaussian white noise representing the bearing's baseline vibration.
        
        Physics: Random vibration from surface roughness, lubricant turbulence,
        and structural resonances creates a broadband noise floor.
        
        Distribution: N(0, 0.05) - zero mean, 0.05 standard deviation
        This represents approximately ±0.15g acceleration noise floor.
        
        Args:
            t: Time vector (used for determining array size)
            
        Returns:
            Gaussian white noise signal
        """
        return np.random.normal(loc=0.0, scale=0.05, size=len(t))
    
    def _generate_shaft_imbalance(self, t: np.ndarray, rpm: float) -> np.ndarray:
        """
        Generate sinusoidal vibration from shaft imbalance (1× RPM component).
        
        Physics: Mass imbalance on the rotating shaft creates a centrifugal force
        that rotates with the shaft, producing a sinusoidal vibration at exactly
        the shaft rotation frequency (1× RPM).
        
        F_imbalance = m * r * ω²  (centrifugal force)
        
        Where:
            m = unbalance mass
            r = radius of mass center from rotation axis
            ω = angular velocity (2π × f_shaft)
        
        Args:
            t: Time vector
            rpm: Shaft rotation speed in revolutions per minute
            
        Returns:
            Sinusoidal signal at shaft frequency
        """
        f_shaft = rpm / 60.0  # Convert RPM to Hz
        amplitude = 0.3  # Moderate imbalance amplitude (in g's)
        return amplitude * np.sin(2 * np.pi * f_shaft * t)
    
    def _generate_fault_impulse_train(
        self,
        t: np.ndarray,
        fault_frequency: float,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Generate impulse train at the fault frequency.
        
        Physics: When a rolling element passes over a localized defect (spall, pit),
        it creates a short-duration impact. These impacts repeat at the characteristic
        fault frequency determined by bearing geometry.
        
        The impulse train is modeled as a series of Dirac delta-like spikes
        occurring at intervals of T = 1/f_fault.
        
        Args:
            t: Time vector
            fault_frequency: Repetition rate of impulses in Hz
            amplitude: Peak amplitude of impulses
            
        Returns:
            Impulse train signal
        """
        if fault_frequency <= 0:
            return np.zeros_like(t)
            
        # Create impulse train using modulo arithmetic
        period = 1.0 / fault_frequency
        phase = t % period
        
        # Generate impulse at each period start (within a small tolerance)
        # Tolerance is set to capture one sample at each impulse location
        dt = 1.0 / self.fs
        impulse_train = np.where(phase < dt, amplitude, 0.0)
        
        return impulse_train
    
    def _generate_impulse_response(self) -> np.ndarray:
        """
        Generate the structural impulse response (exponentially decaying sinusoid).
        
        Physics: When an impulse excites the bearing housing/structure, it "rings"
        at its natural frequency with exponential decay. This is the classic
        single-degree-of-freedom (SDOF) impulse response:
        
        h(t) = A * exp(-ζ * ω_n * t) * sin(ω_d * t)
        
        Where:
            ζ = damping ratio (typically 0.02-0.05 for steel structures)
            ω_n = natural frequency (3-5 kHz typical for bearing housings)
            ω_d = damped natural frequency ≈ ω_n * sqrt(1 - ζ²)
        
        Returns:
            Impulse response array for convolution
        """
        # Structural parameters (typical for steel bearing housing)
        f_natural = 3500.0  # Natural frequency in Hz
        damping_ratio = 0.03  # Low damping for metallic structure
        
        # Duration of impulse response (until it decays to ~1% amplitude)
        # τ = 1/(ζ * ω_n), we use 5τ for full decay
        decay_time = 5.0 / (damping_ratio * 2 * np.pi * f_natural)
        n_ir = int(decay_time * self.fs)
        n_ir = max(n_ir, 50)  # Minimum samples for meaningful response
        
        t_ir = np.arange(n_ir) / self.fs
        
        # Damped natural frequency
        omega_n = 2 * np.pi * f_natural
        omega_d = omega_n * np.sqrt(1 - damping_ratio**2)
        
        # Exponentially decaying sinusoid
        impulse_response = np.exp(-damping_ratio * omega_n * t_ir) * np.sin(omega_d * t_ir)
        
        # Normalize to unit energy
        impulse_response /= np.sqrt(np.sum(impulse_response**2) + 1e-10)
        
        return impulse_response
    
    def _apply_load_zone_modulation(
        self,
        signal: np.ndarray,
        t: np.ndarray,
        rpm: float
    ) -> np.ndarray:
        """
        Apply amplitude modulation due to load zone passage (for inner race faults).
        
        Physics: For an inner race defect, the fault point rotates with the shaft.
        The defect only produces significant impacts when it's in the "load zone"
        (typically the lower half of the bearing under radial load). As the shaft
        rotates, the defect moves in and out of the load zone, creating amplitude
        modulation at the shaft frequency.
        
        Modulation envelope: M(t) = 1 + cos(2π * f_shaft * t)
        
        This gives maximum amplitude when defect is in load zone (cos = 1)
        and minimum (but non-zero) when outside (cos = -1).
        
        Args:
            signal: Input signal to be modulated
            t: Time vector
            rpm: Shaft rotation speed in RPM
            
        Returns:
            Amplitude-modulated signal
        """
        f_shaft = rpm / 60.0
        
        # Modulation envelope: (1 + cos(ω*t))/2 normalized to [0, 1]
        # But we use (1 + cos(ω*t)) to keep some signal even outside load zone
        modulation = 0.5 * (1 + np.cos(2 * np.pi * f_shaft * t))
        
        return signal * modulation
    
    def generate_waveform(
        self,
        rpm: float,
        fault_type: Literal["normal", "outer_race", "inner_race"] = "normal",
        fault_amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Generate a complete vibration waveform with optional fault injection.
        
        This is the core physics engine that synthesizes realistic vibration
        signals based on bearing mechanics and fault models.
        
        Signal Composition:
            Baseline = Noise Floor + Shaft Imbalance (1× RPM)
            Outer Race Fault: Baseline + Impulses at BPFO
            Inner Race Fault: Baseline + Impulses at BPFI with load zone modulation
        
        Args:
            rpm: Shaft rotation speed in revolutions per minute
            fault_type: Type of fault to simulate
                - "normal": Healthy bearing (baseline only)
                - "outer_race": Outer race defect (BPFO impulses)  
                - "inner_race": Inner race defect (BPFI with modulation)
            fault_amplitude: Relative amplitude of fault impulses (0.0 to 2.0)
                Used for simulating fault progression (developing faults)
                
        Returns:
            1D numpy array of vibration amplitude values (length = n_samples)
        """
        # Store current state for the data packager
        self._current_rpm = rpm
        self._fault_amplitude = fault_amplitude
        
        # Generate time vector
        t = self._generate_time_vector()
        
        # === BASELINE SIGNAL ===
        # Component 1: Gaussian white noise (surface roughness, lubricant dynamics)
        noise = self._generate_noise_floor(t)
        
        # Component 2: Shaft imbalance (1× RPM sinusoid)
        imbalance = self._generate_shaft_imbalance(t, rpm)
        
        # Combine baseline components
        waveform = noise + imbalance
        
        # === FAULT INJECTION ===
        if fault_type == "normal":
            # Healthy bearing - return baseline only
            return waveform
        
        # Calculate shaft frequency
        f_shaft = rpm / 60.0
        
        # Get structural impulse response for "ringing" effect
        impulse_response = self._generate_impulse_response()
        
        if fault_type == "outer_race":
            # Outer Race Fault: Fixed defect on stationary outer race
            # Impulses occur at Ball Pass Frequency Outer (BPFO)
            f_bpfo = self.BPFO_RATIO * f_shaft
            
            # Generate impulse train at BPFO
            impulse_train = self._generate_fault_impulse_train(
                t, f_bpfo, amplitude=fault_amplitude
            )
            
            # Convolve with structural response (metal "ringing")
            # This transforms sharp impulses into realistic decaying oscillations
            fault_signal = signal.convolve(impulse_train, impulse_response, mode='same')
            
            # Scale and add to baseline
            waveform += fault_signal * 0.5
            
        elif fault_type == "inner_race":
            # Inner Race Fault: Defect rotates with shaft
            # Impulses at BPFI, modulated by load zone passage
            f_bpfi = self.BPFI_RATIO * f_shaft
            
            # Generate impulse train at BPFI
            impulse_train = self._generate_fault_impulse_train(
                t, f_bpfi, amplitude=fault_amplitude
            )
            
            # Apply load zone modulation BEFORE convolution
            # This simulates the defect moving in and out of the load zone
            modulated_impulses = self._apply_load_zone_modulation(
                impulse_train, t, rpm
            )
            
            # Convolve with structural response
            fault_signal = signal.convolve(modulated_impulses, impulse_response, mode='same')
            
            # Scale and add to baseline
            waveform += fault_signal * 0.5
        
        return waveform
    
    def get_json_payload(
        self,
        rpm: float = None,
        fault_type: Literal["normal", "outer_race", "inner_race"] = "normal",
        fault_amplitude: float = 1.0
    ) -> dict:
        """
        Generate a complete JSON-serializable data packet for streaming.
        
        This method packages the raw vibration waveform with metadata,
        suitable for transmission to a backend for FFT analysis and storage.
        
        Args:
            rpm: Shaft speed in RPM (uses stored value if None)
            fault_type: Type of fault to simulate
            fault_amplitude: Fault severity (0.0 = no fault, 1.0 = nominal, 2.0 = severe)
            
        Returns:
            Dictionary containing:
                - machine_id: Equipment identifier
                - timestamp: ISO 8601 formatted timestamp
                - rpm: Current shaft speed
                - fault_type: Type of fault being simulated
                - sampling_rate: Waveform sampling frequency
                - vibration_raw: List of 2048 float values (the waveform)
        """
        if rpm is None:
            rpm = self._current_rpm
            
        # Generate the waveform
        waveform = self.generate_waveform(rpm, fault_type, fault_amplitude)
        
        # Package as JSON-compatible dictionary
        payload = {
            "machine_id": self.machine_id,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "rpm": float(rpm),
            "fault_type": fault_type,
            "fault_amplitude": float(fault_amplitude),
            "sampling_rate": self.fs,
            "duration_seconds": self.duration,
            "vibration_raw": waveform.tolist()  # Convert numpy array to list
        }
        
        return payload


def simulate_degradation_cycle():
    """
    Simulate a machine degrading from normal operation to developed fault.
    
    Timeline:
        Seconds 0-10:  Normal operation (healthy bearing)
        Seconds 11-20: Developing outer race fault (amplitude increases linearly)
    
    This demonstrates the progressive nature of bearing failures and provides
    data suitable for training prognostic models.
    """
    print("=" * 70)
    print("HIGH-FIDELITY VIBRATION SIMULATOR - Bearing Degradation Demo")
    print("=" * 70)
    print(f"\nPhysics Parameters:")
    print(f"  Sampling Rate: 2048 Hz")
    print(f"  Waveform Duration: 1.0 second")
    print(f"  Samples per Snapshot: 2048")
    print(f"  BPFO Ratio: 5.4× shaft frequency")
    print(f"  BPFI Ratio: 7.2× shaft frequency")
    print("\n" + "-" * 70)
    
    # Initialize simulator
    simulator = VibrationSimulator(
        fs=2048,
        duration=1.0,
        machine_id="CNC-001"
    )
    
    # Operating parameters
    nominal_rpm = 1800.0  # Typical for industrial machinery
    
    # Run simulation for 20 seconds
    total_seconds = 20
    
    for second in range(total_seconds):
        # Determine fault state based on timeline
        if second < 10:
            # Phase 1: Normal operation
            fault_type = "normal"
            fault_amplitude = 0.0
            phase_name = "NORMAL"
        else:
            # Phase 2: Developing outer race fault
            # Amplitude increases linearly from 0.2 to 1.0
            fault_type = "outer_race"
            progress = (second - 10) / 10.0  # 0.0 to 1.0
            fault_amplitude = 0.2 + 0.8 * progress
            phase_name = "DEVELOPING FAULT"
        
        # Add slight RPM variation (±2%)
        rpm_variation = np.random.uniform(-0.02, 0.02) * nominal_rpm
        current_rpm = nominal_rpm + rpm_variation
        
        # Generate data payload
        payload = simulator.get_json_payload(
            rpm=current_rpm,
            fault_type=fault_type,
            fault_amplitude=fault_amplitude
        )
        
        # Convert to JSON string for size measurement
        json_str = json.dumps(payload)
        json_size_kb = len(json_str) / 1024
        
        # Print status
        print(f"\n[T+{second:02d}s] {phase_name}")
        print(f"  RPM: {current_rpm:.1f}")
        print(f"  Fault Type: {fault_type}")
        print(f"  Fault Amplitude: {fault_amplitude:.2f}")
        print(f"  Waveform Points: {len(payload['vibration_raw'])}")
        print(f"  JSON Payload Size: {json_size_kb:.2f} KB")
        
        # Show waveform statistics
        waveform = np.array(payload['vibration_raw'])
        print(f"  Signal Stats: mean={waveform.mean():.4f}, "
              f"std={waveform.std():.4f}, "
              f"peak={np.abs(waveform).max():.4f}")
        
        # Simulate real-time streaming (1 second interval)
        time.sleep(1.0)
    
    print("\n" + "=" * 70)
    print("Simulation Complete")
    print("=" * 70)
    print("\nSummary:")
    print("  - Generated 20 waveform snapshots")
    print("  - Each containing 2048 samples at 2048 Hz")
    print("  - Demonstrated normal → fault progression")
    print("  - Ready for FFT analysis and ML training")


if __name__ == "__main__":
    simulate_degradation_cycle()
