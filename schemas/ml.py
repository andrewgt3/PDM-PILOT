"""
Machine Learning Feature Schemas
================================
Shared Pydantic models for ML feature validation.
Prevents training-serving skew by ensuring same features are used
in training and inference.

TISAX/SOC2 Compliance:
    - Single source of truth for feature definitions
    - Type-safe validation at both training and inference time
    - Prevents silent schema drift
"""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
import numpy as np


class ModelFeatures(BaseModel):
    """
    Exact 26 input features expected by the RUL/Failure prediction model.
    
    This schema is the SINGLE SOURCE OF TRUTH for:
    - Training data validation (train_rul_model.py)
    - Inference input validation (analytics_engine.py, stream_consumer.py)
    - API request validation
    
    Feature Groups:
    1. Frequency Peaks (5): Dominant FFT frequencies
    2. Peak Amplitudes (5): Corresponding amplitudes
    3. Band Powers (3): Low/Mid/High frequency energy
    4. Spectral Features (3): Entropy, Kurtosis, Total Power
    5. Bearing Defect Frequencies (5): BPFO, BPFI, BSF, FTF, Sideband
    6. Degradation (1): Composite health score
    7. Telemetry (4): rotational_speed, temperature, torque, tool_wear
    """
    
    # ==========================================================================
    # FREQUENCY PEAKS (FFT Analysis)
    # ==========================================================================
    peak_freq_1: float = Field(..., ge=0, description="1st dominant frequency (Hz)")
    peak_freq_2: float = Field(..., ge=0, description="2nd dominant frequency (Hz)")
    peak_freq_3: float = Field(..., ge=0, description="3rd dominant frequency (Hz)")
    peak_freq_4: float = Field(..., ge=0, description="4th dominant frequency (Hz)")
    peak_freq_5: float = Field(..., ge=0, description="5th dominant frequency (Hz)")
    
    # ==========================================================================
    # PEAK AMPLITUDES
    # ==========================================================================
    peak_amp_1: float = Field(..., ge=0, description="Amplitude of 1st peak")
    peak_amp_2: float = Field(..., ge=0, description="Amplitude of 2nd peak")
    peak_amp_3: float = Field(..., ge=0, description="Amplitude of 3rd peak")
    peak_amp_4: float = Field(..., ge=0, description="Amplitude of 4th peak")
    peak_amp_5: float = Field(..., ge=0, description="Amplitude of 5th peak")
    
    # ==========================================================================
    # BAND POWERS
    # ==========================================================================
    low_band_power: float = Field(..., ge=0, description="Power in low frequency band")
    mid_band_power: float = Field(..., ge=0, description="Power in mid frequency band")
    high_band_power: float = Field(..., ge=0, description="Power in high frequency band")
    
    # ==========================================================================
    # SPECTRAL FEATURES
    # ==========================================================================
    spectral_entropy: float = Field(..., ge=0, description="Entropy of power spectrum")
    spectral_kurtosis: float = Field(..., description="Kurtosis of power spectrum")
    total_power: float = Field(..., ge=0, description="Total signal power")
    
    # ==========================================================================
    # BEARING DEFECT FREQUENCIES
    # ==========================================================================
    bpfo_amp: float = Field(..., ge=0, description="Ball Pass Frequency Outer race amplitude")
    bpfi_amp: float = Field(..., ge=0, description="Ball Pass Frequency Inner race amplitude")
    bsf_amp: float = Field(..., ge=0, description="Ball Spin Frequency amplitude")
    ftf_amp: float = Field(..., ge=0, description="Fundamental Train Frequency amplitude")
    sideband_strength: float = Field(..., ge=0, description="Sideband modulation strength")
    
    # ==========================================================================
    # DEGRADATION SCORE
    # ==========================================================================
    degradation_score: float = Field(..., ge=0, le=1, description="Composite degradation score [0-1]")
    
    # ==========================================================================
    # TELEMETRY
    # ==========================================================================
    rotational_speed: float = Field(..., ge=0, description="Rotational speed (RPM)")
    temperature: float = Field(..., description="Operating temperature (°C)")
    torque: float = Field(..., ge=0, description="Motor torque (Nm)")
    tool_wear: float = Field(..., ge=0, le=1, description="Tool wear ratio [0-1]")
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Reject unexpected fields
        validate_assignment = True
    
    @classmethod
    def feature_names(cls) -> List[str]:
        """Return ordered list of feature names for model input."""
        return [
            # Frequency peaks
            "peak_freq_1", "peak_freq_2", "peak_freq_3", "peak_freq_4", "peak_freq_5",
            # Peak amplitudes
            "peak_amp_1", "peak_amp_2", "peak_amp_3", "peak_amp_4", "peak_amp_5",
            # Band powers
            "low_band_power", "mid_band_power", "high_band_power",
            # Spectral features
            "spectral_entropy", "spectral_kurtosis", "total_power",
            # Bearing defect frequencies
            "bpfo_amp", "bpfi_amp", "bsf_amp", "ftf_amp", "sideband_strength",
            # Degradation
            "degradation_score",
            # Telemetry
            "rotational_speed", "temperature", "torque", "tool_wear",
        ]
    
    @classmethod
    def expected_count(cls) -> int:
        """Return expected number of features (26)."""
        return 26
    
    def to_array(self) -> "np.ndarray":
        """Convert to numpy array in correct feature order for model.predict()."""
        return np.array([[getattr(self, name) for name in self.feature_names()]])
    
    @classmethod
    def from_dict(cls, data: dict) -> "ModelFeatures":
        """
        Create ModelFeatures from a dictionary, handling missing/extra keys.
        Useful for validating training data rows.
        """
        # Filter to only expected keys
        expected_keys = set(cls.feature_names())
        filtered = {k: v for k, v in data.items() if k in expected_keys}
        
        # Replace NaN/inf with 0
        for k, v in filtered.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                filtered[k] = 0.0
        
        return cls(**filtered)
    
    @classmethod
    def validate_dataframe_columns(cls, columns: List[str]) -> dict:
        """
        Validate that a DataFrame has the required columns.
        
        Returns:
            dict with 'valid', 'missing', 'extra' keys
        """
        expected = set(cls.feature_names())
        actual = set(columns)
        
        missing = expected - actual
        extra = actual - expected
        
        return {
            "valid": len(missing) == 0,
            "missing": list(missing),
            "extra": list(extra),
            "expected_count": cls.expected_count(),
            "actual_count": len(actual)
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_training_features(feature_columns: List[str]) -> None:
    """
    Validate training feature columns match expected schema.
    Raises ValueError if mismatch.
    """
    result = ModelFeatures.validate_dataframe_columns(feature_columns)
    
    if not result["valid"]:
        raise ValueError(
            f"Training features do not match schema. "
            f"Missing: {result['missing']}. "
            f"Expected {result['expected_count']}, got {result['actual_count']}"
        )


def validate_inference_input(features_dict: dict) -> ModelFeatures:
    """
    Validate inference input and return ModelFeatures instance.
    Raises ValidationError if invalid.
    """
    return ModelFeatures.from_dict(features_dict)
