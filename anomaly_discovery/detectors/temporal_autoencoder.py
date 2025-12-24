#!/usr/bin/env python3
"""
Temporal Autoencoder Anomaly Detector
LSTM-based sequence anomaly detection for time-series patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime

# Try to import TensorFlow, fallback to simpler PCA-based approach if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("[TemporalAutoencoder] TensorFlow not available, using PCA fallback")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib


class TemporalAutoencoderDetector:
    """
    LSTM Autoencoder for detecting temporal anomalies in sequences.
    
    Learns the "normal" temporal patterns and flags sequences with
    high reconstruction error as anomalies.
    
    Features:
    - Variable sequence length support
    - Multi-feature reconstruction
    - Per-timestep anomaly localization
    - Reconstruction error distribution modeling
    """
    
    def __init__(
        self,
        sequence_length: int = 20,
        latent_dim: int = 16,
        lstm_units: int = 64,
        dropout_rate: float = 0.2,
        threshold_percentile: float = 95
    ):
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.threshold_percentile = threshold_percentile
        
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.n_features: int = 0
        self.reconstruction_threshold: float = 0.0
        self.error_mean: float = 0.0
        self.error_std: float = 1.0
        self.is_trained: bool = False
        self.training_timestamp: Optional[str] = None
        
        # Fallback for non-TF environments
        self.use_pca_fallback = not HAS_TENSORFLOW
        self.pca_model: Optional[PCA] = None
    
    def _build_model(self, n_features: int) -> Model:
        """Build LSTM Autoencoder architecture."""
        if not HAS_TENSORFLOW:
            return None
        
        # Encoder
        inputs = Input(shape=(self.sequence_length, n_features))
        
        # Encoder LSTM
        encoded = LSTM(self.lstm_units, activation='relu', return_sequences=True)(inputs)
        encoded = Dropout(self.dropout_rate)(encoded)
        encoded = LSTM(self.latent_dim, activation='relu', return_sequences=False)(encoded)
        
        # Bottleneck
        bottleneck = Dense(self.latent_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = RepeatVector(self.sequence_length)(bottleneck)
        decoded = LSTM(self.latent_dim, activation='relu', return_sequences=True)(decoded)
        decoded = Dropout(self.dropout_rate)(decoded)
        decoded = LSTM(self.lstm_units, activation='relu', return_sequences=True)(decoded)
        
        # Output
        outputs = TimeDistributed(Dense(n_features))(decoded)
        
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sliding window sequences from data."""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length])
        return np.array(sequences)
    
    def fit(
        self, 
        data: pd.DataFrame, 
        feature_columns: List[str] = None,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1
    ) -> 'TemporalAutoencoderDetector':
        """
        Train the autoencoder on historical 'normal' data.
        """
        # Determine features
        if feature_columns:
            self.feature_names = feature_columns
        else:
            self.feature_names = data.select_dtypes(include=[np.number]).columns.tolist()
        
        self.n_features = len(self.feature_names)
        
        # Extract and scale features
        X = data[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        sequences = self._create_sequences(X_scaled)
        
        if len(sequences) < 100:
            print(f"[TemporalAutoencoder] Warning: Only {len(sequences)} sequences available")
        
        if self.use_pca_fallback:
            # PCA-based fallback
            flat_sequences = sequences.reshape(len(sequences), -1)
            self.pca_model = PCA(n_components=min(self.latent_dim, flat_sequences.shape[1]))
            reduced = self.pca_model.fit_transform(flat_sequences)
            reconstructed_flat = self.pca_model.inverse_transform(reduced)
            
            # Calculate reconstruction errors
            errors = np.mean((flat_sequences - reconstructed_flat) ** 2, axis=1)
            
        else:
            # TensorFlow LSTM Autoencoder
            self.model = self._build_model(self.n_features)
            
            # Train with early stopping
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            self.model.fit(
                sequences, sequences,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stop],
                verbose=1
            )
            
            # Calculate reconstruction errors on training data
            reconstructed = self.model.predict(sequences, verbose=0)
            errors = np.mean((sequences - reconstructed) ** 2, axis=(1, 2))
        
        # Model error distribution
        self.error_mean = np.mean(errors)
        self.error_std = np.std(errors)
        self.reconstruction_threshold = np.percentile(errors, self.threshold_percentile)
        
        self.is_trained = True
        self.training_timestamp = datetime.now().isoformat()
        
        print(f"[TemporalAutoencoder] Trained on {len(sequences)} sequences")
        print(f"[TemporalAutoencoder] Threshold: {self.reconstruction_threshold:.6f}")
        
        return self
    
    def predict(self, data: pd.DataFrame) -> Dict:
        """
        Detect temporal anomalies in new data.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        # Extract and scale features
        X = data[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        sequences = self._create_sequences(X_scaled)
        
        if len(sequences) == 0:
            return {
                'reconstruction_errors': np.array([]),
                'normalized_scores': np.array([]),
                'is_anomaly': np.array([]),
                'anomaly_count': 0
            }
        
        if self.use_pca_fallback:
            flat_sequences = sequences.reshape(len(sequences), -1)
            reduced = self.pca_model.transform(flat_sequences)
            reconstructed_flat = self.pca_model.inverse_transform(reduced)
            reconstructed = reconstructed_flat.reshape(sequences.shape)
        else:
            reconstructed = self.model.predict(sequences, verbose=0)
        
        # Calculate reconstruction errors
        sequence_errors = np.mean((sequences - reconstructed) ** 2, axis=(1, 2))
        
        # Per-timestep errors for localization
        timestep_errors = np.mean((sequences - reconstructed) ** 2, axis=2)  # (n_sequences, seq_len)
        
        # Per-feature errors
        feature_errors = np.mean((sequences - reconstructed) ** 2, axis=1)  # (n_sequences, n_features)
        
        # Normalize scores (higher = more anomalous)
        normalized_scores = (sequence_errors - self.error_mean) / (self.error_std + 1e-10)
        normalized_scores = 1 / (1 + np.exp(-normalized_scores))  # Sigmoid normalization
        
        # Identify anomalies
        is_anomaly = sequence_errors > self.reconstruction_threshold
        
        result = {
            'reconstruction_errors': sequence_errors,
            'normalized_scores': normalized_scores,
            'is_anomaly': is_anomaly,
            'anomaly_count': is_anomaly.sum(),
            'timestep_errors': timestep_errors,
            'feature_errors': feature_errors,
            'threshold': self.reconstruction_threshold
        }
        
        # Add detailed anomaly info
        if is_anomaly.any():
            anomaly_indices = np.where(is_anomaly)[0]
            result['anomaly_details'] = []
            
            for idx in anomaly_indices:
                # Find most anomalous timesteps
                ts_err = timestep_errors[idx]
                top_timesteps = np.argsort(ts_err)[-3:][::-1]
                
                # Find most anomalous features
                feat_err = feature_errors[idx]
                top_features = np.argsort(feat_err)[-5:][::-1]
                
                detail = {
                    'sequence_index': int(idx),
                    'score': float(normalized_scores[idx]),
                    'reconstruction_error': float(sequence_errors[idx]),
                    'top_anomalous_timesteps': [int(t) for t in top_timesteps],
                    'top_anomalous_features': [
                        {'feature': self.feature_names[f], 'error': float(feat_err[f])}
                        for f in top_features
                    ]
                }
                result['anomaly_details'].append(detail)
        
        return result
    
    def get_sequence_confidence(self, reconstruction_error: float) -> float:
        """Calculate confidence that a sequence is anomalous."""
        z_score = (reconstruction_error - self.error_mean) / (self.error_std + 1e-10)
        # Higher z-score = more confident it's anomalous
        confidence = 1 / (1 + np.exp(-z_score + 2))  # Shift sigmoid
        return float(min(0.99, confidence))
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        state = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'n_features': self.n_features,
            'sequence_length': self.sequence_length,
            'latent_dim': self.latent_dim,
            'reconstruction_threshold': self.reconstruction_threshold,
            'error_mean': self.error_mean,
            'error_std': self.error_std,
            'training_timestamp': self.training_timestamp,
            'use_pca_fallback': self.use_pca_fallback,
            'pca_model': self.pca_model
        }
        
        # Save state
        joblib.dump(state, path.replace('.h5', '_state.pkl'))
        
        # Save Keras model separately if using TF
        if not self.use_pca_fallback and self.model is not None:
            self.model.save(path)
        
        print(f"[TemporalAutoencoder] Model saved to {path}")
    
    def load(self, path: str) -> 'TemporalAutoencoderDetector':
        """Load model from disk."""
        state = joblib.load(path.replace('.h5', '_state.pkl'))
        
        self.scaler = state['scaler']
        self.feature_names = state['feature_names']
        self.n_features = state['n_features']
        self.sequence_length = state['sequence_length']
        self.latent_dim = state['latent_dim']
        self.reconstruction_threshold = state['reconstruction_threshold']
        self.error_mean = state['error_mean']
        self.error_std = state['error_std']
        self.training_timestamp = state['training_timestamp']
        self.use_pca_fallback = state.get('use_pca_fallback', False)
        self.pca_model = state.get('pca_model')
        
        if not self.use_pca_fallback and HAS_TENSORFLOW:
            self.model = load_model(path)
        
        self.is_trained = True
        print(f"[TemporalAutoencoder] Model loaded from {path}")
        
        return self
