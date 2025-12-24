#!/usr/bin/env python3
"""
Ensemble Anomaly Detector
Combines multiple detection algorithms for robust anomaly identification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import os

from .isolation_forest import IsolationForestDetector
from .temporal_autoencoder import TemporalAutoencoderDetector


@dataclass
class AnomalyResult:
    """Structured anomaly detection result."""
    index: int
    timestamp: Optional[datetime]
    machine_id: Optional[str]
    ensemble_score: float
    isolation_forest_score: float
    autoencoder_score: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    confidence: float
    anomaly_type: str  # 'point', 'contextual', 'collective'
    top_features: List[Dict]
    feature_values: Dict
    explanation: str


class EnsembleAnomalyDetector:
    """
    Ensemble detector combining multiple algorithms.
    
    Combines:
    - Isolation Forest (point anomalies)
    - Temporal Autoencoder (sequence anomalies)
    
    Features:
    - Weighted score combination
    - Voting-based and score-based fusion
    - Adaptive thresholding
    - Confidence calibration
    """
    
    def __init__(
        self,
        if_contamination: float = 0.01,
        ae_threshold_percentile: float = 95,
        weights: Dict[str, float] = None,
        min_agreement: int = 1  # Minimum detectors that must flag anomaly
    ):
        self.if_contamination = if_contamination
        self.ae_threshold_percentile = ae_threshold_percentile
        self.weights = weights or {'isolation_forest': 0.5, 'autoencoder': 0.5}
        self.min_agreement = min_agreement
        
        # Initialize detectors
        self.isolation_forest = IsolationForestDetector(contamination=if_contamination)
        self.autoencoder = TemporalAutoencoderDetector(
            threshold_percentile=ae_threshold_percentile
        )
        
        self.is_trained = False
        self.training_timestamp: Optional[str] = None
        self.feature_names: List[str] = []
        
        # Thresholds for severity classification
        self.severity_thresholds = {
            'critical': 0.9,
            'high': 0.75,
            'medium': 0.5,
            'low': 0.25
        }
    
    def fit(
        self, 
        data: pd.DataFrame, 
        feature_columns: List[str] = None,
        ae_epochs: int = 50
    ) -> 'EnsembleAnomalyDetector':
        """
        Train all ensemble members on historical data.
        """
        print("\n" + "=" * 60)
        print("TRAINING ENSEMBLE ANOMALY DETECTOR")
        print("=" * 60)
        
        # Determine features
        if feature_columns:
            self.feature_names = feature_columns
        else:
            self.feature_names = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Train Isolation Forest
        print("\n[1/2] Training Isolation Forest...")
        self.isolation_forest.fit(data, self.feature_names)
        
        # Train Temporal Autoencoder
        print("\n[2/2] Training Temporal Autoencoder...")
        self.autoencoder.fit(data, self.feature_names, epochs=ae_epochs)
        
        self.is_trained = True
        self.training_timestamp = datetime.now().isoformat()
        
        print("\n" + "=" * 60)
        print("ENSEMBLE TRAINING COMPLETE")
        print("=" * 60)
        
        return self
    
    def predict(
        self, 
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        machine_id_col: str = 'machine_id'
    ) -> Dict:
        """
        Detect anomalies using ensemble voting and scoring.
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble not trained. Call fit() first.")
        
        n_samples = len(data)
        
        # Get predictions from each detector
        print("[Ensemble] Running Isolation Forest...")
        if_results = self.isolation_forest.predict(data)
        
        print("[Ensemble] Running Temporal Autoencoder...")
        ae_results = self.autoencoder.predict(data)
        
        # Align results (autoencoder has fewer predictions due to windowing)
        ae_offset = self.autoencoder.sequence_length - 1
        
        # Initialize result arrays
        ensemble_scores = np.zeros(n_samples)
        if_scores = if_results['normalized_scores']
        ae_scores = np.zeros(n_samples)
        
        # Fill in autoencoder scores (with offset)
        if len(ae_results['normalized_scores']) > 0:
            ae_scores[ae_offset:ae_offset + len(ae_results['normalized_scores'])] = ae_results['normalized_scores']
        
        # Combine scores using weighted average
        for i in range(n_samples):
            if i < ae_offset:
                # Only IF available
                ensemble_scores[i] = if_scores[i]
            else:
                # Both available - weighted combination
                ensemble_scores[i] = (
                    self.weights['isolation_forest'] * if_scores[i] +
                    self.weights['autoencoder'] * ae_scores[i]
                )
        
        # Voting: count how many detectors flag each point
        vote_count = np.zeros(n_samples, dtype=int)
        vote_count += if_results['is_anomaly'].astype(int)
        
        ae_is_anomaly = np.zeros(n_samples, dtype=bool)
        if len(ae_results['is_anomaly']) > 0:
            ae_is_anomaly[ae_offset:ae_offset + len(ae_results['is_anomaly'])] = ae_results['is_anomaly']
        vote_count += ae_is_anomaly.astype(int)
        
        # Final anomaly decision: ensemble score threshold + minimum agreement
        score_threshold = 0.5
        is_anomaly = (ensemble_scores > score_threshold) & (vote_count >= self.min_agreement)
        
        # Build detailed results
        anomaly_details = []
        anomaly_indices = np.where(is_anomaly)[0]
        
        for idx in anomaly_indices:
            # Get timestamp and machine_id if available
            timestamp = data.iloc[idx].get(timestamp_col) if timestamp_col in data.columns else None
            machine_id = data.iloc[idx].get(machine_id_col) if machine_id_col in data.columns else None
            
            # Determine severity
            score = ensemble_scores[idx]
            severity = self._get_severity(score)
            
            # Get confidence
            confidence = self._calculate_confidence(score, vote_count[idx])
            
            # Determine anomaly type
            anomaly_type = self._classify_anomaly_type(
                idx, if_results, ae_results, ae_offset, ae_is_anomaly
            )
            
            # Get top contributing features
            top_features = []
            if if_results.get('anomaly_details'):
                for detail in if_results['anomaly_details']:
                    if detail['index'] == idx:
                        top_features = detail.get('top_features', [])
                        break
            
            # Get feature values
            feature_values = {
                name: float(data.iloc[idx][name]) 
                for name in self.feature_names 
                if name in data.columns
            }
            
            # Generate explanation
            explanation = self._generate_explanation(
                idx, score, severity, anomaly_type, top_features, vote_count[idx]
            )
            
            result = AnomalyResult(
                index=int(idx),
                timestamp=timestamp,
                machine_id=machine_id,
                ensemble_score=float(score),
                isolation_forest_score=float(if_scores[idx]),
                autoencoder_score=float(ae_scores[idx]),
                severity=severity,
                confidence=float(confidence),
                anomaly_type=anomaly_type,
                top_features=top_features,
                feature_values=feature_values,
                explanation=explanation
            )
            anomaly_details.append(result)
        
        return {
            'ensemble_scores': ensemble_scores,
            'isolation_forest_scores': if_scores,
            'autoencoder_scores': ae_scores,
            'vote_counts': vote_count,
            'is_anomaly': is_anomaly,
            'anomaly_count': len(anomaly_details),
            'anomaly_details': anomaly_details,
            'detector_contributions': {
                'isolation_forest': int(if_results['is_anomaly'].sum()),
                'autoencoder': int(ae_is_anomaly.sum())
            }
        }
    
    def _get_severity(self, score: float) -> str:
        """Classify severity based on ensemble score."""
        if score >= self.severity_thresholds['critical']:
            return 'critical'
        elif score >= self.severity_thresholds['high']:
            return 'high'
        elif score >= self.severity_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_confidence(self, score: float, vote_count: int) -> float:
        """Calculate confidence in the anomaly detection."""
        # Base confidence from score
        base_confidence = min(0.95, score)
        
        # Boost for agreement between detectors
        agreement_boost = (vote_count - 1) * 0.1  # +10% per additional detector
        
        return min(0.99, base_confidence + agreement_boost)
    
    def _classify_anomaly_type(
        self, 
        idx: int, 
        if_results: Dict, 
        ae_results: Dict,
        ae_offset: int,
        ae_is_anomaly: np.ndarray
    ) -> str:
        """Classify the type of anomaly detected."""
        is_if_anomaly = if_results['is_anomaly'][idx]
        is_ae_anomaly = ae_is_anomaly[idx]
        
        if is_if_anomaly and is_ae_anomaly:
            # Both detect - likely a significant contextual anomaly
            return 'contextual'
        elif is_if_anomaly and not is_ae_anomaly:
            # Only point-based detector - single point anomaly
            return 'point'
        elif is_ae_anomaly and not is_if_anomaly:
            # Only sequence-based - temporal pattern anomaly
            return 'collective'
        else:
            # Ensemble score triggered but neither detector alone
            return 'weak_signal'
    
    def _generate_explanation(
        self,
        idx: int,
        score: float,
        severity: str,
        anomaly_type: str,
        top_features: List[Dict],
        vote_count: int
    ) -> str:
        """Generate human-readable explanation."""
        type_descriptions = {
            'point': 'Unusual single-point reading',
            'contextual': 'Anomalous pattern in context',
            'collective': 'Unusual sequence of readings',
            'weak_signal': 'Subtle deviation detected'
        }
        
        explanation = f"{type_descriptions.get(anomaly_type, 'Anomaly detected')}"
        
        if top_features:
            feature_str = ", ".join([f"{f['feature']}" for f in top_features[:3]])
            explanation += f" in {feature_str}"
        
        explanation += f". Severity: {severity.upper()}. "
        explanation += f"Detected by {vote_count} algorithm(s) with confidence {score:.1%}."
        
        return explanation
    
    def save(self, directory: str) -> None:
        """Save all models to directory."""
        os.makedirs(directory, exist_ok=True)
        
        self.isolation_forest.save(os.path.join(directory, 'isolation_forest.pkl'))
        self.autoencoder.save(os.path.join(directory, 'autoencoder.h5'))
        
        # Save ensemble metadata
        import json
        metadata = {
            'weights': self.weights,
            'min_agreement': self.min_agreement,
            'severity_thresholds': self.severity_thresholds,
            'feature_names': self.feature_names,
            'training_timestamp': self.training_timestamp
        }
        with open(os.path.join(directory, 'ensemble_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[Ensemble] Models saved to {directory}")
    
    def load(self, directory: str) -> 'EnsembleAnomalyDetector':
        """Load all models from directory."""
        self.isolation_forest.load(os.path.join(directory, 'isolation_forest.pkl'))
        self.autoencoder.load(os.path.join(directory, 'autoencoder.h5'))
        
        # Load metadata
        import json
        with open(os.path.join(directory, 'ensemble_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        self.weights = metadata['weights']
        self.min_agreement = metadata['min_agreement']
        self.severity_thresholds = metadata['severity_thresholds']
        self.feature_names = metadata['feature_names']
        self.training_timestamp = metadata['training_timestamp']
        self.is_trained = True
        
        print(f"[Ensemble] Models loaded from {directory}")
        return self
