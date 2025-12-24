#!/usr/bin/env python3
"""
Discovery Engine - Main Orchestrator
Coordinates all anomaly detection and correlation discovery.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import asdict
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

from .detectors import EnsembleAnomalyDetector, IsolationForestDetector
from .analyzers import CorrelationAnalyzer

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")


class DiscoveryEngine:
    """
    Main orchestrator for anomaly discovery and correlation analysis.
    
    Features:
    - Trains and manages detection models
    - Runs continuous anomaly detection
    - Performs correlation analysis
    - Persists discoveries to database
    - Generates insights
    """
    
    def __init__(
        self,
        model_dir: str = 'anomaly_discovery/models/trained',
        feature_columns: List[str] = None
    ):
        self.model_dir = model_dir
        self.feature_columns = feature_columns or [
            'failure_probability', 'rul_days', 'degradation_score',
            'bpfi_amp', 'bpfo_amp', 'bsf_amp', 'ftf_amp',
            'rms_velocity', 'peak_velocity', 'crest_factor',
            'kurtosis', 'rotational_speed', 'vibration_rms'
        ]
        
        self.ensemble = EnsembleAnomalyDetector()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.is_trained = False
    
    def get_db(self):
        """Get database connection."""
        return psycopg2.connect(DATABASE_URL)
    
    def train(
        self,
        days_of_data: int = 30,
        min_samples: int = 1000
    ) -> Dict:
        """
        Train detection models on historical data.
        """
        print("\n" + "=" * 60)
        print("DISCOVERY ENGINE - TRAINING")
        print("=" * 60)
        
        # Fetch historical data
        conn = self.get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT * FROM cwru_features
            WHERE timestamp > NOW() - INTERVAL '%s days'
            ORDER BY timestamp ASC
        """, (days_of_data,))
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if len(rows) < min_samples:
            return {
                'success': False,
                'error': f'Insufficient data: {len(rows)} samples (need {min_samples})'
            }
        
        data = pd.DataFrame(rows)
        
        # Filter to available features
        available_features = [f for f in self.feature_columns if f in data.columns]
        
        print(f"\n[DiscoveryEngine] Training on {len(data)} samples, {len(available_features)} features")
        
        # Train ensemble
        self.ensemble.fit(data, available_features, ae_epochs=30)
        
        # Save models
        os.makedirs(self.model_dir, exist_ok=True)
        self.ensemble.save(self.model_dir)
        
        self.is_trained = True
        
        return {
            'success': True,
            'samples_used': len(data),
            'features_used': available_features,
            'training_timestamp': datetime.now().isoformat()
        }
    
    def detect_anomalies(
        self,
        hours_back: float = 1.0,
        persist: bool = True
    ) -> Dict:
        """
        Run anomaly detection on recent data.
        """
        if not self.is_trained:
            # Try to load saved models
            try:
                self.ensemble.load(self.model_dir)
                self.is_trained = True
            except Exception as e:
                return {'success': False, 'error': f'Models not trained: {e}'}
        
        # Fetch recent data
        conn = self.get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT * FROM cwru_features
            WHERE timestamp > NOW() - INTERVAL '%s hours'
            ORDER BY timestamp ASC
        """, (hours_back,))
        
        rows = cursor.fetchall()
        cursor.close()
        
        if not rows:
            conn.close()
            return {'success': True, 'anomalies': [], 'message': 'No recent data'}
        
        data = pd.DataFrame(rows)
        
        # Run detection
        results = self.ensemble.predict(data)
        
        # Persist anomalies
        if persist and results['anomaly_details']:
            self._persist_anomalies(conn, results['anomaly_details'])
        
        conn.close()
        
        return {
            'success': True,
            'anomaly_count': results['anomaly_count'],
            'anomalies': [asdict(a) for a in results['anomaly_details']],
            'detector_contributions': results['detector_contributions']
        }
    
    def analyze_correlations(
        self,
        days_back: int = 7,
        persist: bool = True
    ) -> Dict:
        """
        Discover cross-machine correlations.
        """
        # Fetch data
        conn = self.get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT * FROM cwru_features
            WHERE timestamp > NOW() - INTERVAL '%s days'
            ORDER BY timestamp ASC
        """, (days_back,))
        
        rows = cursor.fetchall()
        
        # Fetch Work Orders for maintenance correlation
        cursor.execute("""
            SELECT id, machine_id, work_type, created_at, started_at
            FROM work_orders
            WHERE created_at > NOW() - INTERVAL '%s days'
            ORDER BY created_at ASC
        """, (days_back,))
        wo_rows = cursor.fetchall()
        work_orders = pd.DataFrame(wo_rows) if wo_rows else pd.DataFrame()
        
        cursor.close()
        
        if len(rows) < 500:
            conn.close()
            return {'success': False, 'error': 'Insufficient data for correlation analysis'}
        
        data = pd.DataFrame(rows)
        
        # Filter to available features
        available_features = [f for f in self.feature_columns if f in data.columns]
        
        # Run analysis (Modified to use our advanced find_correlations logic if available, 
        # but since we're using the class directly, we need to handle it or update the import)
        # Note: We added find_correlations convenience function in correlation.py which creates the analyzers.
        # Let's use the analyzers directly here to keep it explicit.
        
        # 1. Standard Analysis
        results = self.correlation_analyzer.analyze_correlations(
            data,
            machine_col='machine_id',
            timestamp_col='timestamp',
            feature_columns=available_features
        )
        
        # 2. Maintenance Analysis
        from .analyzers.correlation import MaintenanceCorrelationAnalyzer
        if not work_orders.empty:
            # We need anomaly scores. Ideally we fetch them or compute them. 
            # For now, let's look if 'failure_probability' or vibration is in data.
            # Or better, fetch actual anomaly detections to use as the target
            
            # Use 'failure_probability' if available as a proxy for machine health score
            if 'failure_probability' not in data.columns and 'vibration_rms' in data.columns:
                data['ensemble_score'] = data['vibration_rms'] # Proxy
            elif 'failure_probability' in data.columns:
                data['ensemble_score'] = data['failure_probability']
                
            maint_analyzer = MaintenanceCorrelationAnalyzer(days_lookback=days_back)
            try:
                maint_discoveries = maint_analyzer.analyze_impact(data, work_orders)
                results['discoveries'].extend(maint_discoveries)
                
                # Update summary
                if 'summary' not in results: results['summary'] = {}
                results['summary']['maintenance_correlations'] = len(maint_discoveries)
            except Exception as e:
                print(f"[DiscoveryEngine] Maintenance correlation failed: {e}")

        
        # Persist discoveries
        if persist and results['discoveries']:
            self._persist_correlations(conn, results['discoveries'])
            
        conn.close()
        
        return {
            'success': True,
            'correlation_count': len(results['discoveries']),
            'correlations': [asdict(c) for c in results['discoveries'][:20]],  # Top 20
            'summary': results['summary']
        }
    
    def generate_insight(
        self,
        anomaly_ids: List[str] = None,
        correlation_ids: List[str] = None
    ) -> Dict:
        """
        Generate human-readable insights from discoveries.
        """
        conn = self.get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get recent anomalies
        cursor.execute("""
            SELECT * FROM anomaly_detections
            WHERE created_at > NOW() - INTERVAL '24 hours'
            ORDER BY ensemble_score DESC
            LIMIT 10
        """)
        anomalies = cursor.fetchall()
        
        # Get recent correlations
        cursor.execute("""
            SELECT * FROM correlation_discoveries
            WHERE created_at > NOW() - INTERVAL '7 days'
            ORDER BY ABS(correlation_coefficient) DESC
            LIMIT 10
        """)
        correlations = cursor.fetchall()
        
        # Generate insights based on patterns
        insights = []
        
        # Insight: Multiple high-severity anomalies
        high_severity = [a for a in anomalies if a.get('severity') in ['high', 'critical']]
        if len(high_severity) >= 3:
            machines = list(set([a['machine_id'] for a in high_severity]))
            insight = {
                'title': f'Cluster of {len(high_severity)} High-Severity Anomalies Detected',
                'summary': f'Multiple machines ({", ".join(machines)}) showing abnormal behavior in the last 24 hours.',
                'priority': 'high',
                'insight_type': 'anomaly',
                'machine_ids': machines,
                'recommended_actions': [
                    'Review machine operating parameters',
                    'Check for common environmental factors',
                    'Consider preventive maintenance window'
                ]
            }
            insights.append(insight)
        
        # Insight: Strong causal relationship discovered
        causal = [c for c in correlations if c.get('granger_causal')]
        if causal:
            c = causal[0]
            insight = {
                'title': f'Causal Relationship: {c["source_machine_id"]} → {c["target_machine_id"]}',
                'summary': (
                    f'{c["source_feature"]} on {c["source_machine_id"]} appears to cause '
                    f'changes in {c["target_feature"]} on {c["target_machine_id"]} '
                    f'with {c["lag_hours"]}h delay.'
                ),
                'priority': 'medium',
                'insight_type': 'correlation',
                'machine_ids': [c['source_machine_id'], c['target_machine_id']],
                'recommended_actions': [
                    f'Monitor {c["source_machine_id"]} as leading indicator',
                    f'Consider {c["source_feature"]} thresholds for early warning'
                ]
            }
            insights.append(insight)
        
        # Persist insights
        for insight in insights:
            self._persist_insight(cursor, insight)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return {
            'success': True,
            'insights_generated': len(insights),
            'insights': insights
        }
    
    def _persist_anomalies(self, conn, anomalies: List) -> None:
        """Save anomaly detections to database."""
        cursor = conn.cursor()
        
        for a in anomalies:
            # Generate detection ID
            detection_id = f"AD-{datetime.now().strftime('%Y%m%d%H%M%S')}-{a.index:04d}"
            
            cursor.execute("""
                INSERT INTO anomaly_detections
                (detection_id, machine_id, timestamp, isolation_forest_score, 
                 autoencoder_score, ensemble_score, anomaly_type, severity,
                 confidence, anomalous_features, feature_deviations)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (detection_id) DO NOTHING
            """, (
                detection_id,
                a.machine_id or 'unknown',
                a.timestamp or datetime.now(),
                a.isolation_forest_score,
                a.autoencoder_score,
                a.ensemble_score,
                a.anomaly_type,
                a.severity,
                a.confidence,
                json.dumps(a.feature_values),
                json.dumps({f['feature']: f['importance'] for f in a.top_features[:5]})
            ))
        
        conn.commit()
        cursor.close()
        print(f"[DiscoveryEngine] Persisted {len(anomalies)} anomalies")
    
    def _persist_correlations(self, conn, correlations: List) -> None:
        """Save correlation discoveries to database."""
        cursor = conn.cursor()
        
        for i, c in enumerate(correlations[:50]):  # Limit to top 50
            correlation_id = f"CD-{datetime.now().strftime('%Y%m%d')}-{i:04d}"
            
            cursor.execute("""
                INSERT INTO correlation_discoveries
                (correlation_id, source_machine_id, source_feature,
                 target_machine_id, target_feature, correlation_coefficient,
                 p_value, lag_hours, granger_causal, discovery_type,
                 strength, confidence, sample_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (correlation_id) DO NOTHING
            """, (
                correlation_id,
                c.source_machine,
                c.source_feature,
                c.target_machine,
                c.target_feature,
                c.correlation,
                c.p_value,
                c.lag_hours,
                c.granger_causal,
                c.discovery_type,
                c.strength,
                c.confidence,
                c.sample_count
            ))
        
        conn.commit()
        cursor.close()
        print(f"[DiscoveryEngine] Persisted {min(len(correlations), 50)} correlations")
    
    def _persist_insight(self, cursor, insight: Dict) -> None:
        """Save insight to database."""
        insight_id = f"INS-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        cursor.execute("""
            INSERT INTO discovery_insights
            (insight_id, title, summary, insight_type, priority,
             machine_ids, recommended_actions, generated_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'discovery_engine')
            ON CONFLICT (insight_id) DO NOTHING
        """, (
            insight_id,
            insight['title'],
            insight['summary'],
            insight['insight_type'],
            insight['priority'],
            insight.get('machine_ids', []),
            insight.get('recommended_actions', [])
        ))


# Convenience function
def run_discovery(hours_back: float = 1.0, train_first: bool = False) -> Dict:
    """Quick discovery run."""
    engine = DiscoveryEngine()
    
    if train_first:
        result = engine.train()
        if not result['success']:
            return result
    
    return engine.detect_anomalies(hours_back=hours_back)
