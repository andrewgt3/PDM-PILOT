#!/usr/bin/env python3
"""
Anomaly Discovery Background Worker
Runs continuously to detect anomalies and discover correlations.

Usage:
    python3 discovery_worker.py

Options:
    --detection-interval 300    Seconds between detection runs (default: 5 min)
    --correlation-interval 3600 Seconds between correlation analysis (default: 1 hour)
"""

import os
import sys
import time
import argparse
import signal
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DiscoveryWorker:
    """Background worker for continuous anomaly detection."""
    
    def __init__(
        self,
        detection_interval: int = 300,
        correlation_interval: int = 3600,
        detection_window_hours: float = 1.0
    ):
        self.detection_interval = detection_interval
        self.correlation_interval = correlation_interval
        self.detection_window_hours = detection_window_hours
        
        self.running = True
        self.last_detection = 0
        self.last_correlation = 0
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        # Lazy load engine
        self._engine = None
    
    @property
    def engine(self):
        if self._engine is None:
            from anomaly_discovery.discovery_engine import DiscoveryEngine
            self._engine = DiscoveryEngine()
            # Try to load existing models
            try:
                self._engine.ensemble.load(self._engine.model_dir)
                self._engine.is_trained = True
                print("[Worker] Loaded existing models")
            except Exception:
                print("[Worker] No existing models found, will train on first run")
        return self._engine
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\n[Worker] Shutdown signal received, stopping...")
        self.running = False
    
    def run(self):
        """Main worker loop."""
        print("=" * 60)
        print("ANOMALY DISCOVERY WORKER")
        print("=" * 60)
        print(f"Detection interval: {self.detection_interval}s")
        print(f"Correlation interval: {self.correlation_interval}s")
        print(f"Detection window: {self.detection_window_hours}h")
        print("=" * 60)
        print("\n[Worker] Starting... Press Ctrl+C to stop.\n")
        
        # Initial training if needed
        if not self.engine.is_trained:
            print("[Worker] Training models on historical data...")
            result = self.engine.train(days_of_data=7, min_samples=100)
            if result['success']:
                print(f"[Worker] Training complete: {result['samples_used']} samples")
            else:
                print(f"[Worker] Training failed: {result.get('error', 'Unknown error')}")
        
        while self.running:
            current_time = time.time()
            
            # Run anomaly detection
            if current_time - self.last_detection >= self.detection_interval:
                self._run_detection()
                self.last_detection = current_time
            
            # Run correlation analysis (less frequently)
            if current_time - self.last_correlation >= self.correlation_interval:
                self._run_correlation_analysis()
                self.last_correlation = current_time
            
            # Sleep for a bit
            time.sleep(10)
        
        print("[Worker] Stopped.")
    
    def _run_detection(self):
        """Run anomaly detection cycle."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] Running anomaly detection...")
        
        try:
            result = self.engine.detect_anomalies(
                hours_back=self.detection_window_hours,
                persist=True
            )
            
            if result['success']:
                count = result.get('anomaly_count', 0)
                if count > 0:
                    print(f"[{timestamp}] ⚠️  Detected {count} anomalies")
                    for a in result.get('anomalies', [])[:3]:
                        print(f"    - {a.get('machine_id', 'Unknown')}: {a.get('severity', 'unknown')} ({a.get('anomaly_type', 'unknown')})")
                else:
                    print(f"[{timestamp}] ✓ No anomalies detected")
            else:
                print(f"[{timestamp}] Detection failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"[{timestamp}] Detection error: {e}")
    
    def _run_correlation_analysis(self):
        """Run correlation analysis cycle."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] Running correlation analysis...")
        
        try:
            result = self.engine.analyze_correlations(
                days_back=3,
                persist=True
            )
            
            if result['success']:
                count = result.get('correlation_count', 0)
                causal = result.get('summary', {}).get('causal_relationships', 0)
                print(f"[{timestamp}] Found {count} correlations ({causal} causal)")
            else:
                print(f"[{timestamp}] Correlation analysis failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"[{timestamp}] Correlation analysis error: {e}")
    
    def run_once(self):
        """Run detection once (for testing)."""
        if not self.engine.is_trained:
            self.engine.train(days_of_data=7, min_samples=100)
        
        self._run_detection()
        return self.engine.detect_anomalies(hours_back=1.0, persist=False)


def main():
    parser = argparse.ArgumentParser(description='Anomaly Discovery Background Worker')
    parser.add_argument(
        '--detection-interval',
        type=int,
        default=300,
        help='Seconds between detection runs (default: 300)'
    )
    parser.add_argument(
        '--correlation-interval',
        type=int,
        default=3600,
        help='Seconds between correlation analysis (default: 3600)'
    )
    parser.add_argument(
        '--window',
        type=float,
        default=1.0,
        help='Hours of data to analyze per detection (default: 1.0)'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run detection once and exit'
    )
    
    args = parser.parse_args()
    
    worker = DiscoveryWorker(
        detection_interval=args.detection_interval,
        correlation_interval=args.correlation_interval,
        detection_window_hours=args.window
    )
    
    if args.once:
        worker.run_once()
    else:
        worker.run()


if __name__ == "__main__":
    main()
