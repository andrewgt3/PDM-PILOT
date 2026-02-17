#!/usr/bin/env python3
"""
Populate AI Discovery Page
Train models and run anomaly detection to populate the frontend.
"""

import requests
import time
import sys

API_BASE = "http://localhost:8000/api/discovery"

def check_status():
    """Check if models are trained."""
    try:
        res = requests.get(f"{API_BASE}/status")
        if res.ok:
            data = res.json()
            print(f"✓ Model Status: {'Trained' if data.get('is_trained') else 'Not Trained'}")
            return data.get('is_trained', False)
        else:
            print(f"❌ Status check failed: {res.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return False

def train_models():
    """Train anomaly detection models."""
    print("\n" + "="*70)
    print("TRAINING ANOMALY DETECTION MODELS")
    print("="*70)
    
    try:
        res = requests.post(
            f"{API_BASE}/train",
            json={"days_of_data": 7, "min_samples": 100}
        )
        
        if res.ok:
            data = res.json()
            print(f"✓ Training started: {data.get('message')}")
            print(f"  Training ID: {data.get('training_id')}")
            
            # Poll for completion
            print("\nWaiting for training to complete...")
            for i in range(30):  # Wait up to 30 seconds
                time.sleep(1)
                if check_status():
                    print("\n✅ Training complete!")
                    return True
                print(".", end="", flush=True)
            
            print("\n⚠️  Training taking longer than expected. Check status manually.")
            return False
        else:
            print(f"❌ Training failed: {res.status_code}")
            print(f"   Response: {res.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False

def run_detection():
    """Run anomaly detection on recent data."""
    print("\n" + "="*70)
    print("RUNNING ANOMALY DETECTION")
    print("="*70)
    
    try:
        res = requests.post(
            f"{API_BASE}/detect",
            json={"hours_back": 4, "persist": True}
        )
        
        if res.ok:
            data = res.json()
            print(f"✓ Detection complete!")
            print(f"  Anomalies detected: {data.get('anomaly_count', 0)}")
            print(f"  Critical: {data.get('critical_count', 0)}")
            print(f"  High: {data.get('high_count', 0)}")
            return True
        else:
            print(f"❌ Detection failed: {res.status_code}")
            print(f"   Response: {res.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        return False

def get_anomalies():
    """Fetch and display detected anomalies."""
    try:
        res = requests.get(f"{API_BASE}/anomalies?limit=10")
        if res.ok:
            data = res.json()
            anomalies = data.get('data', [])
            print(f"\n📊 Found {len(anomalies)} anomalies")
            
            for i, a in enumerate(anomalies[:5], 1):
                print(f"\n  {i}. {a.get('machine_id')} - {a.get('severity', 'unknown').upper()}")
                print(f"     Type: {a.get('anomaly_type')}")
                print(f"     Score: {a.get('ensemble_score', 0)*100:.0f}%")
            
            return len(anomalies)
        else:
            print(f"❌ Failed to fetch anomalies: {res.status_code}")
            return 0
    except Exception as e:
        print(f"❌ Error fetching anomalies: {e}")
        return 0

def main():
    print("="*70)
    print("AI DISCOVERY PAGE SETUP")
    print("="*70)
    print("\nThis script will:")
    print("  1. Check if models are trained")
    print("  2. Train models if needed")
    print("  3. Run anomaly detection")
    print("  4. Display results")
    print()
    
    # Step 1: Check status
    is_trained = check_status()
    
    # Step 2: Train if needed
    if not is_trained:
        print("\n⚠️  Models not trained. Starting training...")
        if not train_models():
            print("\n❌ Training failed. Cannot proceed with detection.")
            sys.exit(1)
    else:
        print("\n✓ Models already trained. Skipping training step.")
    
    # Step 3: Run detection
    print("\nRunning detection on recent data...")
    time.sleep(2)  # Give training a moment to finalize
    
    if run_detection():
        # Step 4: Display results
        time.sleep(2)  # Give detection a moment to persist
        count = get_anomalies()
        
        print("\n" + "="*70)
        print("✅ SUCCESS!")
        print("="*70)
        print(f"\nThe AI Discovery page now has {count} anomalies to display.")
        print("\nRefresh your browser at: http://localhost:5173")
        print("Navigate to: AI Discovery (in sidebar)")
        print()
    else:
        print("\n❌ Detection failed. Check the API server logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
