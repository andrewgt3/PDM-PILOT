#!/usr/bin/env python3
"""
Insert sample anomaly data directly into the database
to populate the AI Discovery page for demonstration.
"""

import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
import random
import uuid
import json

# Database connection (local PostgreSQL)
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "gaia"
DB_USER = "postgres"
DB_PASSWORD = "postgres"

# Machine IDs from the fleet
MACHINES = ["WB-001", "WB-002", "HP-200", "PR-101", "TS-001", "CV-100"]

def create_tables(conn):
    """Create anomaly detection tables if they don't exist."""
    cursor = conn.cursor()
    
    # Create anomaly_detections table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_detections (
            detection_id VARCHAR(50) PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            machine_id VARCHAR(50) NOT NULL,
            anomaly_type VARCHAR(50),
            severity VARCHAR(20),
            ensemble_score FLOAT,
            confidence FLOAT,
            feature_deviations JSONB,
            reviewed BOOLEAN DEFAULT FALSE,
            is_true_positive BOOLEAN,
            reviewed_by VARCHAR(100),
            reviewed_at TIMESTAMPTZ,
            notes TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    
    # Create correlation_discoveries table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS correlation_discoveries (
            correlation_id SERIAL PRIMARY KEY,
            source_machine_id VARCHAR(50),
            target_machine_id VARCHAR(50),
            source_feature VARCHAR(100),
            target_feature VARCHAR(100),
            correlation_coefficient FLOAT,
            lag_hours INT,
            strength VARCHAR(20),
            granger_causal BOOLEAN,
            confidence FLOAT,
            explanation TEXT,
            discovery_type VARCHAR(50),
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    
    # Create discovery_insights table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS discovery_insights (
            insight_id VARCHAR(50) PRIMARY KEY,
            insight_type VARCHAR(50),
            priority VARCHAR(20),
            title VARCHAR(200),
            summary TEXT,
            recommended_actions JSONB,
            acknowledged BOOLEAN DEFAULT FALSE,
            acknowledged_by VARCHAR(100),
            acknowledged_at TIMESTAMPTZ,
            action_taken TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    
    conn.commit()
    cursor.close()
    print("✓ Tables created/verified")

def insert_sample_anomalies(conn):
    """Insert sample anomaly detections."""
    cursor = conn.cursor()
    
    anomalies = [
        {
            "machine_id": "CV-100",
            "anomaly_type": "vibration_spike",
            "severity": "critical",
            "ensemble_score": 0.92,
            "confidence": 0.88,
            "hours_ago": 2
        },
        {
            "machine_id": "TS-001",
            "anomaly_type": "temperature_drift",
            "severity": "high",
            "ensemble_score": 0.78,
            "confidence": 0.82,
            "hours_ago": 4
        },
        {
            "machine_id": "HP-200",
            "anomaly_type": "pressure_anomaly",
            "severity": "high",
            "ensemble_score": 0.75,
            "confidence": 0.79,
            "hours_ago": 6
        },
        {
            "machine_id": "WB-001",
            "anomaly_type": "bearing_fault",
            "severity": "medium",
            "ensemble_score": 0.65,
            "confidence": 0.71,
            "hours_ago": 8
        },
        {
            "machine_id": "PR-101",
            "anomaly_type": "flow_irregularity",
            "severity": "medium",
            "ensemble_score": 0.58,
            "confidence": 0.68,
            "hours_ago": 12
        },
    ]
    
    for a in anomalies:
        detection_id = f"DET-{uuid.uuid4().hex[:12].upper()}"
        timestamp = datetime.now() - timedelta(hours=a["hours_ago"])
        
        feature_deviations = {
            "vibration_x": round(random.uniform(0.6, 0.95), 2),
            "temperature": round(random.uniform(0.5, 0.85), 2),
            "peak_freq_1": round(random.uniform(0.4, 0.75), 2),
            "bpfo_amp": round(random.uniform(0.3, 0.65), 2)
        }
        
        cursor.execute("""
            INSERT INTO anomaly_detections 
            (detection_id, timestamp, machine_id, anomaly_type, severity, 
             ensemble_score, confidence, feature_deviations, reviewed, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (detection_id) DO NOTHING
        """, (
            detection_id, timestamp, a["machine_id"], a["anomaly_type"],
            a["severity"], a["ensemble_score"], a["confidence"],
            json.dumps(feature_deviations), False
        ))
    
    conn.commit()
    cursor.close()
    print(f"✓ Inserted {len(anomalies)} sample anomalies")

def insert_sample_correlations(conn):
    """Insert sample cross-machine correlations."""
    cursor = conn.cursor()
    
    correlations = [
        {
            "source": "CV-100",
            "target": "TS-001",
            "source_feature": "vibration_amplitude",
            "target_feature": "torque_variation",
            "coefficient": 0.87,
            "lag": 2,
            "strength": "very_strong",
            "causal": True,
            "explanation": "Conveyor vibration propagates to torque station within 2 hours"
        },
        {
            "source": "HP-200",
            "target": "WB-001",
            "source_feature": "pressure_fluctuation",
            "target_feature": "weld_quality",
            "coefficient": 0.72,
            "lag": 4,
            "strength": "strong",
            "causal": True,
            "explanation": "Hydraulic press pressure affects downstream weld quality"
        },
        {
            "source": "PR-101",
            "target": "PR-101",
            "source_feature": "flow_rate",
            "target_feature": "temperature",
            "coefficient": 0.65,
            "lag": 0,
            "strength": "moderate",
            "causal": False,
            "explanation": "Paint flow rate correlates with applicator temperature"
        },
    ]
    
    for c in correlations:
        cursor.execute("""
            INSERT INTO correlation_discoveries 
            (source_machine_id, target_machine_id, source_feature, target_feature,
             correlation_coefficient, lag_hours, strength, granger_causal, 
             confidence, explanation, discovery_type, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """, (
            c["source"], c["target"], c["source_feature"], c["target_feature"],
            c["coefficient"], c["lag"], c["strength"], c["causal"],
            0.85, c["explanation"], "cross_machine"
        ))
    
    conn.commit()
    cursor.close()
    print(f"✓ Inserted {len(correlations)} sample correlations")

def insert_sample_insights(conn):
    """Insert sample AI insights."""
    cursor = conn.cursor()
    
    insights = [
        {
            "type": "cascade_failure_risk",
            "priority": "critical",
            "title": "Potential Cascade Failure Detected",
            "summary": "CV-100 vibration anomaly may trigger failures in downstream equipment (TS-001, LA-003). Immediate inspection recommended.",
            "actions": ["Inspect CV-100 bearings", "Check torque station alignment", "Schedule preventive maintenance"]
        },
        {
            "type": "maintenance_optimization",
            "priority": "high",
            "title": "Coordinated Maintenance Opportunity",
            "summary": "HP-200 and WB-001 show correlated degradation patterns. Scheduling simultaneous maintenance could reduce downtime by 40%.",
            "actions": ["Schedule joint maintenance window", "Order parts for both machines", "Coordinate with production planning"]
        },
    ]
    
    for i in insights:
        insight_id = f"INS-{uuid.uuid4().hex[:12].upper()}"
        
        cursor.execute("""
            INSERT INTO discovery_insights 
            (insight_id, insight_type, priority, title, summary, 
             recommended_actions, acknowledged, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (insight_id) DO NOTHING
        """, (
            insight_id, i["type"], i["priority"], i["title"],
            i["summary"], json.dumps(i["actions"]), False
        ))
    
    conn.commit()
    cursor.close()
    print(f"✓ Inserted {len(insights)} sample insights")

def main():
    print("="*70)
    print("POPULATING AI DISCOVERY PAGE WITH SAMPLE DATA")
    print("="*70)
    print()
    
    try:
        # Connect to database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        print(f"✓ Connected to database: {DB_NAME}@{DB_HOST}")
        
        # Create tables
        create_tables(conn)
        
        # Insert sample data
        insert_sample_anomalies(conn)
        insert_sample_correlations(conn)
        insert_sample_insights(conn)
        
        conn.close()
        
        print()
        print("="*70)
        print("✅ SUCCESS!")
        print("="*70)
        print()
        print("The AI Discovery page now has sample data:")
        print("  • 5 Anomalies (1 critical, 2 high, 2 medium)")
        print("  • 3 Cross-machine correlations")
        print("  • 2 AI-generated insights")
        print()
        print("Refresh your browser at: http://localhost:5173")
        print("Navigate to: AI Discovery (in sidebar)")
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
