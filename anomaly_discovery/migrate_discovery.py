#!/usr/bin/env python3
"""
Database Migration: Anomaly Discovery Engine
Creates tables for storing anomalies, correlations, patterns, and insights.
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")


def run_migration():
    print("=" * 60)
    print("ANOMALY DISCOVERY ENGINE - DATABASE MIGRATION")
    print("=" * 60)
    
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    # =========================================================================
    # 1. ANOMALY_DETECTIONS - Detected anomalies from all algorithms
    # =========================================================================
    print("\n[1/4] Creating anomaly_detections table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_detections (
            id SERIAL PRIMARY KEY,
            detection_id VARCHAR(30) UNIQUE NOT NULL,
            machine_id VARCHAR(20) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            
            -- Detection scores from each algorithm
            isolation_forest_score FLOAT,
            autoencoder_score FLOAT,
            dbscan_outlier BOOLEAN DEFAULT FALSE,
            ensemble_score FLOAT NOT NULL,  -- Combined score 0-1
            
            -- Anomaly classification
            anomaly_type VARCHAR(50),  -- 'point', 'contextual', 'collective', 'drift'
            severity VARCHAR(20),       -- 'low', 'medium', 'high', 'critical'
            confidence FLOAT,           -- 0-1 confidence in detection
            
            -- Feature context (what was anomalous)
            anomalous_features JSONB,   -- {"bpfi_amp": 0.42, "temperature": 85.2}
            feature_deviations JSONB,   -- {"bpfi_amp": 3.2, "temperature": 2.1} (std devs)
            baseline_values JSONB,      -- Normal baseline for comparison
            
            -- Investigation status
            reviewed BOOLEAN DEFAULT FALSE,
            reviewed_by VARCHAR(100),
            reviewed_at TIMESTAMPTZ,
            is_true_positive BOOLEAN,   -- NULL = unknown, True/False after review
            notes TEXT,
            
            -- Linking
            related_alarm_id VARCHAR(20),
            related_work_order_id VARCHAR(20),
            
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_anomaly_machine ON anomaly_detections(machine_id);
        CREATE INDEX IF NOT EXISTS idx_anomaly_timestamp ON anomaly_detections(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_anomaly_score ON anomaly_detections(ensemble_score DESC);
        CREATE INDEX IF NOT EXISTS idx_anomaly_reviewed ON anomaly_detections(reviewed);
    """)
    print("   ✓ anomaly_detections table created")
    
    # =========================================================================
    # 2. CORRELATION_DISCOVERIES - Cross-machine correlations
    # =========================================================================
    print("\n[2/4] Creating correlation_discoveries table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS correlation_discoveries (
            id SERIAL PRIMARY KEY,
            correlation_id VARCHAR(30) UNIQUE NOT NULL,
            
            -- The two entities being correlated
            source_machine_id VARCHAR(20) NOT NULL,
            source_feature VARCHAR(50) NOT NULL,
            target_machine_id VARCHAR(20) NOT NULL,
            target_feature VARCHAR(50) NOT NULL,
            
            -- Correlation metrics
            correlation_coefficient FLOAT NOT NULL,  -- Pearson correlation
            p_value FLOAT,                           -- Statistical significance
            lag_hours FLOAT DEFAULT 0,               -- Time lag (source leads target by X hours)
            
            -- Granger causality results
            granger_f_statistic FLOAT,
            granger_p_value FLOAT,
            granger_causal BOOLEAN,                  -- Does source Granger-cause target?
            
            -- Mutual information (non-linear correlation)
            mutual_information FLOAT,
            
            -- Discovery metadata
            discovery_type VARCHAR(50),              -- 'temporal', 'spatial', 'causal', 'synchronous'
            strength VARCHAR(20),                    -- 'weak', 'moderate', 'strong', 'very_strong'
            confidence FLOAT,
            
            -- Time window analyzed
            analysis_start TIMESTAMPTZ,
            analysis_end TIMESTAMPTZ,
            sample_count INTEGER,
            
            -- Status
            validated BOOLEAN DEFAULT FALSE,
            is_spurious BOOLEAN,                     -- Marked as false correlation
            engineering_notes TEXT,
            
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_corr_source ON correlation_discoveries(source_machine_id);
        CREATE INDEX IF NOT EXISTS idx_corr_target ON correlation_discoveries(target_machine_id);
        CREATE INDEX IF NOT EXISTS idx_corr_strength ON correlation_discoveries(correlation_coefficient DESC);
    """)
    print("   ✓ correlation_discoveries table created")
    
    # =========================================================================
    # 3. PATTERN_LIBRARY - Known patterns and signatures
    # =========================================================================
    print("\n[3/4] Creating pattern_library table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pattern_library (
            id SERIAL PRIMARY KEY,
            pattern_id VARCHAR(30) UNIQUE NOT NULL,
            
            -- Pattern identification
            name VARCHAR(200) NOT NULL,
            description TEXT,
            category VARCHAR(50),                    -- 'failure_precursor', 'seasonal', 'operational', 'environmental'
            
            -- Pattern definition
            pattern_type VARCHAR(50) NOT NULL,       -- 'threshold', 'sequence', 'trend', 'frequency', 'template'
            feature_pattern JSONB NOT NULL,          -- The actual pattern definition
            
            -- Matching criteria
            equipment_types TEXT[],                  -- Which equipment types this applies to
            min_confidence FLOAT DEFAULT 0.7,
            
            -- Pattern provenance
            source VARCHAR(50),                      -- 'manual', 'discovered', 'imported'
            discovered_from_anomaly_id VARCHAR(30),
            
            -- Metrics
            match_count INTEGER DEFAULT 0,
            true_positive_count INTEGER DEFAULT 0,
            false_positive_count INTEGER DEFAULT 0,
            precision FLOAT,                         -- TP / (TP + FP)
            
            -- Status
            is_active BOOLEAN DEFAULT TRUE,
            validated_by VARCHAR(100),
            validated_at TIMESTAMPTZ,
            
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_pattern_category ON pattern_library(category);
        CREATE INDEX IF NOT EXISTS idx_pattern_active ON pattern_library(is_active);
    """)
    print("   ✓ pattern_library table created")
    
    # =========================================================================
    # 4. DISCOVERY_INSIGHTS - Human-readable insights
    # =========================================================================
    print("\n[4/4] Creating discovery_insights table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS discovery_insights (
            id SERIAL PRIMARY KEY,
            insight_id VARCHAR(30) UNIQUE NOT NULL,
            
            -- Insight content
            title VARCHAR(300) NOT NULL,
            summary TEXT NOT NULL,
            detailed_explanation TEXT,
            
            -- Classification
            insight_type VARCHAR(50),                -- 'anomaly', 'correlation', 'trend', 'prediction', 'recommendation'
            priority VARCHAR(20),                    -- 'low', 'medium', 'high', 'critical'
            confidence FLOAT,
            
            -- Affected equipment
            machine_ids TEXT[],
            equipment_types TEXT[],
            
            -- Evidence
            supporting_anomaly_ids TEXT[],
            supporting_correlation_ids TEXT[],
            data_evidence JSONB,                     -- Charts, statistics, etc.
            
            -- Time relevance
            time_window_start TIMESTAMPTZ,
            time_window_end TIMESTAMPTZ,
            
            -- Actions
            recommended_actions TEXT[],
            estimated_impact VARCHAR(200),           -- "Potential $50K savings" or "Prevent 2 hours downtime"
            
            -- Status
            acknowledged BOOLEAN DEFAULT FALSE,
            acknowledged_by VARCHAR(100),
            acknowledged_at TIMESTAMPTZ,
            action_taken TEXT,
            
            -- Generation metadata  
            generated_by VARCHAR(50),                -- 'rule_engine', 'ml_model', 'llm'
            model_version VARCHAR(20),
            
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_insight_type ON discovery_insights(insight_type);
        CREATE INDEX IF NOT EXISTS idx_insight_priority ON discovery_insights(priority);
        CREATE INDEX IF NOT EXISTS idx_insight_ack ON discovery_insights(acknowledged);
    """)
    print("   ✓ discovery_insights table created")
    
    # Commit all changes
    conn.commit()
    cursor.close()
    conn.close()
    
    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE!")
    print("=" * 60)
    print("\nTables created:")
    print("  • anomaly_detections      - Multi-algorithm anomaly scores")
    print("  • correlation_discoveries - Cross-machine correlations")
    print("  • pattern_library         - Known failure signatures")
    print("  • discovery_insights      - Human-readable insights")


if __name__ == "__main__":
    run_migration()
