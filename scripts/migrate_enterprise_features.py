#!/usr/bin/env python3
"""
Database Migration: Enterprise Features
Creates tables for alarms, work orders, shift schedules, and failure events.

Run with: python migrate_enterprise_features.py
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
    print("ENTERPRISE FEATURES - DATABASE MIGRATION")
    print("=" * 60)
    
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    # =========================================================================
    # 1. PDM_ALARMS - System-generated alarms from predictions
    # =========================================================================
    print("\n[1/5] Creating pdm_alarms table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pdm_alarms (
            id SERIAL PRIMARY KEY,
            alarm_id VARCHAR(20) UNIQUE NOT NULL,
            machine_id VARCHAR(20) NOT NULL,
            severity VARCHAR(20) NOT NULL,  -- 'critical', 'warning', 'info'
            code VARCHAR(20) NOT NULL,       -- e.g., 'PDM-001'
            message TEXT NOT NULL,
            source VARCHAR(50) DEFAULT 'PDM System',
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            acknowledged BOOLEAN DEFAULT FALSE,
            acknowledged_by VARCHAR(100),
            acknowledged_at TIMESTAMPTZ,
            active BOOLEAN DEFAULT TRUE,
            resolved_at TIMESTAMPTZ,
            
            -- Trigger data
            trigger_type VARCHAR(50),        -- 'failure_probability', 'degradation', 'bearing_fault'
            trigger_value FLOAT,
            threshold_value FLOAT
        );
        
        CREATE INDEX IF NOT EXISTS idx_alarms_machine ON pdm_alarms(machine_id);
        CREATE INDEX IF NOT EXISTS idx_alarms_active ON pdm_alarms(active);
        CREATE INDEX IF NOT EXISTS idx_alarms_timestamp ON pdm_alarms(timestamp DESC);
    """)
    print("   ✓ pdm_alarms table created")
    
    # =========================================================================
    # 2. WORK_ORDERS - Maintenance work orders
    # =========================================================================
    print("\n[2/5] Creating work_orders table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS work_orders (
            id SERIAL PRIMARY KEY,
            work_order_id VARCHAR(20) UNIQUE NOT NULL,
            machine_id VARCHAR(20) NOT NULL,
            title VARCHAR(200) NOT NULL,
            description TEXT,
            priority VARCHAR(20) NOT NULL,   -- 'critical', 'high', 'medium', 'low'
            work_type VARCHAR(50) NOT NULL,  -- 'corrective', 'preventive', 'inspection'
            status VARCHAR(20) DEFAULT 'pending',  -- 'pending', 'scheduled', 'in_progress', 'completed', 'cancelled'
            
            -- Scheduling
            created_at TIMESTAMPTZ DEFAULT NOW(),
            scheduled_date DATE,
            started_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ,
            
            -- Assignment
            assigned_to VARCHAR(100),
            created_by VARCHAR(100) DEFAULT 'PDM System',
            
            -- Details
            estimated_duration_hours FLOAT,
            actual_duration_hours FLOAT,
            parts_used TEXT,  -- JSON array of part numbers
            notes TEXT,
            
            -- Link to alarm that triggered this
            source_alarm_id VARCHAR(20)
        );
        
        CREATE INDEX IF NOT EXISTS idx_wo_machine ON work_orders(machine_id);
        CREATE INDEX IF NOT EXISTS idx_wo_status ON work_orders(status);
        CREATE INDEX IF NOT EXISTS idx_wo_created ON work_orders(created_at DESC);
    """)
    print("   ✓ work_orders table created")
    
    # =========================================================================
    # 3. FAILURE_EVENTS - Track when machines reach failure state
    # =========================================================================
    print("\n[3/5] Creating failure_events table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS failure_events (
            id SERIAL PRIMARY KEY,
            machine_id VARCHAR(20) NOT NULL,
            event_type VARCHAR(50) NOT NULL,  -- 'predicted_failure', 'actual_failure', 'near_miss'
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            
            -- State at time of event
            failure_probability FLOAT,
            degradation_score FLOAT,
            bpfi_amp FLOAT,
            bpfo_amp FLOAT,
            
            -- Resolution
            resolved BOOLEAN DEFAULT FALSE,
            resolution_timestamp TIMESTAMPTZ,
            resolution_work_order_id VARCHAR(20),
            downtime_hours FLOAT
        );
        
        CREATE INDEX IF NOT EXISTS idx_failures_machine ON failure_events(machine_id);
        CREATE INDEX IF NOT EXISTS idx_failures_timestamp ON failure_events(timestamp DESC);
    """)
    print("   ✓ failure_events table created")
    
    # =========================================================================
    # 4. SHIFT_SCHEDULE - Production shift configuration
    # =========================================================================
    print("\n[4/5] Creating shift_schedule table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS shift_schedule (
            id SERIAL PRIMARY KEY,
            shift_name VARCHAR(50) NOT NULL,
            start_time TIME NOT NULL,
            end_time TIME NOT NULL,
            days_of_week INTEGER[] DEFAULT '{1,2,3,4,5}',  -- 1=Mon, 7=Sun
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Insert default shifts if empty
        INSERT INTO shift_schedule (shift_name, start_time, end_time, days_of_week)
        SELECT 'Day', '06:00', '14:00', '{1,2,3,4,5}'
        WHERE NOT EXISTS (SELECT 1 FROM shift_schedule WHERE shift_name = 'Day');
        
        INSERT INTO shift_schedule (shift_name, start_time, end_time, days_of_week)
        SELECT 'Afternoon', '14:00', '22:00', '{1,2,3,4,5}'
        WHERE NOT EXISTS (SELECT 1 FROM shift_schedule WHERE shift_name = 'Afternoon');
        
        INSERT INTO shift_schedule (shift_name, start_time, end_time, days_of_week)
        SELECT 'Night', '22:00', '06:00', '{1,2,3,4,5}'
        WHERE NOT EXISTS (SELECT 1 FROM shift_schedule WHERE shift_name = 'Night');
    """)
    print("   ✓ shift_schedule table created with default shifts")
    
    # =========================================================================
    # 5. PRODUCTION_MODE - Track overtime/reduced production
    # =========================================================================
    print("\n[5/5] Creating production_mode table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS production_mode (
            id SERIAL PRIMARY KEY,
            mode VARCHAR(20) NOT NULL,  -- 'normal', 'overtime', 'reduced'
            weekly_hours FLOAT DEFAULT 120,
            wear_factor FLOAT DEFAULT 1.0,  -- Multiplier for wear rate
            effective_from TIMESTAMPTZ DEFAULT NOW(),
            effective_until TIMESTAMPTZ,
            notes TEXT
        );
        
        -- Insert default normal mode
        INSERT INTO production_mode (mode, weekly_hours, wear_factor)
        SELECT 'normal', 120, 1.0
        WHERE NOT EXISTS (SELECT 1 FROM production_mode WHERE effective_until IS NULL);
    """)
    print("   ✓ production_mode table created")
    
    # Commit all changes
    conn.commit()
    cursor.close()
    conn.close()
    
    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE!")
    print("=" * 60)
    print("\nTables created:")
    print("  • pdm_alarms         - System-generated alarms")
    print("  • work_orders        - Maintenance work orders")
    print("  • failure_events     - Track machine failures for MTBF")
    print("  • shift_schedule     - Production shift times")
    print("  • production_mode    - Overtime/normal/reduced tracking")


if __name__ == "__main__":
    run_migration()
