"""
Fleet-Wide Synthetic Data Generator with Cascade Failure Scenario
==================================================================
Simulates 4 robots with a conveyor jam cascade failure + RUL tracking.

Scenario:
- Day 5, 14:00: Conveyor jam occurs
- Robot 1: Torque spikes 300% (fighting jam)
- Robot 2 & 3: Torque drops to 0 (starved)
- Robot 4: Vibration increases 50% (compensating)

Author: PlantAGI Team
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =============================================================================
# CONFIGURATION
# =============================================================================

ROBOTS = ['ROBOT_1', 'ROBOT_2', 'ROBOT_3', 'ROBOT_4']
START_DATE = datetime(2025, 12, 14, 0, 0, 0)
DURATION_DAYS = 7
SAMPLE_RATE_SECONDS = 1  # 1 sample per second

# Cascade event timing
CASCADE_EVENT_DAY = 5
CASCADE_EVENT_HOUR = 14
CASCADE_EVENT_MINUTE = 0

# RUL Initial values (hours)
RUL_INITIAL = {
    'ROBOT_1': 200,   # Degrading
    'ROBOT_2': 5000,  # Healthy
    'ROBOT_3': 150,   # Warning
    'ROBOT_4': 24     # Critical
}

# Output files
SENSOR_OUTPUT = "fleet_sensor_data.csv"
EVENTS_OUTPUT = "fleet_events.csv"

# =============================================================================
# BASELINE SENSOR VALUES
# =============================================================================

BASELINE = {
    'vibration_x': 0.15,
    'vibration_y': 0.12,
    'vibration_z': 0.10,
    'joint_1_torque': 45.0,
    'joint_2_torque': 38.0,
    'joint_3_torque': 32.0,
    'motor_temp_c': 65.0,
    'current_draw_a': 8.5
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def add_noise(value, noise_pct=0.05):
    """Add random noise to a value."""
    return value * (1 + np.random.uniform(-noise_pct, noise_pct))

def calculate_rul_decay(initial_rul, hours_elapsed, decay_rate=1.0):
    """Calculate remaining useful life with decay."""
    return max(0, initial_rul - (hours_elapsed * decay_rate))

# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_fleet_data():
    """Generate sensor data for all 4 robots."""
    print("=" * 80)
    print("FLEET DATA GENERATOR - 4 ROBOTS WITH CASCADE FAILURE")
    print("=" * 80)
    print()
    
    # Calculate total samples
    total_seconds = DURATION_DAYS * 24 * 3600
    total_samples = total_seconds // SAMPLE_RATE_SECONDS
    
    # Cascade event timestamp
    cascade_time = START_DATE + timedelta(
        days=CASCADE_EVENT_DAY - 1,
        hours=CASCADE_EVENT_HOUR,
        minutes=CASCADE_EVENT_MINUTE
    )
    
    print(f"Configuration:")
    print(f"  Robots: {len(ROBOTS)}")
    print(f"  Duration: {DURATION_DAYS} days")
    print(f"  Sample rate: {SAMPLE_RATE_SECONDS}s")
    print(f"  Total samples per robot: {total_samples:,}")
    print(f"  Cascade event: {cascade_time}")
    print()
    
    all_sensor_data = []
    all_events = []
    
    for robot_id in ROBOTS:
        print(f"Generating data for {robot_id}...")
        
        robot_data = []
        initial_rul = RUL_INITIAL[robot_id]
        
        for i in range(total_samples):
            timestamp = START_DATE + timedelta(seconds=i * SAMPLE_RATE_SECONDS)
            hours_elapsed = (timestamp - START_DATE).total_seconds() / 3600
            
            # Calculate RUL
            rul = calculate_rul_decay(initial_rul, hours_elapsed, decay_rate=1.0)
            
            # Baseline values with noise
            vibration_x = add_noise(BASELINE['vibration_x'])
            vibration_y = add_noise(BASELINE['vibration_y'])
            vibration_z = add_noise(BASELINE['vibration_z'])
            joint_1_torque = add_noise(BASELINE['joint_1_torque'])
            joint_2_torque = add_noise(BASELINE['joint_2_torque'])
            joint_3_torque = add_noise(BASELINE['joint_3_torque'])
            motor_temp = add_noise(BASELINE['motor_temp_c'])
            current_draw = add_noise(BASELINE['current_draw_a'])
            
            # CASCADE FAILURE SCENARIO
            if timestamp >= cascade_time:
                time_since_jam = (timestamp - cascade_time).total_seconds() / 60  # minutes
                
                if time_since_jam <= 120:  # 2-hour cascade effect
                    if robot_id == 'ROBOT_1':
                        # Fighting the jam - torque spikes 300%
                        joint_1_torque *= 4.0
                        joint_2_torque *= 3.5
                        joint_3_torque *= 3.0
                        vibration_x *= 2.5
                        vibration_y *= 2.0
                        current_draw *= 3.0
                        motor_temp += 20
                        
                    elif robot_id in ['ROBOT_2', 'ROBOT_3']:
                        # Starved of parts - torque drops to near zero
                        joint_1_torque *= 0.05
                        joint_2_torque *= 0.05
                        joint_3_torque *= 0.05
                        current_draw *= 0.2
                        motor_temp -= 10
                        
                    elif robot_id == 'ROBOT_4':
                        # Compensating - vibration increases 50%
                        vibration_x *= 1.5
                        vibration_y *= 1.5
                        vibration_z *= 1.5
                        joint_1_torque *= 1.3
                        current_draw *= 1.4
                        motor_temp += 5
            
            # Natural degradation based on RUL
            if rul < 100:
                degradation_factor = 1 + (100 - rul) / 100
                vibration_x *= degradation_factor
                motor_temp += (100 - rul) * 0.1
            
            robot_data.append({
                'timestamp': timestamp,
                'robot_id': robot_id,
                'vibration_x': round(vibration_x, 4),
                'vibration_y': round(vibration_y, 4),
                'vibration_z': round(vibration_z, 4),
                'joint_1_torque': round(joint_1_torque, 2),
                'joint_2_torque': round(joint_2_torque, 2),
                'joint_3_torque': round(joint_3_torque, 2),
                'motor_temp_c': round(motor_temp, 2),
                'current_draw_a': round(current_draw, 2),
                'rul_hours': round(rul, 1)
            })
        
        all_sensor_data.extend(robot_data)
        print(f"  ✓ Generated {len(robot_data):,} samples for {robot_id}")
    
    # Create events DataFrame
    all_events.append({
        'timestamp': cascade_time,
        'event_type': 'Conveyor_Jam_Main_Line',
        'severity': 'CRITICAL',
        'affected_assets': 'ALL',
        'description': 'Main conveyor line jammed - cascade failure across fleet'
    })
    
    # Add individual robot events
    for robot_id in ROBOTS:
        initial_rul = RUL_INITIAL[robot_id]
        
        if initial_rul <= 24:
            all_events.append({
                'timestamp': START_DATE,
                'event_type': f'{robot_id}_RUL_CRITICAL',
                'severity': 'CRITICAL',
                'affected_assets': robot_id,
                'description': f'{robot_id} has critical RUL (<24 hours)'
            })
        elif initial_rul <= 150:
            all_events.append({
                'timestamp': START_DATE,
                'event_type': f'{robot_id}_RUL_WARNING',
                'severity': 'WARNING',
                'affected_assets': robot_id,
                'description': f'{robot_id} has low RUL (<150 hours)'
            })
    
    # Convert to DataFrames
    df_sensors = pd.DataFrame(all_sensor_data)
    df_events = pd.DataFrame(all_events)
    
    return df_sensors, df_events

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Execute the fleet data generation."""
    start_time = datetime.now()
    
    # Generate data
    df_sensors, df_events = generate_fleet_data()
    
    print()
    print("=" * 80)
    print("SAVING DATA")
    print("=" * 80)
    
    # Save sensor data
    df_sensors.to_csv(SENSOR_OUTPUT, index=False)
    print(f"✓ Saved sensor data: {SENSOR_OUTPUT}")
    print(f"  Rows: {len(df_sensors):,}")
    print(f"  Size: {len(df_sensors) * len(df_sensors.columns):,} cells")
    
    # Save events
    df_events.to_csv(EVENTS_OUTPUT, index=False)
    print(f"✓ Saved events: {EVENTS_OUTPUT}")
    print(f"  Events: {len(df_events)}")
    
    # Statistics
    print()
    print("=" * 80)
    print("FLEET STATISTICS")
    print("=" * 80)
    
    for robot in ROBOTS:
        robot_data = df_sensors[df_sensors['robot_id'] == robot]
        print(f"\n{robot}:")
        print(f"  Samples: {len(robot_data):,}")
        print(f"  Initial RUL: {RUL_INITIAL[robot]} hours")
        print(f"  Final RUL: {robot_data['rul_hours'].iloc[-1]:.1f} hours")
        print(f"  Avg Vibration X: {robot_data['vibration_x'].mean():.3f}g")
        print(f"  Max Torque J1: {robot_data['joint_1_torque'].max():.1f} Nm")
        print(f"  Avg Motor Temp: {robot_data['motor_temp_c'].mean():.1f}°C")
    
    duration = (datetime.now() - start_time).total_seconds()
    print()
    print(f"✓ Generation completed in {duration:.2f} seconds")
    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Load data:")
    print(f"   python3 etl_pipeline.py --fleet")
    print()
    print("2. Run analytics:")
    print(f"   python3 analytics_engine.py --cascade")
    print()
    print("3. View dashboard:")
    print(f"   streamlit run dashboard_streamlit.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
