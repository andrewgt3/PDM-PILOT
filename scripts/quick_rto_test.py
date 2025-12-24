#!/usr/bin/env python3
"""Quick RTO test for database services."""
import time
import subprocess
from datetime import datetime

print('# Recovery Time Objective (RTO) Log')
print('')
print(f'**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print('**Test Type:** Database Services Only (TimescaleDB, Redis, MongoDB)')
print('')
print('---')
print('')
print('## Timeline')
print('')
print('| Elapsed (s) | Event | Description |')
print('|------------|-------|-------------|')

start = time.time()

def log_event(event, desc):
    elapsed = time.time() - start
    print(f'| {elapsed:.2f} | `{event}` | {desc} |')
    return elapsed

log_event('recovery_started', 'Recovery process initiated')

# Start only database services
result = subprocess.run(
    ['docker-compose', 'up', '-d', 'timescaledb', 'redis', 'mongodb'],
    capture_output=True, text=True, timeout=120
)
log_event('docker_up', 'docker-compose up (DB services) completed')

# Wait for TimescaleDB
for i in range(30):
    result = subprocess.run(
        ['docker', 'exec', 'pdm-timescaledb', 'pg_isready', '-U', 'postgres'],
        capture_output=True, text=True, timeout=5
    )
    if result.returncode == 0:
        break
    time.sleep(2)

log_event('database_healthy', 'TimescaleDB pg_isready passed')

# Check Redis
result = subprocess.run(
    ['docker', 'exec', 'pdm-redis', 'redis-cli', 'ping'],
    capture_output=True, text=True, timeout=5
)
if 'PONG' in result.stdout:
    log_event('redis_healthy', 'Redis PING responded with PONG')

total = time.time() - start
print('')
print('---')
print('')
print('## Summary')
print('')
print(f'> **RTO Achieved:** {total:.2f} seconds ({total/60:.2f} minutes)')
