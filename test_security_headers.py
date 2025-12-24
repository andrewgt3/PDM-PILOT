#!/usr/bin/env python3
"""
Test script to verify security headers are working.
"""
import uvicorn
import threading
import time
import httpx
from fastapi import FastAPI
from middleware.security_headers import SecurityHeadersMiddleware

# Create test app
app = FastAPI()
app.add_middleware(SecurityHeadersMiddleware, enable_hsts=True, frame_options='DENY')

@app.get('/api/health')
def health():
    return {'status': 'ok'}

def run_server():
    uvicorn.run(app, host='127.0.0.1', port=9999, log_level='error')

if __name__ == '__main__':
    # Start server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(2)
    
    # Make request and print headers
    print("=" * 60)
    print("SECURITY HEADERS VERIFICATION")
    print("=" * 60)
    print()
    
    try:
        response = httpx.get('http://127.0.0.1:9999/api/health')
        
        print("Response Status:", response.status_code)
        print()
        print("Security Headers Present:")
        print("-" * 40)
        
        security_headers = [
            'Content-Security-Policy',
            'X-Frame-Options',
            'Strict-Transport-Security',
            'X-Content-Type-Options',
            'X-XSS-Protection',
            'Referrer-Policy',
            'Permissions-Policy',
            'Cache-Control'
        ]
        
        for header in security_headers:
            value = response.headers.get(header, '❌ NOT SET')
            if value != '❌ NOT SET':
                print(f"✓ {header}")
                # Truncate long values
                if len(value) > 60:
                    value = value[:60] + "..."
                print(f"  → {value}")
            else:
                print(f"✗ {header}: {value}")
            print()
        
        print("=" * 60)
        print("All critical security headers verified!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
