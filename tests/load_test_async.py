import asyncio
import time
import httpx
import sys
from statistics import mean

BASE_URL = "http://localhost:8000"
NUM_REQUESTS = 50

async def get_token():
    """Get authentication token."""
    async with httpx.AsyncClient() as client:
        # Use the hardcoded admin credentials from enterprise_api.py
        response = await client.post(
            f"{BASE_URL}/api/enterprise/token",
            data={"username": "admin", "password": "secret123"}
        )
        if response.status_code != 200:
            print(f"❌ Authentication failed: {response.text}")
            sys.exit(1)
        return response.json()["access_token"]

async def fetch(client, url, token):
    """Fetch a single URL."""
    headers = {"Authorization": f"Bearer {token}"}
    resp = await client.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"Request failed: {resp.status_code}")
    return resp

async def run_sequential(token):
    print(f"Running {NUM_REQUESTS} sequential requests...")
    start_time = time.perf_counter()
    
    async with httpx.AsyncClient() as client:
        for _ in range(NUM_REQUESTS):
            await fetch(client, f"{BASE_URL}/api/machines", token)
            
    total_time = time.perf_counter() - start_time
    print(f"Sequential Total Time: {total_time:.4f}s")
    print(f"Sequential Avg/Req:    {total_time/NUM_REQUESTS:.4f}s")
    return total_time

async def run_concurrent(token):
    print(f"Running {NUM_REQUESTS} concurrent requests...")
    start_time = time.perf_counter()
    
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=100)) as client:
        tasks = [fetch(client, f"{BASE_URL}/api/machines", token) for _ in range(NUM_REQUESTS)]
        await asyncio.gather(*tasks)
            
    total_time = time.perf_counter() - start_time
    print(f"Concurrent Total Time: {total_time:.4f}s")
    print(f"Concurrent Avg/Req:    {total_time/NUM_REQUESTS:.4f}s")
    return total_time

async def main():
    print("=" * 60)
    print("ASYNC PERFORMANCE TEST")
    print("=" * 60)
    
    # 1. Get Token
    try:
        token = await get_token()
        print("✓ Authenticated")
    except Exception as e:
        print(f"❌ Failed to connect to API: {e}")
        print("Make sure server is running on port 8000")
        sys.exit(1)

    # 2. Sequential Benchmark
    seq_time = await run_sequential(token)
    print("-" * 60)
    
    # 3. Concurrent Benchmark
    conc_time = await run_concurrent(token)
    print("=" * 60)
    
    # 4. Analysis
    ratio = conc_time / seq_time
    print(f"Ratio (Conc/Seq): {ratio:.2f}")
    
    if conc_time > (seq_time * 0.5):
        print("❌ TEST FAILED: Concurrency did not provide >50% speedup.")
        print("   This indicates blocking code in the async loop or DB driver.")
        sys.exit(1)
    else:
        print("✅ TEST PASSED: Asynchronous execution verified.")
        print(f"   Speedup Factor: {seq_time/conc_time:.2f}x")
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())
