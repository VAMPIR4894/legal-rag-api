#!/usr/bin/env python3
"""
Quick API test script - tests the FastAPI endpoints
"""
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"
AUTH = ("legal_user", "super_secure_password123")

print("=" * 80)
print("FASTAPI ENDPOINT TESTING")
print("=" * 80)

# Test 1: Health Check
print("\n1. Testing Health Endpoint (GET /api/v1/health)")
print("-" * 80)
try:
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("✅ Health check successful!")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Query endpoint (requires auth)
print("\n2. Testing Query Endpoint (POST /api/v1/query)")
print("-" * 80)
query_data = {
    "query": "What are the legal requirements for property disputes?"
}

try:
    response = requests.post(
        f"{BASE_URL}/query",
        json=query_data,
        auth=AUTH,
        timeout=60
    )
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nQuery: {result['query']}")
        print(f"\nAnswer: {result['answer_with_explanation'][:500]}...")
        print(f"\nRetrieved {len(result['sources'])} sources")
        print("✅ Query successful!")
    else:
        print(f"Response: {response.text}")
        print("⚠️  Query returned non-200 status")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 80)
print("API TESTING COMPLETE")
print("=" * 80)
