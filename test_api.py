#!/usr/bin/env python3
"""
Quick test script to verify Jolpica-F1 API endpoints are working correctly.
"""

import json
import requests
import time
from datetime import datetime

def test_jolpica_endpoints():
    """Test various Jolpica-F1 API endpoints."""
    
    base_url = "https://api.jolpi.ca/ergast/f1"
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'gemma-f1-expert-test/1.0'
    })
    
    endpoints_to_test = [
        ("seasons", "seasons.json"),
        ("circuits", "circuits.json"),
        ("2024_races", "2024/races.json"),
        ("2024_drivers", "2024/drivers.json"),
        ("2024_constructors", "2024/constructors.json"),
        ("2024_driver_standings", "2024/driverstandings.json"),
        ("2024_constructor_standings", "2024/constructorstandings.json"),
        ("2024_results", "2024/results.json"),
        ("2024_qualifying", "2024/qualifying.json"),
    ]
    
    print("🧪 Testing Jolpica-F1 API Endpoints")
    print("=" * 40)
    
    results = {}
    
    for name, endpoint in endpoints_to_test:
        url = f"{base_url}/{endpoint}"
        print(f"\nTesting {name}: {url}")
        
        try:
            time.sleep(0.2)  # Rate limiting
            response = session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Check if response has expected structure
            if "MRData" in data:
                total = data["MRData"].get("total", "unknown")
                print(f"✅ Success - Total records: {total}")
                results[name] = {"status": "success", "total": total}
                
                # Show sample data structure
                mr_data = data["MRData"]
                for key in mr_data.keys():
                    if key != "total" and key != "limit" and key != "offset":
                        if isinstance(mr_data[key], dict) and mr_data[key]:
                            sample_key = list(mr_data[key].keys())[0] if mr_data[key] else "none"
                            print(f"   Contains: {key} -> {sample_key}")
            else:
                print(f"⚠️  Unexpected response structure")
                results[name] = {"status": "warning", "issue": "unexpected_structure"}
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed: {e}")
            results[name] = {"status": "failed", "error": str(e)}
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            results[name] = {"status": "failed", "error": f"JSON error: {e}"}
    
    # Test a specific race result
    print(f"\nTesting specific race result...")
    try:
        time.sleep(0.2)
        url = f"{base_url}/2024/1/results.json"  # First race of 2024
        response = session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if races and races[0].get("Results"):
            winner = races[0]["Results"][0]
            driver_name = f"{winner['Driver']['givenName']} {winner['Driver']['familyName']}"
            constructor = winner["Constructor"]["name"]
            print(f"✅ 2024 Race 1 Winner: {driver_name} ({constructor})")
            results["specific_race"] = {"status": "success", "winner": driver_name}
        else:
            print("⚠️  No race results found")
            results["specific_race"] = {"status": "warning", "issue": "no_results"}
            
    except Exception as e:
        print(f"❌ Specific race test failed: {e}")
        results["specific_race"] = {"status": "failed", "error": str(e)}
    
    # Summary
    print("\n📊 Test Summary")
    print("-" * 20)
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    total_tests = len(results)
    print(f"Successful: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 All API endpoints working correctly!")
        return True
    else:
        print("⚠️  Some endpoints had issues - check logs above")
        return False

if __name__ == "__main__":
    success = test_jolpica_endpoints()
    exit(0 if success else 1)
