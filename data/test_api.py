#!/usr/bin/env python3
"""
Quick test script for Jolpica-F1 API endpoints.

This script tests the API endpoints with a small sample of data
to verify the implementation is working correctly.

Usage:
    python test_api.py
"""

import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Any


class JolpicaF1TestClient:
    """Test client for Jolpica-F1 API."""
    
    BASE_URL = "https://api.jolpi.ca/ergast/f1"
    FALLBACK_URL = "http://ergast.com/api/f1"
    RATE_LIMIT_DELAY = 0.2
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'gemma-f1-expert/1.0 (Educational Project)'
        })
    
    def _make_request(self, endpoint: str) -> Dict[str, Any]:
        """Make a rate-limited request to the API."""
        url = f"{self.BASE_URL}/{endpoint}.json"
        
        try:
            time.sleep(self.RATE_LIMIT_DELAY)
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            print(f"Primary API failed for {endpoint}: {e}")
            # Try fallback URL
            try:
                fallback_url = f"{self.FALLBACK_URL}/{endpoint}.json"
                time.sleep(self.RATE_LIMIT_DELAY)
                response = self.session.get(fallback_url, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as fallback_e:
                print(f"Fallback API also failed: {fallback_e}")
                return {}
    
    def test_current_standings(self):
        """Test current driver standings."""
        print("🧪 Testing current driver standings...")
        data = self._make_request("2024/driverstandings")  # Use 2024 since 2025 season might not be available yet
        
        if data:
            standings = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
            if standings and standings[0].get("DriverStandings"):
                drivers = standings[0]["DriverStandings"][:3]  # Top 3
                print(f"✅ Found {len(standings[0]['DriverStandings'])} drivers in standings")
                for i, driver in enumerate(drivers, 1):
                    name = f"{driver['Driver']['givenName']} {driver['Driver']['familyName']}"
                    points = driver['points']
                    print(f"   {i}. {name} - {points} points")
                return True
            else:
                print("❌ No driver standings found")
                return False
        else:
            print("❌ Failed to fetch driver standings")
            return False
    
    def test_recent_races(self):
        """Test recent race data."""
        print("\n🧪 Testing recent race data...")
        
        # Get 2024 races (we know this season is complete)
        year = 2024
        data = self._make_request(f"{year}")
        
        if data:
            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if races:
                print(f"✅ Found {len(races)} races in {year}")
                
                # Test first race results
                first_race = races[0]
                round_num = first_race["round"]
                race_name = first_race["raceName"]
                
                print(f"   Testing results for: {race_name}")
                
                # Get race results
                results_data = self._make_request(f"{year}/{round_num}/results")
                if results_data:
                    race_results = results_data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                    if race_results and race_results[0].get("Results"):
                        winner = race_results[0]["Results"][0]
                        driver_name = f"{winner['Driver']['givenName']} {winner['Driver']['familyName']}"
                        constructor = winner["Constructor"]["name"]
                        print(f"   ✅ Winner: {driver_name} ({constructor})")
                        return True
                    else:
                        print("   ❌ No race results found")
                        return False
                else:
                    print("   ❌ Failed to fetch race results")
                    return False
            else:
                print(f"❌ No races found for {year}")
                return False
        else:
            print(f"❌ Failed to fetch races for {year}")
            return False
    
    def test_qualifying(self):
        """Test qualifying data."""
        print("\n🧪 Testing qualifying data...")
        
        # Test 2024 first race qualifying
        year = 2024
        round_num = 1
        
        data = self._make_request(f"{year}/{round_num}/qualifying")
        
        if data:
            qual_results = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if qual_results and qual_results[0].get("QualifyingResults"):
                pole_sitter = qual_results[0]["QualifyingResults"][0]
                driver_name = f"{pole_sitter['Driver']['givenName']} {pole_sitter['Driver']['familyName']}"
                print(f"✅ Pole position: {driver_name}")
                return True
            else:
                print("❌ No qualifying results found")
                return False
        else:
            print("❌ Failed to fetch qualifying data")
            return False


def main():
    """Run API tests."""
    print("🏎️  Jolpica-F1 API Test Suite")
    print("=" * 40)
    
    client = JolpicaF1TestClient()
    tests_passed = 0
    total_tests = 3
    
    # Run tests
    if client.test_current_standings():
        tests_passed += 1
    
    if client.test_recent_races():
        tests_passed += 1
    
    if client.test_qualifying():
        tests_passed += 1
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All API tests passed! Ready to collect full dataset.")
        return True
    else:
        print("⚠️  Some tests failed. Check API endpoints.")
        return False


if __name__ == "__main__":
    main()
