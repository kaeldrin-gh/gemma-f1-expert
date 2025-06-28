#!/usr/bin/env python3
"""
Test the efficient Jolpica-F1 data collection for a small subset.
"""

import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Any
from tqdm import tqdm


class JolpicaF1Client:
    """Client for fetching data from the Jolpica-F1 API."""
    
    BASE_URL = "https://api.jolpi.ca/ergast/f1"
    FALLBACK_URL = "http://ergast.com/api/f1"  # Legacy Ergast URL
    RATE_LIMIT_DELAY = 0.3  # 3 requests/second (under 4 req/sec burst limit)
    SUSTAINED_LIMIT_DELAY = 7.5  # 500 requests/hour = 7.2s between requests
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'gemma-f1-expert/1.0 (Educational Project)'
        })
        self.request_count = 0
        self.start_time = time.time()
    
    def _make_request(self, endpoint: str) -> Dict[str, Any]:
        """Make a rate-limited request to the API."""
        # Check sustained rate limit (500 requests/hour)
        elapsed_time = time.time() - self.start_time
        if self.request_count >= 500 and elapsed_time < 3600:
            sleep_time = 3600 - elapsed_time + 1
            print(f"⏳ Sustained rate limit reached. Sleeping for {sleep_time:.1f}s...")
            time.sleep(sleep_time)
            self.request_count = 0
            self.start_time = time.time()
        
        url = f"{self.BASE_URL}/{endpoint}.json"
        
        try:
            # Respect burst limit (4 requests/second)
            time.sleep(self.RATE_LIMIT_DELAY)
            self.request_count += 1
            
            print(f"🔗 Request #{self.request_count}: {endpoint}")
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            if "429" in str(e):
                print(f"🚫 Rate limit exceeded. Waiting 60s before retry...")
                time.sleep(60)
                return self._make_request(endpoint)  # Retry after rate limit reset
            
            print(f"Primary API failed for {endpoint}: {e}")
            return {}
    
    def get_season_races(self, year: int) -> List[Dict[str, Any]]:
        """Get all races for a given season."""
        data = self._make_request(f"{year}/races")
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        return races
    
    def get_season_results(self, year: int) -> Dict[str, Any]:
        """Get all race results for a season (more efficient than per-race queries)."""
        return self._make_request(f"{year}/results")
    
    def get_driver_standings(self, year: int) -> Dict[str, Any]:
        """Get final driver standings for a season."""
        return self._make_request(f"{year}/driverstandings")


def test_efficient_collection():
    """Test efficient data collection for 2024 only."""
    client = JolpicaF1Client()
    
    print("Testing efficient F1 data collection for 2024...")
    print("Rate limits: 4 req/sec burst, 500 req/hour sustained")
    
    year = 2024
    start_time = time.time()
    
    # Test efficient season-level queries
    print(f"\n📊 Testing season {year} queries:")
    
    # Get races (1 request)
    races = client.get_season_races(year)
    print(f"✅ Races: {len(races)} races found")
    
    # Get all results for the season (1 request instead of ~24)
    results = client.get_season_results(year)
    results_count = len(results.get("MRData", {}).get("RaceTable", {}).get("Races", []))
    print(f"✅ Season Results: {results_count} race results")
    
    # Get driver standings (1 request)
    standings = client.get_driver_standings(year)
    drivers_count = len(standings.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [{}])[0].get("DriverStandings", []))
    print(f"✅ Driver Standings: {drivers_count} drivers")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n📈 Efficiency Summary:")
    print(f"   Total requests: {client.request_count}")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Avg per request: {total_time/client.request_count:.2f}s")
    print(f"   Data collected: {len(races)} races, {results_count} results, {drivers_count} drivers")
    
    # Compare vs individual race queries
    estimated_individual = len(races) * 2 + 2  # 2 requests per race + standings
    print(f"   Old method would need: {estimated_individual} requests")
    print(f"   Efficiency gain: {estimated_individual/client.request_count:.1f}x fewer requests")
    
    return True


if __name__ == "__main__":
    test_efficient_collection()
