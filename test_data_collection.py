#!/usr/bin/env python3
"""
Quick data collection test - collects just 2024 data for testing.
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
            try:
                fallback_url = f"{self.FALLBACK_URL}/{endpoint}.json"
                time.sleep(self.RATE_LIMIT_DELAY)
                response = self.session.get(fallback_url, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as fallback_e:
                print(f"Fallback API also failed: {fallback_e}")
                return {}
    
    def get_season_races(self, year: int) -> List[Dict[str, Any]]:
        """Get all races for a given season."""
        data = self._make_request(f"{year}/races")
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        return races
    
    def get_race_results(self, year: int, round_num: int) -> Dict[str, Any]:
        """Get race results for a specific race."""
        return self._make_request(f"{year}/{round_num}/results")
    
    def get_driver_standings(self, year: int) -> Dict[str, Any]:
        """Get final driver standings for a season."""
        return self._make_request(f"{year}/driverstandings")

def test_data_collection():
    """Test data collection for 2024 season only."""
    print("🏎️  Testing F1 Data Collection (2024 only)")
    print("=" * 50)
    
    client = JolpicaF1Client()
    
    # Test current standings
    print("📊 Fetching 2024 driver standings...")
    standings = client.get_driver_standings(2024)
    
    if standings and "MRData" in standings:
        standings_list = standings.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
        if standings_list and standings_list[0].get("DriverStandings"):
            driver_standings = standings_list[0]["DriverStandings"]
            print(f"✅ Found {len(driver_standings)} drivers in standings")
            
            # Show top 5
            print("🏆 Top 5 drivers in 2024:")
            for i, driver in enumerate(driver_standings[:5]):
                name = f"{driver['Driver']['givenName']} {driver['Driver']['familyName']}"
                points = driver['points']
                constructor = driver['Constructors'][0]['name']
                print(f"   {i+1}. {name} - {points} pts ({constructor})")
        else:
            print("❌ No driver standings found")
            return False
    else:
        print("❌ Failed to fetch standings")
        return False
    
    # Test race data collection (first 3 races only)
    print("\n🏁 Fetching 2024 races...")
    races = client.get_season_races(2024)
    
    if races:
        print(f"✅ Found {len(races)} races in 2024")
        
        print("\n🏎️  Testing race results for first 3 races...")
        race_data = []
        
        for race in races[:3]:  # Only first 3 races for testing
            round_num = int(race["round"])
            race_name = race["raceName"]
            
            print(f"   Fetching {race_name} (Round {round_num})...")
            
            results = client.get_race_results(2024, round_num)
            
            if results and "MRData" in results:
                race_results = results.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                if race_results and race_results[0].get("Results"):
                    winner = race_results[0]["Results"][0]
                    driver_name = f"{winner['Driver']['givenName']} {winner['Driver']['familyName']}"
                    constructor = winner["Constructor"]["name"]
                    
                    race_info = {
                        "round": round_num,
                        "race_name": race_name,
                        "winner": driver_name,
                        "constructor": constructor,
                        "date": race.get("date", "")
                    }
                    race_data.append(race_info)
                    
                    print(f"     ✅ Winner: {driver_name} ({constructor})")
                else:
                    print(f"     ❌ No results found for {race_name}")
            else:
                print(f"     ❌ Failed to fetch results for {race_name}")
        
        # Save test data
        test_data = {
            "metadata": {
                "test_date": datetime.now().isoformat(),
                "source": "Jolpica-F1 API",
                "test_scope": "2024 season, first 3 races"
            },
            "standings": standings,
            "race_data": race_data
        }
        
        output_file = "data/test_jolpica_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Test data saved to: {output_file}")
        print(f"📊 Collected {len(race_data)} race results")
        
        return True
    else:
        print("❌ Failed to fetch races")
        return False

if __name__ == "__main__":
    success = test_data_collection()
    if success:
        print("\n🎉 Data collection test successful!")
    else:
        print("\n❌ Data collection test failed!")
    exit(0 if success else 1)
