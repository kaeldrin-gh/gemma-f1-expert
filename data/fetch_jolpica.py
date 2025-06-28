#!/usr/bin/env python3
"""
Fetch Formula 1 data from the Jolpica-F1 API.

This script collects race results, driver standings, constructor standings,
and fastest lap data from 2000 to the present. It respects the API rate
limit of 200 requests per hour for unauthenticated users.

Usage:
    python fetch_jolpica.py

Output:
    jolpica_raw.json - Raw F1 data in JSON format
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
    RATE_LIMIT_DELAY = 0.2  # 200 requests/hour = 0.2s between requests
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'gemma-f1-expert/1.0 (Educational Project)'
        })
    
    def _make_request(self, endpoint: str) -> Dict[str, Any]:
        """Make a rate-limited request to the API."""
        url = f"{self.BASE_URL}/{endpoint}.json"
        
        try:
            time.sleep(self.RATE_LIMIT_DELAY)  # Respect rate limit
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
    
    def get_seasons(self, start_year: int = 2000) -> List[int]:
        """Get list of available seasons from start_year to present."""
        current_year = datetime.now().year
        return list(range(start_year, current_year + 1))
    
    def get_season_races(self, year: int) -> List[Dict[str, Any]]:
        """Get all races for a given season."""
        data = self._make_request(f"{year}")
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        return races
    
    def get_race_results(self, year: int, round_num: int) -> Dict[str, Any]:
        """Get race results for a specific race."""
        return self._make_request(f"{year}/{round_num}/results")
    
    def get_qualifying_results(self, year: int, round_num: int) -> Dict[str, Any]:
        """Get qualifying results for a specific race."""
        return self._make_request(f"{year}/{round_num}/qualifying")
    
    def get_fastest_laps(self, year: int, round_num: int) -> Dict[str, Any]:
        """Get fastest lap data for a specific race."""
        return self._make_request(f"{year}/{round_num}/fastest/1")
    
    def get_driver_standings(self, year: int) -> Dict[str, Any]:
        """Get final driver standings for a season."""
        return self._make_request(f"{year}/driverStandings")
    
    def get_constructor_standings(self, year: int) -> Dict[str, Any]:
        """Get final constructor standings for a season."""
        return self._make_request(f"{year}/constructorStandings")
    
    def get_current_standings(self) -> Dict[str, Any]:
        """Get current season driver standings."""
        current_year = datetime.now().year
        return self._make_request(f"{current_year}/driverStandings")


def fetch_comprehensive_data() -> Dict[str, Any]:
    """Fetch comprehensive F1 data from Jolpica API."""
    client = JolpicaF1Client()
    data = {
        "metadata": {
            "fetch_date": datetime.now().isoformat(),
            "source": "Jolpica-F1 API",
            "rate_limit": "200 requests/hour"
        },
        "seasons": {},
        "current_standings": {}
    }
    
    seasons = client.get_seasons(start_year=2000)
    print(f"Fetching data for {len(seasons)} seasons: {seasons[0]}-{seasons[-1]}")
    
    # Fetch current standings first
    print("Fetching current season standings...")
    data["current_standings"] = client.get_current_standings()
    
    # Fetch historical data
    for year in tqdm(seasons, desc="Processing seasons"):
        season_data = {
            "races": [],
            "driver_standings": {},
            "constructor_standings": {}
        }
        
        # Get races for the season
        races = client.get_season_races(year)
        
        for race in tqdm(races, desc=f"Processing {year} races", leave=False):
            round_num = int(race["round"])
            race_info = {
                "race_info": race,
                "results": client.get_race_results(year, round_num),
                "qualifying": client.get_qualifying_results(year, round_num),
                "fastest_lap": client.get_fastest_laps(year, round_num)
            }
            season_data["races"].append(race_info)
        
        # Get season standings
        season_data["driver_standings"] = client.get_driver_standings(year)
        season_data["constructor_standings"] = client.get_constructor_standings(year)
        
        data["seasons"][str(year)] = season_data
        
        # Progress update
        if year % 5 == 0:
            print(f"Completed season {year}")
    
    return data


def main():
    """Main function to fetch and save F1 data."""
    print("Starting Jolpica-F1 data collection...")
    print("Rate limit: 200 requests/hour (0.2s delay between requests)")
    
    try:
        data = fetch_comprehensive_data()
        
        # Save to file
        output_file = "data/jolpica_raw.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Print summary
        total_seasons = len(data["seasons"])
        total_races = sum(len(season["races"]) for season in data["seasons"].values())
        
        print(f"\n✅ Data collection complete!")
        print(f"📊 Summary:")
        print(f"   - Seasons: {total_seasons}")
        print(f"   - Total races: {total_races}")
        print(f"   - Output file: {output_file}")
        print(f"   - File size: {get_file_size(output_file):.1f} MB")
        
    except Exception as e:
        print(f"❌ Error during data collection: {e}")
        raise


def get_file_size(filepath: str) -> float:
    """Get file size in MB."""
    import os
    try:
        return os.path.getsize(filepath) / (1024 * 1024)
    except OSError:
        return 0.0


if __name__ == "__main__":
    main()
