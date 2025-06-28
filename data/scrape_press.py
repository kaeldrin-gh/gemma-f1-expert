#!/usr/bin/env python3
"""
Scrape Formula 1 press releases from FIA and team RSS feeds.

This script collects recent press releases, race summaries, and technical
updates from official F1 sources to supplement the factual race data
with explanatory content.

Usage:
    python scrape_press.py

Output:
    press_raw.json - Raw press release data in JSON format
"""

import json
import time
import requests
import feedparser
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin
from tqdm import tqdm


class F1PressScaper:
    """Scraper for F1 press releases and news feeds."""
    
    # RSS feeds for F1 content
    RSS_FEEDS = {
        "fia": "https://www.fia.com/rss.xml",
        "formula1": "https://www.formula1.com/en/latest/all.xml",
        "teams": {
            "red_bull": "https://www.redbull.com/int-en/tags/f1/feed",
            "mercedes": "https://www.mercedesamgf1.com/en/news/feed/",
            "ferrari": "https://www.ferrari.com/en/formula1/news/feed",
            "mclaren": "https://www.mclaren.com/racing/team/news/feed/",
            "alpine": "https://www.alpinecars.com/en/f1-team/news/feed/",
            "aston_martin": "https://www.astonmartinf1.com/en-GB/news/feed",
            "alphatauri": "https://scuderiaalphatauri.com/en/news/feed/",
            "alfa_romeo": "https://www.sauber-group.com/motorsport/f1-team/news/feed/",
            "haas": "https://www.haasf1team.com/news/feed",
            "williams": "https://www.williamsf1.com/news/feed/"
        }
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.cutoff_date = datetime.now() - timedelta(days=730)  # Last 2 seasons
    
    def fetch_rss_feed(self, url: str, source_name: str) -> List[Dict[str, Any]]:
        """Fetch and parse an RSS feed."""
        try:
            print(f"Fetching {source_name} RSS feed...")
            time.sleep(1)  # Be respectful to servers
            
            # Some feeds might not be proper RSS, try direct request first
            response = self.session.get(url, timeout=10)
            if response.status_code == 404:
                print(f"⚠️  RSS feed not found for {source_name}: {url}")
                return []
            
            # Parse RSS feed
            feed = feedparser.parse(response.content)
            
            if feed.bozo:
                print(f"⚠️  Invalid RSS feed for {source_name}")
                return []
            
            articles = []
            for entry in feed.entries:
                # Parse publication date
                pub_date = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    pub_date = datetime(*entry.updated_parsed[:6])
                
                # Filter by date (last 2 seasons only)
                if pub_date and pub_date < self.cutoff_date:
                    continue
                
                article = {
                    "title": entry.get("title", "").strip(),
                    "link": entry.get("link", ""),
                    "description": entry.get("description", "").strip(),
                    "summary": entry.get("summary", "").strip(),
                    "published": pub_date.isoformat() if pub_date else None,
                    "source": source_name,
                    "content": self._extract_content(entry)
                }
                
                # Only include F1-related content
                if self._is_f1_related(article):
                    articles.append(article)
            
            print(f"✅ Found {len(articles)} relevant articles from {source_name}")
            return articles
            
        except Exception as e:
            print(f"❌ Error fetching {source_name} RSS: {e}")
            return []
    
    def _extract_content(self, entry) -> str:
        """Extract full content from RSS entry."""
        content = ""
        
        # Try different content fields
        if hasattr(entry, 'content') and entry.content:
            content = entry.content[0].value if isinstance(entry.content, list) else entry.content
        elif hasattr(entry, 'summary') and entry.summary:
            content = entry.summary
        elif hasattr(entry, 'description') and entry.description:
            content = entry.description
        
        # Clean HTML tags (basic cleaning)
        import re
        content = re.sub(r'<[^>]+>', '', content)
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content
    
    def _is_f1_related(self, article: Dict[str, Any]) -> bool:
        """Check if article is F1-related."""
        f1_keywords = [
            'formula 1', 'f1', 'grand prix', 'gp', 'racing', 'driver',
            'championship', 'circuit', 'qualifying', 'practice', 'drs',
            'fia', 'steward', 'penalty', 'fastest lap', 'pole position',
            'constructor', 'team', 'season', 'regulation', 'technical'
        ]
        
        text = f"{article['title']} {article['description']} {article['summary']}".lower()
        return any(keyword in text for keyword in f1_keywords)
    
    def scrape_all_feeds(self) -> List[Dict[str, Any]]:
        """Scrape all configured RSS feeds."""
        all_articles = []
        
        # Scrape main feeds
        for source, url in [("fia", self.RSS_FEEDS["fia"]), 
                           ("formula1", self.RSS_FEEDS["formula1"])]:
            articles = self.fetch_rss_feed(url, source)
            all_articles.extend(articles)
        
        # Scrape team feeds
        for team, url in tqdm(self.RSS_FEEDS["teams"].items(), desc="Scraping team feeds"):
            articles = self.fetch_rss_feed(url, f"team_{team}")
            all_articles.extend(articles)
        
        return all_articles
    
    def create_synthetic_content(self) -> List[Dict[str, Any]]:
        """Create synthetic F1 explanatory content for training."""
        synthetic_articles = [
            {
                "title": "Understanding DRS in Formula 1",
                "content": "DRS (Drag Reduction System) is a driver-controlled device that reduces aerodynamic drag to increase straight-line speed. It can only be used in designated DRS zones during races when a driver is within one second of the car ahead. The rear wing flap opens, reducing downforce and drag, allowing for easier overtaking. DRS is disabled in wet conditions for safety reasons.",
                "source": "synthetic_education",
                "category": "technical_explanation",
                "published": datetime.now().isoformat()
            },
            {
                "title": "Formula 1 Tyre Strategy Basics",
                "content": "F1 races use different tyre compounds: soft (red), medium (yellow), and hard (white). Softer tyres provide more grip but wear faster. Teams must use at least two different compounds during a race. Tyre strategy involves deciding when to pit for fresh tyres, balancing speed against tyre degradation. Weather conditions heavily influence tyre choice.",
                "source": "synthetic_education",
                "category": "strategy_explanation",
                "published": datetime.now().isoformat()
            },
            {
                "title": "Formula 1 Points System Explained",
                "content": "The current F1 points system awards points to the top 10 finishers: 25 for 1st, 18 for 2nd, 15 for 3rd, 12 for 4th, 10 for 5th, 8 for 6th, 6 for 7th, 4 for 8th, 2 for 9th, and 1 for 10th. An additional point is awarded for the fastest lap if the driver finishes in the points. Both Drivers' and Constructors' Championships use this system.",
                "source": "synthetic_education",
                "category": "rules_explanation",
                "published": datetime.now().isoformat()
            },
            {
                "title": "Formula 1 Qualifying Format",
                "content": "F1 qualifying consists of three sessions: Q1 (18 minutes), Q2 (15 minutes), and Q3 (12 minutes). The five slowest drivers in Q1 are eliminated, followed by the five slowest in Q2. The remaining 10 drivers compete in Q3 for pole position. Q2 tyre choice determines race start tyres for Q3 participants.",
                "source": "synthetic_education",
                "category": "format_explanation",
                "published": datetime.now().isoformat()
            }
        ]
        
        return synthetic_articles


def main():
    """Main function to scrape press releases and save data."""
    print("Starting F1 press release scraping...")
    print(f"Collecting articles from last 2 seasons (since {datetime.now() - timedelta(days=730):%Y-%m-%d})")
    
    scraper = F1PressScaper()
    
    try:
        # Scrape RSS feeds
        articles = scraper.scrape_all_feeds()
        
        # Add synthetic educational content
        synthetic_content = scraper.create_synthetic_content()
        articles.extend(synthetic_content)
        
        # Prepare output data
        data = {
            "metadata": {
                "scrape_date": datetime.now().isoformat(),
                "sources": list(scraper.RSS_FEEDS.keys()),
                "total_sources": len(scraper.RSS_FEEDS) + len(scraper.RSS_FEEDS["teams"]),
                "cutoff_date": scraper.cutoff_date.isoformat()
            },
            "articles": articles,
            "summary": {
                "total_articles": len(articles),
                "by_source": {}
            }
        }
        
        # Generate source summary
        for article in articles:
            source = article["source"]
            data["summary"]["by_source"][source] = data["summary"]["by_source"].get(source, 0) + 1
        
        # Save to file
        output_file = "data/press_raw.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n✅ Press scraping complete!")
        print(f"📊 Summary:")
        print(f"   - Total articles: {len(articles)}")
        print(f"   - Sources: {len(data['summary']['by_source'])}")
        print(f"   - Output file: {output_file}")
        print(f"   - File size: {get_file_size(output_file):.1f} MB")
        
        print("\n📋 Articles by source:")
        for source, count in sorted(data["summary"]["by_source"].items()):
            print(f"   - {source}: {count}")
        
    except Exception as e:
        print(f"❌ Error during press scraping: {e}")
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
