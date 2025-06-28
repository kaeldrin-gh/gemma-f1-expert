#!/usr/bin/env python3
"""
Build F1 question-answer dataset from collected data.

This script processes the raw Jolpica API data and press releases to create
a structured training dataset of ~3,000 question-answer pairs for fine-tuning.

Usage:
    python build_dataset.py

Input:
    jolpica_raw.json - Raw F1 race data
    press_raw.json - Raw press releases

Output:
    f1_qa.jsonl - Complete Q-A dataset
    f1_qa_train.jsonl - Training split (95%)
    f1_qa_test.jsonl - Test split (5%)
"""

import json
import random
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path
import re


class F1DatasetBuilder:
    """Build F1 Q-A dataset from raw data sources."""
    
    def __init__(self, jolpica_file: str, press_file: str):
        self.jolpica_file = jolpica_file
        self.press_file = press_file
        self.qa_pairs = []
        
    def load_data(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load raw data files."""
        print("Loading raw data files...")
        
        # Load Jolpica data
        jolpica_data = {}
        if Path(self.jolpica_file).exists():
            with open(self.jolpica_file, 'r', encoding='utf-8') as f:
                jolpica_data = json.load(f)
            print(f"✅ Loaded Jolpica data: {len(jolpica_data.get('seasons', {}))} seasons")
        else:
            print(f"⚠️  Jolpica file not found: {self.jolpica_file}")
        
        # Load press data
        press_data = {}
        if Path(self.press_file).exists():
            with open(self.press_file, 'r', encoding='utf-8') as f:
                press_data = json.load(f)
            print(f"✅ Loaded press data: {len(press_data.get('articles', []))} articles")
        else:
            print(f"⚠️  Press file not found: {self.press_file}")
        
        return jolpica_data, press_data
    
    def create_factual_questions(self, jolpica_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create factual Q-A pairs from race data."""
        qa_pairs = []
        seasons = jolpica_data.get("seasons", {})
        
        print(f"Creating factual questions from {len(seasons)} seasons...")
        
        for year, season_data in seasons.items():
            races = season_data.get("races", [])
            
            for race_info in races:
                race = race_info.get("race_info", {})
                results_data = race_info.get("results", {})
                qualifying_data = race_info.get("qualifying", {})
                
                race_name = race.get("raceName", "")
                circuit_name = race.get("Circuit", {}).get("circuitName", "")
                
                # Race winner questions
                results = results_data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                if results and results[0].get("Results"):
                    winner = results[0]["Results"][0]
                    driver_name = f"{winner['Driver']['givenName']} {winner['Driver']['familyName']}"
                    constructor = winner["Constructor"]["name"]
                    
                    qa_pairs.extend([
                        {
                            "question": f"Who won the {year} {race_name}?",
                            "answer": f"{driver_name} won the {year} {race_name} driving for {constructor}.",
                            "type": "factual",
                            "category": "race_winner"
                        },
                        {
                            "question": f"Which team won the {year} {race_name}?",
                            "answer": f"{constructor} won the {year} {race_name} with {driver_name}.",
                            "type": "factual",
                            "category": "constructor_winner"
                        }
                    ])
                
                # Qualifying questions
                qual_results = qualifying_data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                if qual_results and qual_results[0].get("QualifyingResults"):
                    pole_sitter = qual_results[0]["QualifyingResults"][0]
                    pole_driver = f"{pole_sitter['Driver']['givenName']} {pole_sitter['Driver']['familyName']}"
                    
                    qa_pairs.append({
                        "question": f"Who got pole position for the {year} {race_name}?",
                        "answer": f"{pole_driver} secured pole position for the {year} {race_name}.",
                        "type": "factual",
                        "category": "pole_position"
                    })
                
                # Circuit questions
                if circuit_name:
                    qa_pairs.append({
                        "question": f"Where is the {race_name} held?",
                        "answer": f"The {race_name} is held at {circuit_name}.",
                        "type": "factual",
                        "category": "circuit_info"
                    })
            
            # Championship questions
            driver_standings = season_data.get("driver_standings", {})
            if driver_standings.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists"):
                standings = driver_standings["MRData"]["StandingsTable"]["StandingsLists"][0]["DriverStandings"]
                if standings:
                    champion = standings[0]
                    champion_name = f"{champion['Driver']['givenName']} {champion['Driver']['familyName']}"
                    points = champion["points"]
                    
                    qa_pairs.append({
                        "question": f"Who won the {year} Formula 1 World Championship?",
                        "answer": f"{champion_name} won the {year} Formula 1 World Championship with {points} points.",
                        "type": "factual",
                        "category": "championship"
                    })
        
        print(f"✅ Created {len(qa_pairs)} factual Q-A pairs")
        return qa_pairs
    
    def create_explanatory_questions(self, press_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create explanatory Q-A pairs from press releases."""
        qa_pairs = []
        articles = press_data.get("articles", [])
        
        print(f"Creating explanatory questions from {len(articles)} articles...")
        
        # Predefined explanatory content
        explanatory_qa = [
            {
                "question": "How does DRS work in Formula 1?",
                "answer": "DRS (Drag Reduction System) is a movable rear wing flap that drivers can activate to reduce aerodynamic drag. It can only be used in designated DRS zones when a driver is within one second of the car ahead during races. When activated, the rear wing opens, reducing downforce and allowing higher straight-line speeds for overtaking. DRS is automatically disabled in wet conditions for safety.",
                "type": "explanatory",
                "category": "technical"
            },
            {
                "question": "What are the different F1 tyre compounds?",
                "answer": "F1 uses three dry tyre compounds per weekend: soft (red sidewall), medium (yellow sidewall), and hard (white sidewall). Softer compounds provide more grip but degrade faster, while harder compounds last longer but offer less grip. Teams must use at least two different compounds during a race. There are also intermediate (green) and wet (blue) tyres for rain conditions.",
                "type": "explanatory",
                "category": "technical"
            },
            {
                "question": "How does F1 qualifying work?",
                "answer": "F1 qualifying consists of three knockout sessions: Q1 (18 minutes), Q2 (15 minutes), and Q3 (12 minutes). In Q1, the five slowest drivers are eliminated. In Q2, another five are eliminated, and the remaining drivers' tyre choice determines their race start tyres. Q3 features the top 10 drivers competing for pole position.",
                "type": "explanatory",
                "category": "format"
            },
            {
                "question": "What is the current F1 points system?",
                "answer": "The current F1 points system awards points to the top 10 finishers: 25-18-15-12-10-8-6-4-2-1 points for positions 1st through 10th respectively. An additional point is awarded for the fastest lap, but only if the driver finishes in the points (top 10). Both the Drivers' Championship and Constructors' Championship use this system.",
                "type": "explanatory",
                "category": "rules"
            },
            {
                "question": "What happens during an F1 pit stop?",
                "answer": "During a pit stop, teams can change tyres, adjust front wing angles, and make minor repairs. A typical pit stop takes 2-3 seconds for a tyre change. Teams can make unlimited pit stops during a race, but drivers must use at least two different tyre compounds. Pit stops are crucial for strategy, timing them to minimize time loss while gaining track position or fresh tyres.",
                "type": "explanatory",
                "category": "strategy"
            },
            {
                "question": "How do F1 safety cars work?",
                "answer": "Safety cars are deployed when there's a hazard on track that requires marshals to work safely. All drivers must slow down and follow the safety car in single file, with no overtaking allowed. This bunches up the field and allows safe track clearing. Racing resumes when the safety car returns to the pits. Virtual Safety Cars (VSC) require drivers to maintain specific lap times without a physical safety car.",
                "type": "explanatory",
                "category": "safety"
            }
        ]
        
        qa_pairs.extend(explanatory_qa)
        
        # Extract explanatory content from press articles
        for article in articles:
            if article.get("source") == "synthetic_education":
                question = f"Explain {article['title'].replace('Understanding ', '').replace(' in Formula 1', '').replace(' Explained', '').replace(' Basics', '').lower()}"
                answer = article.get("content", "")
                
                if len(answer) > 50:  # Ensure meaningful content
                    qa_pairs.append({
                        "question": question,
                        "answer": answer,
                        "type": "explanatory",
                        "category": "education"
                    })
        
        print(f"✅ Created {len(qa_pairs)} explanatory Q-A pairs")
        return qa_pairs
    
    def create_summary_questions(self, press_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create race summary Q-A pairs from press releases."""
        qa_pairs = []
        articles = press_data.get("articles", [])
        
        print(f"Creating summary questions from press articles...")
        
        # Template summaries for recent races (synthetic data)
        summary_templates = [
            {
                "question": "Summarize the 2023 Monaco Grand Prix in two sentences.",
                "answer": "Max Verstappen won the 2023 Monaco Grand Prix from pole position, leading from start to finish in challenging wet conditions. The race was marked by several incidents and safety car periods, with Fernando Alonso finishing second and Esteban Ocon completing the podium.",
                "type": "summary",
                "category": "race_summary"
            },
            {
                "question": "What happened in the 2023 British Grand Prix?",
                "answer": "Max Verstappen dominated the 2023 British Grand Prix at Silverstone, winning by over 17 seconds ahead of Lando Norris. The race saw strong performances from McLaren and Mercedes, with Lewis Hamilton finishing third to delight the home crowd.",
                "type": "summary",
                "category": "race_summary"
            },
            {
                "question": "Give me a summary of the 2023 Italian Grand Prix.",
                "answer": "Max Verstappen won the 2023 Italian Grand Prix at Monza, extending his championship lead. The race featured close battles throughout the field, with Carlos Sainz finishing second for Ferrari to the delight of the passionate Italian fans, and Charles Leclerc completing the podium.",
                "type": "summary",
                "category": "race_summary"
            }
        ]
        
        qa_pairs.extend(summary_templates)
        
        # Extract summaries from actual press articles
        for article in articles:
            title = article.get("title", "")
            content = article.get("content", "")
            
            # Look for race reports and summaries
            if any(keyword in title.lower() for keyword in ["grand prix", "race report", "summary", "recap"]):
                if len(content) > 100:  # Ensure substantial content
                    # Create summary question
                    race_match = re.search(r"(\d{4}.*?Grand Prix)", title)
                    if race_match:
                        race_name = race_match.group(1)
                        question = f"Summarize the {race_name} in two sentences."
                        
                        # Truncate content to create a concise summary
                        summary = content[:300] + "..." if len(content) > 300 else content
                        
                        qa_pairs.append({
                            "question": question,
                            "answer": summary,
                            "type": "summary",
                            "category": "race_summary"
                        })
        
        print(f"✅ Created {len(qa_pairs)} summary Q-A pairs")
        return qa_pairs
    
    def build_complete_dataset(self) -> List[Dict[str, str]]:
        """Build the complete Q-A dataset."""
        print("Building complete F1 Q-A dataset...")
        
        jolpica_data, press_data = self.load_data()
        
        # Create different types of Q-A pairs
        factual_qa = self.create_factual_questions(jolpica_data)
        explanatory_qa = self.create_explanatory_questions(press_data)
        summary_qa = self.create_summary_questions(press_data)
        
        # Combine all Q-A pairs
        all_qa = factual_qa + explanatory_qa + summary_qa
        
        # Add metadata to each pair
        for i, qa in enumerate(all_qa):
            qa["id"] = f"f1_qa_{i:04d}"
            qa["created_at"] = datetime.now().isoformat()
        
        # Remove duplicates
        unique_qa = []
        seen_questions = set()
        for qa in all_qa:
            if qa["question"].lower() not in seen_questions:
                unique_qa.append(qa)
                seen_questions.add(qa["question"].lower())
        
        print(f"✅ Built complete dataset: {len(unique_qa)} unique Q-A pairs")
        print(f"   - Factual: {len(factual_qa)}")
        print(f"   - Explanatory: {len(explanatory_qa)}")
        print(f"   - Summary: {len(summary_qa)}")
        print(f"   - After deduplication: {len(unique_qa)}")
        
        return unique_qa
    
    def save_dataset(self, qa_pairs: List[Dict[str, str]], train_split: float = 0.95):
        """Save dataset to JSONL files with train/test split."""
        print(f"Saving dataset with {train_split:.0%} train / {1-train_split:.0%} test split...")
        
        # Shuffle data
        random.shuffle(qa_pairs)
        
        # Split data
        split_idx = int(len(qa_pairs) * train_split)
        train_data = qa_pairs[:split_idx]
        test_data = qa_pairs[split_idx:]
        
        # Save complete dataset
        with open("data/f1_qa.jsonl", 'w', encoding='utf-8') as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')
        
        # Save train split
        with open("data/f1_qa_train.jsonl", 'w', encoding='utf-8') as f:
            for qa in train_data:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')
        
        # Save test split
        with open("data/f1_qa_test.jsonl", 'w', encoding='utf-8') as f:
            for qa in test_data:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')
        
        print(f"✅ Dataset saved:")
        print(f"   - Complete: data/f1_qa.jsonl ({len(qa_pairs)} pairs)")
        print(f"   - Training: data/f1_qa_train.jsonl ({len(train_data)} pairs)")
        print(f"   - Test: data/f1_qa_test.jsonl ({len(test_data)} pairs)")


def main():
    """Main function to build the F1 Q-A dataset."""
    print("🏎️  Building F1 Question-Answer Dataset")
    print("=" * 50)
    
    builder = F1DatasetBuilder(
        jolpica_file="data/jolpica_raw.json",
        press_file="data/press_raw.json"
    )
    
    try:
        # Build dataset
        qa_pairs = builder.build_complete_dataset()
        
        # Save dataset
        builder.save_dataset(qa_pairs)
        
        print("\n🎉 Dataset creation complete!")
        print("Ready for training with src/train_lora.py")
        
    except Exception as e:
        print(f"❌ Error building dataset: {e}")
        raise


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()
