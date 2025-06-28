#!/usr/bin/env python3
"""
Quick dataset builder test using the collected test data.
"""

import json
import random
from datetime import datetime
from pathlib import Path

def create_test_dataset():
    """Create a small test dataset from our collected data."""
    
    # Check if we have test data
    test_data_file = "data/test_jolpica_data.json"
    press_data_file = "data/press_raw.json"
    
    if not Path(test_data_file).exists():
        print(f"❌ Test data file not found: {test_data_file}")
        return False
    
    print("📊 Building test dataset...")
    
    # Load test data
    with open(test_data_file, 'r', encoding='utf-8') as f:
        jolpica_data = json.load(f)
    
    press_data = {}
    if Path(press_data_file).exists():
        with open(press_data_file, 'r', encoding='utf-8') as f:
            press_data = json.load(f)
    
    qa_pairs = []
    
    # Create factual questions from race data
    race_data = jolpica_data.get("race_data", [])
    for race in race_data:
        year = "2024"  # We know it's 2024 from our test
        race_name = race["race_name"]
        winner = race["winner"]
        constructor = race["constructor"]
        
        qa_pairs.extend([
            {
                "question": f"Who won the {year} {race_name}?",
                "answer": f"{winner} won the {year} {race_name} driving for {constructor}.",
                "type": "factual",
                "category": "race_winner"
            },
            {
                "question": f"Which team won the {year} {race_name}?",
                "answer": f"{constructor} won the {year} {race_name} with {winner}.",
                "type": "factual",
                "category": "constructor_winner"
            }
        ])
    
    # Add championship questions from standings
    standings_data = jolpica_data.get("standings", {})
    if standings_data and "MRData" in standings_data:
        standings_list = standings_data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
        if standings_list and standings_list[0].get("DriverStandings"):
            top_drivers = standings_list[0]["DriverStandings"][:3]  # Top 3
            
            for i, driver in enumerate(top_drivers):
                name = f"{driver['Driver']['givenName']} {driver['Driver']['familyName']}"
                points = driver['points']
                position = i + 1
                
                qa_pairs.append({
                    "question": f"Who is in {position}{'st' if position==1 else 'nd' if position==2 else 'rd'} place in the 2024 F1 championship?",
                    "answer": f"{name} is in {position}{'st' if position==1 else 'nd' if position==2 else 'rd'} place in the 2024 F1 championship with {points} points.",
                    "type": "factual",
                    "category": "championship"
                })
    
    # Add explanatory content
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
            "question": "What is Formula 1?",
            "answer": "Formula 1 is the highest class of international auto racing for single-seater formula racing cars. It features a series of races called Grands Prix held on purpose-built circuits and closed public roads around the world. F1 cars are the fastest regulated road-course racing cars in the world, featuring advanced aerodynamics, hybrid power units, and cutting-edge technology.",
            "type": "explanatory",
            "category": "general"
        }
    ]
    
    qa_pairs.extend(explanatory_qa)
    
    # Add metadata to each pair
    for i, qa in enumerate(qa_pairs):
        qa["id"] = f"f1_qa_test_{i:04d}"
        qa["created_at"] = datetime.now().isoformat()
    
    print(f"✅ Created {len(qa_pairs)} Q-A pairs")
    
    # Show examples
    print("\n📝 Example Q-A pairs:")
    for i, qa in enumerate(qa_pairs[:5]):
        print(f"\n{i+1}. [{qa['type'].upper()}] {qa['question']}")
        print(f"   Answer: {qa['answer'][:100]}{'...' if len(qa['answer']) > 100 else ''}")
    
    # Create train/test split
    random.shuffle(qa_pairs)
    split_idx = int(len(qa_pairs) * 0.8)  # 80% train, 20% test
    
    train_data = qa_pairs[:split_idx]
    test_data = qa_pairs[split_idx:]
    
    # Save datasets
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Save complete dataset
    with open("data/f1_qa_test.jsonl", 'w', encoding='utf-8') as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')
    
    # Save train split
    with open("data/f1_qa_train_test.jsonl", 'w', encoding='utf-8') as f:
        for qa in train_data:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')
    
    # Save test split
    with open("data/f1_qa_test_split.jsonl", 'w', encoding='utf-8') as f:
        for qa in test_data:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Test dataset saved:")
    print(f"   - Complete: data/f1_qa_test.jsonl ({len(qa_pairs)} pairs)")
    print(f"   - Training: data/f1_qa_train_test.jsonl ({len(train_data)} pairs)")
    print(f"   - Test: data/f1_qa_test_split.jsonl ({len(test_data)} pairs)")
    
    return True

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    success = create_test_dataset()
    if success:
        print("\n🎉 Test dataset creation successful!")
    else:
        print("\n❌ Test dataset creation failed!")
    exit(0 if success else 1)
