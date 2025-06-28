#!/usr/bin/env python3
"""
Dataset preparation utilities for F1 Q-A fine-tuning.

This module provides functions to format and preprocess the F1 question-answer
dataset for training with Unsloth and Transformers.

Functions:
    format_for_training: Convert Q-A pairs to chat format
    load_dataset: Load and format JSONL dataset
    create_prompt_template: Create consistent prompt formatting
"""

import json
from typing import List, Dict, Any, Iterator
from datasets import Dataset


def create_prompt_template() -> str:
    """Create consistent prompt template for F1 assistant."""
    return """You are an expert Formula 1 assistant. Answer questions about F1 accurately and concisely.

User: {question}
Assistant: {answer}"""


def format_qa_for_chat(question: str, answer: str) -> Dict[str, List[Dict[str, str]]]:
    """Format Q-A pair for chat-based training."""
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
    }


def format_qa_for_instruct(question: str, answer: str) -> Dict[str, str]:
    """Format Q-A pair for instruction-based training."""
    template = create_prompt_template()
    return {
        "text": template.format(question=question, answer=answer),
        "input": question,
        "output": answer
    }


def load_jsonl_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load Q-A pairs from JSONL file."""
    qa_pairs = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    qa_pairs.append(json.loads(line))
        
        print(f"✅ Loaded {len(qa_pairs)} Q-A pairs from {file_path}")
        return qa_pairs
        
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON in {file_path}: {e}")
        return []


def prepare_training_data(file_path: str, format_type: str = "instruct") -> Dataset:
    """Prepare training data in the specified format."""
    qa_pairs = load_jsonl_dataset(file_path)
    
    if not qa_pairs:
        raise ValueError(f"No data loaded from {file_path}")
    
    formatted_data = []
    
    for qa in qa_pairs:
        question = qa.get("question", "")
        answer = qa.get("answer", "")
        
        if not question or not answer:
            continue
        
        if format_type == "chat":
            formatted_data.append(format_qa_for_chat(question, answer))
        elif format_type == "instruct":
            formatted_data.append(format_qa_for_instruct(question, answer))
        else:
            raise ValueError(f"Unsupported format_type: {format_type}")
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(formatted_data)
    
    print(f"✅ Prepared {len(dataset)} training examples in '{format_type}' format")
    return dataset


def create_evaluation_samples(file_path: str, num_samples: int = 50) -> List[Dict[str, str]]:
    """Create evaluation samples for manual review."""
    qa_pairs = load_jsonl_dataset(file_path)
    
    if not qa_pairs:
        return []
    
    # Sample different types of questions
    factual = [qa for qa in qa_pairs if qa.get("type") == "factual"]
    explanatory = [qa for qa in qa_pairs if qa.get("type") == "explanatory"]
    summary = [qa for qa in qa_pairs if qa.get("type") == "summary"]
    
    samples = []
    samples_per_type = num_samples // 3
    
    # Sample from each type
    import random
    if factual:
        samples.extend(random.sample(factual, min(samples_per_type, len(factual))))
    if explanatory:
        samples.extend(random.sample(explanatory, min(samples_per_type, len(explanatory))))
    if summary:
        samples.extend(random.sample(summary, min(samples_per_type, len(summary))))
    
    # Fill remaining with random samples
    remaining = num_samples - len(samples)
    if remaining > 0:
        other_samples = [qa for qa in qa_pairs if qa not in samples]
        if other_samples:
            samples.extend(random.sample(other_samples, min(remaining, len(other_samples))))
    
    print(f"✅ Created {len(samples)} evaluation samples")
    return samples


def print_dataset_stats(file_path: str):
    """Print statistics about the dataset."""
    qa_pairs = load_jsonl_dataset(file_path)
    
    if not qa_pairs:
        return
    
    # Count by type
    type_counts = {}
    category_counts = {}
    
    for qa in qa_pairs:
        qa_type = qa.get("type", "unknown")
        category = qa.get("category", "unknown")
        
        type_counts[qa_type] = type_counts.get(qa_type, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
    
    print(f"\n📊 Dataset Statistics for {file_path}:")
    print(f"Total Q-A pairs: {len(qa_pairs)}")
    
    print("\nBy type:")
    for qa_type, count in sorted(type_counts.items()):
        print(f"  {qa_type}: {count}")
    
    print("\nBy category:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")
    
    # Calculate average lengths
    question_lengths = [len(qa.get("question", "")) for qa in qa_pairs]
    answer_lengths = [len(qa.get("answer", "")) for qa in qa_pairs]
    
    print(f"\nAverage question length: {sum(question_lengths) / len(question_lengths):.1f} chars")
    print(f"Average answer length: {sum(answer_lengths) / len(answer_lengths):.1f} chars")


if __name__ == "__main__":
    # Example usage and testing
    print("🔧 Testing dataset preparation utilities...")
    
    # Test with training data
    train_file = "data/f1_qa_train.jsonl"
    test_file = "data/f1_qa_test.jsonl"
    
    for file_path in [train_file, test_file]:
        print(f"\n{'='*50}")
        print_dataset_stats(file_path)
    
    # Test data preparation
    try:
        dataset = prepare_training_data(train_file, format_type="instruct")
        print(f"\n✅ Successfully prepared training dataset: {len(dataset)} examples")
        
        # Show example
        if len(dataset) > 0:
            example = dataset[0]
            print("\n📝 Example formatted training data:")
            print(f"Text: {example['text'][:200]}...")
            
    except Exception as e:
        print(f"❌ Error preparing training data: {e}")
    
    print("\n🎉 Dataset preparation utilities test complete!")
