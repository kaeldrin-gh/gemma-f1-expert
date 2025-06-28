#!/usr/bin/env python3
"""
Evaluation script for the fine-tuned F1 expert model.

This script evaluates the model's performance on held-out test data,
including exact match accuracy for factual questions and manual
assessment of explanatory content quality.

Usage:
    python evaluate.py [--model_path MODEL] [--test_file TEST_FILE]

Example:
    python evaluate.py --model_path models/gemma-f1-lora --test_file data/f1_qa_test.jsonl
"""

import argparse
import json
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re


class F1ModelEvaluator:
    """Evaluator for F1 expert model."""
    
    def __init__(self, model_path: str, base_model: str = "google/gemma-3n-E2B"):
        self.model_path = model_path
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        print(f"Loading model from: {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                load_in_4bit=True if torch.cuda.is_available() else False
            )
            
            # Load LoRA adapters
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            
            print("✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Fallback to base model only
            print("Falling back to base model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
    
    def generate_answer(self, question: str, max_length: int = 256) -> str:
        """Generate answer for a given question."""
        # Format prompt
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
You are an expert Formula 1 assistant. Answer the following question accurately and concisely.

{question}

### Response:
"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (after "### Response:")
        response_start = response.find("### Response:")
        if response_start != -1:
            answer = response[response_start + len("### Response:"):].strip()
        else:
            answer = response.strip()
        
        return answer
    
    def load_test_data(self, test_file: str) -> List[Dict[str, Any]]:
        """Load test dataset."""
        test_data = []
        
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        test_data.append(json.loads(line))
            
            print(f"✅ Loaded {len(test_data)} test examples")
            return test_data
            
        except FileNotFoundError:
            print(f"❌ Test file not found: {test_file}")
            return []
    
    def exact_match_evaluation(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate exact match accuracy on factual questions."""
        print("Evaluating exact match accuracy...")
        
        factual_questions = [qa for qa in test_data if qa.get("type") == "factual"]
        
        if not factual_questions:
            print("⚠️  No factual questions found in test data")
            return {"exact_match": 0.0, "total_factual": 0}
        
        correct = 0
        total = len(factual_questions)
        
        print(f"Evaluating {total} factual questions...")
        
        for i, qa in enumerate(factual_questions):
            question = qa["question"]
            expected = qa["answer"].lower().strip()
            
            # Generate model answer
            generated = self.generate_answer(question).lower().strip()
            
            # Check for exact match (flexible matching)
            if self._flexible_match(expected, generated):
                correct += 1
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{total}")
        
        accuracy = correct / total
        
        print(f"✅ Exact match accuracy: {accuracy:.3f} ({correct}/{total})")
        
        return {
            "exact_match": accuracy,
            "correct": correct,
            "total_factual": total
        }
    
    def _flexible_match(self, expected: str, generated: str) -> bool:
        """Check if generated answer matches expected with flexibility."""
        # Remove common variations
        expected_clean = re.sub(r'\s+', ' ', expected.lower().strip())
        generated_clean = re.sub(r'\s+', ' ', generated.lower().strip())
        
        # Exact match
        if expected_clean == generated_clean:
            return True
        
        # Check if expected answer is contained in generated
        if expected_clean in generated_clean:
            return True
        
        # Check key entities (names, years, etc.)
        expected_entities = re.findall(r'\b[A-Z][a-z]+\b|\b\d{4}\b', expected)
        generated_entities = re.findall(r'\b[A-Z][a-z]+\b|\b\d{4}\b', generated)
        
        if expected_entities and generated_entities:
            # Check if most entities match
            matches = sum(1 for entity in expected_entities if entity.lower() in generated.lower())
            return matches >= len(expected_entities) * 0.7
        
        return False
    
    def manual_evaluation(self, test_data: List[Dict[str, Any]], num_samples: int = 10):
        """Manual evaluation of response quality."""
        print(f"\nManual evaluation of {num_samples} random samples:")
        print("=" * 60)
        
        # Sample different types
        samples = random.sample(test_data, min(num_samples, len(test_data)))
        
        for i, qa in enumerate(samples, 1):
            question = qa["question"]
            expected = qa["answer"]
            qa_type = qa.get("type", "unknown")
            
            generated = self.generate_answer(question)
            
            print(f"\n{i}. [{qa_type.upper()}] {question}")
            print(f"Expected: {expected}")
            print(f"Generated: {generated}")
            print("-" * 60)
    
    def category_analysis(self, test_data: List[Dict[str, Any]]):
        """Analyze performance by question category."""
        print("\nCategory-wise analysis:")
        print("=" * 40)
        
        categories = {}
        for qa in test_data:
            category = qa.get("category", "unknown")
            if category not in categories:
                categories[category] = []
            categories[category].append(qa)
        
        for category, questions in categories.items():
            print(f"\n{category.upper()}: {len(questions)} questions")
            
            if len(questions) <= 3:  # Evaluate small categories completely
                correct = 0
                for qa in questions:
                    expected = qa["answer"].lower().strip()
                    generated = self.generate_answer(qa["question"]).lower().strip()
                    if self._flexible_match(expected, generated):
                        correct += 1
                
                accuracy = correct / len(questions)
                print(f"Accuracy: {accuracy:.3f} ({correct}/{len(questions)})")
    
    def comprehensive_evaluation(self, test_file: str):
        """Run comprehensive evaluation."""
        print("🏎️  F1 Expert Model Evaluation")
        print("=" * 50)
        
        # Load test data
        test_data = self.load_test_data(test_file)
        
        if not test_data:
            return
        
        # Exact match evaluation
        exact_match_results = self.exact_match_evaluation(test_data)
        
        # Category analysis
        self.category_analysis(test_data)
        
        # Manual evaluation
        self.manual_evaluation(test_data, num_samples=5)
        
        # Summary
        print("\n🎯 EVALUATION SUMMARY")
        print("=" * 30)
        print(f"Total test examples: {len(test_data)}")
        print(f"Factual questions: {exact_match_results['total_factual']}")
        print(f"Exact match accuracy: {exact_match_results['exact_match']:.3f}")
        
        # Performance assessment
        accuracy = exact_match_results['exact_match']
        if accuracy >= 0.85:
            print("🎉 Excellent performance!")
        elif accuracy >= 0.70:
            print("✅ Good performance")
        elif accuracy >= 0.50:
            print("⚠️  Fair performance - consider more training")
        else:
            print("❌ Poor performance - model needs improvement")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate F1 expert model")
    parser.add_argument("--model_path", default="models/gemma-f1-lora", help="Path to fine-tuned model")
    parser.add_argument("--base_model", default="google/gemma-3n-E2B", help="Base model name")
    parser.add_argument("--test_file", default="data/f1_qa_test.jsonl", help="Test data file")
    parser.add_argument("--manual_samples", type=int, default=5, help="Number of manual evaluation samples")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"❌ Model not found: {args.model_path}")
        print("Please train the model first: python src/train_lora.py")
        return
    
    # Check if test file exists
    if not Path(args.test_file).exists():
        print(f"❌ Test file not found: {args.test_file}")
        print("Please create the dataset first: python data/build_dataset.py")
        return
    
    try:
        # Initialize evaluator
        evaluator = F1ModelEvaluator(args.model_path, args.base_model)
        
        # Run evaluation
        evaluator.comprehensive_evaluation(args.test_file)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    # Set random seed for reproducible manual evaluation
    random.seed(42)
    main()
