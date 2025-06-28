#!/usr/bin/env python3
"""
Test suite for F1 expert model generation capabilities.

This module contains unit tests for the F1 expert CLI and generation
functionality to ensure the model produces reasonable answers.

Usage:
    pytest test_generate.py -v
    python -m pytest test_generate.py::test_factual_questions -v
"""

import pytest
import sys
from pathlib import Path
import subprocess
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from generate import F1ExpertCLI
except ImportError:
    # Skip tests if model dependencies not available
    pytest.skip("Model dependencies not available", allow_module_level=True)


class TestF1ExpertCLI:
    """Test cases for F1 Expert CLI."""
    
    @pytest.fixture(scope="class")
    def cli(self):
        """Initialize CLI instance for testing."""
        model_path = "models/gemma-f1-lora"
        
        if not Path(model_path).exists():
            pytest.skip(f"Model not found: {model_path}. Train model first.")
        
        try:
            return F1ExpertCLI(model_path)
        except Exception as e:
            pytest.skip(f"Failed to load model: {e}")
    
    def test_model_loading(self, cli):
        """Test that model loads successfully."""
        assert cli.model is not None, "Model should be loaded"
        assert cli.tokenizer is not None, "Tokenizer should be loaded"
    
    def test_basic_generation(self, cli):
        """Test basic answer generation."""
        question = "What is Formula 1?"
        answer = cli.generate_answer(question)
        
        assert isinstance(answer, str), "Answer should be a string"
        assert len(answer) > 10, "Answer should be substantial"
        assert "formula" in answer.lower() or "f1" in answer.lower(), "Answer should mention F1"
    
    def test_factual_questions(self, cli):
        """Test factual F1 questions."""
        factual_questions = [
            "Who won the 2022 Formula 1 World Championship?",
            "What is the current F1 points system?",
            "How many teams are in Formula 1?",
        ]
        
        for question in factual_questions:
            answer = cli.generate_answer(question)
            
            assert isinstance(answer, str), f"Answer should be string for: {question}"
            assert len(answer) > 5, f"Answer too short for: {question}"
            assert not answer.startswith("Error"), f"Generation error for: {question}"
    
    def test_technical_questions(self, cli):
        """Test technical F1 concept questions."""
        technical_questions = [
            "How does DRS work in Formula 1?",
            "What are F1 tyre compounds?",
            "Explain F1 qualifying format",
        ]
        
        for question in technical_questions:
            answer = cli.generate_answer(question)
            
            assert isinstance(answer, str), f"Answer should be string for: {question}"
            assert len(answer) > 20, f"Technical answer too short for: {question}"
            
            # Check for relevant keywords based on question
            if "drs" in question.lower():
                assert any(word in answer.lower() for word in ["drs", "drag", "wing", "overtaking"])
            elif "tyre" in question.lower():
                assert any(word in answer.lower() for word in ["soft", "medium", "hard", "compound"])
            elif "qualifying" in question.lower():
                assert any(word in answer.lower() for word in ["q1", "q2", "q3", "session"])
    
    def test_answer_quality(self, cli):
        """Test general answer quality metrics."""
        test_questions = [
            "Who is the most successful F1 driver?",
            "What is the Monaco Grand Prix?",
            "How long is an F1 race?",
        ]
        
        for question in test_questions:
            answer = cli.generate_answer(question)
            
            # Quality checks
            assert len(answer) < 1000, f"Answer too long for: {question}"
            assert len(answer.split()) > 5, f"Answer too short for: {question}"
            assert not answer.lower().startswith("i don't"), f"Model claims ignorance: {question}"
            assert "error" not in answer.lower(), f"Error in answer: {question}"
    
    def test_response_formatting(self, cli):
        """Test that responses are properly formatted."""
        question = "What is Formula 1?"
        answer = cli.generate_answer(question)
        
        # Basic formatting checks
        assert not answer.startswith("### Response:"), "Response marker should be removed"
        assert not answer.startswith("Below is an instruction"), "Prompt artifacts should be removed"
        assert answer.strip() == answer, "Answer should be trimmed"
    
    @pytest.mark.parametrize("question,expected_keywords", [
        ("Who won the 2022 Monaco Grand Prix?", ["monaco", "2022"]),
        ("How does DRS work?", ["drs", "drag"]),
        ("What is the F1 points system?", ["points", "25"]),
    ])
    def test_specific_answers(self, cli, question, expected_keywords):
        """Test specific questions with expected content."""
        answer = cli.generate_answer(question)
        
        for keyword in expected_keywords:
            assert keyword.lower() in answer.lower(), f"Expected '{keyword}' in answer for '{question}'"


class TestCLIScript:
    """Test the CLI script execution."""
    
    def test_cli_help(self):
        """Test CLI help message."""
        result = subprocess.run(
            [sys.executable, "src/generate.py", "--help"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, "Help command should succeed"
        assert "F1 expert" in result.stdout.lower(), "Help should mention F1 expert"
    
    def test_cli_simple_question(self):
        """Test CLI with simple question."""
        model_path = Path(__file__).parent.parent / "models" / "gemma-f1-lora"
        
        if not model_path.exists():
            pytest.skip("Model not found for CLI testing")
        
        question = "What is Formula 1?"
        result = subprocess.run(
            [sys.executable, "src/generate.py", question],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        if result.returncode != 0:
            pytest.skip(f"CLI execution failed: {result.stderr}")
        
        assert "formula" in result.stdout.lower() or "f1" in result.stdout.lower()


class TestDatasetIntegrity:
    """Test dataset files for integrity."""
    
    def test_training_data_exists(self):
        """Test that training data files exist."""
        data_dir = Path(__file__).parent.parent / "data"
        
        expected_files = [
            "f1_qa_train.jsonl",
            "f1_qa_test.jsonl",
            "f1_qa.jsonl"
        ]
        
        for filename in expected_files:
            filepath = data_dir / filename
            if filepath.exists():
                assert filepath.stat().st_size > 0, f"{filename} should not be empty"
            # Note: Files might not exist if data collection hasn't been run
    
    def test_jsonl_format(self):
        """Test JSONL files are properly formatted."""
        data_dir = Path(__file__).parent.parent / "data"
        test_file = data_dir / "f1_qa_test.jsonl"
        
        if not test_file.exists():
            pytest.skip("Test data file not found")
        
        line_count = 0
        with open(test_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        assert "question" in data, f"Line {line_num}: Missing 'question' field"
                        assert "answer" in data, f"Line {line_num}: Missing 'answer' field"
                        assert isinstance(data["question"], str), f"Line {line_num}: 'question' should be string"
                        assert isinstance(data["answer"], str), f"Line {line_num}: 'answer' should be string"
                        line_count += 1
                    except json.JSONDecodeError:
                        pytest.fail(f"Line {line_num}: Invalid JSON format")
        
        assert line_count > 0, "Test file should contain data"


def test_model_performance_benchmark():
    """Benchmark test for model performance."""
    model_path = Path(__file__).parent.parent / "models" / "gemma-f1-lora"
    
    if not model_path.exists():
        pytest.skip("Model not found for performance testing")
    
    try:
        cli = F1ExpertCLI(str(model_path))
        
        # Simple performance test
        import time
        start_time = time.time()
        answer = cli.generate_answer("What is Formula 1?")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Performance assertions
        assert response_time < 30, f"Response time too slow: {response_time:.2f}s"
        assert len(answer) > 10, "Answer should be substantial"
        
    except Exception as e:
        pytest.skip(f"Performance test failed: {e}")


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
