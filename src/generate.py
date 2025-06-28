#!/usr/bin/env python3
"""
CLI interface for asking questions to the F1 expert model.

This script provides a command-line interface to query the fine-tuned
F1 model with questions and get expert responses.

Usage:
    python generate.py "Who won the 2023 Monaco Grand Prix?"
    python generate.py "How does DRS work in Formula 1?"
    python generate.py --interactive  # Start interactive session

Example:
    python generate.py "What is the current F1 points system?"
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class F1ExpertCLI:
    """Command-line interface for F1 expert model."""
    
    def __init__(self, model_path: str = "models/gemma-f1-lora", base_model: str = "google/gemma-3n"):
        self.model_path = model_path
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model."""
        print("🏎️  Loading F1 Expert Model...")
        
        if not Path(self.model_path).exists():
            print(f"❌ Model not found: {self.model_path}")
            print("Please train the model first: python src/train_lora.py")
            sys.exit(1)
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
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
            print("Falling back to base model...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                print("✅ Base model loaded")
            except Exception as base_e:
                print(f"❌ Failed to load base model: {base_e}")
                sys.exit(1)
    
    def generate_answer(self, question: str) -> str:
        """Generate answer for a question."""
        # Create prompt
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
You are an expert Formula 1 assistant. Answer the following question accurately and concisely.

{question}

### Response:
"""
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                top_p=0.9,
                early_stopping=True
            )
        
        # Decode and clean response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (everything after "### Response:")
        response_marker = "### Response:"
        if response_marker in response:
            answer = response.split(response_marker, 1)[1].strip()
        else:
            answer = response.strip()
        
        # Clean up common issues
        answer = self._clean_response(answer)
        
        return answer
    
    def _clean_response(self, response: str) -> str:
        """Clean up the generated response."""
        # Remove any remaining prompt artifacts
        response = response.replace("### Instruction:", "").strip()
        response = response.replace("Below is an instruction", "").strip()
        
        # Remove excessive whitespace
        import re
        response = re.sub(r'\n\s*\n', '\n', response)
        response = re.sub(r'\s+', ' ', response)
        
        # Truncate at natural stopping points
        sentences = response.split('. ')
        if len(sentences) > 1:
            # Keep response concise (max 3 sentences for most answers)
            response = '. '.join(sentences[:3])
            if not response.endswith('.'):
                response += '.'
        
        return response.strip()
    
    def interactive_mode(self):
        """Start interactive question-answering session."""
        print("\n🏁 F1 Expert Interactive Mode")
        print("Ask me anything about Formula 1!")
        print("Type 'quit', 'exit', or 'q' to end the session.")
        print("-" * 50)
        
        while True:
            try:
                question = input("\n🏎️  Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q', '']:
                    print("Goodbye! 🏁")
                    break
                
                if len(question) < 3:
                    print("Please ask a longer question.")
                    continue
                
                print("\n🤖 F1 Expert:", end=" ")
                answer = self.generate_answer(question)
                print(answer)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 🏁")
                break
            except Exception as e:
                print(f"\n❌ Error generating answer: {e}")
    
    def answer_question(self, question: str):
        """Answer a single question."""
        if not question.strip():
            print("Please provide a question.")
            return
        
        print(f"🏎️  Question: {question}")
        print("\n🤖 F1 Expert:", end=" ")
        
        try:
            answer = self.generate_answer(question)
            print(answer)
        except Exception as e:
            print(f"❌ Error generating answer: {e}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Ask questions to the F1 expert model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py "Who won the 2023 Monaco Grand Prix?"
  python generate.py "How does DRS work in Formula 1?"
  python generate.py --interactive
        """
    )
    
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask the F1 expert"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start interactive question-answering session"
    )
    parser.add_argument(
        "--model_path",
        default="models/gemma-f1-lora",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--base_model",
        default="google/gemma-3n",
        help="Base model name"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.interactive and not args.question:
        print("❌ Please provide a question or use --interactive mode")
        parser.print_help()
        sys.exit(1)
    
    try:
        # Initialize CLI
        cli = F1ExpertCLI(args.model_path, args.base_model)
        
        if args.interactive:
            cli.interactive_mode()
        else:
            cli.answer_question(args.question)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
