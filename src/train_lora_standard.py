#!/usr/bin/env python3
"""
Alternative LoRA fine-tuning script for Gemma-3 F1 expert using standard transformers + PEFT.

This script fine-tunes Google's Gemma-3 model using LoRA (Low-Rank Adaptation)
on the F1 question-answer dataset. Works with any CUDA-compatible GPU.

Usage:
    python train_lora_standard.py [--model_name MODEL] [--output_dir OUTPUT]

Example:
    python train_lora_standard.py --model_name google/gemma-3n --output_dir models/gemma-f1-lora
"""

import argparse
import os
import json
from datetime import datetime
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_model_and_tokenizer(model_name: str = "google/gemma-3n-E2B", max_seq_length: int = 512):
    """Load and configure model with LoRA adapters."""
    print(f"🔧 Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load model with optimal settings for your RTX 5070 Ti
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 for memory efficiency
        device_map="auto",  # Automatically distribute across GPU
        trust_remote_code=True,
        use_cache=False,  # Disable cache for training
    )
    
    # Configure LoRA for efficient fine-tuning
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # LoRA rank - balance between performance and memory
        lora_alpha=16,  # LoRA scaling factor
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
            "gate_proj", "up_proj", "down_proj",     # MLP layers
        ],
        lora_dropout=0.05,  # Prevent overfitting
        bias="none",
        inference_mode=False,
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer


def load_dataset(data_file: str):
    """Load and prepare the F1 Q&A dataset."""
    print(f"📊 Loading dataset: {data_file}")
    
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    print(f"✅ Loaded {len(data)} training examples")
    return data


def format_prompt(question: str, answer: str) -> str:
    """Format the prompt for instruction tuning."""
    return f"""You are an expert Formula 1 assistant. Answer questions about F1 accurately and concisely.

User: {question}
Assistant: {answer}"""


def preprocess_function(examples, tokenizer, max_length=512):
    """Preprocess the dataset for training."""
    texts = []
    for question, answer in zip(examples["question"], examples["answer"]):
        formatted_text = format_prompt(question, answer)
        texts.append(formatted_text)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Labels are the same as input_ids for causal LM
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized


def create_trainer(model, tokenizer, train_dataset, output_dir):
    """Create and configure the trainer."""
    
    # Training arguments optimized for RTX 5070 Ti (16GB VRAM)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Conservative batch size
        gradient_accumulation_steps=4,  # Effective batch size = 2 * 4 = 8
        warmup_steps=50,
        learning_rate=2e-4,
        fp16=True,  # Mixed precision for memory efficiency
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        save_total_limit=3,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,  # Disable wandb/tensorboard
        dataloader_pin_memory=False,  # Reduce memory usage
        gradient_checkpointing=True,  # Trade compute for memory
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal language modeling
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    return trainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-3 for F1 Q&A")
    parser.add_argument("--model_name", default="google/gemma-3n-E2B", 
                       help="Base model name")
    parser.add_argument("--data_file", default="data/f1_qa_train.jsonl", 
                       help="Training data file")
    parser.add_argument("--output_dir", default="models/gemma-f1-lora", 
                       help="Output directory")
    parser.add_argument("--max_length", type=int, default=512, 
                       help="Maximum sequence length")
    
    args = parser.parse_args()
    
    print("🏎️  Gemma F1 Expert - LoRA Training")
    print("=" * 50)
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_file}")
    print(f"Output: {args.output_dir}")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU detected - training will be slow")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    try:
        model, tokenizer = setup_model_and_tokenizer(args.model_name, args.max_length)
    except Exception as e:
        print(f"❌ Failed to load model {args.model_name}: {e}")
        print("💡 Try using a smaller model like 'google/gemma-2-2b'")
        return
    
    # Load dataset
    if not Path(args.data_file).exists():
        print(f"❌ Training data not found: {args.data_file}")
        print("💡 Run: python data/build_dataset.py")
        return
    
    raw_data = load_dataset(args.data_file)
    
    # Convert to HuggingFace dataset
    questions = [item["question"] for item in raw_data]
    answers = [item["answer"] for item in raw_data]
    
    dataset_dict = {
        "question": questions,
        "answer": answers
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Preprocess dataset
    print("🔄 Preprocessing dataset...")
    def preprocess_batch(examples):
        return preprocess_function(examples, tokenizer, args.max_length)
    
    train_dataset = dataset.map(
        preprocess_batch,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    print(f"✅ Preprocessed {len(train_dataset)} examples")
    
    # Create trainer
    trainer = create_trainer(model, tokenizer, train_dataset, args.output_dir)
    
    # Start training
    print("🚀 Starting training...")
    start_time = datetime.now()
    
    try:
        trainer.train()
        
        # Save the final model
        print("💾 Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        
        # Save training info
        training_info = {
            "model_name": args.model_name,
            "training_data": args.data_file,
            "num_examples": len(train_dataset),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "output_dir": args.output_dir
        }
        
        with open(f"{args.output_dir}/training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        print("🎉 Training completed successfully!")
        print(f"📁 Model saved to: {args.output_dir}")
        
        # Test generation
        print("🧪 Testing model generation...")
        test_question = "What is DRS in Formula 1?"
        test_prompt = f"""You are an expert Formula 1 assistant. Answer questions about F1 accurately and concisely.

User: {test_question}
Assistant:"""
        
        inputs = tokenizer(test_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test Question: {test_question}")
        print(f"Model Response: {response.split('Assistant:')[-1].strip()}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
