#!/usr/bin/env python3
"""
LoRA fine-tuning script for Gemma-3 F1 expert using Unsloth.

This script fine-tunes Google's Gemma-3 model using LoRA (Low-Rank Adaptation)
on the F1 question-answer dataset. Optimized for 8-16GB GPU memory.

Usage:
    python train_lora.py [--model_name MODEL] [--output_dir OUTPUT]

Example:
    python train_lora.py --model_name google/gemma-3n --output_dir models/gemma-f1-lora
"""

import argparse
import os
import json
from datetime import datetime
from pathlib import Path
import torch
from datasets import Dataset

# Unsloth imports
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments


def setup_model_and_tokenizer(model_name: str, max_seq_length: int = 256):
    """Load and configure model with LoRA adapters."""
    print(f"Loading model: {model_name}")
    
    # Load model in 4-bit for memory efficiency
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
    )
    
    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=4,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,  # LoRA scaling
        lora_dropout=0.05,  # LoRA dropout
        bias="none",  # No bias training
        use_gradient_checkpointing="unsloth",  # Use Unsloth's optimized checkpointing
        random_state=42,
    )
    
    print("✅ Model and LoRA adapters configured")
    return model, tokenizer


def load_and_format_dataset(data_file: str, tokenizer) -> Dataset:
    """Load and format dataset for training."""
    print(f"Loading dataset from: {data_file}")
    
    # Load Q-A pairs
    qa_pairs = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                qa_pairs.append(json.loads(line))
    
    print(f"Loaded {len(qa_pairs)} Q-A pairs")
    
    # Format for instruction tuning
    formatted_data = []
    instruction_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
You are an expert Formula 1 assistant. Answer the following question accurately and concisely.

{question}

### Response:
{answer}"""
    
    for qa in qa_pairs:
        question = qa.get("question", "")
        answer = qa.get("answer", "")
        
        if question and answer:
            formatted_text = instruction_template.format(
                question=question,
                answer=answer
            )
            formatted_data.append({"text": formatted_text})
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(formatted_data)
    
    print(f"✅ Formatted {len(dataset)} training examples")
    return dataset


def train_model(model, tokenizer, dataset: Dataset, output_dir: str):
    """Train the model with LoRA."""
    print("Setting up training...")
    
    # Training arguments optimized for 8-16GB GPU
    training_args = TrainingArguments(
        per_device_train_batch_size=8,      # Batch size per device
        gradient_accumulation_steps=1,       # Gradient accumulation
        warmup_steps=100,                    # Warmup steps
        max_steps=500,                       # Total training steps (adjust based on dataset size)
        num_train_epochs=2,                  # Number of epochs
        learning_rate=2e-4,                  # Learning rate
        fp16=not torch.cuda.is_available(),  # Use FP16 if CUDA available
        bf16=torch.cuda.is_available(),      # Use BF16 on modern GPUs
        logging_steps=10,                    # Log every N steps
        optim="adamw_8bit",                  # 8-bit AdamW optimizer
        weight_decay=0.01,                   # Weight decay
        lr_scheduler_type="cosine",          # Learning rate scheduler
        seed=42,                             # Random seed
        output_dir=output_dir,               # Output directory
        save_steps=100,                      # Save checkpoint every N steps
        save_total_limit=2,                  # Keep only last 2 checkpoints
        dataloader_num_workers=0,            # Number of data loading workers
        remove_unused_columns=False,         # Keep all columns
        report_to=None,                      # Disable wandb/tensorboard
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=256,
        dataset_num_proc=2,
        packing=False,  # Disable packing for simplicity
        args=training_args,
    )
    
    print("🚀 Starting training...")
    start_time = datetime.now()
    
    # Train the model
    trainer.train()
    
    end_time = datetime.now()
    training_duration = end_time - start_time
    print(f"✅ Training completed in {training_duration}")
    
    return trainer


def save_model(model, tokenizer, output_dir: str, push_to_hub: bool = False, hub_model_id: str = None):
    """Save the trained model and adapters."""
    print(f"Saving model to: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save LoRA adapters only (much smaller than full model)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training metadata
    metadata = {
        "base_model": model.config.name_or_path if hasattr(model.config, 'name_or_path') else "google/gemma-3n",
        "lora_rank": 4,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "max_seq_length": 256,
        "training_date": datetime.now().isoformat(),
        "framework": "unsloth",
        "description": "F1 expert fine-tuned on racing Q-A data"
    }
    
    with open(f"{output_dir}/training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Model and adapters saved")
    
    # Optional: Push to Hugging Face Hub
    if push_to_hub and hub_model_id:
        try:
            print(f"Pushing to Hub: {hub_model_id}")
            model.push_to_hub(hub_model_id, token=True)
            tokenizer.push_to_hub(hub_model_id, token=True)
            print("✅ Model pushed to Hugging Face Hub")
        except Exception as e:
            print(f"⚠️  Failed to push to Hub: {e}")


def estimate_training_time(dataset_size: int, batch_size: int = 8, epochs: int = 2) -> str:
    """Estimate training time based on dataset size."""
    steps_per_epoch = dataset_size // batch_size
    total_steps = steps_per_epoch * epochs
    
    # Rough estimate: ~0.5-1 second per step on modern GPU
    estimated_minutes = (total_steps * 0.75) / 60
    
    if estimated_minutes < 60:
        return f"~{estimated_minutes:.0f} minutes"
    else:
        return f"~{estimated_minutes/60:.1f} hours"


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-3 for F1 expertise")
    parser.add_argument("--model_name", default="google/gemma-3n", help="Base model name")
    parser.add_argument("--data_file", default="data/f1_qa_train.jsonl", help="Training data file")
    parser.add_argument("--output_dir", default="models/gemma-f1-lora", help="Output directory")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", help="Hugging Face Hub model ID")
    
    args = parser.parse_args()
    
    print("🏎️  Gemma-3 F1 Expert Training")
    print("=" * 50)
    print(f"Base model: {args.model_name}")
    print(f"Data file: {args.data_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"Max sequence length: {args.max_seq_length}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("⚠️  No GPU detected. Training will be slow on CPU.")
    
    try:
        # Load and setup model
        model, tokenizer = setup_model_and_tokenizer(
            args.model_name, 
            args.max_seq_length
        )
        
        # Load and format dataset
        dataset = load_and_format_dataset(args.data_file, tokenizer)
        
        # Estimate training time
        estimated_time = estimate_training_time(len(dataset))
        print(f"Estimated training time: {estimated_time}")
        
        # Train model
        trainer = train_model(model, tokenizer, dataset, args.output_dir)
        
        # Save model
        save_model(
            model, 
            tokenizer, 
            args.output_dir,
            args.push_to_hub,
            args.hub_model_id
        )
        
        print("\n🎉 Training complete!")
        print(f"Model saved to: {args.output_dir}")
        print("\nNext steps:")
        print("1. Test the model: python src/generate.py 'Who won the 2023 Monaco GP?'")
        print("2. Run evaluation: python src/evaluate.py")
        print("3. Launch web app: streamlit run src/webapp.py")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
