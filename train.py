#!/usr/bin/env python3
"""
Fine-tuning Script for COBOL->Python Translation
Uses Qwen2.5-Coder-1.5B-Instruct with LoRA for efficient training.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from tqdm import tqdm
from rich.console import Console
from rich.logging import RichHandler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_SEQ_LENGTH = 2048


def setup_device():
    """Setup and return the compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.print(f"[green]Using GPU: {gpu_name} ({gpu_memory:.1f} GB)[/green]")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        console.print("[green]Using Apple Silicon MPS[/green]")
    else:
        device = torch.device("cpu")
        console.print("[yellow]Warning: Using CPU (training will be slow)[/yellow]")
    return device


def load_tokenizer():
    """Load and configure the tokenizer."""
    console.print(f"[blue]Loading tokenizer from {MODEL_NAME}...[/blue]")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    console.print(f"[green]Tokenizer loaded. Vocab size: {len(tokenizer)}[/green]")
    return tokenizer


def load_model(device: torch.device, use_gradient_checkpointing: bool = True):
    """Load the model with memory optimizations."""
    console.print(f"[blue]Loading model from {MODEL_NAME}...[/blue]")

    # Determine dtype based on device
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    console.print(f"[yellow]Using dtype: {dtype}[/yellow]")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
        attn_implementation="eager",  # Use eager attention for compatibility
    )

    if device.type != "cuda":
        model = model.to(device)

    # Enable gradient checkpointing for memory efficiency
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        console.print("[green]Gradient checkpointing enabled[/green]")

    # Print model size
    param_count = sum(p.numel() for p in model.parameters())
    console.print(f"[green]Model loaded. Parameters: {param_count/1e6:.1f}M[/green]")

    return model


def apply_lora(model, lora_rank: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
    """Apply LoRA adapters to the model."""
    console.print(f"[blue]Applying LoRA (rank={lora_rank}, alpha={lora_alpha})...[/blue]")

    # Define LoRA configuration
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"[green]LoRA applied. Trainable: {trainable_params/1e6:.2f}M / {total_params/1e6:.1f}M ({100*trainable_params/total_params:.2f}%)[/green]")

    return model


def load_data(data_dir: str):
    """Load the processed training and validation data."""
    console.print(f"[blue]Loading data from {data_dir}...[/blue]")

    train_file = Path(data_dir) / "train.jsonl"
    val_file = Path(data_dir) / "val.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found at {train_file}. Run select_data.py first.")

    train_dataset = load_dataset("json", data_files=str(train_file), split="train")

    if val_file.exists():
        val_dataset = load_dataset("json", data_files=str(val_file), split="train")
    else:
        # Split train data if no validation file
        split_data = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_data["train"]
        val_dataset = split_data["test"]

    console.print(f"[green]Loaded {len(train_dataset)} train samples, {len(val_dataset)} val samples[/green]")

    return train_dataset, val_dataset


def format_chat_template(example: Dict, tokenizer) -> Dict:
    """Format the example using the model's chat template."""
    messages = example["messages"]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": text}


def tokenize_function(examples: Dict, tokenizer, max_length: int) -> Dict:
    """Tokenize the formatted examples."""
    # Tokenize
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def prepare_datasets(train_dataset, val_dataset, tokenizer, max_length: int):
    """Prepare datasets for training."""
    console.print("[blue]Preparing datasets...[/blue]")

    # Format with chat template
    train_dataset = train_dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        desc="Formatting train data"
    )
    val_dataset = val_dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        desc="Formatting val data"
    )

    # Tokenize
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train data"
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing val data"
    )

    console.print(f"[green]Datasets prepared. Train: {len(train_dataset)}, Val: {len(val_dataset)}[/green]")

    return train_dataset, val_dataset


def create_training_args(
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.1,
    save_steps: int = 100,
    logging_steps: int = 10,
    eval_steps: int = 100,
) -> TrainingArguments:
    """Create training arguments optimized for consumer GPUs."""

    # Calculate effective batch size
    effective_batch_size = batch_size * gradient_accumulation_steps
    console.print(f"[yellow]Effective batch size: {effective_batch_size}[/yellow]")

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        report_to="none",  # Disable wandb/tensorboard for simplicity
        remove_unused_columns=False,
        optim="adamw_torch",
        max_grad_norm=1.0,
        push_to_hub=False,
    )


def train(
    data_dir: str = "data/processed",
    output_dir: str = "checkpoints",
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    lora_rank: int = 16,
    max_length: int = MAX_SEQ_LENGTH,
    resume_from_checkpoint: Optional[str] = None,
):
    """Main training function."""
    console.print("[bold green]" + "=" * 60)
    console.print("[bold green]COBOL->Python Fine-tuning with LoRA[/bold green]")
    console.print("[bold green]" + "=" * 60)

    # Setup
    device = setup_device()
    tokenizer = load_tokenizer()

    # Load data
    train_dataset, val_dataset = load_data(data_dir)

    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(
        train_dataset, val_dataset, tokenizer, max_length
    )

    # Load model
    model = load_model(device, use_gradient_checkpointing=True)

    # Apply LoRA
    model = apply_lora(model, lora_rank=lora_rank)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
    )

    # Training arguments
    training_args = create_training_args(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    console.print("\n[bold green]Starting training...[/bold green]")

    if resume_from_checkpoint:
        console.print(f"[yellow]Resuming from checkpoint: {resume_from_checkpoint}[/yellow]")

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted. Saving checkpoint...[/yellow]")
        trainer.save_model(f"{output_dir}/interrupted_checkpoint")
        console.print(f"[green]Checkpoint saved to {output_dir}/interrupted_checkpoint[/green]")
        return

    # Save final model
    final_output_dir = f"{output_dir}/final"
    console.print(f"\n[blue]Saving final model to {final_output_dir}...[/blue]")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    # Save training info
    training_info = {
        "base_model": MODEL_NAME,
        "lora_rank": lora_rank,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size * gradient_accumulation_steps,
        "max_length": max_length,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "timestamp": datetime.now().isoformat(),
    }

    with open(f"{final_output_dir}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)

    console.print("\n[bold green]Training complete![/bold green]")
    console.print(f"[bold]Model saved to:[/bold] {final_output_dir}")
    console.print(f"[bold]Next step:[/bold] Run 'python test.py --model {final_output_dir}' to test")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen for COBOL->Python translation")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory containing processed training data")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--max-length", type=int, default=MAX_SEQ_LENGTH,
                        help="Maximum sequence length")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        max_length=args.max_length,
        resume_from_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
