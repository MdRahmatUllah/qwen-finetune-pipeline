#!/usr/bin/env python3
"""
QLoRA fine-tuning script for Qwen2.5-7B using TRL SFTTrainer.

Usage:
    python scripts/04_train_qlora.py
    python scripts/04_train_qlora.py --config configs/train_qlora.yaml
"""

import json
import yaml
import torch
import typer
from pathlib import Path
from rich.console import Console
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

console = Console()


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def format_chat(example: dict, tokenizer) -> dict:
    """
    Format conversation data using the model's chat template.
    
    Args:
        example: Dictionary with 'conversations' key
        tokenizer: Tokenizer with chat template
        
    Returns:
        Dictionary with 'text' key containing formatted chat
    """
    try:
        text = tokenizer.apply_chat_template(
            example["conversations"],
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to format example: {e}[/yellow]")
        return {"text": ""}


def main(config: str = "configs/train_qlora.yaml"):
    """
    Train Qwen2.5-7B with QLoRA on SFT dataset.
    
    Args:
        config: Path to training configuration YAML file
    """
    console.print("[bold blue]QLoRA Fine-tuning for Qwen2.5-7B[/bold blue]\n")
    
    # Load configuration
    if not Path(config).exists():
        console.print(f"[red]Error: Config file '{config}' not found![/red]")
        raise typer.Exit(1)
    
    cfg = load_config(config)
    console.print(f"[green]✓ Loaded configuration from {config}[/green]")
    
    # Check if dataset exists
    dataset_path = cfg["dataset_path"]
    if not Path(dataset_path).exists():
        console.print(f"[red]Error: Dataset '{dataset_path}' not found![/red]")
        console.print("[dim]Run scripts 01-03 first to generate the training dataset[/dim]")
        raise typer.Exit(1)
    
    console.print(f"[green]✓ Found dataset: {dataset_path}[/green]")
    
    # Load tokenizer
    console.print(f"\n[cyan]Loading tokenizer: {cfg['model_name']}[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True, trust_remote_code=True)
    
    # Set pad token to eos token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    console.print("[green]✓ Tokenizer loaded[/green]")
    
    # Load dataset
    console.print(f"\n[cyan]Loading dataset from {dataset_path}[/cyan]")
    ds = load_dataset("json", data_files=dataset_path, split="train")
    console.print(f"[green]✓ Loaded {len(ds)} training examples[/green]")
    
    # Configure 4-bit quantization
    console.print("\n[cyan]Configuring 4-bit quantization (QLoRA)[/cyan]")

    # Determine compute dtype based on hardware support
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        console.print("[green]Using bfloat16 compute dtype[/green]")
    else:
        compute_dtype = torch.float16
        console.print("[yellow]Using float16 compute dtype (bf16 not supported)[/yellow]")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["quantization"]["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype
    )
    console.print("[green]✓ Quantization config ready[/green]")

    # Load base model with quantization
    console.print(f"\n[cyan]Loading base model: {cfg['model_name']}[/cyan]")
    console.print("[dim]This may take a few minutes...[/dim]")

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=quant_config if torch.cuda.is_available() else None,
        torch_dtype=compute_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    console.print("[green]✓ Model loaded with 4-bit quantization[/green]" if torch.cuda.is_available() else "[yellow]✓ Model loaded (CPU mode - no quantization)[/yellow]")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    console.print("\n[cyan]Configuring LoRA adapters[/cyan]")
    lora_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    console.print("[green]✓ LoRA adapters configured[/green]")
    
    # Format dataset with chat template
    console.print("\n[cyan]Formatting dataset with chat template[/cyan]")
    formatted_ds = ds.map(
        lambda x: format_chat(x, tokenizer),
        remove_columns=ds.column_names
    )
    # Filter out empty examples
    formatted_ds = formatted_ds.filter(lambda x: len(x["text"]) > 0)
    console.print(f"[green]✓ Formatted {len(formatted_ds)} examples[/green]")
    
    # Training arguments
    console.print("\n[cyan]Setting up training arguments[/cyan]")

    # Detect precision support
    use_bf16 = False
    use_fp16 = False

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            use_bf16 = True
            console.print("[green]✓ Using bfloat16 precision (bf16)[/green]")
        else:
            use_fp16 = True
            console.print("[yellow]⚠ bf16 not supported, falling back to float16 (fp16)[/yellow]")
    else:
        console.print("[red]⚠ CUDA not available! Training will be very slow on CPU.[/red]")
        console.print("[yellow]Consider installing PyTorch with CUDA support for GPU acceleration.[/yellow]")

    # Use SFTConfig instead of TrainingArguments for TRL 0.24.0+
    training_args = SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        num_train_epochs=cfg["train"]["num_train_epochs"],
        learning_rate=cfg["train"]["learning_rate"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        lr_scheduler_type=cfg["train"]["lr_scheduler_type"],
        logging_steps=cfg["train"]["logging_steps"],
        save_steps=cfg["train"]["save_steps"],
        max_steps=cfg["train"]["max_steps"],
        bf16=use_bf16,
        fp16=use_fp16,
        optim="paged_adamw_32bit" if torch.cuda.is_available() else "adamw_torch",
        gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        logging_dir=f"{cfg['output_dir']}/logs",
        save_total_limit=3,
        seed=cfg["seed"],
        report_to="none",  # Disable wandb/tensorboard by default
        # SFT-specific parameters (moved from SFTTrainer.__init__ in TRL 0.24.0+)
        dataset_text_field="text",
        max_length=cfg["train"]["max_seq_length"],
        packing=cfg["train"]["packing"],
    )
    console.print("[green]✓ Training arguments configured[/green]")
    
    # Initialize SFT Trainer
    console.print("\n[cyan]Initializing SFT Trainer[/cyan]")
    # In TRL 0.24.0+, dataset_text_field, max_length, and packing are in SFTConfig, not SFTTrainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=formatted_ds,
        args=training_args
    )
    console.print("[green]✓ Trainer initialized[/green]")
    
    # Start training
    console.print("\n[bold green]Starting training...[/bold green]")
    console.print("[dim]This will take several hours depending on your hardware[/dim]\n")
    
    trainer.train()
    
    # Save final model
    console.print("\n[cyan]Saving final model and tokenizer[/cyan]")
    trainer.model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    
    console.print(f"\n[bold green]✓ Training complete![/bold green]")
    console.print(f"[dim]Model saved to: {Path(cfg['output_dir']).absolute()}[/dim]")


if __name__ == "__main__":
    typer.run(main)

