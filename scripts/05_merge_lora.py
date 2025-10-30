#!/usr/bin/env python3
"""
Merge LoRA adapters back into the base model to create a full HuggingFace model.

The script automatically detects and uses the latest checkpoint if the adapter path
contains checkpoint subdirectories (e.g., checkpoint-12, checkpoint-24).

Usage:
    python scripts/05_merge_lora.py
    python scripts/05_merge_lora.py --base Qwen/Qwen2.5-7B --adapter models/lora --output models/merged
    python scripts/05_merge_lora.py --adapter models/lora/checkpoint-12  # Use specific checkpoint
"""

import typer
import torch
from pathlib import Path
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

console = Console()


def main(
    base: str = "Qwen/Qwen2.5-7B",
    adapter: str = "models/lora",
    output: str = "models/merged"
):
    """
    Merge LoRA adapters into base model and save as full model.

    Args:
        base: Base model name or path (HuggingFace model ID)
        adapter: Path to LoRA adapter directory or checkpoint subdirectory.
                 If the path contains checkpoint-* subdirectories, the latest one will be used.
        output: Output directory for merged model
    """
    console.print("[bold blue]Merging LoRA Adapters into Base Model[/bold blue]\n")

    adapter_path = Path(adapter)
    output_path = Path(output)

    # Check if adapter directory exists
    if not adapter_path.exists():
        console.print(f"[red]Error: Adapter directory '{adapter}' not found![/red]")
        console.print("[dim]Run 04_train_qlora.py first to train the model[/dim]")
        raise typer.Exit(1)

    # Check for adapter files - handle checkpoint subdirectories
    adapter_config = adapter_path / "adapter_config.json"

    if not adapter_config.exists():
        # Look for checkpoint subdirectories
        console.print(f"[yellow]No adapter_config.json in '{adapter}', checking for checkpoints...[/yellow]")

        checkpoints = sorted(
            [d for d in adapter_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else 0,
            reverse=True  # Latest checkpoint first
        )

        if not checkpoints:
            console.print(f"[red]Error: No adapter_config.json or checkpoint directories found in '{adapter}'![/red]")
            console.print("[dim]The adapter directory may be incomplete or corrupted[/dim]")
            console.print("[dim]Expected either:[/dim]")
            console.print(f"[dim]  - {adapter}/adapter_config.json[/dim]")
            console.print(f"[dim]  - {adapter}/checkpoint-*/adapter_config.json[/dim]")
            raise typer.Exit(1)

        # Use the latest checkpoint
        adapter_path = checkpoints[0]
        adapter_config = adapter_path / "adapter_config.json"

        if not adapter_config.exists():
            console.print(f"[red]Error: No adapter_config.json found in checkpoint '{adapter_path.name}'![/red]")
            raise typer.Exit(1)

        console.print(f"[green]✓ Found checkpoint: {adapter_path.name}[/green]")

    console.print(f"[green]✓ Found LoRA adapters in {adapter_path}[/green]")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    console.print(f"\n[cyan]Loading tokenizer from {base}[/cyan]")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True, trust_remote_code=True)
        console.print("[green]✓ Tokenizer loaded[/green]")
    except Exception as e:
        console.print(f"[red]Error loading tokenizer: {e}[/red]")
        raise typer.Exit(1)
    
    # Load base model
    console.print(f"\n[cyan]Loading base model: {base}[/cyan]")
    console.print("[dim]This may take a few minutes...[/dim]")
    
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=torch.float16,  # Use fp16 for efficiency
            device_map="cpu",  # Load to CPU to avoid VRAM issues during merge
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        console.print("[green]✓ Base model loaded[/green]")
    except Exception as e:
        console.print(f"[red]Error loading base model: {e}[/red]")
        raise typer.Exit(1)
    
    # Load PEFT model (base + adapters)
    console.print(f"\n[cyan]Loading LoRA adapters from {adapter}[/cyan]")
    try:
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        console.print("[green]✓ LoRA adapters loaded[/green]")
    except Exception as e:
        console.print(f"[red]Error loading PEFT adapters: {e}[/red]")
        raise typer.Exit(1)
    
    # Merge adapters into base model
    console.print("\n[cyan]Merging LoRA weights into base model[/cyan]")
    console.print("[dim]This operation may take several minutes...[/dim]")
    
    try:
        merged_model = peft_model.merge_and_unload()
        console.print("[green]✓ LoRA weights merged successfully[/green]")
    except Exception as e:
        console.print(f"[red]Error during merge: {e}[/red]")
        raise typer.Exit(1)
    
    # Save merged model
    console.print(f"\n[cyan]Saving merged model to {output}[/cyan]")
    console.print("[dim]Saving in safetensors format...[/dim]")
    
    try:
        merged_model.save_pretrained(
            str(output_path),
            safe_serialization=True,  # Use safetensors format
            max_shard_size="5GB"  # Shard large models
        )
        console.print("[green]✓ Model saved[/green]")
    except Exception as e:
        console.print(f"[red]Error saving model: {e}[/red]")
        raise typer.Exit(1)
    
    # Save tokenizer
    console.print(f"\n[cyan]Saving tokenizer to {output}[/cyan]")
    try:
        tokenizer.save_pretrained(str(output_path))
        console.print("[green]✓ Tokenizer saved[/green]")
    except Exception as e:
        console.print(f"[red]Error saving tokenizer: {e}[/red]")
        raise typer.Exit(1)
    
    # Summary
    console.print("\n[bold green]✓ Merge complete![/bold green]")
    console.print(f"[dim]Merged model saved to: {output_path.absolute()}[/dim]")
    console.print("\n[cyan]Next steps:[/cyan]")
    console.print("  1. Run 06_convert_to_gguf.sh to convert to GGUF format")
    console.print("  2. Run 07_make_ollama_model.sh to create Ollama model")


if __name__ == "__main__":
    typer.run(main)

