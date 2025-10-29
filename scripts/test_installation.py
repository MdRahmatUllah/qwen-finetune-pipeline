#!/usr/bin/env python3
"""
Test script to verify installation and environment setup.

Usage:
    python scripts/test_installation.py
"""

import sys
from rich.console import Console
from rich.table import Table

console = Console()


def check_python_version():
    """Check if Python version is 3.10 or 3.11."""
    version = sys.version_info
    if version.major == 3 and version.minor in [10, 11]:
        return True, f"{version.major}.{version.minor}.{version.micro}"
    return False, f"{version.major}.{version.minor}.{version.micro}"


def check_package(package_name, import_name=None):
    """Check if a package is installed and importable."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True, "✓"
    except ImportError:
        return False, "✗"


def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            return True, f"✓ ({device_count} GPU(s), {device_name})"
        else:
            return False, "✗ (CUDA not available)"
    except ImportError:
        return False, "✗ (PyTorch not installed)"


def main():
    """Run all installation checks."""
    console.print("\n[bold blue]Qwen Fine-tuning Pipeline - Installation Check[/bold blue]\n")
    
    # Check Python version
    py_ok, py_version = check_python_version()
    if py_ok:
        console.print(f"[green]✓[/green] Python version: {py_version}")
    else:
        console.print(f"[red]✗[/red] Python version: {py_version} (requires 3.10 or 3.11)")
    
    console.print()
    
    # Create table for package checks
    table = Table(title="Package Installation Status")
    table.add_column("Package", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Import Name", style="dim")
    
    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("trl", "trl"),
        ("peft", "peft"),
        ("accelerate", "accelerate"),
        ("bitsandbytes", "bitsandbytes"),
        ("datasets", "datasets"),
        ("docling", "docling"),
        ("docling-core", "docling_core"),
        ("pydantic", "pydantic"),
        ("typer", "typer"),
        ("rich", "rich"),
        ("pyyaml", "yaml"),
    ]
    
    all_ok = True
    for pkg_name, import_name in packages:
        ok, status = check_package(pkg_name, import_name)
        if not ok:
            all_ok = False
        
        status_color = "green" if ok else "red"
        table.add_row(pkg_name, f"[{status_color}]{status}[/{status_color}]", import_name)
    
    console.print(table)
    console.print()
    
    # Check CUDA
    cuda_ok, cuda_status = check_cuda()
    if cuda_ok:
        console.print(f"[green]{cuda_status}[/green]")
    else:
        console.print(f"[yellow]{cuda_status}[/yellow]")
        console.print("[dim]Note: CUDA is required for training but not for data preparation[/dim]")
    
    console.print()
    
    # Check directory structure
    console.print("[bold]Directory Structure:[/bold]")
    from pathlib import Path
    
    required_dirs = [
        "data/raw_pdfs",
        "data/md",
        "data/cleaned",
        "data/sft",
        "models/lora",
        "models/merged",
        "models/gguf",
        "configs",
        "scripts",
        "ollama",
    ]
    
    dirs_ok = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            console.print(f"  [green]✓[/green] {dir_path}")
        else:
            console.print(f"  [red]✗[/red] {dir_path}")
            dirs_ok = False
    
    console.print()
    
    # Summary
    if py_ok and all_ok and dirs_ok:
        console.print("[bold green]✓ All checks passed! You're ready to go.[/bold green]")
        console.print("\n[cyan]Next steps:[/cyan]")
        console.print("  1. Place PDFs in data/raw_pdfs/")
        console.print("  2. Run: python scripts/01_pdf_to_md.py")
        console.print("  Or run the complete pipeline: make all")
    else:
        console.print("[bold yellow]⚠ Some checks failed. Please install missing dependencies.[/bold yellow]")
        console.print("\n[cyan]To install dependencies:[/cyan]")
        console.print("  pip install -r requirements.txt")
    
    console.print()


if __name__ == "__main__":
    main()

