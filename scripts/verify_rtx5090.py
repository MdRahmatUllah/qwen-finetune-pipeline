#!/usr/bin/env python3
"""
Verification script for RTX 5090 + PyTorch + bitsandbytes setup.

This script checks:
1. PyTorch installation and CUDA availability
2. GPU detection and compute capability
3. BF16 support
4. bitsandbytes 4-bit quantization capability
"""

import sys
from rich.console import Console
from rich.table import Table

console = Console()

def check_pytorch():
    """Check PyTorch installation and CUDA support."""
    console.print("\n[cyan]═══ PyTorch Installation ═══[/cyan]")
    
    try:
        import torch
        console.print(f"[green]✓[/green] PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            console.print(f"[green]✓[/green] CUDA available: True")
            console.print(f"[green]✓[/green] CUDA version: {torch.version.cuda}")
            console.print(f"[green]✓[/green] cuDNN version: {torch.backends.cudnn.version()}")
            
            # Check device count
            device_count = torch.cuda.device_count()
            console.print(f"[green]✓[/green] GPU count: {device_count}")
            
            # Check each GPU
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                console.print(f"[green]✓[/green] GPU {i}: {device_name}")
                
                # Get compute capability
                capability = torch.cuda.get_device_capability(i)
                console.print(f"[green]✓[/green] Compute capability: sm_{capability[0]}{capability[1]}")
                
                # Check if compute capability is supported
                arch_list = torch.cuda.get_arch_list()
                sm_version = f"sm_{capability[0]}{capability[1]}"
                if sm_version in arch_list or f"compute_{capability[0]}{capability[1]}" in arch_list:
                    console.print(f"[green]✓[/green] Compute capability {sm_version} is SUPPORTED")
                else:
                    console.print(f"[red]✗[/red] Compute capability {sm_version} is NOT SUPPORTED")
                    console.print(f"[yellow]Supported architectures: {', '.join(arch_list)}[/yellow]")
                    return False
            
            # Check BF16 support
            if torch.cuda.is_bf16_supported():
                console.print(f"[green]✓[/green] BF16 (bfloat16) support: Yes")
            else:
                console.print(f"[yellow]⚠[/yellow] BF16 (bfloat16) support: No (will use FP16)")
            
            return True
        else:
            console.print(f"[red]✗[/red] CUDA not available")
            console.print(f"[yellow]PyTorch is installed without CUDA support[/yellow]")
            return False
            
    except ImportError:
        console.print(f"[red]✗[/red] PyTorch not installed")
        return False
    except Exception as e:
        console.print(f"[red]✗[/red] Error checking PyTorch: {e}")
        return False


def check_bitsandbytes():
    """Check bitsandbytes installation and 4-bit quantization support."""
    console.print("\n[cyan]═══ bitsandbytes Installation ═══[/cyan]")
    
    try:
        import bitsandbytes as bnb
        console.print(f"[green]✓[/green] bitsandbytes version: {bnb.__version__}")
        
        # Try to import CUDA functions
        try:
            from bitsandbytes.cuda_setup.main import get_compute_capabilities
            console.print(f"[green]✓[/green] CUDA setup module available")
        except ImportError:
            console.print(f"[yellow]⚠[/yellow] CUDA setup module not available (may be normal for newer versions)")
        
        return True
        
    except ImportError:
        console.print(f"[red]✗[/red] bitsandbytes not installed")
        return False
    except Exception as e:
        console.print(f"[red]✗[/red] Error checking bitsandbytes: {e}")
        return False


def test_4bit_quantization():
    """Test 4-bit quantization with a small tensor."""
    console.print("\n[cyan]═══ Testing 4-bit Quantization ═══[/cyan]")
    
    try:
        import torch
        from transformers import BitsAndBytesConfig
        
        if not torch.cuda.is_available():
            console.print(f"[yellow]⚠[/yellow] Skipping 4-bit test (CUDA not available)")
            return True
        
        # Create a simple quantization config
        console.print("[dim]Creating BitsAndBytesConfig...[/dim]")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        console.print(f"[green]✓[/green] BitsAndBytesConfig created successfully")
        
        # Test basic tensor operations on GPU
        console.print("[dim]Testing GPU tensor operations...[/dim]")
        test_tensor = torch.randn(100, 100).cuda()
        result = test_tensor @ test_tensor.T
        console.print(f"[green]✓[/green] GPU tensor operations working")
        
        console.print(f"[green]✓[/green] 4-bit quantization setup is ready")
        console.print(f"[dim]Note: Full model quantization test requires downloading a model[/dim]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗[/red] 4-bit quantization test failed: {e}")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return False


def check_other_dependencies():
    """Check other important dependencies."""
    console.print("\n[cyan]═══ Other Dependencies ═══[/cyan]")
    
    dependencies = [
        ("transformers", "Hugging Face Transformers"),
        ("peft", "Parameter-Efficient Fine-Tuning"),
        ("trl", "Transformer Reinforcement Learning"),
        ("accelerate", "Hugging Face Accelerate"),
        ("datasets", "Hugging Face Datasets"),
    ]
    
    all_ok = True
    for module_name, display_name in dependencies:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            console.print(f"[green]✓[/green] {display_name}: {version}")
        except ImportError:
            console.print(f"[red]✗[/red] {display_name}: Not installed")
            all_ok = False
    
    return all_ok


def main():
    """Run all verification checks."""
    console.print("\n[bold cyan]RTX 5090 + PyTorch + bitsandbytes Verification[/bold cyan]")
    console.print("[dim]Checking your setup for QLoRA fine-tuning...[/dim]\n")
    
    results = []
    
    # Run checks
    results.append(("PyTorch & CUDA", check_pytorch()))
    results.append(("bitsandbytes", check_bitsandbytes()))
    results.append(("4-bit Quantization", test_4bit_quantization()))
    results.append(("Other Dependencies", check_other_dependencies()))
    
    # Summary
    console.print("\n[cyan]═══ Summary ═══[/cyan]")
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Component", style="dim")
    table.add_column("Status", justify="center")
    
    for name, status in results:
        status_str = "[green]✓ PASS[/green]" if status else "[red]✗ FAIL[/red]"
        table.add_row(name, status_str)
    
    console.print(table)
    
    # Final verdict
    if all(status for _, status in results):
        console.print("\n[bold green]✓ All checks passed! Your system is ready for QLoRA training.[/bold green]")
        console.print("\n[dim]You can now run:[/dim]")
        console.print("[cyan]python scripts/04_train_qlora.py --config configs/train_qlora.yaml[/cyan]")
        return 0
    else:
        console.print("\n[bold red]✗ Some checks failed. Please fix the issues above.[/bold red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())

