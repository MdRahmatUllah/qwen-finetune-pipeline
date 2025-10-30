#!/usr/bin/env python3
"""
Convert PDFs to Markdown using Docling.

Usage:
    python scripts/01_pdf_to_md.py
    python scripts/01_pdf_to_md.py --src data/raw_pdfs --dst data/md --conf configs/docling.yaml
    python scripts/01_pdf_to_md.py --src C:\MyDocuments\Research --dst data/md --preserve-structure
"""

import pathlib
import json
import yaml
import typer
import torch
from rich.console import Console
from rich.progress import track
from rich.table import Table
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

console = Console()


def main(
    src: str = "data/raw_pdfs",
    dst: str = "data/md",
    conf: str = "configs/docling.yaml",
    preserve_structure: bool = False,
    recursive: bool = True
):
    """
    Convert all PDFs in src directory (recursively) to Markdown files in dst directory.

    Args:
        src: Source directory containing PDF files (searches recursively by default)
        dst: Destination directory for Markdown output
        conf: Path to Docling configuration YAML file
        preserve_structure: If True, preserve source subdirectory structure in destination
        recursive: If True, search for PDFs recursively in all subdirectories (default: True)
    """
    src_path = pathlib.Path(src)
    dst_path = pathlib.Path(dst)
    conf_path = pathlib.Path(conf)

    # Create destination directory
    dst_path.mkdir(parents=True, exist_ok=True)

    # Check if source directory exists
    if not src_path.exists():
        console.print(f"[red]Error: Source directory '{src}' does not exist![/red]")
        raise typer.Exit(1)

    # Get list of PDF files (recursively or non-recursively)
    if recursive:
        pdf_files = list(src_path.rglob("*.pdf"))
        search_mode = "recursively"
    else:
        pdf_files = list(src_path.glob("*.pdf"))
        search_mode = "in top-level directory"

    if not pdf_files:
        console.print(f"[yellow]Warning: No PDF files found {search_mode} in '{src}'[/yellow]")
        console.print(f"[dim]Please place PDF files in the '{src}' directory[/dim]")
        if recursive:
            console.print(f"[dim]Tip: Use --no-recursive to search only the top-level directory[/dim]")
        return

    console.print(f"[green]Found {len(pdf_files)} PDF file(s) {search_mode}[/green]")

    # Display directory structure info if preserving structure
    if preserve_structure and recursive:
        unique_dirs = set(pdf.parent.relative_to(src_path) for pdf in pdf_files)
        console.print(f"[dim]Subdirectories found: {len(unique_dirs)}[/dim]")
        if len(unique_dirs) <= 10:
            for dir_path in sorted(unique_dirs):
                console.print(f"[dim]  - {dir_path}[/dim]")

    # Initialize Docling converter
    try:
        # Detect and configure GPU acceleration
        device = AcceleratorDevice.CPU
        device_name = "CPU"

        # Check for CUDA GPU availability
        if torch.cuda.is_available():
            device = AcceleratorDevice.CUDA
            device_name = f"CUDA (GPU: {torch.cuda.get_device_name(0)})"
            console.print(f"[green]✓ GPU acceleration enabled: {device_name}[/green]")
        else:
            console.print(f"[yellow]⚠ GPU not available, using CPU[/yellow]")
            console.print(f"[dim]  To enable GPU: Install PyTorch with CUDA support[/dim]")

        # Configure accelerator options
        # Use more threads for CPU, fewer for GPU (GPU handles parallelism internally)
        num_threads = 4 if device == AcceleratorDevice.CUDA else 8
        accelerator_options = AcceleratorOptions(
            num_threads=num_threads,
            device=device
        )

        # Create pipeline options from configuration
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options

        # Load and apply configuration if available
        if conf_path.exists():
            with open(conf_path, 'r') as f:
                config = yaml.safe_load(f)

            # Apply PDF configuration settings
            pdf_config = config.get('pdf', {})
            if 'do_ocr' in pdf_config:
                pipeline_options.do_ocr = pdf_config['do_ocr']
            if 'do_table_structure' in pdf_config:
                pipeline_options.do_table_structure = pdf_config['do_table_structure']

            # Apply table structure options
            table_config = pdf_config.get('table_structure', {})
            if 'do_cell_matching' in table_config:
                pipeline_options.table_structure_options.do_cell_matching = table_config['do_cell_matching']

            console.print(f"[dim]Using configuration from {conf}[/dim]")
        else:
            console.print(f"[yellow]Config file not found, using default settings[/yellow]")

        # Create converter with PDF format options
        conv = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        console.print(f"[dim]Accelerator: {device_name} ({num_threads} threads)[/dim]")
    except Exception as e:
        console.print(f"[red]Error initializing Docling: {e}[/red]")
        raise typer.Exit(1)
    
    # Process each PDF
    success_count = 0
    error_count = 0

    for pdf in track(pdf_files, description="Converting PDFs..."):
        try:
            # Convert PDF to document
            result = conv.convert(str(pdf))

            # Export to Markdown
            md_content = result.document.export_to_markdown()

            # Determine output file path
            if preserve_structure and recursive:
                # Preserve directory structure
                relative_path = pdf.relative_to(src_path)
                output_file = dst_path / relative_path.parent / (pdf.stem + ".md")
                # Create subdirectories if needed
                output_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Flatten all files to destination directory
                output_file = dst_path / (pdf.stem + ".md")

            # Write to output file
            output_file.write_text(md_content, encoding="utf-8")

            # Display relative path for better readability
            if preserve_structure and recursive:
                display_path = str(output_file.relative_to(dst_path))
            else:
                display_path = output_file.name

            console.print(f"[green]✓[/green] {pdf.name} → {display_path}")
            success_count += 1

        except Exception as e:
            console.print(f"[red]✗[/red] {pdf.name}: {str(e)}")
            error_count += 1

    # Summary
    console.print("\n[bold]Conversion Summary:[/bold]")
    console.print(f"  [green]Success: {success_count}[/green]")
    if error_count > 0:
        console.print(f"  [red]Errors: {error_count}[/red]")
    console.print(f"  [dim]Output directory: {dst_path.absolute()}[/dim]")
    if preserve_structure:
        console.print(f"  [dim]Directory structure: preserved[/dim]")
    else:
        console.print(f"  [dim]Directory structure: flattened[/dim]")


if __name__ == "__main__":
    typer.run(main)

