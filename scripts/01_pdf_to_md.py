#!/usr/bin/env python3
"""
Convert PDFs to Markdown using Docling.

Usage:
    python scripts/01_pdf_to_md.py
    python scripts/01_pdf_to_md.py --src data/raw_pdfs --dst data/md --conf configs/docling.yaml
"""

import pathlib
import json
import yaml
import typer
from rich.console import Console
from rich.progress import track
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

console = Console()


def main(
    src: str = "data/raw_pdfs",
    dst: str = "data/md",
    conf: str = "configs/docling.yaml"
):
    """
    Convert all PDFs in src directory to Markdown files in dst directory.
    
    Args:
        src: Source directory containing PDF files
        dst: Destination directory for Markdown output
        conf: Path to Docling configuration YAML file
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
    
    # Get list of PDF files
    pdf_files = list(src_path.glob("*.pdf"))
    
    if not pdf_files:
        console.print(f"[yellow]Warning: No PDF files found in '{src}'[/yellow]")
        console.print(f"[dim]Please place PDF files in the '{src}' directory[/dim]")
        return
    
    console.print(f"[green]Found {len(pdf_files)} PDF file(s) to process[/green]")
    
    # Initialize Docling converter
    try:
        # Create pipeline options from configuration
        pipeline_options = PdfPipelineOptions()

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
            
            # Write to output file
            output_file = dst_path / (pdf.stem + ".md")
            output_file.write_text(md_content, encoding="utf-8")
            
            console.print(f"[green]✓[/green] {pdf.name} → {output_file.name}")
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


if __name__ == "__main__":
    typer.run(main)

