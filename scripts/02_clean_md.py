#!/usr/bin/env python3
"""
Clean Markdown files by removing boilerplate, normalizing whitespace, and fixing formatting.

Usage:
    python scripts/02_clean_md.py
    python scripts/02_clean_md.py --src data/md --dst data/cleaned
"""

import re
import pathlib
import typer
from rich.console import Console
from rich.progress import track

console = Console()


def clean(text: str) -> str:
    """
    Clean markdown text by applying various normalization rules.
    
    Args:
        text: Raw markdown text
        
    Returns:
        Cleaned markdown text
    """
    t = text
    
    # Collapse excess blank lines (3+ newlines → 2 newlines)
    t = re.sub(r'\n{3,}', '\n\n', t)
    
    # Remove page numbers (e.g., "Page 1", "Page 42")
    t = re.sub(r'(?m)^\s*Page\s+\d+\s*$', '', t)
    
    # Remove stray horizontal rules (lines with only dashes)
    t = re.sub(r'(?m)^\s*-{3,}\s*$', '', t)
    
    # De-hyphenate line wraps (word- \n word → wordword)
    t = re.sub(r'(\w)-\n(\w)', r'\1\2', t)
    
    # Convert setext-style headings to ATX-style
    # Example: "Heading\n===" → "# Heading"
    t = re.sub(r'(?m)^([A-Za-z].+)\s*\n={3,}$', r'# \1', t)
    t = re.sub(r'(?m)^([A-Za-z].+)\s*\n-{3,}$', r'## \1', t)
    
    # Remove common footer patterns
    t = re.sub(r'(?m)^\s*©.*$', '', t)  # Copyright notices
    t = re.sub(r'(?m)^\s*Copyright.*$', '', t, flags=re.IGNORECASE)
    
    # Normalize multiple spaces to single space (but preserve indentation)
    t = re.sub(r'(?m)([^\n\s])\s{2,}([^\n])', r'\1 \2', t)
    
    # Remove trailing whitespace from lines
    t = re.sub(r'(?m)[ \t]+$', '', t)
    
    # Ensure code fences are properly formatted
    t = re.sub(r'(?m)^```\s*(\w+)\s*$', r'```\1', t)
    
    # Remove excessive whitespace at start/end
    t = t.strip()
    
    # Ensure file ends with single newline
    if t and not t.endswith('\n'):
        t += '\n'
    
    return t


def main(
    src: str = "data/md",
    dst: str = "data/cleaned"
):
    """
    Clean all Markdown files in src directory and save to dst directory.
    
    Args:
        src: Source directory containing Markdown files
        dst: Destination directory for cleaned Markdown output
    """
    src_path = pathlib.Path(src)
    dst_path = pathlib.Path(dst)
    
    # Create destination directory
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # Check if source directory exists
    if not src_path.exists():
        console.print(f"[red]Error: Source directory '{src}' does not exist![/red]")
        raise typer.Exit(1)
    
    # Get list of Markdown files
    md_files = list(src_path.glob("*.md"))
    
    if not md_files:
        console.print(f"[yellow]Warning: No Markdown files found in '{src}'[/yellow]")
        console.print(f"[dim]Run 01_pdf_to_md.py first to generate Markdown files[/dim]")
        return
    
    console.print(f"[green]Found {len(md_files)} Markdown file(s) to clean[/green]")
    
    # Process each Markdown file
    success_count = 0
    error_count = 0
    
    for md_file in track(md_files, description="Cleaning Markdown files..."):
        try:
            # Read raw markdown
            raw_content = md_file.read_text(encoding="utf-8")
            
            # Clean the content
            cleaned_content = clean(raw_content)
            
            # Write to output file
            output_file = dst_path / md_file.name
            output_file.write_text(cleaned_content, encoding="utf-8")
            
            # Calculate size reduction
            size_before = len(raw_content)
            size_after = len(cleaned_content)
            reduction = ((size_before - size_after) / size_before * 100) if size_before > 0 else 0
            
            console.print(
                f"[green]✓[/green] {md_file.name} "
                f"[dim]({size_before:,} → {size_after:,} chars, {reduction:.1f}% reduction)[/dim]"
            )
            success_count += 1
            
        except Exception as e:
            console.print(f"[red]✗[/red] {md_file.name}: {str(e)}")
            error_count += 1
    
    # Summary
    console.print("\n[bold]Cleaning Summary:[/bold]")
    console.print(f"  [green]Success: {success_count}[/green]")
    if error_count > 0:
        console.print(f"  [red]Errors: {error_count}[/red]")
    console.print(f"  [dim]Output directory: {dst_path.absolute()}[/dim]")


if __name__ == "__main__":
    typer.run(main)

