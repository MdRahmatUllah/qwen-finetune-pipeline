#!/usr/bin/env python3
"""
Chunk cleaned Markdown files into JSONL format for SFT training.

Usage:
    python scripts/03_chunk_to_jsonl.py
    python scripts/03_chunk_to_jsonl.py --src data/cleaned --outdir data/sft
"""

import json
import pathlib
import re
import typer
from hashlib import md5
from rich.console import Console
from rich.progress import track

console = Console()


def sentence_split(text: str) -> list[str]:
    """
    Split text into sentences using basic punctuation rules.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Split on sentence-ending punctuation followed by whitespace or paragraph breaks
    sentences = re.split(r'(?<=[.!?])\s+\n?|\n{2,}', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, target: int = 500, overlap: int = 100, min_chunk: int = 50) -> list[str]:
    """
    Chunk text into overlapping segments based on word count.
    
    This is a simplistic word-based chunker. For production, consider using
    a proper tokenizer (e.g., from transformers) for more accurate token counts.
    
    Args:
        text: Input text to chunk
        target: Target number of words per chunk
        overlap: Number of words to overlap between chunks
        min_chunk: Minimum number of words for a valid chunk
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    i = 0
    
    while i < len(words):
        # Calculate end position
        end = min(i + target, len(words))
        
        # Extract chunk
        chunk = " ".join(words[i:end])
        
        # Only add if chunk meets minimum size
        if len(words[i:end]) >= min_chunk:
            chunks.append(chunk)
        
        # Move to next position with overlap
        if end >= len(words):
            break
        i = max(end - overlap, i + 1)
    
    return chunks


def create_sft_example(chunk: str, source_file: str) -> dict:
    """
    Create an SFT training example in conversation format.
    
    Args:
        chunk: Text chunk
        source_file: Source filename for metadata
        
    Returns:
        Dictionary with conversations and metadata
    """
    # Truncate chunk to reasonable length (4000 chars)
    truncated_chunk = chunk
    
    # Create conversation format for SFT
    example = {
        "conversations": [
            {
                "role": "system",
                "content": "You are a helpful domain expert."
            },
            {
                "role": "user",
                "content": f"Summarize the following content clearly and factually:\n\n{truncated_chunk}"
            },
            {
                "role": "assistant",
                "content": truncated_chunk
            }
        ],
        "meta": {
            "source": source_file,
            "hash": md5(chunk.encode()).hexdigest()
        }
    }
    
    return example


def main(
    src: str = "data/cleaned",
    outdir: str = "data/sft",
    target_words: int = 500,
    overlap_words: int = 100,
    min_words: int = 50
):
    """
    Chunk cleaned Markdown files and create SFT training dataset in JSONL format.
    
    Args:
        src: Source directory containing cleaned Markdown files
        outdir: Output directory for JSONL dataset
        target_words: Target number of words per chunk
        overlap_words: Number of words to overlap between chunks
        min_words: Minimum number of words for a valid chunk
    """
    src_path = pathlib.Path(src)
    out_path = pathlib.Path(outdir)
    
    # Create output directory
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Check if source directory exists
    if not src_path.exists():
        console.print(f"[red]Error: Source directory '{src}' does not exist![/red]")
        raise typer.Exit(1)
    
    # Get list of Markdown files
    md_files = list(src_path.glob("*.md"))
    
    if not md_files:
        console.print(f"[yellow]Warning: No Markdown files found in '{src}'[/yellow]")
        console.print(f"[dim]Run 02_clean_md.py first to generate cleaned Markdown files[/dim]")
        return
    
    console.print(f"[green]Found {len(md_files)} Markdown file(s) to process[/green]")
    console.print(f"[dim]Chunking parameters: target={target_words} words, overlap={overlap_words} words, min={min_words} words[/dim]")
    
    # Output file
    output_file = out_path / "train.jsonl"
    
    total_chunks = 0
    total_files = 0
    
    # Process each Markdown file
    with output_file.open("w", encoding="utf-8") as writer:
        for md_file in track(md_files, description="Chunking files..."):
            try:
                # Read cleaned markdown
                text = md_file.read_text(encoding="utf-8")
                
                # Chunk the text
                chunks = chunk_text(
                    text,
                    target=target_words,
                    overlap=overlap_words,
                    min_chunk=min_words
                )
                
                # Create SFT examples and write to JSONL
                file_chunks = 0
                for chunk in chunks:
                    # Skip very small chunks
                    if len(chunk.split()) < min_words:
                        continue
                    
                    # Create SFT example
                    example = create_sft_example(chunk, md_file.name)
                    
                    # Write as JSONL
                    writer.write(json.dumps(example, ensure_ascii=False) + "\n")
                    file_chunks += 1
                    total_chunks += 1
                
                console.print(f"[green]✓[/green] {md_file.name} → {file_chunks} chunks")
                total_files += 1
                
            except Exception as e:
                console.print(f"[red]✗[/red] {md_file.name}: {str(e)}")
    
    # Summary
    console.print("\n[bold]Chunking Summary:[/bold]")
    console.print(f"  [green]Files processed: {total_files}[/green]")
    console.print(f"  [green]Total chunks created: {total_chunks}[/green]")
    console.print(f"  [dim]Output file: {output_file.absolute()}[/dim]")
    
    if total_chunks > 0:
        console.print(f"\n[green]✓ Dataset ready for training![/green]")
    else:
        console.print(f"\n[yellow]⚠ No chunks were created. Check your input files.[/yellow]")


if __name__ == "__main__":
    typer.run(main)

