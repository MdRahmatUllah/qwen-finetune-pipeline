# Makefile for Qwen Fine-tuning Pipeline
# Automates the complete workflow from PDFs to Ollama model

.PHONY: help pdf md clean chunk sft train merge gguf ollama all check-pdfs setup

# Default target - show help
help:
	@echo "Qwen Fine-tuning Pipeline - Available targets:"
	@echo ""
	@echo "  make setup      - Set up Python environment and install dependencies"
	@echo "  make pdf        - Check for PDFs in data/raw_pdfs/"
	@echo "  make md         - Convert PDFs to Markdown (step 1)"
	@echo "  make clean      - Clean Markdown files (step 2)"
	@echo "  make chunk      - Chunk Markdown to JSONL (step 3)"
	@echo "  make sft        - Run all data preparation steps (md + clean + chunk)"
	@echo "  make train      - Train QLoRA model (step 4)"
	@echo "  make merge      - Merge LoRA adapters into base model (step 5)"
	@echo "  make gguf       - Convert to GGUF format (step 6)"
	@echo "  make ollama     - Create Ollama model (step 7)"
	@echo "  make all        - Run complete pipeline (sft + train + merge + gguf + ollama)"
	@echo ""
	@echo "Quick start:"
	@echo "  1. Place PDFs in data/raw_pdfs/"
	@echo "  2. Run: make all"
	@echo ""

# Check if PDFs are present
check-pdfs:
	@if [ -z "$$(ls -A data/raw_pdfs/*.pdf 2>/dev/null)" ]; then \
		echo "Error: No PDF files found in data/raw_pdfs/"; \
		echo "Please add PDF files before running the pipeline"; \
		exit 1; \
	fi
	@echo "✓ Found PDF files in data/raw_pdfs/"

# Setup Python environment
setup:
	@echo "Setting up Python environment..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"
	@echo ""
	@echo "Optional: Configure accelerate for multi-GPU training"
	@echo "Run: accelerate config"

# Step 1: Convert PDFs to Markdown
md: check-pdfs
	@echo "=== Step 1: Converting PDFs to Markdown ==="
	python scripts/01_pdf_to_md.py
	@echo ""

# Step 2: Clean Markdown files
clean:
	@echo "=== Step 2: Cleaning Markdown files ==="
	python scripts/02_clean_md.py
	@echo ""

# Step 3: Chunk Markdown to JSONL
chunk:
	@echo "=== Step 3: Chunking to JSONL ==="
	python scripts/03_chunk_to_jsonl.py
	@echo ""

# Combined data preparation (steps 1-3)
sft: md clean chunk
	@echo "=== Data preparation complete ==="
	@echo "✓ SFT dataset ready in data/sft/train.jsonl"
	@echo ""

# Step 4: Train QLoRA model
train:
	@echo "=== Step 4: Training QLoRA model ==="
	@echo "This will take several hours depending on your hardware"
	python scripts/04_train_qlora.py
	@echo ""

# Step 5: Merge LoRA adapters
merge:
	@echo "=== Step 5: Merging LoRA adapters ==="
	python scripts/05_merge_lora.py
	@echo ""

# Step 6: Convert to GGUF
gguf:
	@echo "=== Step 6: Converting to GGUF ==="
	bash scripts/06_convert_to_gguf.sh
	@echo ""

# Step 7: Create Ollama model
ollama:
	@echo "=== Step 7: Creating Ollama model ==="
	bash scripts/07_make_ollama_model.sh
	@echo ""

# Run complete pipeline
all: sft train merge gguf ollama
	@echo "=========================================="
	@echo "=== Pipeline Complete! ==="
	@echo "=========================================="
	@echo ""
	@echo "Your custom Qwen model is ready!"
	@echo "Run: ollama run qwen2p5-7b-sft"
	@echo ""

# Alias for pdf target
pdf: check-pdfs
	@echo "PDFs are ready in data/raw_pdfs/"
	@echo "Run 'make md' to convert them to Markdown"

