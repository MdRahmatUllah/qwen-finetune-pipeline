# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-29

### Added

#### Core Pipeline
- Complete 7-step pipeline from PDF to Ollama model
- PDF to Markdown conversion using Docling
- Markdown cleaning and normalization
- Text chunking with overlap for SFT dataset creation
- QLoRA fine-tuning with 4-bit quantization
- LoRA adapter merging
- GGUF conversion using llama.cpp
- Ollama model creation and deployment

#### Scripts
- `01_pdf_to_md.py` - PDF to Markdown conversion
- `02_clean_md.py` - Markdown cleaning
- `03_chunk_to_jsonl.py` - Dataset creation
- `04_train_qlora.py` - QLoRA training
- `05_merge_lora.py` - LoRA merging
- `06_convert_to_gguf.sh` - GGUF conversion (Bash)
- `06_convert_to_gguf.ps1` - GGUF conversion (PowerShell)
- `07_make_ollama_model.sh` - Ollama deployment (Bash)
- `07_make_ollama_model.ps1` - Ollama deployment (PowerShell)
- `test_installation.py` - Installation verification

#### Configuration
- `configs/docling.yaml` - Docling PDF processing config
- `configs/chunking.yaml` - Text chunking parameters
- `configs/train_qlora.yaml` - Training hyperparameters
- `ollama/Modelfile` - Ollama model definition

#### Documentation
- Comprehensive README.md with full documentation
- QUICKSTART_WINDOWS.md for Windows users
- LICENSE (MIT)
- .env.example for environment configuration
- IMPLEMENTATION_SUMMARY.md

#### Project Structure
- Modern Python packaging with pyproject.toml
- requirements.txt with all dependencies
- Makefile for workflow automation
- .gitignore for proper version control
- Complete directory structure with .gitkeep files

#### Developer Tools
- Black formatter configuration
- Ruff linter configuration
- MyPy type checker configuration
- Pytest configuration

#### Features
- Cross-platform support (Linux, Mac, Windows)
- Rich console output with progress tracking
- Comprehensive error handling
- Memory-efficient training (32GB VRAM)
- Flexible configuration via YAML
- Command-line arguments for all scripts

### Technical Details

#### Dependencies
- PyTorch 2.1.0+ with CUDA support
- Transformers 4.36.0+
- TRL 0.7.0+ for SFT training
- PEFT 0.7.0+ for LoRA
- Accelerate 0.25.0+ for distributed training
- Bitsandbytes 0.41.0+ for quantization
- Docling 1.0.0+ for PDF processing
- Rich, Typer for CLI

#### Model Support
- Base model: Qwen/Qwen2.5-7B
- Context length: Up to 128k (training at 4k-8k recommended)
- Quantization: 4-bit NF4 with double quantization
- LoRA: r=16, alpha=32, dropout=0.05

#### Output Formats
- HuggingFace safetensors (merged model)
- GGUF fp16 (full precision)
- GGUF Q4_K_M (quantized, recommended)

### Known Limitations

- Flash Attention may not work on Windows
- Training requires NVIDIA GPU with 32GB+ VRAM
- GGUF conversion requires building llama.cpp
- Large models require significant disk space (50GB+)

### Future Enhancements

Potential improvements for future versions:
- Support for other base models (Llama, Mistral, etc.)
- Multi-GPU training support
- Evaluation metrics and validation
- Hyperparameter tuning scripts
- Docker containerization
- Web UI for easier use
- Continued pre-training (CPT) option
- Integration with other deployment platforms

---

## Version History

- **0.1.0** (2025-10-29) - Initial release with complete pipeline

[0.1.0]: https://github.com/yourusername/qwen-finetune-pipeline/releases/tag/v0.1.0

