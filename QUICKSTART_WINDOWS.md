# Quick Start Guide for Windows

This guide provides Windows-specific instructions for running the Qwen fine-tuning pipeline.

## Prerequisites

### 1. Install Python 3.10 or 3.11

Download from [python.org](https://www.python.org/downloads/) and ensure "Add Python to PATH" is checked during installation.

### 2. Install CUDA Toolkit

Download CUDA 11.8 or 12.1+ from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).

### 3. Install Git

Download from [git-scm.com](https://git-scm.com/download/win).

### 4. Install Ollama

Download from [ollama.ai](https://ollama.ai/download/windows).

## Setup

### 1. Clone the Repository

```powershell
git clone https://github.com/yourusername/qwen-finetune-pipeline.git
cd qwen-finetune-pipeline
```

### 2. Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Note:** If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** Flash Attention may not work on Windows. The pipeline will work without it, just slower.

## Running the Pipeline

### Option 1: Step-by-Step (Recommended for First Time)

```powershell
# 1. Place your PDFs in data/raw_pdfs/
Copy-Item "C:\path\to\your\pdfs\*.pdf" -Destination "data\raw_pdfs\"

# 2. Convert PDFs to Markdown
python scripts\01_pdf_to_md.py

# 3. Clean Markdown
python scripts\02_clean_md.py

# 4. Create training dataset
python scripts\03_chunk_to_jsonl.py

# 5. Train the model (this takes several hours!)
python scripts\04_train_qlora.py

# 6. Merge LoRA adapters
python scripts\05_merge_lora.py

# 7. Convert to GGUF
powershell scripts\06_convert_to_gguf.ps1

# 8. Create Ollama model
powershell scripts\07_make_ollama_model.ps1
```

### Option 2: Using Make (if you have Make installed)

If you have Make installed (via Chocolatey, WSL, or Git Bash):

```bash
# Run complete pipeline
make all

# Or run individual steps
make md      # Convert PDFs
make clean   # Clean markdown
make chunk   # Create dataset
make train   # Train model
make merge   # Merge adapters
make gguf    # Convert to GGUF
make ollama  # Create Ollama model
```

## Building llama.cpp on Windows

For the GGUF conversion step, you'll need to build llama.cpp:

### Using Visual Studio

1. Install [Visual Studio 2022 Community](https://visualstudio.microsoft.com/downloads/)
2. Install "Desktop development with C++" workload
3. Open PowerShell in the project directory:

```powershell
cd llama.cpp
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Using CMake and MinGW

Alternatively, install CMake and MinGW-w64, then:

```powershell
cd llama.cpp
mkdir build
cd build
cmake .. -G "MinGW Makefiles"
cmake --build .
```

## Troubleshooting

### CUDA Out of Memory

Edit `configs\train_qlora.yaml` and reduce:
- `per_device_train_batch_size: 1`
- `max_seq_length: 2048` (from 4096)

### Docling Installation Issues

If Docling fails to install, try:

```powershell
pip install --upgrade pip setuptools wheel
pip install docling --no-cache-dir
```

### PyTorch CUDA Not Available

Reinstall PyTorch with CUDA support:

```powershell
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Permission Errors

Run PowerShell as Administrator if you encounter permission errors.

## Performance Tips

### Use SSD Storage

Store the project on an SSD for faster data loading during training.

### Monitor GPU Usage

Use Task Manager (Performance tab) or GPU-Z to monitor VRAM usage during training.

### Adjust Batch Size

If training is too slow, you can increase `gradient_accumulation_steps` in the config to simulate larger batches.

## Next Steps

After creating your Ollama model:

```powershell
# Run the model
ollama run qwen2p5-7b-sft

# Test it with a prompt
> Tell me about the content from the training documents
```

## Getting Help

- Check the main [README.md](README.md) for detailed documentation
- Review [Troubleshooting section](README.md#-troubleshooting) in README
- Open an issue on GitHub if you encounter problems

## Useful Windows Commands

```powershell
# Check Python version
python --version

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU info
nvidia-smi

# List installed packages
pip list

# Check disk space
Get-PSDrive C
```

