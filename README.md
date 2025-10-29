# Qwen Fine-tuning Pipeline

Complete end-to-end pipeline for fine-tuning **Qwen2.5-7B** on custom PDF documents and deploying to **Ollama** for local inference.

## 🚀 Features

- **PDF → Markdown**: High-quality extraction using [Docling](https://docling-project.github.io/docling/) with table/layout detection
- **QLoRA Training**: Memory-efficient 4-bit fine-tuning with LoRA adapters (works on 32GB VRAM)
- **GGUF Conversion**: Convert to GGUF format for efficient local inference
- **Ollama Integration**: Deploy as a custom Ollama model for easy local use
- **Automated Pipeline**: Makefile and scripts for complete workflow automation

## 📋 Requirements

- **Python**: 3.10 or 3.11
- **GPU**: NVIDIA GPU with 32GB+ VRAM (for training)
- **CUDA**: CUDA 11.8+ or 12.1+
- **Ollama**: For final model deployment ([install here](https://ollama.ai))

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/MdRahmatUllah/qwen-finetune-pipeline.git
cd qwen-finetune-pipeline
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Accelerate (optional, for multi-GPU)

```bash
accelerate config
```

### 5. Verify Installation

```bash
python scripts/test_installation.py
```

This will check that all dependencies are installed correctly and CUDA is available.

## 📁 Project Structure

```
qwen-finetune-pipeline/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
├── Makefile
├── data/
│   ├── raw_pdfs/            # Put your PDFs here
│   ├── md/                  # Docling → Markdown output
│   ├── cleaned/             # Cleaned markdown
│   ├── chunks/              # Chunked JSONL
│   └── sft/                 # Final SFT dataset
├── scripts/
│   ├── 01_pdf_to_md.py
│   ├── 02_clean_md.py
│   ├── 03_chunk_to_jsonl.py
│   ├── 04_train_qlora.py
│   ├── 05_merge_lora.py
│   ├── 06_convert_to_gguf.sh
│   └── 07_make_ollama_model.sh
├── configs/
│   ├── docling.yaml
│   ├── chunking.yaml
│   └── train_qlora.yaml
├── training/
│   └── run_logs/
├── models/
│   ├── base/                # HF cache/symlink for base model
│   ├── lora/                # PEFT adapter checkpoints
│   ├── merged/              # Merged safetensors
│   └── gguf/                # Converted GGUF files
└── ollama/
    └── Modelfile
```

## 🎯 Quick Start

### Option 1: Run Complete Pipeline

```bash
# 1. Place PDFs in data/raw_pdfs/
cp /path/to/your/*.pdf data/raw_pdfs/

# 2. Run everything
make all
```

### Option 2: Step-by-Step

```bash
# Step 1: Convert PDFs to Markdown
make md

# Step 2: Clean Markdown
make clean

# Step 3: Chunk to JSONL
make chunk

# Step 4: Train QLoRA model (takes several hours)
make train

# Step 5: Merge LoRA adapters
make merge

# Step 6: Convert to GGUF
make gguf

# Step 7: Create Ollama model
make ollama
```

## 📝 Detailed Workflow

### 1. PDF → Markdown (Docling)

Converts PDFs to clean Markdown with advanced layout detection:

```bash
python scripts/01_pdf_to_md.py
```

**Features:**
- Table detection and extraction
- Footnote inference
- Code block extraction
- Heading normalization

### 2. Clean Markdown

Removes boilerplate, normalizes whitespace, fixes formatting:

```bash
python scripts/02_clean_md.py
```

**Cleaning operations:**
- Remove page numbers and footers
- De-hyphenate line wraps
- Normalize whitespace
- Convert setext headings to ATX

### 3. Chunk to JSONL

Creates SFT training dataset in conversation format:

```bash
python scripts/03_chunk_to_jsonl.py
```

**Output format:**
```json
{
  "conversations": [
    {"role": "system", "content": "You are a helpful domain expert."},
    {"role": "user", "content": "Summarize the following content..."},
    {"role": "assistant", "content": "..."}
  ],
  "meta": {"source": "document.md", "hash": "..."}
}
```

### 4. QLoRA Training

Fine-tune Qwen2.5-7B with 4-bit quantization:

```bash
python scripts/04_train_qlora.py
```

**Configuration** (`configs/train_qlora.yaml`):
- Model: `Qwen/Qwen2.5-7B`
- Quantization: 4-bit NF4 with double quantization
- LoRA: r=16, alpha=32
- Batch size: 1 with gradient accumulation
- Max sequence length: 4096 tokens

### 5. Merge LoRA Adapters

Merge LoRA weights back into base model:

```bash
python scripts/05_merge_lora.py
```

Creates a full HuggingFace model in `models/merged/`.

### 6. Convert to GGUF

Convert to GGUF format for Ollama:

```bash
bash scripts/06_convert_to_gguf.sh
# On Windows:
powershell scripts/06_convert_to_gguf.ps1
```

Generates:
- `qwen2p5_7b_merged.gguf` (fp16)
- `qwen2p5_7b_merged.Q4_K_M.gguf` (quantized, recommended)

### 7. Create Ollama Model

Deploy to Ollama:

```bash
bash scripts/07_make_ollama_model.sh
# On Windows:
powershell scripts/07_make_ollama_model.ps1
```

## 🎮 Using Your Model

```bash
# Run the model
ollama run qwen2p5-7b-sft

# Example prompts
> Tell me about the content from the training documents
> Summarize the key points from the PDFs
> What information do you have about [topic]?
```

## ⚙️ Configuration

### Training Configuration

Edit `configs/train_qlora.yaml` to customize:

- Learning rate
- Batch size and gradient accumulation
- Number of epochs
- LoRA parameters (r, alpha, dropout)
- Max sequence length

### Docling Configuration

Edit `configs/docling.yaml` for PDF processing:

- Table detection
- Footnote inference
- Code extraction
- Image handling

## 💡 Tips & Best Practices

### Memory Requirements

- **Training**: 32GB VRAM minimum (QLoRA 4-bit)
- **Merging**: Can run on CPU
- **Conversion**: CPU-only, ~32GB RAM recommended

### Context Length

- Training at 4-8k tokens is stable and memory-efficient
- Qwen2.5 base supports up to 128k context
- Inference can use longer contexts than training

### Dataset Quality

- Use high-quality, domain-specific PDFs
- More data = better results (aim for 100+ pages)
- Clean, well-formatted documents work best

## 🔧 Troubleshooting

### CUDA Out of Memory

- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Reduce `max_seq_length`
- Enable `gradient_checkpointing`

### Docling Errors

- Ensure PDFs are not password-protected
- Check PDF file integrity
- Try updating Docling: `pip install -U docling`

### Ollama Issues

- Ensure Ollama is installed and running
- Check GGUF file exists in `models/gguf/`
- Verify Modelfile path is correct

## 📚 Resources

- [Docling Documentation](https://docling-project.github.io/docling/)
- [Qwen2.5-7B Model Card](https://huggingface.co/Qwen/Qwen2.5-7B)
- [TRL SFTTrainer Docs](https://huggingface.co/docs/trl/en/sft_trainer)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Ollama Documentation](https://ollama.ai)

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## ⭐ Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen) for the base model
- [Docling Project](https://github.com/docling-project/docling) for PDF processing
- [HuggingFace](https://huggingface.co) for transformers, PEFT, and TRL
- [Ollama](https://ollama.ai) for local model deployment

