# Repo layout (OSS-style)

```
qwen-finetune-pipeline/
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ .env.example
├─ .gitignore
├─ data/
│  ├─ raw_pdfs/            # put your PDFs here
│  ├─ md/                  # docling → markdown output
│  ├─ cleaned/             # cleaned markdown
│  ├─ chunks/              # chunked JSONL (pretraining or SFT-ready)
│  └─ sft/                 # final SFT dataset (jsonl)
├─ scripts/
│  ├─ 01_pdf_to_md.py
│  ├─ 02_clean_md.py
│  ├─ 03_chunk_to_jsonl.py
│  ├─ 04_train_qlora.py
│  ├─ 05_merge_lora.py
│  ├─ 06_convert_to_gguf.sh
│  └─ 07_make_ollama_model.sh
├─ configs/
│  ├─ docling.yaml
│  ├─ chunking.yaml
│  └─ train_qlora.yaml
├─ training/
│  └─ run_logs/
├─ models/
│  ├─ base/                # HF cache/symlink for base model
│  ├─ lora/                # PEFT adapter checkpoints
│  ├─ merged/              # merged safetensors
│  └─ gguf/                # converted GGUF files
└─ ollama/
   └─ Modelfile
```

---

# 0) Environment

* Python 3.10/3.11
* PyTorch (CUDA), `transformers`, `trl`, `peft`, `accelerate`, `bitsandbytes`, `datasets`, `flash-attn` (optional but nice), `docling` + `docling-core`, `pydantic`, `typer`, `rich`.

```
pip install -r requirements.txt
accelerate config
```

**Docling** converts PDFs to Markdown/HTML/JSON with advanced layout/reading-order and table handling. ([docling-project.github.io][1])

**Model:** `Qwen/Qwen2.5-7B` (base). Model card notes 128k context on base; Instruct defaults to 32k. We’ll fine-tune the **base**. ([Hugging Face][2])

**QLoRA stack:** TRL SFTTrainer + PEFT + bitsandbytes (4-bit) has a documented path and configs. ([Hugging Face][3])

---

# 1) PDF → Markdown with Docling

**Config:** `configs/docling.yaml` (example)

```yaml
pdf:
  detect_tables: true
  infer_footnotes: true
  extract_code: true
export:
  markdown:
    include_images: false
    heading_normalization: true
```

**Script:** `scripts/01_pdf_to_md.py` (outline)

```python
import pathlib, json, typer
from docling.document_converter import DocumentConverter
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

def main(src="data/raw_pdfs", dst="data/md", conf="configs/docling.yaml"):
    src = pathlib.Path(src); dst = pathlib.Path(dst); dst.mkdir(parents=True, exist_ok=True)
    conv = DocumentConverter(pipeline=StandardPdfPipeline.from_yaml(conf))
    for pdf in src.glob("*.pdf"):
        doc = conv.convert(pdf)
        md = doc.export_to_markdown()
        (dst / (pdf.stem + ".md")).write_text(md, encoding="utf-8")

if __name__ == "__main__":
    typer.run(main)
```

Docling usage & exports shown in the docs (Markdown/HTML/DocTags/JSON). ([docling-project.github.io][1])

Run:

```
python scripts/01_pdf_to_md.py
```

---

# 2) Clean Markdown

Goals: strip boilerplate, normalize whitespace, fix broken headings, join hyphenated line breaks, remove page numbers/footers, standardize code fences.

**Script:** `scripts/02_clean_md.py` (outline)

```python
import re, pathlib, typer

def clean(text: str) -> str:
    t = text
    t = re.sub(r'\n{3,}', '\n\n', t)          # collapse excess blank lines
    t = re.sub(r'(?m)^\s*Page\s+\d+\s*$', '', t)  # remove "Page N"
    t = re.sub(r'(?m)^\s*-+\s*$', '', t)      # stray rules
    t = re.sub(r'(\w)-\n(\w)', r'\1\2', t)    # de-hyphenate line wraps
    t = re.sub(r'(?m)^\s*([A-Za-z].+)\s*\n={3,}$', r'# \1', t)  # setext→ATX
    return t.strip()

def main(src="data/md", dst="data/cleaned"):
    src = pathlib.Path(src); dst = pathlib.Path(dst); dst.mkdir(parents=True, exist_ok=True)
    for md in src.glob("*.md"):
        raw = md.read_text(encoding="utf-8")
        md_out = clean(raw)
        (dst / md.name).write_text(md_out, encoding="utf-8")

if __name__ == "__main__":
    typer.run(main)
```

Run:

```
python scripts/02_clean_md.py
```

---

# 3) Chunk → JSONL for SFT (or CPT)

Two common paths:

* **SFT**: supervised pairs (instruction/input → output). If you only have raw text, create QA summaries, section Q&A, or instruction-style pairs programmatically.
* **CPT/DAPT**: next-token LM on unlabeled text; pack into windows.

For **general SFT** on knowledge docs, a robust default is **semantic-aware chunking + metadata**:

**Chunking config:** `configs/chunking.yaml`

```yaml
target_token_length: 800
min_chunk: 400
overlap: 120
splitters:
  - type: "markdown_header"
  - type: "semantic_sentence"     # fall back to sentence splitter
  - type: "length"
metadata:
  include:
    - source
    - title
    - page_start
    - page_end
```

**Script:** `scripts/03_chunk_to_jsonl.py` (outline)

```python
import json, pathlib, typer, re
from hashlib import md5

def sentence_split(s):  # minimal; swap with better splitter if you like
    return re.split(r'(?<=[.!?])\s+\n?|\n{2,}', s)

def chunk_text(text, target=800, overlap=120, tokenizer=None):
    # simplistic token-ish splitter using words; replace with HF tokenizer for precision
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        end = min(i + target, len(words))
        chunk = " ".join(words[i:end])
        chunks.append(chunk)
        i = max(end - overlap, end)
    return chunks

def main(src="data/cleaned", outdir="data/sft"):
    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "train.jsonl"
    with out_path.open("w", encoding="utf-8") as w:
        for md in pathlib.Path(src).glob("*.md"):
            text = md.read_text(encoding="utf-8")
            for c in chunk_text(text):
                if len(c.split()) < 80: 
                    continue
                # Simple SFT pair: ask-as-you-tell (instruction -> answer)
                item = {
                    "conversations": [
                        {"role":"system","content":"You are a helpful domain expert."},
                        {"role":"user","content": "Summarize the following content clearly and factually:\n\n" + c[:4000]},
                        {"role":"assistant","content": c[:4000]}
                    ],
                    "meta": {
                        "source": str(md.name),
                        "hash": md5(c.encode()).hexdigest()
                    }
                }
                w.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    typer.run(main)
```

Run:

```
python scripts/03_chunk_to_jsonl.py
```

> If you prefer **CPT** instead of SFT, emit `{ "text": <chunk> }` JSONL and switch the training objective accordingly.

---

# 4) QLoRA Finetune (Qwen2.5-7B, 4-bit)

* Base: `Qwen/Qwen2.5-7B`
* Trainer: TRL `SFTTrainer` with `bnb_4bit`, LoRA `r=16`, gradient checkpointing, FlashAttention if installed.
* Docs: TRL SFTTrainer & bitsandbytes QLoRA/FSDP integration. ([Hugging Face][3])

**Training config** `configs/train_qlora.yaml` (example)

```yaml
model_name: Qwen/Qwen2.5-7B
output_dir: models/lora
dataset_path: data/sft/train.jsonl
hf_format: "chat_template"     # apply_chat_template for Qwen
seed: 42

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: "bfloat16"

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj - k_proj - v_proj - o_proj
    - gate_proj - up_proj - down_proj

train:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 64
  num_train_epochs: 2
  learning_rate: 2.0e-4
  warmup_ratio: 0.03
  lr_scheduler_type: cosine
  logging_steps: 10
  save_steps: 500
  max_steps: -1
  gradient_checkpointing: true
  max_seq_length: 4096
  packing: true
```

**Script:** `scripts/04_train_qlora.py` (outline)

```python
import json, yaml, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments

cfg = yaml.safe_load(open("configs/train_qlora.yaml"))

tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("json", data_files=cfg["dataset_path"], split="train")

qconf = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], quantization_config=qconf, torch_dtype=torch.bfloat16, device_map="auto")

lconf = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lconf)

def format_chat(example):
    return tokenizer.apply_chat_template(example["conversations"], tokenize=False, add_generation_prompt=False)

trainer = SFTTrainer(
   model=model,
   tokenizer=tokenizer,
   train_dataset=ds.map(lambda x: {"text": format_chat(x)}),
   dataset_text_field="text",
   max_seq_length=4096,
   packing=True,
   args=TrainingArguments(
      output_dir=cfg["output_dir"],
      per_device_train_batch_size=1,
      gradient_accumulation_steps=64,
      num_train_epochs=2,
      learning_rate=2e-4,
      warmup_ratio=0.03,
      logging_steps=10,
      save_steps=500,
      bf16=True,
      optim="paged_adamw_32bit"
   )
)

trainer.train()
trainer.model.save_pretrained("models/lora")
tokenizer.save_pretrained("models/lora")
```

(QLoRA/PEFT/TRL references.) ([Hugging Face][4])

---

# 5) Merge LoRA → full HF model

After training, create a single safetensors model by merging adapters into the base:

**Script:** `scripts/05_merge_lora.py` (outline)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "Qwen/Qwen2.5-7B"
adapter_dir = "models/lora"
out_dir = "models/merged"

tok = AutoTokenizer.from_pretrained(base, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="cpu")
peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
merged = peft_model.merge_and_unload()   # <-- key API
merged.save_pretrained(out_dir, safe_serialization=True)
tok.save_pretrained(out_dir)
```

PEFT `merge_and_unload()` merges LoRA into base weights. ([Hugging Face Forums][5])

Run:

```
python scripts/05_merge_lora.py
```

---

# 6) Convert merged HF → GGUF (for Ollama)

**Qwen → GGUF:** Qwen docs describe using **llama.cpp**’s `convert-hf-to-gguf.py` for Qwen3/2.5 lines. (Vision models have special notes; we’re doing text-only 7B.) ([qwen.readthedocs.io][6])

**Script:** `scripts/06_convert_to_gguf.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
MERGED_DIR="models/merged"
GGUF_DIR="models/gguf"
mkdir -p "$GGUF_DIR"

# clone llama.cpp if needed
[ -d llama.cpp ] || git clone https://github.com/ggerganov/llama.cpp.git
pip install -r llama.cpp/requirements.txt

# Convert fp16 → GGUF
python llama.cpp/convert-hf-to-gguf.py "$MERGED_DIR" --outfile "$GGUF_DIR/qwen2p5_7b_merged.gguf" --outtype f16

# (Optional) quantize to Q4_K_M for smaller footprint
./llama.cpp/quantize "$GGUF_DIR/qwen2p5_7b_merged.gguf" "$GGUF_DIR/qwen2p5_7b_merged.Q4_K_M.gguf" Q4_K_M
```

(General GGUF conversion guidance & discussions.) ([GitHub][7])

> Note: llama.cpp issue threads show which Qwen variants are supported; text-only Qwen2/2.5/3 are supported via convert-hf-to-gguf (VL variants require different handling). ([GitHub][8])

---

# 7) Create an Ollama model and run it

Two common ways:

**A) Use a local GGUF + `Modelfile`**

`ollama/Modelfile`:

```
FROM ./models/gguf/qwen2p5_7b_merged.Q4_K_M.gguf
TEMPLATE """
{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|><|im_start|>assistant
{{ end }}
"""
PARAMETER num_ctx 8192
PARAMETER temperature 0.7
```

**Script:** `scripts/07_make_ollama_model.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
ollama create qwen2p5-7b-sft -f ollama/Modelfile
ollama run qwen2p5-7b-sft
```

**B) Alternative—Modelfile that points to a remote GGUF/HF asset**: see community tutorials (convert to GGUF first, then reference in `FROM`). ([Medium][9])

---

# 8) Makefile (optional, nice DX)

```
.PHONY: pdf md clean chunk sft train merge gguf ollama all

pdf:        ## put PDFs in data/raw_pdfs/
	@echo "PDFs ready"

md:
	python scripts/01_pdf_to_md.py

clean:
	python scripts/02_clean_md.py

chunk:
	python scripts/03_chunk_to_jsonl.py

sft: md clean chunk

train:
	python scripts/04_train_qlora.py

merge:
	python scripts/05_merge_lora.py

gguf:
	bash scripts/06_convert_to_gguf.sh

ollama:
	bash scripts/07_make_ollama_model.sh

all: sft train merge gguf ollama
```

---

# 9) Practical tips

* **Context length**: Train at 4–8k tokens for stability/VRAM. You can still *infer* longer if you choose suitable variants/config, but training at huge lengths balloons memory. Qwen2.5 base lists **128k** context; Instruct often ships with **32k** (YaRN can extrapolate). ([Hugging Face][2])
* **Tokenization**: Always `apply_chat_template` for SFT chat data with Qwen family (TRL example docs). ([Hugging Face][3])
* **Memory**: 32 GB VRAM works with QLoRA (4-bit), batch size 1 and gradient accumulation to reach your effective batch. bitsandbytes/PEFT recipes are documented. ([Hugging Face][4])
* **Merging**: Use PEFT’s `merge_and_unload()` for a clean HF model before GGUF. ([Hugging Face Forums][5])
* **Ollama**: If a model isn’t in the official library, you can import via GGUF + `Modelfile`. Tutorials show the process. ([GPU Mart][10])

---

# 10) Links & resources

* **Docling (PDF → Markdown/JSON)**: site & GitHub. ([docling-project.github.io][1])
* **Docling Core (APIs & chunking utilities)**. ([GitHub][11])
* **Qwen2.5-7B (base)** model card. ([Hugging Face][2])
* **Qwen2.5-7B-Instruct** (context notes / YaRN). ([Hugging Face][12])
* **TRL SFTTrainer** (SFT on chat data). ([Hugging Face][3])
* **bitsandbytes FSDP-QLoRA doc** (integration guide). ([Hugging Face][4])
* **PEFT merge LoRA → base** (HF forum, issues, examples). ([Hugging Face Forums][5])
* **llama.cpp GGUF conversion** (general guide) & **Qwen → GGUF** page. ([GitHub][7])
* **Ollama custom import tutorials** (GGUF + Modelfile). ([GPU Mart][10])

---

# 11) Minimal checklist (copy/paste to README)

1. Put PDFs into `data/raw_pdfs/`.
2. `python scripts/01_pdf_to_md.py` → Markdown in `data/md`. ([docling-project.github.io][1])
3. `python scripts/02_clean_md.py` → cleaned MD in `data/cleaned`.
4. `python scripts/03_chunk_to_jsonl.py` → `data/sft/train.jsonl`.
5. `python scripts/04_train_qlora.py` → saves LoRA to `models/lora`. ([Hugging Face][3])
6. `python scripts/05_merge_lora.py` → merged HF model in `models/merged`. ([Hugging Face Forums][5])
7. `bash scripts/06_convert_to_gguf.sh` → GGUF in `models/gguf`. ([qwen.readthedocs.io][6])
8. `bash scripts/07_make_ollama_model.sh` → `ollama run qwen2p5-7b-sft`. ([GPU Mart][10])

---

[1]: https://docling-project.github.io/docling/?utm_source=chatgpt.com "Docling - GitHub Pages"
[2]: https://huggingface.co/Qwen/Qwen2.5-7B?utm_source=chatgpt.com "Qwen/Qwen2.5-7B"
[3]: https://huggingface.co/docs/trl/en/sft_trainer?utm_source=chatgpt.com "SFT Trainer"
[4]: https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora?utm_source=chatgpt.com "FSDP-QLoRA"
[5]: https://discuss.huggingface.co/t/help-with-merging-lora-weights-back-into-base-model/40968?utm_source=chatgpt.com "Help with merging LoRA weights back into base model :-)"
[6]: https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html?utm_source=chatgpt.com "llama.cpp - Qwen docs"
[7]: https://github.com/ggml-org/llama.cpp/discussions/2948?utm_source=chatgpt.com "How to convert HuggingFace model to GGUF format · ggml ..."
[8]: https://github.com/ggerganov/llama.cpp/issues/11541?utm_source=chatgpt.com "when llama.cpp can support convert qwen2.5 VL 7B/72B ..."
[9]: https://medium.com/%40udemirezen/converting-hugging-face-models-for-use-with-ollama-a-detailed-tutorial-4e64b66eea27?utm_source=chatgpt.com "Converting Hugging Face Models for Use with Ollama"
[10]: https://www.gpu-mart.com/blog/import-models-from-huggingface-to-ollama?utm_source=chatgpt.com "How to Import Models from Hugging Face to Ollama"
[11]: https://github.com/docling-project/docling-core?utm_source=chatgpt.com "docling-project/docling-core: A python library to define and ..."
[12]: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct?utm_source=chatgpt.com "Qwen/Qwen2.5-7B-Instruct"
