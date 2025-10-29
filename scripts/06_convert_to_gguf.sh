#!/usr/bin/env bash
# Convert merged HuggingFace model to GGUF format using llama.cpp
#
# Usage:
#   bash scripts/06_convert_to_gguf.sh
#   bash scripts/06_convert_to_gguf.sh models/merged models/gguf

set -euo pipefail

# Configuration
MERGED_DIR="${1:-models/merged}"
GGUF_DIR="${2:-models/gguf}"
LLAMA_CPP_DIR="llama.cpp"

echo "=== Converting HuggingFace Model to GGUF ==="
echo ""

# Check if merged model exists
if [ ! -d "$MERGED_DIR" ]; then
    echo "Error: Merged model directory '$MERGED_DIR' not found!"
    echo "Run 05_merge_lora.py first to create the merged model"
    exit 1
fi

echo "✓ Found merged model in $MERGED_DIR"

# Create GGUF output directory
mkdir -p "$GGUF_DIR"
echo "✓ Output directory: $GGUF_DIR"
echo ""

# Clone llama.cpp if not present
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "Cloning llama.cpp repository..."
    git clone https://github.com/ggerganov/llama.cpp.git
    echo "✓ llama.cpp cloned"
else
    echo "✓ llama.cpp already present"
fi

# Install Python requirements for llama.cpp
echo ""
echo "Installing llama.cpp Python requirements..."
pip install -q -r "$LLAMA_CPP_DIR/requirements.txt"
echo "✓ Requirements installed"

# Convert to GGUF (fp16)
echo ""
echo "Converting to GGUF format (fp16)..."
echo "This may take several minutes..."
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    "$MERGED_DIR" \
    --outfile "$GGUF_DIR/qwen2p5_7b_merged.gguf" \
    --outtype f16

echo "✓ Conversion to fp16 GGUF complete"

# Optional: Quantize to Q4_K_M for smaller size
echo ""
echo "Quantizing to Q4_K_M (recommended for Ollama)..."
echo "This will create a smaller, faster model with minimal quality loss"

# Build quantize tool if needed
if [ ! -f "$LLAMA_CPP_DIR/quantize" ] && [ ! -f "$LLAMA_CPP_DIR/quantize.exe" ]; then
    echo "Building llama.cpp quantize tool..."
    cd "$LLAMA_CPP_DIR"
    make quantize
    cd ..
    echo "✓ Quantize tool built"
fi

# Run quantization
if [ -f "$LLAMA_CPP_DIR/quantize" ]; then
    QUANTIZE_BIN="$LLAMA_CPP_DIR/quantize"
elif [ -f "$LLAMA_CPP_DIR/quantize.exe" ]; then
    QUANTIZE_BIN="$LLAMA_CPP_DIR/quantize.exe"
else
    echo "Warning: quantize binary not found, skipping quantization"
    echo "You can manually quantize later if needed"
    QUANTIZE_BIN=""
fi

if [ -n "$QUANTIZE_BIN" ]; then
    "$QUANTIZE_BIN" \
        "$GGUF_DIR/qwen2p5_7b_merged.gguf" \
        "$GGUF_DIR/qwen2p5_7b_merged.Q4_K_M.gguf" \
        Q4_K_M
    echo "✓ Quantization complete"
fi

# Summary
echo ""
echo "=== Conversion Complete ==="
echo ""
echo "Generated files:"
ls -lh "$GGUF_DIR"/*.gguf
echo ""
echo "Next step: Run 07_make_ollama_model.sh to create an Ollama model"

