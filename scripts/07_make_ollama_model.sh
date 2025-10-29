#!/usr/bin/env bash
# Create and run an Ollama model from GGUF file
#
# Usage:
#   bash scripts/07_make_ollama_model.sh
#   bash scripts/07_make_ollama_model.sh qwen2p5-7b-sft

set -euo pipefail

# Configuration
MODEL_NAME="${1:-qwen2p5-7b-sft}"
MODELFILE="ollama/Modelfile"
GGUF_FILE="models/gguf/qwen2p5_7b_merged.Q4_K_M.gguf"

echo "=== Creating Ollama Model ==="
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Error: Ollama is not installed!"
    echo "Please install Ollama from: https://ollama.ai"
    exit 1
fi

echo "✓ Ollama is installed"

# Check if GGUF file exists
if [ ! -f "$GGUF_FILE" ]; then
    echo "Error: GGUF file not found: $GGUF_FILE"
    echo ""
    echo "Trying fallback to fp16 version..."
    GGUF_FILE="models/gguf/qwen2p5_7b_merged.gguf"
    
    if [ ! -f "$GGUF_FILE" ]; then
        echo "Error: No GGUF file found!"
        echo "Run 06_convert_to_gguf.sh first to create the GGUF file"
        exit 1
    fi
fi

echo "✓ Found GGUF file: $GGUF_FILE"

# Check if Modelfile exists
if [ ! -f "$MODELFILE" ]; then
    echo "Error: Modelfile not found: $MODELFILE"
    echo "The Modelfile should be created in the ollama/ directory"
    exit 1
fi

echo "✓ Found Modelfile: $MODELFILE"
echo ""

# Create the Ollama model
echo "Creating Ollama model: $MODEL_NAME"
echo "This may take a few minutes..."
ollama create "$MODEL_NAME" -f "$MODELFILE"

echo ""
echo "✓ Model created successfully!"
echo ""

# List available models
echo "Available Ollama models:"
ollama list
echo ""

# Prompt to run the model
echo "=== Model Ready ==="
echo ""
echo "To run the model, use:"
echo "  ollama run $MODEL_NAME"
echo ""
echo "Example prompts:"
echo "  - Tell me about the content from the training documents"
echo "  - Summarize the key points from the PDFs"
echo "  - What information do you have about [topic]?"
echo ""

# Ask if user wants to run now
read -p "Do you want to run the model now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting interactive session with $MODEL_NAME..."
    echo "Type 'exit' or press Ctrl+D to quit"
    echo ""
    ollama run "$MODEL_NAME"
fi

