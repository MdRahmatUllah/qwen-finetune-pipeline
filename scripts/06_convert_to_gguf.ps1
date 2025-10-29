# Convert merged HuggingFace model to GGUF format using llama.cpp
#
# Usage:
#   powershell scripts/06_convert_to_gguf.ps1
#   powershell scripts/06_convert_to_gguf.ps1 -MergedDir models/merged -GgufDir models/gguf
#   powershell -File scripts\06_convert_to_gguf.ps1

param(
    [string]$MergedDir = "models/merged",
    [string]$GgufDir = "models/gguf",
    [string]$LlamaCppDir = "llama.cpp"
)

$ErrorActionPreference = "Stop"

Write-Host "=== Converting HuggingFace Model to GGUF ===" -ForegroundColor Blue
Write-Host ""

# Check if merged model exists
if (-not (Test-Path $MergedDir)) {
    Write-Host "Error: Merged model directory '$MergedDir' not found!" -ForegroundColor Red
    Write-Host "Run 05_merge_lora.py first to create the merged model" -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Found merged model in $MergedDir" -ForegroundColor Green

# Create GGUF output directory
New-Item -ItemType Directory -Force -Path $GgufDir | Out-Null
Write-Host "[OK] Output directory: $GgufDir" -ForegroundColor Green
Write-Host ""

# Clone llama.cpp if not present
if (-not (Test-Path $LlamaCppDir)) {
    Write-Host "Cloning llama.cpp repository..." -ForegroundColor Cyan
    git clone https://github.com/ggml-org/llama.cpp.git
    Write-Host "[OK] llama.cpp cloned" -ForegroundColor Green
} else {
    Write-Host "[OK] llama.cpp already present" -ForegroundColor Green
}

# Install Python requirements for llama.cpp
Write-Host ""
Write-Host "Installing llama.cpp Python requirements..." -ForegroundColor Cyan
pip install -q -r "$LlamaCppDir/requirements.txt"
Write-Host "[OK] Requirements installed" -ForegroundColor Green

# Convert to GGUF (fp16)
Write-Host ""
Write-Host "Converting to GGUF format (fp16)..." -ForegroundColor Cyan
Write-Host "This may take several minutes..." -ForegroundColor Yellow

python "$LlamaCppDir/convert_hf_to_gguf.py" `
    $MergedDir `
    --outfile "$GgufDir/qwen2p5_7b_merged_2.gguf" `
    --outtype f16

Write-Host "[OK] Conversion to fp16 GGUF complete" -ForegroundColor Green

# Optional: Quantize to Q4_K_M for smaller size
Write-Host ""
Write-Host "Quantizing to Q4_K_M (recommended for Ollama)..." -ForegroundColor Cyan
Write-Host "This will create a smaller, faster model with minimal quality loss" -ForegroundColor Yellow

# Check for quantize executable
$QuantizeExe = $null
if (Test-Path "$LlamaCppDir/quantize.exe") {
    $QuantizeExe = "$LlamaCppDir/quantize.exe"
} elseif (Test-Path "$LlamaCppDir/build/bin/Release/quantize.exe") {
    $QuantizeExe = "$LlamaCppDir/build/bin/Release/quantize.exe"
}

if ($null -eq $QuantizeExe) {
    Write-Host "Warning: quantize.exe not found" -ForegroundColor Yellow
    Write-Host "You need to build llama.cpp first. See: https://github.com/ggml-org/llama.cpp.git#build" -ForegroundColor Yellow
    Write-Host "Skipping quantization step" -ForegroundColor Yellow
} else {
    & $QuantizeExe `
        "$GgufDir/qwen2p5_7b_merged_2.gguf" `
        "$GgufDir/qwen2p5_7b_merged_2.Q4_K_M.gguf" `
        Q4_K_M
    Write-Host "[OK] Quantization complete" -ForegroundColor Green
}

# Summary
Write-Host ""
Write-Host "=== Conversion Complete ===" -ForegroundColor Blue
Write-Host ""
Write-Host "Generated files:" -ForegroundColor Cyan
Get-ChildItem "$GgufDir/*.gguf" | ForEach-Object {
    $sizeMB = [math]::Round($_.Length/1MB, 2)
    Write-Host "  $($_.Name) - $sizeMB MB"
}
Write-Host ""
Write-Host "Next step: Run 07_make_ollama_model.ps1 to create an Ollama model" -ForegroundColor Cyan

