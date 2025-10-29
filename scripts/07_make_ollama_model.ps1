# Create and run an Ollama model from GGUF file
#
# Usage:
#   powershell scripts/07_make_ollama_model.ps1
#   powershell -File scripts\07_make_ollama_model.ps1

param(
    [string]$ModelName = "qwen25-7b-sft",
    [string]$Modelfile = "ollama/Modelfile",
    [string]$GgufFile = "models/gguf/qwen2p5_7b_merged_2.gguf"
)

$ErrorActionPreference = "Stop"

Write-Host "=== Creating Ollama Model ===" -ForegroundColor Blue
Write-Host ""

# Check if Ollama is installed
try {
    $null = Get-Command ollama -ErrorAction Stop
    Write-Host "[OK] Ollama is installed" -ForegroundColor Green
} catch {
    Write-Host "Error: Ollama is not installed!" -ForegroundColor Red
    Write-Host "Please install Ollama from: https://ollama.ai" -ForegroundColor Yellow
    exit 1
}

# Check if GGUF file exists
if (-not (Test-Path $GgufFile)) {
    Write-Host "Error: GGUF file not found: $GgufFile" -ForegroundColor Red
    Write-Host ""
    Write-Host "Trying fallback to fp16 version..." -ForegroundColor Yellow
    $GgufFile = "models/gguf/qwen2p5_7b_merged_2.gguf"
    
    if (-not (Test-Path $GgufFile)) {
        Write-Host "Error: No GGUF file found!" -ForegroundColor Red
        Write-Host "Run 06_convert_to_gguf.ps1 first to create the GGUF file" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "[OK] Found GGUF file: $GgufFile" -ForegroundColor Green

# Check if Modelfile exists
if (-not (Test-Path $Modelfile)) {
    Write-Host "Error: Modelfile not found: $Modelfile" -ForegroundColor Red
    Write-Host "The Modelfile should be created in the ollama/ directory" -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Found Modelfile: $Modelfile" -ForegroundColor Green
Write-Host ""

# Create a temporary Modelfile with absolute path to GGUF file
# Ollama requires absolute paths on Windows
Write-Host "Preparing Modelfile with absolute path..." -ForegroundColor Cyan
$absoluteGgufPath = (Resolve-Path $GgufFile).Path
$tempModelfile = "ollama/Modelfile.tmp"

# Read the original Modelfile and replace the FROM line with absolute path
$modelfileContent = Get-Content $Modelfile -Raw
$modelfileContent = $modelfileContent -replace 'FROM\s+\./models/gguf/[^\r\n]+', "FROM $absoluteGgufPath"
$modelfileContent | Set-Content $tempModelfile -NoNewline

Write-Host "[OK] Using GGUF file: $absoluteGgufPath" -ForegroundColor Green
Write-Host ""

# Create the Ollama model
Write-Host "Creating Ollama model: $ModelName" -ForegroundColor Cyan
Write-Host "This may take a few minutes..." -ForegroundColor Yellow
Write-Host ""

ollama create $ModelName -f $tempModelfile

# Clean up temporary file
Remove-Item $tempModelfile -ErrorAction SilentlyContinue

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Error: Failed to create Ollama model!" -ForegroundColor Red
    Write-Host "The 'ollama create' command failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "  1. Check that the GGUF file path in the Modelfile is correct" -ForegroundColor Yellow
    Write-Host "  2. Verify the Modelfile syntax is valid" -ForegroundColor Yellow
    Write-Host "  3. Make sure the GGUF file is not corrupted" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Modelfile location: $Modelfile" -ForegroundColor Cyan
    Write-Host "GGUF file location: $GgufFile" -ForegroundColor Cyan
    exit 1
}

# Verify the model was actually created
Write-Host ""
$modelExists = ollama list | Select-String -Pattern $ModelName -Quiet
if (-not $modelExists) {
    Write-Host "Error: Model creation reported success but model not found in 'ollama list'!" -ForegroundColor Red
    Write-Host "This may indicate a problem with the Modelfile or GGUF file." -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Model created successfully!" -ForegroundColor Green
Write-Host ""

# List available models
Write-Host "Available Ollama models:" -ForegroundColor Cyan
ollama list
Write-Host ""

# Summary
Write-Host "=== Model Ready ===" -ForegroundColor Blue
Write-Host ""
Write-Host "To run the model, use:" -ForegroundColor Cyan
Write-Host "  ollama run $ModelName" -ForegroundColor White
Write-Host ""
Write-Host "Example prompts:" -ForegroundColor Cyan
Write-Host "  - Tell me about the content from the training documents" -ForegroundColor White
Write-Host "  - Summarize the key points from the PDFs" -ForegroundColor White
Write-Host "  - What information do you have about [topic]?" -ForegroundColor White
Write-Host ""

# Ask if user wants to run now
$response = Read-Host "Do you want to run the model now? (y/n)"
if ($response -eq 'y' -or $response -eq 'Y') {
    Write-Host ""
    Write-Host "Starting interactive session with $ModelName..." -ForegroundColor Cyan
    Write-Host "Type '/bye' to quit" -ForegroundColor Yellow
    Write-Host ""
    ollama run $ModelName
}

