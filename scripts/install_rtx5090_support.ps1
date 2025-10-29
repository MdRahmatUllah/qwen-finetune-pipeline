# PowerShell script to install PyTorch with RTX 5090 (sm_120) support
# This script installs PyTorch nightly with CUDA 12.8 for Blackwell architecture

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "RTX 5090 PyTorch Installation Script" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if we're in a virtual environment
if (-not $env:VIRTUAL_ENV) {
    Write-Host "WARNING: You are not in a virtual environment!" -ForegroundColor Yellow
    Write-Host "It's recommended to use a virtual environment." -ForegroundColor Yellow
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        Write-Host "Exiting..." -ForegroundColor Red
        exit 1
    }
}

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Cyan
$pythonVersion = python --version 2>&1
Write-Host "Found: $pythonVersion" -ForegroundColor Green

# Step 1: Uninstall existing PyTorch
Write-Host "`nStep 1: Uninstalling existing PyTorch..." -ForegroundColor Cyan
pip uninstall torch torchvision torchaudio -y

# Step 2: Install PyTorch nightly with CUDA 12.8
Write-Host "`nStep 2: Installing PyTorch nightly with CUDA 12.8 (sm_120 support)..." -ForegroundColor Cyan
Write-Host "This may take several minutes..." -ForegroundColor Yellow
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: Failed to install PyTorch nightly" -ForegroundColor Red
    exit 1
}

# Step 3: Upgrade bitsandbytes
Write-Host "`nStep 3: Upgrading bitsandbytes..." -ForegroundColor Cyan
pip install --upgrade bitsandbytes

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nWARNING: Failed to upgrade bitsandbytes" -ForegroundColor Yellow
    Write-Host "Continuing anyway..." -ForegroundColor Yellow
}

# Step 4: Verify installation
Write-Host "`nStep 4: Verifying installation..." -ForegroundColor Cyan
Write-Host "Running verification script...`n" -ForegroundColor Yellow

python scripts/verify_rtx5090.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "Installation completed successfully!" -ForegroundColor Green
    Write-Host "========================================`n" -ForegroundColor Green
    Write-Host "Your RTX 5090 is now ready for QLoRA training!" -ForegroundColor Green
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "1. Place your PDFs in data/raw_pdfs/" -ForegroundColor White
    Write-Host "2. Run the pipeline: make all" -ForegroundColor White
    Write-Host "   OR run step-by-step:" -ForegroundColor White
    Write-Host "   - python scripts/01_pdf_to_md.py" -ForegroundColor White
    Write-Host "   - python scripts/02_clean_md.py" -ForegroundColor White
    Write-Host "   - python scripts/03_chunk_to_jsonl.py" -ForegroundColor White
    Write-Host "   - python scripts/04_train_qlora.py`n" -ForegroundColor White
} else {
    Write-Host "`n========================================" -ForegroundColor Red
    Write-Host "Installation completed with errors" -ForegroundColor Red
    Write-Host "========================================`n" -ForegroundColor Red
    Write-Host "Please review the errors above and try again." -ForegroundColor Yellow
    exit 1
}

