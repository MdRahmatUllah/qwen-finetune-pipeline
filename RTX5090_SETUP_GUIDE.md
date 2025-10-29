# RTX 5090 Setup Guide for QLoRA Training

This guide helps you set up PyTorch with proper CUDA support for the NVIDIA RTX 5090 (Blackwell architecture, compute capability sm_120).

## The Problem

The RTX 5090 has **compute capability sm_120** (Blackwell architecture), which is not supported by stable PyTorch releases as of October 2025. You'll encounter this error:

```
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90.
```

This causes the following errors:
- `RuntimeError: CUDA error: no kernel image is available for execution on the device`
- `ValueError: Your setup doesn't support bf16/gpu`
- bitsandbytes 4-bit quantization failures

## The Solution

Use **PyTorch nightly builds with CUDA 12.8**, which include sm_120 support for Blackwell GPUs.

---

## Quick Installation (Automated)

### Option 1: Use the PowerShell Script

```powershell
# Run the automated installation script
.\scripts\install_rtx5090_support.ps1
```

This script will:
1. Uninstall existing PyTorch
2. Install PyTorch nightly with CUDA 12.8
3. Upgrade bitsandbytes
4. Verify the installation

---

## Manual Installation (Step-by-Step)

### Prerequisites

- ✅ NVIDIA RTX 5090 GPU
- ✅ NVIDIA Driver 581.57 or later (with CUDA 13.0 support)
- ✅ Python 3.9 - 3.13
- ✅ Virtual environment activated

### Step 1: Check Your Current Setup

```powershell
# Check Python version
python --version

# Check current PyTorch version
pip show torch

# Check NVIDIA driver and CUDA version
nvidia-smi
```

### Step 2: Uninstall Existing PyTorch

```powershell
pip uninstall torch torchvision torchaudio -y
```

### Step 3: Install PyTorch Nightly with CUDA 12.8

```powershell
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Important Notes:**
- This installs PyTorch **nightly** (development version)
- CUDA 12.8 includes sm_120 support for RTX 5090
- `torchvision` and `torchaudio` with CUDA 12.8 are still in development (as of Feb 2025)
- For QLoRA training, you only need `torch` (vision/audio not required)

### Step 4: Upgrade bitsandbytes

```powershell
pip install --upgrade bitsandbytes
```

### Step 5: Verify Installation

```powershell
python scripts/verify_rtx5090.py
```

Expected output:
```
✓ PyTorch version: 2.7.0.dev20250220+cu128
✓ CUDA available: True
✓ CUDA version: 12.8
✓ GPU 0: NVIDIA GeForce RTX 5090
✓ Compute capability: sm_120
✓ Compute capability sm_120 is SUPPORTED
✓ BF16 (bfloat16) support: Yes
✓ bitsandbytes version: 0.48.1
✓ 4-bit quantization setup is ready
```

---

## Verification Commands

### Check PyTorch CUDA Support

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'Compute: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}'); print(f'BF16: {torch.cuda.is_bf16_supported()}')"
```

### Check Supported Architectures

```powershell
python -c "import torch; print('Supported:', torch.cuda.get_arch_list())"
```

You should see `sm_120` or `compute_120` in the list.

### Test GPU Tensor Operations

```powershell
python -c "import torch; x = torch.randn(1000, 1000).cuda(); y = x @ x.T; print(f'GPU test passed: {y.shape}')"
```

---

## Troubleshooting

### Issue: "Could not find a version that satisfies the requirement torch"

**Cause:** Your Python version is not supported by nightly builds.

**Solution:** Use Python 3.9 - 3.13. Check with:
```powershell
python --version
```

### Issue: "CUDA not available" after installation

**Cause:** PyTorch CPU version was installed instead of CUDA version.

**Solution:** 
1. Verify you used the correct index URL: `https://download.pytorch.org/whl/nightly/cu128`
2. Check installed version: `pip show torch` (should show `+cu128` in version)
3. Reinstall if needed

### Issue: "no kernel image is available" persists

**Cause:** Old PyTorch version still installed or wrong CUDA version.

**Solution:**
1. Completely uninstall PyTorch: `pip uninstall torch torchvision torchaudio -y`
2. Clear pip cache: `pip cache purge`
3. Reinstall PyTorch nightly with CUDA 12.8
4. Verify with `python scripts/verify_rtx5090.py`

### Issue: bitsandbytes errors

**Cause:** bitsandbytes may not be compatible with nightly PyTorch.

**Solution:**
1. Try upgrading: `pip install --upgrade bitsandbytes`
2. If that fails, try installing from source:
   ```powershell
   pip install git+https://github.com/TimDettmers/bitsandbytes.git
   ```

### Issue: Training is slow despite GPU being detected

**Cause:** Model may be loading on CPU instead of GPU.

**Solution:**
1. Check GPU utilization: `nvidia-smi` (should show high GPU usage during training)
2. Verify `device_map="auto"` is set in model loading
3. Check training script output for "Using bfloat16 precision (bf16)"

---

## Performance Expectations

With RTX 5090 (32GB VRAM) and proper setup:

| Metric | Expected Value |
|--------|---------------|
| **Precision** | bfloat16 (bf16) |
| **Quantization** | 4-bit NF4 |
| **VRAM Usage** | ~8-12GB for Qwen2.5-7B |
| **Training Speed** | ~2-4 hours for typical fine-tuning |
| **Batch Size** | 4-8 (with gradient accumulation) |
| **GPU Utilization** | 80-95% |

---

## Known Limitations

1. **Nightly Builds**: You're using development versions of PyTorch
   - May have bugs or instability
   - API may change between builds
   - Not recommended for production deployments

2. **torchvision/torchaudio**: Not available with CUDA 12.8 yet
   - Only affects vision/audio tasks
   - Does not affect QLoRA text model training

3. **Windows Support**: CUDA 12.8 Windows support is recent (Feb 2025)
   - Linux support is more mature
   - Consider WSL2 if you encounter issues

---

## Alternative: Use WSL2 (Linux)

If you encounter issues with Windows, you can use WSL2:

```bash
# In WSL2 Ubuntu
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

Linux nightly builds have been available longer and may be more stable.

---

## Next Steps After Installation

Once verification passes:

1. **Test PDF conversion:**
   ```powershell
   python scripts/01_pdf_to_md.py --src data/raw_pdfs --dst data/md
   ```

2. **Run the full pipeline:**
   ```powershell
   make all
   ```
   Or step-by-step:
   ```powershell
   python scripts/01_pdf_to_md.py
   python scripts/02_clean_md.py
   python scripts/03_chunk_to_jsonl.py
   python scripts/04_train_qlora.py
   python scripts/05_merge_lora.py
   ```

3. **Monitor training:**
   - Watch GPU usage: `nvidia-smi -l 1`
   - Check training logs in `training/run_logs/`
   - Training should show "Using bfloat16 precision (bf16)"

---

## References

- [PyTorch Forums: sm_120 Support](https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099)
- [PyTorch Nightly Downloads](https://download.pytorch.org/whl/nightly/torch/)
- [NVIDIA RTX 5090 Specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/)
- [bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)

---

## Support

If you encounter issues:

1. Run the verification script: `python scripts/verify_rtx5090.py`
2. Check PyTorch forums for latest updates on sm_120 support
3. Ensure you have the latest NVIDIA drivers (581.57+)
4. Consider using WSL2 if Windows issues persist

---

**Last Updated:** October 29, 2025  
**PyTorch Nightly Version:** 2.7.0.dev20250220+cu128  
**CUDA Version:** 12.8  
**Supported Compute Capabilities:** sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, sm_120

