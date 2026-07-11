# setup_spark.ps1
# =================
# Installs dependencies required for SparK + YOLO11 pretraining.
# Assumes PyTorch with CUDA 12.1 is already installed in the environment.

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Installing SparK + YOLO dependencies" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Install ultralytics for YOLO models
Write-Host "`nInstalling Ultralytics (YOLO11)..." -ForegroundColor Yellow
conda run -n ag-foundation pip install ultralytics

# Install spconv for CUDA 12.1
# NOTE: Using the prebuilt wheel which avoids C++ compilation on Windows
Write-Host "`nInstalling spconv (Sparse Convolutions) for CUDA 12.1..." -ForegroundColor Yellow
conda run -n ag-foundation pip install spconv-cu121

Write-Host "`nTesting spconv installation..." -ForegroundColor Yellow
conda run -n ag-foundation python -c "import spconv; import spconv.pytorch as spconv_core; print('spconv successfully loaded!')"

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  SparK Environment Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
