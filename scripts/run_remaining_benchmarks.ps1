$ErrorActionPreference = "Stop"

Write-Host "===============================================" -ForegroundColor Green
Write-Host " Starting Remaining AG-Foundation Benchmarks   " -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green

# 2. OBB Detection (PlantSeg)
Write-Host "`n--- Task 2: Detection (OBB) ---" -ForegroundColor Magenta
Write-Host "Running MIM Fine-tuning..." -ForegroundColor Cyan
conda run -n ag-foundation python -m ag_foundation train-det --config configs/det_plantseg_mim.yaml
Write-Host "Running ImageNet Fine-tuning..." -ForegroundColor Cyan
conda run -n ag-foundation python -m ag_foundation train-det --config configs/det_plantseg_imagenet.yaml

# 3. Counting (Corn Kernels)
Write-Host "`n--- Task 3: Regression / Counting ---" -ForegroundColor Magenta
Write-Host "Running MIM Fine-tuning..." -ForegroundColor Cyan
conda run -n ag-foundation python -m ag_foundation train-count --config configs/count_corn_mim.yaml
Write-Host "Running ImageNet Fine-tuning..." -ForegroundColor Cyan
conda run -n ag-foundation python -m ag_foundation train-count --config configs/count_corn_imagenet.yaml

# 4. Temporal Analysis (Longitudinal Nutrient)
Write-Host "`n--- Task 4: Temporal Analysis ---" -ForegroundColor Magenta
Write-Host "Running MIM Fine-tuning..." -ForegroundColor Cyan
conda run -n ag-foundation python -m ag_foundation train-temporal --config configs/temporal_nutrient_mim.yaml
Write-Host "Running ImageNet Fine-tuning..." -ForegroundColor Cyan
conda run -n ag-foundation python -m ag_foundation train-temporal --config configs/temporal_nutrient_imagenet.yaml

Write-Host "`n===============================================" -ForegroundColor Green
Write-Host " All Benchmarks Completed Successfully!        " -ForegroundColor Green
Write-Host " Check the E:\AG_Dataset\runs\ directory for results. " -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green
