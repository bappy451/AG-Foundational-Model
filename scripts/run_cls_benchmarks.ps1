Write-Host "===============================================" -ForegroundColor Green
Write-Host " Starting AG-Foundation Classification Benchmark " -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green
Write-Host ""

Write-Host "[1/2] Running Fine-tuning with AG-Foundation MIM Weights..." -ForegroundColor Cyan
python -m ag_foundation train-cls --config configs/finetune_classification.yaml

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: First training run failed. Aborting benchmark." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "[2/2] Running Fine-tuning with ImageNet Baseline Weights..." -ForegroundColor Cyan
python -m ag_foundation train-cls --config configs/finetune_classification_imagenet.yaml

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Second training run failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "===============================================" -ForegroundColor Green
Write-Host " All Benchmarks Completed Successfully!        " -ForegroundColor Green
Write-Host " Check the runs/ directory for the metrics.csv " -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green
