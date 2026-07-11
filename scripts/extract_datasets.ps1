# Dataset Extraction Script
# Run once before starting any benchmark training

param(
    [switch]$SkipPlantCLEF  # PlantCLEF is 7.5GB tar — pass -SkipPlantCLEF to skip
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Green
Write-Host "  AG-Foundation: Extract Evaluation Datasets" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

$evalRoot = "E:\AG_Dataset\Evaluation"

# 1. PlantSeg Disease Detection (OBB)
$plantSegZip = "$evalRoot\PlantSeg Disease Detection.v2i.yolov8-obb.zip"
$plantSegDst  = "$evalRoot\PlantSeg"
if (-not (Test-Path "$plantSegDst\train")) {
    Write-Host "[1/4] Extracting PlantSeg Disease Detection..." -ForegroundColor Cyan
    Expand-Archive -Path $plantSegZip -DestinationPath $plantSegDst -Force
    Write-Host "  -> Extracted to $plantSegDst" -ForegroundColor Green
} else {
    Write-Host "[1/4] PlantSeg already extracted. Skipping." -ForegroundColor Yellow
}

# 2. Corn Kernel Counting
$cornZip = "$evalRoot\corn-kernel-counting\corn_kenel_counting_dataset.zip"
$cornDst  = "$evalRoot\corn-kernel-counting\data"
if (-not (Test-Path $cornDst)) {
    Write-Host "[2/4] Extracting Corn Kernel Counting dataset..." -ForegroundColor Cyan
    Expand-Archive -Path $cornZip -DestinationPath $cornDst -Force
    Write-Host "  -> Extracted to $cornDst" -ForegroundColor Green
} else {
    Write-Host "[2/4] Corn dataset already extracted. Skipping." -ForegroundColor Yellow
}

# 3. Longitudinal Nutrient Deficiency
$nutrientZip = "$evalRoot\longitudinal-nutrient-deficiency\Longitudinal_Nutrient_Deficiency.zip"
$nutrientDst  = "$evalRoot\longitudinal-nutrient-deficiency\data"
if (-not (Test-Path $nutrientDst)) {
    Write-Host "[3/4] Extracting Longitudinal Nutrient Deficiency dataset..." -ForegroundColor Cyan
    Expand-Archive -Path $nutrientZip -DestinationPath $nutrientDst -Force
    Write-Host "  -> Extracted to $nutrientDst" -ForegroundColor Green
} else {
    Write-Host "[3/4] Nutrient dataset already extracted. Skipping." -ForegroundColor Yellow
}

# 4. PlantCLEF2025test (test-only, large 7.5GB tar)
$plantCLEFTar = "$evalRoot\PlantCLEF2025test.tar"
$plantCLEFDst = "$evalRoot\PlantCLEF2025test"
if (-not $SkipPlantCLEF) {
    if (-not (Test-Path $plantCLEFDst) -or (Get-ChildItem $plantCLEFDst).Count -eq 0) {
        Write-Host "[4/4] Extracting PlantCLEF2025test (7.5 GB - this will take a while)..." -ForegroundColor Cyan
        New-Item -ItemType Directory -Force -Path $plantCLEFDst | Out-Null
        tar -xf $plantCLEFTar -C $plantCLEFDst
        Write-Host "  -> Extracted to $plantCLEFDst" -ForegroundColor Green
    } else {
        Write-Host "[4/4] PlantCLEF2025test already extracted. Skipping." -ForegroundColor Yellow
    }
} else {
    Write-Host "[4/4] PlantCLEF2025test extraction skipped (-SkipPlantCLEF)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  All datasets ready!" -ForegroundColor Green
Write-Host "  Next step: .\scripts\run_all_benchmarks.ps1" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
