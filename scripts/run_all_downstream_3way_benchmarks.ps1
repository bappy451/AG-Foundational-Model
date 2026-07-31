# run_all_downstream_3way_benchmarks.ps1
#
# Runs a comprehensive 3-Way benchmark comparing:
#   1. DINO Pretrained LP-FT (Ours - AG-Foundation)
#   2. ImageNet Official Baseline
#   3. Random Initialization (From Scratch)
#
# Across ALL 5 evaluation datasets in E:\AG_Dataset\01_Evaluation:
#   - Task 1a: Classification_Medicinal_Plant (Species Classification)
#   - Task 1b: PlantSeg_cls (Agricultural Disease Classification, 298 classes)
#   - Task 2 : PlantSeg (Oriented Bounding Box Disease Detection)
#   - Task 3 : corn-kernel-counting (Spatial Regression / Counting)
#   - Task 4 : longitudinal-nutrient-deficiency (Temporal Progression Analysis)
#
# Automatically skips jobs that already have a completed metrics.csv.
#
# Usage:
#   conda activate ag-foundation
#   $env:PYTHONPATH = "src"
#   .\scripts\run_all_downstream_3way_benchmarks.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot  = $PSScriptRoot | Split-Path -Parent
$PYTHON    = "python"
$SRC       = "$RepoRoot\src"

if ($env:PYTHONPATH -notmatch [regex]::Escape($SRC)) {
    $env:PYTHONPATH = $SRC
}

Write-Host ""
Write-Host "==========================================================================" -ForegroundColor Green
Write-Host "   STARTING AG-FOUNDATION 3-WAY DOWNSTREAM BENCHMARK SUITE (5 DATASETS)   " -ForegroundColor Green
Write-Host "==========================================================================" -ForegroundColor Green
Write-Host ""

# STEP 1: Prepare PlantSeg classification layout if needed
$PlantSegCls = "E:\AG_Dataset\01_Evaluation\PlantSeg_cls"
if (-not (Test-Path $PlantSegCls)) {
    Write-Host "[PREP] Converting PlantSeg OBB dataset to ImageFolder layout..." -ForegroundColor Yellow
    & $PYTHON "$RepoRoot\scripts\prepare_plantseg_cls.py"
    Write-Host "[PREP] Done." -ForegroundColor Green
}

$jobs = @(
    # 1. Medicinal Plant Classification (train-cls)
    @{ task="train-cls"; label="MedicinalPlant | DINO (ours)";  config="configs\benchmark_medicinal_plant_dino.yaml";     outdir="E:\AG_Dataset\AG-Foundational-Model\runs\benchmark\medicinal_plant_dino" },
    @{ task="train-cls"; label="MedicinalPlant | ImageNet";      config="configs\benchmark_medicinal_plant_imagenet.yaml"; outdir="E:\AG_Dataset\AG-Foundational-Model\runs\benchmark\medicinal_plant_imagenet" },
    @{ task="train-cls"; label="MedicinalPlant | Random Init";   config="configs\benchmark_medicinal_plant_random.yaml";   outdir="E:\AG_Dataset\AG-Foundational-Model\runs\benchmark\medicinal_plant_random" },

    # 2. PlantSeg Agricultural Disease Classification (train-cls)
    @{ task="train-cls"; label="PlantSeg_cls   | DINO (ours)";   config="configs\benchmark_plantseg_dino.yaml";            outdir="E:\AG_Dataset\AG-Foundational-Model\runs\benchmark\plantseg_dino" },
    @{ task="train-cls"; label="PlantSeg_cls   | ImageNet";      config="configs\benchmark_plantseg_imagenet.yaml";         outdir="E:\AG_Dataset\AG-Foundational-Model\runs\benchmark\plantseg_imagenet" },
    @{ task="train-cls"; label="PlantSeg_cls   | Random Init";   config="configs\benchmark_plantseg_random.yaml";           outdir="E:\AG_Dataset\AG-Foundational-Model\runs\benchmark\plantseg_random" },

    # 3. PlantSeg OBB Disease Detection (train-det)
    @{ task="train-det"; label="PlantSeg_det   | DINO (ours)";   config="configs\benchmark_det_plantseg_dino.yaml";        outdir="E:\AG_Dataset\AG-Foundational-Model\runs\benchmark\plantseg_det_dino" },
    @{ task="train-det"; label="PlantSeg_det   | ImageNet";      config="configs\benchmark_det_plantseg_imagenet.yaml";     outdir="E:\AG_Dataset\AG-Foundational-Model\runs\benchmark\plantseg_det_imagenet" },
    @{ task="train-det"; label="PlantSeg_det   | Random Init";   config="configs\benchmark_det_plantseg_random.yaml";       outdir="E:\AG_Dataset\AG-Foundational-Model\runs\benchmark\plantseg_det_random" },

    # 4. Corn Kernel Counting (train-count)
    @{ task="train-count"; label="CornCounting   | DINO (ours)"; config="configs\benchmark_count_corn_dino.yaml";          outdir="E:\AG_Dataset\AG-Foundational-Model\runs\benchmark\count_corn_dino" },
    @{ task="train-count"; label="CornCounting   | ImageNet";    config="configs\benchmark_count_corn_imagenet.yaml";       outdir="E:\AG_Dataset\AG-Foundational-Model\runs\benchmark\count_corn_imagenet" },
    @{ task="train-count"; label="CornCounting   | Random Init"; config="configs\benchmark_count_corn_random.yaml";         outdir="E:\AG_Dataset\AG-Foundational-Model\runs\benchmark\count_corn_random" },

    # 5. Longitudinal Nutrient Deficiency (train-temporal)
    @{ task="train-temporal"; label="NutrientTemp   | DINO (ours)"; config="configs\benchmark_temporal_nutrient_dino.yaml";    outdir="E:\AG_Dataset\AG-Foundational-Model\runs\benchmark\temporal_nutrient_dino" },
    @{ task="train-temporal"; label="NutrientTemp   | ImageNet";    config="configs\benchmark_temporal_nutrient_imagenet.yaml"; outdir="E:\AG_Dataset\AG-Foundational-Model\runs\benchmark\temporal_nutrient_imagenet" },
    @{ task="train-temporal"; label="NutrientTemp   | Random Init"; config="configs\benchmark_temporal_nutrient_random.yaml";   outdir="E:\AG_Dataset\AG-Foundational-Model\runs\benchmark\temporal_nutrient_random" }
)

foreach ($job in $jobs) {
    $metricsPath = Join-Path $job.outdir "metrics.csv"
    if (Test-Path $metricsPath) {
        Write-Host ""
        Write-Host ("=" * 75) -ForegroundColor DarkGray
        Write-Host ("  [SKIP] TASK: {0} | {1} (Already completed: {2})" -f $job.task, $job.label, $metricsPath) -ForegroundColor DarkGray
        Write-Host ("=" * 75) -ForegroundColor DarkGray
        continue
    }

    Write-Host ""
    Write-Host ("=" * 75) -ForegroundColor Cyan
    Write-Host ("  TASK: {0} | RUNNING: {1}" -f $job.task, $job.label) -ForegroundColor Cyan
    Write-Host ("  Config: {0}" -f $job.config) -ForegroundColor Cyan
    Write-Host ("=" * 75) -ForegroundColor Cyan

    $env:PYTHONUNBUFFERED = "1"
    & $PYTHON -u -m ag_foundation ($job.task) --config ($job.config)
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "[$($job.label)] Training exited with code $LASTEXITCODE."
    }
}

Write-Host ""
Write-Host "==========================================================================" -ForegroundColor Green
Write-Host "   ALL 15 3-WAY BENCHMARK EXPERIMENTS COMPLETED ACROSS 5 DATASETS!        " -ForegroundColor Green
Write-Host "   Check E:\AG_Dataset\runs\benchmark\ for individual metrics.csv files   " -ForegroundColor Green
Write-Host "==========================================================================" -ForegroundColor Green
