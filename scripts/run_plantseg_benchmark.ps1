# scripts\run_plantseg_benchmark.ps1
#
# Runs the full downstream classification benchmark on PlantSeg_cls (298 agricultural disease classes)
# comparing:
#   1. DINO Pre-trained LP-FT (Ours)       -> configs/benchmark_plantseg_dino.yaml (5 epochs frozen LP -> LLRD fine-tuning)
#   2. ImageNet Official Baseline Weights  -> configs/benchmark_plantseg_imagenet.yaml (standard fine-tuning)
#   3. Random Initialization (Lower Bound) -> configs/benchmark_plantseg_random.yaml (from-scratch training)
#
# Usage (from repo root):
#   conda activate ag-foundation
#   $env:PYTHONPATH = "src"
#   .\scripts\run_plantseg_benchmark.ps1
#
# Features:
#   - Full modern interactive progress bar (command_progress_context)
#   - Automatic metrics parsing and summary table generation

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot  = $PSScriptRoot | Split-Path -Parent
$PYTHON    = "python"
$SRC       = "$RepoRoot\src"

if ($env:PYTHONPATH -notmatch [regex]::Escape($SRC)) {
    $env:PYTHONPATH = $SRC
}

function Run-PlantSegJob {
    param(
        [string]$Label,
        [string]$ConfigPath
    )

    Write-Host "`n======================================================================" -ForegroundColor Cyan
    Write-Host "  RUNNING BENCHMARK: $Label" -ForegroundColor Cyan
    Write-Host "  Config           : $ConfigPath" -ForegroundColor Cyan
    Write-Host "======================================================================" -ForegroundColor Cyan

    & $PYTHON -m ag_foundation train-cls --config $ConfigPath
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "[$Label] Training exited with code $LASTEXITCODE. Checking if metrics exist..."
    }

    $tmpScript = [System.IO.Path]::GetTempFileName() + ".py"
    $pyCode = @"
import sys, csv, pathlib, yaml
cfg_path = r'$ConfigPath'
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
out_dir = pathlib.Path(cfg['runtime']['output_dir'])
metrics_path = out_dir / 'metrics.csv'
if not metrics_path.exists():
    print('N/A')
    sys.exit(0)
best = 0.0
with open(metrics_path) as f:
    for row in csv.DictReader(f):
        v = float(row.get('val_acc1', 0))
        if v > best:
            best = v
print(f'{best:.2f}')
"@
    Set-Content -Path $tmpScript -Value $pyCode -Encoding UTF8
    $bestAcc = & $PYTHON $tmpScript
    Remove-Item $tmpScript -Force
    return $bestAcc.Trim()
}

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host " STARTING PLANTSEG DISEASE CLASSIFICATION DOWNSTREAM BENCHMARK       " -ForegroundColor Green
Write-Host " Dataset: E:\AG_Dataset\01_Evaluation\PlantSeg_cls (298 classes)     " -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green

$results = @{}

$jobs = @(
    @{ label = "PlantSeg | DINO LP-FT (Ours)";      config = "configs\benchmark_plantseg_dino.yaml" },
    @{ label = "PlantSeg | ImageNet Baseline";       config = "configs\benchmark_plantseg_imagenet.yaml" },
    @{ label = "PlantSeg | Random Init Baseline";    config = "configs\benchmark_plantseg_random.yaml" }
)

foreach ($job in $jobs) {
    $cfgFull = Join-Path $RepoRoot $job.config
    $acc = Run-PlantSegJob -Label $job.label -ConfigPath $cfgFull
    $results[$job.label] = if ($null -eq $acc) { "ERROR" } else { $acc }
}

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host "  FINAL PLANTSEG COMPARISON TABLE                                    " -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host ""
Write-Host ("{0,-40} {1,14}" -f "Method / Initialization", "Best Val Acc1%") -ForegroundColor White
Write-Host ("-" * 56)
foreach ($key in $results.Keys | Sort-Object) {
    $acc = $results[$key]
    $color = "White"
    if ($key -match "DINO") { $color = "Green" }
    elseif ($key -match "ImageNet") { $color = "Yellow" }
    else { $color = "Gray" }
    Write-Host ("{0,-40} {1,14}%" -f $key, $acc) -ForegroundColor $color
}
Write-Host ""

$summaryDir = "$RepoRoot\runs\benchmark"
$null = New-Item -ItemType Directory -Force -Path $summaryDir
$summaryPath = "$summaryDir\plantseg_benchmark_summary.txt"
$lines = @(
    "PLANTSEG BENCHMARK RESULTS - $(Get-Date -Format 'yyyy-MM-dd HH:mm')",
    "=" * 56,
    ("{0,-40} {1,14}" -f "Method / Initialization", "Best Val Acc1%"),
    ("-" * 56)
)
foreach ($key in $results.Keys | Sort-Object) {
    $lines += ("{0,-40} {1,14}%" -f $key, $results[$key])
}
$lines | Set-Content -Path $summaryPath -Encoding UTF8
Write-Host "Summary saved to: $summaryPath" -ForegroundColor Cyan
Write-Host "All individual metrics CSVs saved in: $summaryDir\" -ForegroundColor Cyan
