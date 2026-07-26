# run_downstream_benchmark.ps1
#
# Runs the full downstream classification benchmark comparing:
#   - DINO pre-trained backbone (your AG-Foundation model)
#   - ImageNet pre-trained backbone (standard timm baseline)
#   - Random initialization (lower bound)
#
# Evaluated on:
#   - Classification_Medicinal_Plant  (species classification)
#   - PlantSeg                        (plant disease classification from OBB dataset)
#
# Usage (from repo root):
#   conda activate ag-foundation
#   $env:PYTHONPATH = "src"
#   .\scripts\run_downstream_benchmark.ps1
#
# Results are saved per run in runs/benchmark/<dataset>_<init>/metrics.csv
# A final comparison table is printed at the end.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot  = $PSScriptRoot | Split-Path -Parent
$PYTHON    = "python"
$SRC       = "$RepoRoot\src"

if ($env:PYTHONPATH -notmatch [regex]::Escape($SRC)) {
    $env:PYTHONPATH = $SRC
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run one train-cls job and return best val_acc1
# ─────────────────────────────────────────────────────────────────────────────
function Run-Benchmark {
    param(
        [string]$Label,
        [string]$ConfigPath
    )

    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  RUNNING: $Label" -ForegroundColor Cyan
    Write-Host "  Config : $ConfigPath" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan

    & $PYTHON -m ag_foundation train-cls --config $ConfigPath | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "[$Label] Training exited with code $LASTEXITCODE - checking for any existing metrics.csv."
    }

    # Re-parse via a temp script for safety
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

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Prepare PlantSeg classification layout (idempotent)
# ─────────────────────────────────────────────────────────────────────────────
$PlantSegCls = "E:\AG_Dataset\01_Evaluation\PlantSeg_cls"
if (-not (Test-Path $PlantSegCls)) {
    Write-Host ""
    Write-Host "[PREP] Converting PlantSeg OBB dataset to ImageFolder layout..." -ForegroundColor Yellow
    & $PYTHON "$RepoRoot\scripts\prepare_plantseg_cls.py"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "PlantSeg preparation failed."
        exit 1
    }
    Write-Host "[PREP] Done." -ForegroundColor Green
} else {
    Write-Host "[PREP] PlantSeg_cls already exists, skipping preparation." -ForegroundColor Green
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Run all 6 benchmark jobs
# ─────────────────────────────────────────────────────────────────────────────
$results = @{}

$jobs = @(
    @{ label = "MedicinalPlant | DINO (ours)";  config = "configs\benchmark_medicinal_plant_dino.yaml" },
    @{ label = "MedicinalPlant | ImageNet";      config = "configs\benchmark_medicinal_plant_imagenet.yaml" },
    @{ label = "MedicinalPlant | Random Init";   config = "configs\benchmark_medicinal_plant_random.yaml" },
    @{ label = "PlantSeg       | DINO (ours)";   config = "configs\benchmark_plantseg_dino.yaml" },
    @{ label = "PlantSeg       | ImageNet";      config = "configs\benchmark_plantseg_imagenet.yaml" },
    @{ label = "PlantSeg       | Random Init";   config = "configs\benchmark_plantseg_random.yaml" }
)

foreach ($job in $jobs) {
    $cfgFull = Join-Path $RepoRoot $job.config
    $acc = Run-Benchmark -Label $job.label -ConfigPath $cfgFull
    $results[$job.label] = if ($null -eq $acc) { "ERROR" } else { $acc }
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Print final comparison table
# ─────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Green
Write-Host "  BENCHMARK RESULTS SUMMARY" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Green
Write-Host ""
Write-Host ("{0,-45} {1,12}" -f "Experiment", "Best Val Acc1") -ForegroundColor White
Write-Host ("-" * 60)
foreach ($key in $results.Keys | Sort-Object) {
    $acc = $results[$key]
    $color = "White"
    if ($key -match "DINO") { $color = "Green" }
    elseif ($key -match "ImageNet") { $color = "Yellow" }
    else { $color = "Gray" }
    Write-Host ("{0,-45} {1,12}%" -f $key, $acc) -ForegroundColor $color
}
Write-Host ""
Write-Host "Results CSVs saved in: $RepoRoot\runs\benchmark\" -ForegroundColor Cyan

# Also save summary to file
$summaryPath = "$RepoRoot\runs\benchmark\benchmark_summary.txt"
$null = New-Item -ItemType Directory -Force -Path "$RepoRoot\runs\benchmark"
$lines = @("BENCHMARK RESULTS - $(Get-Date -Format 'yyyy-MM-dd HH:mm')", "=" * 60)
foreach ($key in $results.Keys | Sort-Object) {
    $lines += ("{0,-45} {1,8}%" -f $key, $results[$key])
}
$lines | Set-Content -Path $summaryPath -Encoding UTF8
Write-Host "Summary also saved to: $summaryPath" -ForegroundColor Cyan
