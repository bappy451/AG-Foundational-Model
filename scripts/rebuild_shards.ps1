# rebuild_shards.ps1
# ==================
# Complete shard rebuild pipeline for AG-Foundation pretraining dataset.
#
# Prerequisites:
#   conda activate ag-foundation
#
# Usage:
#   .\scripts\rebuild_shards.ps1

param(
    [switch]$SkipCatalog,
    [switch]$SkipShardDelete,
    [switch]$SkipBuild,
    [switch]$Resume,
    [int]$Workers = 8
)

$ErrorActionPreference = "Stop"
$ROOT = "E:\AG_Dataset\AG-Foundational-Model"
$PRETRAINING_ROOT = "E:\AG_Dataset\AG-Foundational-Model\Pretraining"
$SHARDS_DIR = "E:\AG_Dataset\shards"
$CATALOG_V2 = "$PRETRAINING_ROOT\catalog_v2.csv"

function Write-Step {
    param([string]$Msg)
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  $Msg" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host ""
}

function Write-Ok  { param([string]$Msg); Write-Host "  [OK] $Msg" -ForegroundColor Green }
function Write-Warn{ param([string]$Msg); Write-Host "  [WARN] $Msg" -ForegroundColor Yellow }
function Write-Err { param([string]$Msg); Write-Host "  [ERROR] $Msg" -ForegroundColor Red }

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Magenta
Write-Host "  AG-Foundation: Rebuild Pretraining Shards" -ForegroundColor Magenta
Write-Host ("=" * 70) -ForegroundColor Magenta
Write-Host "  Pretraining root : $PRETRAINING_ROOT"
Write-Host "  Shards output    : $SHARDS_DIR"
Write-Host "  Workers          : $Workers"
Write-Host "  Resume mode      : $Resume"
Write-Host ("=" * 70) -ForegroundColor Magenta

if ($Resume) {
    Write-Err "Resume mode is disabled for shard rebuilds."
    Write-Warn "The previous resume logic could overwrite existing shard names and skip the wrong records."
    Write-Warn "Run without -Resume for a clean rebuild, or use -SkipBuild to verify existing shards only."
    exit 1
}

if (-not $SkipCatalog) {
    Write-Step "Step 1/3: Building clean catalog (catalog_v2.csv)"
    Write-Host "  This scans every ZIP and TAR archive. With the new native 'tar -tf'" -ForegroundColor Yellow
    Write-Host "  optimization, scanning the 600GB PlantCLEF archives now takes ~3-5 minutes" -ForegroundColor Yellow
    Write-Host "  instead of 2-4 hours. Please be patient while the index is built." -ForegroundColor Yellow
    Write-Host ""

    $catalogStart = Get-Date
    conda run -n ag-foundation python "$ROOT\scripts\build_pretraining_catalog.py" --pretraining-root $PRETRAINING_ROOT --output $CATALOG_V2

    if ($LASTEXITCODE -ne 0) {
        Write-Err "Catalog build failed with exit code $LASTEXITCODE. Aborting."
        exit 1
    }

    $catalogElapsed = ((Get-Date) - $catalogStart).TotalMinutes
    $catalogElapsedStr = [math]::Round($catalogElapsed, 1).ToString()
    Write-Ok "Catalog built in $catalogElapsedStr minutes -> $CATALOG_V2"
    Write-Host ""
    $lines = (Get-Content $CATALOG_V2 | Measure-Object -Line).Lines - 1
    Write-Host "  Total records in catalog_v2.csv: $lines" -ForegroundColor Green
}
else {
    Write-Warn "Skipping catalog build (-SkipCatalog). Using: $CATALOG_V2"
    if (-not (Test-Path $CATALOG_V2)) {
        Write-Err "catalog_v2.csv not found at $CATALOG_V2. Cannot continue."
        exit 1
    }
    $lines = (Get-Content $CATALOG_V2 | Measure-Object -Line).Lines - 1
    Write-Host "  Existing catalog records: $lines" -ForegroundColor Green
}

if (-not $SkipBuild) {
    if (-not $SkipShardDelete -and -not $Resume) {
        Write-Step "Step 2a/3: Clearing existing shards"
        $existingShards = @(Get-ChildItem "$SHARDS_DIR\*.tar" -ErrorAction SilentlyContinue)
        if ($existingShards.Count -gt 0) {
            $totalBytes = ($existingShards | Measure-Object -Property Length -Sum).Sum
            $totalGBStr = [math]::Round($totalBytes / 1e9, 1).ToString()
            $cnt = $existingShards.Count
            Write-Warn "About to DELETE $cnt existing shards ($totalGBStr GB)."
            Write-Host "  These are the OLD broken shards with scale variance issues." -ForegroundColor Yellow
            Write-Host "  Press Ctrl+C to cancel, or wait 10 seconds to proceed..." -ForegroundColor Yellow
            Start-Sleep -Seconds 10

            Write-Host "  Deleting old shards..."
            Remove-Item "$SHARDS_DIR\*.tar" -Force
            Write-Ok "Deleted $cnt old shards."
        }
        else {
            Write-Ok "No existing shards found -- directory is clean."
        }

        New-Item -ItemType Directory -Force -Path $SHARDS_DIR | Out-Null
    }
    elseif ($Resume) {
        Write-Warn "Resume mode: existing complete shards will be kept, partial shard will be deleted."
    }
    else {
        Write-Warn "-SkipShardDelete: existing shards will NOT be deleted."
    }

    Write-Step "Step 2b/3: Building new shards"
    Write-Host "  Pipeline: open image -> validate -> RGB fix -> bounded resize (1024px) -> JPEG encode"
    Write-Host "  All output images: RGB, max 1024px min-side, JPEG quality=92" -ForegroundColor Green
    Write-Host "  Scheduling: up to $Workers archive-focused workers; each worker finishes one archive/source before switching." -ForegroundColor Green
    Write-Host "  Live output: progress bar, active archive count, skip totals, and shard estimates." -ForegroundColor Green
    Write-Host "  This will take 6-12 hours on i9-14900KF with $Workers workers." -ForegroundColor Yellow
    Write-Host ""

    $buildStart = Get-Date
    $condaArgs = @(
        "run", "-n", "ag-foundation", "python",
        "$ROOT\scripts\..\src\ag_foundation\data\build_wds_shards.py",
        "--catalog", "$CATALOG_V2",
        "--pretraining-root", "$PRETRAINING_ROOT",
        "--output-prefix", "$SHARDS_DIR\dataset",
        "--max-size", "1000000000",
        "--max-count", "10000",
        "--workers", "$Workers",
        "--progress-every", "1000"
    )
    if ($Resume) {
        $condaArgs += "--resume"
    }
    
    & conda $condaArgs

    if ($LASTEXITCODE -ne 0) {
        Write-Err "Shard build failed with exit code $LASTEXITCODE."
        Write-Warn "Fix the issue above, then rerun the clean rebuild. Resume is intentionally disabled."
        exit 1
    }

    $buildElapsed = ((Get-Date) - $buildStart).TotalHours
    $buildElapsedStr = [math]::Round($buildElapsed, 2).ToString()
    Write-Ok "Shard build complete in $buildElapsedStr hours."
}
else {
    Write-Warn "Skipping shard build (-SkipBuild). Running verification only."
}

Write-Step "Step 3/3: Verifying shards"

$newShards = @(Get-ChildItem "$SHARDS_DIR\*.tar" -ErrorAction SilentlyContinue)
if ($newShards.Count -eq 0) {
    Write-Err "No shards found in $SHARDS_DIR after build. Something went wrong."
    exit 1
}

$totalBytesNew = ($newShards | Measure-Object -Property Length -Sum).Sum
$totalNewGBStr = [math]::Round($totalBytesNew / 1e9, 1).ToString()
$cntNew = $newShards.Count
Write-Host "  Found $cntNew shards ($totalNewGBStr GB) -- running verification..."
Write-Host "  Verification samples every shard and checks RGB, minimum size, and min-side resize policy." -ForegroundColor Green

conda run -n ag-foundation python "$ROOT\scripts\verify_shards.py" --shards-dir $SHARDS_DIR --target-size 1000000000

if ($LASTEXITCODE -ne 0) {
    Write-Err "Verification found issues! Check output above."
    Write-Warn "Do NOT start pretraining until shard issues are resolved."
    exit 1
}

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Green
Write-Host "  SHARD REBUILD COMPLETE" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Green
Write-Host "  Shards location : $SHARDS_DIR"
Write-Host "  Shard count     : $cntNew"
Write-Host "  Total size      : $totalNewGBStr GB"
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor Cyan
Write-Host "    Run a 5-epoch smoke test to confirm clean loss curve:"
Write-Host "    python -m ag_foundation train-dino --config configs/wds_dino_pretrain_v2.yaml"
Write-Host ""
Write-Host "    Or run full pretraining:"
Write-Host "    python -m ag_foundation train-dino --config configs/wds_dino_pretrain_v2.yaml"
Write-Host ("=" * 70) -ForegroundColor Green
