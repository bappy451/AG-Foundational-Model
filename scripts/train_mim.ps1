Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "windows_common.ps1")

function Show-Usage {
@'
Usage:
  .\scripts\train_mim.ps1 --config PATH [--python PATH]
  .\scripts\train_mim.ps1 --data-root PATH --output-dir PATH [options] [--python PATH]

What it does:
  - launches the unified ag-foundation CLI
  - trains a MAE-style masked image modeling run
  - supports RGB and multispectral data on cpu, cuda, or mps
  - can append command output to command.log for reproducibility

Options:
  --log-file PATH       Append wrapper logs to a custom file.
  --no-log              Disable wrapper logging for one run.

Examples:
  .\scripts\train_mim.ps1 --config .\configs\train_mim.example.yaml
  .\scripts\train_mim.ps1 --config .\configs\pretraining_seedlings_smoke.yaml --resume
  .\scripts\train_mim.ps1 --data-root D:\data\ag.zip --output-dir .\runs\ag-mim --channels 4 --precision fp16 --model-name S
'@ | Write-Output
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$codeDir = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))
$defaultLogFile = Join-Path $codeDir "command.log"
$invocationCwd = (Get-Location).Path
$originalArgs = @($args)
$pythonExe = ""
$showUsage = $false
$logFile = $defaultLogFile
$loggingEnabled = $true
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$forwardArgs = New-Object System.Collections.Generic.List[string]

$index = 0
while ($index -lt $args.Count) {
    $token = $args[$index]
    switch -Regex ($token) {
        '^--python=(.*)$' {
            $pythonExe = $Matches[1]
            $index += 1
            continue
        }
        '^--python$' {
            if ($index + 1 -ge $args.Count -or $args[$index + 1].StartsWith('-')) { throw "--python requires a path." }
            $pythonExe = $args[$index + 1]
            $index += 2
            continue
        }
        '^--log-file=(.*)$' {
            $logFile = $Matches[1]
            $index += 1
            continue
        }
        '^--log-file$' {
            if ($index + 1 -ge $args.Count -or $args[$index + 1].StartsWith('-')) { throw "--log-file requires a path." }
            $logFile = $args[$index + 1]
            $index += 2
            continue
        }
        '^--no-log$' {
            $loggingEnabled = $false
            $index += 1
            continue
        }
        '^(-h|--help|-\\?)$' {
            $showUsage = $true
            $index += 1
            continue
        }
        default {
            $forwardArgs.Add($token) | Out-Null
            $index += 1
            continue
        }
    }
}

if ($showUsage) {
    Show-Usage
    exit 0
}

$pythonExe = Resolve-PreferredPython -CodeDir $codeDir -RequestedPython $pythonExe
Assert-PythonModules -PythonExe $pythonExe -Modules @("torch", "timm")
$pythonScript = Join-Path $scriptDir "ag_foundation.py"
$commandArgs = @($pythonScript, "train-mim") + @($forwardArgs)
$env:AG_FOUNDATION_WRAPPER_LOGGING = "1"

if ($loggingEnabled) {
    $logDir = Split-Path -Parent $logFile
    if (-not [string]::IsNullOrWhiteSpace($logDir)) {
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    }

    $commandText = @($pythonExe) + $commandArgs
    Add-Content -Path $logFile -Value ""
    Add-Content -Path $logFile -Value "================================================================================"
    Add-Content -Path $logFile -Value "Command Log"
    Add-Content -Path $logFile -Value "================================================================================"
    Add-Content -Path $logFile -Value ("Started   : " + (Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"))
    Add-Content -Path $logFile -Value ("Command   : " + ($commandText -join " "))
    Add-Content -Path $logFile -Value ("CWD       : " + $invocationCwd)
    Add-Content -Path $logFile -Value ("Log file  : " + $logFile)
    Add-Content -Path $logFile -Value "================================================================================"
    Write-Output "[logging] Appending command output to $logFile"
    & $pythonExe @($commandArgs) 2>&1 | Tee-Object -FilePath $logFile -Append
    $exitCode = $LASTEXITCODE
    Add-Content -Path $logFile -Value ("[logging] Finished (exit={0}, finished={1}, duration={2:n2}s)" -f $exitCode, (Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"), $stopwatch.Elapsed.TotalSeconds)
    exit $exitCode
}

& $pythonExe @($commandArgs)
exit $LASTEXITCODE
