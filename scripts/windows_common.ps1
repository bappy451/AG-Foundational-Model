Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-PreferredPython {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CodeDir,
        [string]$RequestedPython = ""
    )

    if (-not [string]::IsNullOrWhiteSpace($RequestedPython)) {
        return $RequestedPython
    }

    $venvPython = Join-Path $CodeDir ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }

    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
        $condaPython = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path $condaPython) {
            return $condaPython
        }
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $pyLauncher) {
        return $pyLauncher.Source
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $python) {
        return $python.Source
    }

    throw "No Python interpreter was found. Create .venv first or pass --python PATH."
}

function Assert-PythonModules {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonExe,
        [Parameter(Mandatory = $true)]
        [string[]]$Modules
    )

    $moduleList = $Modules -join ","
    $script = @"
import importlib.util
import sys

missing = [name for name in sys.argv[1].split(",") if importlib.util.find_spec(name) is None]
if missing:
    print(", ".join(missing))
    raise SystemExit(1)
"@
    $output = & $PythonExe -c $script $moduleList 2>&1
    if ($LASTEXITCODE -ne 0) {
        $missing = ($output | Out-String).Trim()
        if ([string]::IsNullOrWhiteSpace($missing)) {
            $missing = $moduleList
        }
        throw "Selected Python is missing required module(s): $missing. Python: $PythonExe. Install dependencies with: $PythonExe -m pip install -e '.[dev,ml]'"
    }
}
