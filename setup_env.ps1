<#
.SYNOPSIS
    Creates the Windows virtual environment (.venv) for the network fault diagnosis project.

.DESCRIPTION
    Run from the project root:
        powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
#>
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "==> Python version:" -ForegroundColor Cyan
& $Python --version

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "==> Creating virtual environment in .venv" -ForegroundColor Cyan
    & $Python -m venv .venv
}

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Host "==> Upgrading pip" -ForegroundColor Cyan
& $venvPy -m pip install --upgrade pip

Write-Host "==> Installing requirements" -ForegroundColor Cyan
& $venvPy -m pip install -r requirements.txt

Write-Host "==> Verifying install" -ForegroundColor Cyan
& $venvPy -c "import sklearn, numpy, pandas, streamlit; print('scikit-learn', sklearn.__version__, '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| streamlit', streamlit.__version__)"

Write-Host ""
Write-Host "Done. Activate the environment with:" -ForegroundColor Green
Write-Host "    .\.venv\Scripts\Activate.ps1"
