# Requires PowerShell 5+ or PowerShell 7+
$ErrorActionPreference = 'Stop'

Write-Host 'Setting up ProteoForge Analysis environment (Windows)'
Write-Host '==============================================='

# Verify project root
if (-not (Test-Path -Path 'requirements.txt')) {
    throw 'Run this script from the project root containing requirements.txt'
}

# Resolve Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { $python = Get-Command py -ErrorAction SilentlyContinue }
if (-not $python) { throw 'Python is not installed or not on PATH. Install Python 3.10+.' }

# Create venv if missing
if (-not (Test-Path -Path '.venv')) {
    Write-Host 'Creating Python virtual environment (.venv)'
    & $python.Path -m venv .venv
}

$venvPy = Join-Path '.venv' 'Scripts/python.exe'
if (-not (Test-Path -Path $venvPy)) {
    throw "Virtual environment Python not found at $venvPy"
}

Write-Host 'Installing Python packages from requirements.txt'
& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install -r requirements.txt

# R environment (requires Rscript on PATH)
$rscript = Get-Command Rscript -ErrorAction SilentlyContinue
if ($rscript) {
    Write-Host 'Configuring R environment via setup_env.R (renv + pak)'
    & $rscript.Path setup_env.R
} else {
    Write-Warning 'Rscript not found on PATH. Skipping R environment setup.'
    Write-Host 'Install R (>= 4.5.0) from https://cran.r-project.org/ and ensure Rscript is on PATH.'
}

Write-Host ''
Write-Host 'Setup complete'
Write-Host '--------------'
Write-Host 'Python venv: .venv\'
Write-Host 'R env (renv): renv\ (if R was available)'
Write-Host ''
Write-Host 'Activate Python (PowerShell): .\.venv\Scripts\Activate.ps1'
