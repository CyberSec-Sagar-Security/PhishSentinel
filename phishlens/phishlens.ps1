<#
.SYNOPSIS
    PhishLens — One-click PowerShell launcher for Windows
.DESCRIPTION
    Provides a menu-driven interface to all PhishLens operations:
    setup, training, web app, testing, and cleanup.
.EXAMPLE
    .\phishlens.ps1                  # Interactive menu
    .\phishlens.ps1 -Action setup    # Direct setup
    .\phishlens.ps1 -Action app      # Launch Streamlit
    .\phishlens.ps1 -Action train    # Quick train
    .\phishlens.ps1 -Action test     # Run tests
#>

param(
    [ValidateSet("menu","setup","download","app","train","test","verify","clean")]
    [string]$Action = "menu"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Paths ────────────────────────────────────────────────────────────────────
$ScriptDir  = $PSScriptRoot
$VenvDir    = Join-Path $ScriptDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip    = Join-Path $VenvDir "Scripts\pip.exe"
$Req        = Join-Path $ScriptDir "requirements.txt"
$AppFile    = Join-Path $ScriptDir "app.py"
$TrainFile  = Join-Path $ScriptDir "train.py"

# ── Colours ──────────────────────────────────────────────────────────────────
function Write-Color([string]$msg, [string]$color="White") {
    Write-Host $msg -ForegroundColor $color
}
function Write-Header([string]$msg) {
    Write-Color "`n$('='*60)" Cyan
    Write-Color "  $msg" Cyan
    Write-Color "$('='*60)" Cyan
}
function Write-OK([string]$msg)   { Write-Host "  [OK] " -ForegroundColor Green  -NoNewline; Write-Host $msg }
function Write-WARN([string]$msg) { Write-Host "  [!!] " -ForegroundColor Yellow -NoNewline; Write-Host $msg }
function Write-FAIL([string]$msg) { Write-Host "  [XX] " -ForegroundColor Red    -NoNewline; Write-Host $msg }

# ── Venv helpers ─────────────────────────────────────────────────────────────
function Test-Venv { Test-Path $VenvPython }

function New-Venv {
    Write-Header "Creating virtual environment"
    python -m venv $VenvDir
    Write-OK "Virtual environment created at $VenvDir"
}

function Invoke-Venv {
    if (-not (Test-Venv)) { New-Venv }
    $activate = Join-Path $VenvDir "Scripts\Activate.ps1"
    & $activate
}

# ─────────────────────────────────────────────────────────────────────────────
# ACTION: setup
# ─────────────────────────────────────────────────────────────────────────────
function Invoke-Setup {
    Write-Header "PhishLens Full Setup"

    # 1. Python check
    $pyVer = python --version 2>&1
    Write-OK "System Python: $pyVer"

    # 2. Venv
    if (-not (Test-Venv)) { New-Venv }
    else { Write-OK "Virtual environment already exists" }

    # 3. Upgrade pip
    Write-Header "Upgrading pip / setuptools / wheel"
    & $VenvPython -m pip install --upgrade pip setuptools wheel
    Write-OK "pip upgraded"

    # 4. Install requirements
    Write-Header "Installing requirements (may take 5-10 min first time)"
    & $VenvPython -m pip install -r $Req --no-warn-script-location
    if ($LASTEXITCODE -ne 0) {
        Write-FAIL "pip install failed — check errors above"
        exit 1
    }
    Write-OK "All packages installed"

    # 5. .env
    $envFile    = Join-Path $ScriptDir ".env"
    $envExample = Join-Path $ScriptDir ".env.example"
    if (-not (Test-Path $envFile)) {
        if (Test-Path $envExample) {
            Copy-Item $envExample $envFile
            Write-WARN ".env copied from .env.example — fill in your API keys!"
        }
    } else {
        Write-OK ".env exists"
    }

    # 6. Directories
    @("data\raw","models","reports") | ForEach-Object {
        $d = Join-Path $ScriptDir $_
        if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
        Write-OK "Directory: $_"
    }

    # 7. Quick verification
    Write-Header "Verifying core imports"
    $checks = @("numpy","pandas","sklearn","xgboost","lightgbm","catboost",
                "torch","streamlit","mlflow","shap","lime","loguru")
    $pass = 0 ; $fail = 0
    foreach ($pkg in $checks) {
        & $VenvPython -c "import $pkg" *> $null
        $rc = $LASTEXITCODE
        if ($rc -eq 0) { Write-OK $pkg ; $pass++ }
        else           { Write-FAIL $pkg ; $fail++ }
    }

    Write-Header "Setup Complete"
    Write-Color "  Libraries: $pass OK  /  $fail FAILED" $(if($fail -eq 0){"Green"}else{"Red"})
    if ($fail -eq 0) {
        Write-Color "`n  PhishLens is ready!  Run:  .\phishlens.ps1 -Action app" Green
    } else {
        Write-WARN "Some imports failed. Try re-running setup or check requirements.txt."
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# ACTION: app  — launch Streamlit
# ─────────────────────────────────────────────────────────────────────────────
function Invoke-App {
    Write-Header "Launching PhishLens Web Interface"
    if (-not (Test-Venv)) {
        Write-WARN "Virtual environment not found. Running setup first..."
        Invoke-Setup
    }
    Write-OK "Starting Streamlit on http://localhost:8501"
    Write-WARN "Press Ctrl+C to stop the server"
    Set-Location $ScriptDir
    & $VenvPython -m streamlit run $AppFile --server.port 8501
}

# ─────────────────────────────────────────────────────────────────────────────
# ACTION: download — download training datasets
# ─────────────────────────────────────────────────────────────────────────────
function Invoke-Download {
    Write-Header "PhishLens Dataset Downloader"
    if (-not (Test-Venv)) {
        Write-WARN "Run .\phishlens.ps1 -Action setup first"
        exit 1
    }
    Set-Location $ScriptDir
    Write-OK "Starting download of SpamAssassin + Enron + phishing_pot + Umbrella datasets..."
    Write-WARN "This may take several minutes depending on your connection."
    & $VenvPython (Join-Path $ScriptDir "download_datasets.py")
    if ($LASTEXITCODE -eq 0) {
        Write-OK "All datasets downloaded and processed successfully!"
        Write-Color "`n  Run training:  .\phishlens.ps1 -Action train" Cyan
    } else {
        Write-FAIL "Dataset download encountered errors. Check output above."
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# ACTION: train — quick training run
# ─────────────────────────────────────────────────────────────────────────────
function Invoke-Train {
    Write-Header "PhishLens Training"
    if (-not (Test-Venv)) {
        Write-WARN "Run .\phishlens.ps1 -Action setup first"
        exit 1
    }

    # Prefer pre-built processed CSV from download_datasets.py
    $ProcessedTrain = Join-Path $ScriptDir "data\processed\train.csv"
    if (Test-Path $ProcessedTrain) {
        $DataDir = Join-Path $ScriptDir "data\processed"
        Write-OK "Using pre-built dataset: data/processed/train.csv"
    } else {
        $DataDir = Join-Path $ScriptDir "data\raw"
        Write-WARN "No processed dataset found. For best results, run: .\phishlens.ps1 -Action download"
        Write-WARN "Data directory: $DataDir"
    }

    Write-Host ""
    Write-Host "  Available training modes:" -ForegroundColor Cyan
    Write-Host "  [1] Quick train (XGBoost only, no Optuna, no network)"
    Write-Host "  [2] Full train  (all models + Optuna, ~30 min)"
    Write-Host "  [3] Offline smoke test (synthetic data, no real dataset needed)"
    Write-Host ""
    $choice = Read-Host "Select mode (1/2/3)"

    switch ($choice) {
        "1" {
            & $VenvPython $TrainFile --data-dir $DataDir --models xgboost --no-network --eval --save models
        }
        "2" {
            & $VenvPython $TrainFile --data-dir $DataDir --models all --tune --eval --adversarial --save models
        }
        "3" {
            Write-OK "Running pipeline smoke test with synthetic emails..."
            & $VenvPython -c @"
import sys, os
sys.path.insert(0, r'$ScriptDir')
os.chdir(r'$ScriptDir')
import pandas as pd, numpy as np
from src.features.pipeline import FeaturePipeline
from src.utils.config import DEFAULT_CONFIG

emails = [
    {'raw_email': 'From: phish@evil.xyz\nSubject: Urgent!\n\nClick http://steal.xyz/', 'label': 1},
    {'raw_email': 'From: info@legit.com\nSubject: Newsletter\n\nHello!', 'label': 0},
] * 5
df = pd.DataFrame(emails)
pipe = FeaturePipeline(DEFAULT_CONFIG, use_network=False, use_intelligence_apis=False, use_gemini=False, use_tfidf=False)
X, names = pipe.fit_transform(df)
print(f'Feature matrix shape: {X.shape}')
print(f'Feature count      : {len(names)}')
print(f'NaN values         : {np.isnan(X).sum()}')
print('Smoke test PASSED' if X.shape[0] == 10 and not np.isnan(X).any() else 'FAILED')
"@
        }
        default {
            Write-WARN "Invalid choice"
        }
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# ACTION: test — run pytest
# ─────────────────────────────────────────────────────────────────────────────
function Invoke-Tests {
    Write-Header "Running Test Suite"
    if (-not (Test-Venv)) {
        Write-WARN "Run .\phishlens.ps1 -Action setup first"
        exit 1
    }
    Set-Location $ScriptDir
    & $VenvPython -m pytest tests/ -v --tb=short --timeout=60 `
        --cov=src --cov-report=term-missing
}

# ─────────────────────────────────────────────────────────────────────────────
# ACTION: verify — full verification (imports, tests, API keys)
# ─────────────────────────────────────────────────────────────────────────────
function Invoke-Verify {
    Write-Header "PhishLens Full Verification"
    Set-Location $ScriptDir
    & $VenvPython (Join-Path $ScriptDir "install_and_verify.py") --verify-only
}

# ─────────────────────────────────────────────────────────────────────────────
# ACTION: clean — remove venv and caches
# ─────────────────────────────────────────────────────────────────────────────
function Invoke-Clean {
    Write-Header "Cleaning up"
    Write-WARN "This will delete the virtual environment and all __pycache__ folders."
    $confirm = Read-Host "Type YES to confirm"
    if ($confirm -eq "YES") {
        if (Test-Path $VenvDir) { Remove-Item -Recurse -Force $VenvDir ; Write-OK ".venv removed" }
        Get-ChildItem -Path $ScriptDir -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
        Write-OK "__pycache__ folders removed"
        Get-ChildItem -Path $ScriptDir -Recurse -Filter "*.pyc" | Remove-Item -Force
        Write-OK ".pyc files removed"
    } else {
        Write-WARN "Cancelled"
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# ACTION: menu — interactive
# ─────────────────────────────────────────────────────────────────────────────
function Show-Menu {
    Clear-Host
    Write-Color @"

  ██████╗ ██╗  ██╗██╗███████╗██╗  ██╗██╗     ███████╗███╗   ██╗███████╗
  ██╔══██╗██║  ██║██║██╔════╝██║  ██║██║     ██╔════╝████╗  ██║██╔════╝
  ██████╔╝███████║██║███████╗███████║██║     █████╗  ██╔██╗ ██║███████╗
  ██╔═══╝ ██╔══██║██║╚════██║██╔══██║██║     ██╔══╝  ██║╚██╗██║╚════██║
  ██║     ██║  ██║██║███████║██║  ██║███████╗███████╗██║ ╚████║███████║
  ╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝
                ML Phishing Email Detection System
"@ Cyan

    $venvStatus = if (Test-Venv) { "[READY]" } else { "[NOT SET UP]" }
    Write-Color "  Virtual env: $venvStatus`n" $(if (Test-Venv) { "Green" } else { "Red" })

    Write-Host "  [1] Full Setup (install dependencies + verify)"
    Write-Host "  [2] Download Datasets (SpamAssassin, Enron, phishing_pot)"
    Write-Host "  [3] Launch Web Interface (Streamlit)"
    Write-Host "  [4] Train Models"
    Write-Host "  [5] Run Tests"
    Write-Host "  [6] Verify (imports, tests, API keys)"
    Write-Host "  [7] Clean (remove venv + caches)"
    Write-Host "  [0] Exit"
    Write-Host ""

    $choice = Read-Host "  Select option"
    switch ($choice) {
        "1" { Invoke-Setup }
        "2" { Invoke-Download }
        "3" { Invoke-App }
        "4" { Invoke-Train }
        "5" { Invoke-Tests }
        "6" { Invoke-Verify }
        "7" { Invoke-Clean }
        "0" { exit 0 }
        default { Write-WARN "Invalid option" }
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────
switch ($Action) {
    "menu"     { Show-Menu }
    "setup"    { Invoke-Setup }
    "download" { Invoke-Download }
    "app"      { Invoke-App }
    "train"    { Invoke-Train }
    "test"     { Invoke-Tests }
    "verify"   { Invoke-Verify }
    "clean"    { Invoke-Clean }
}
