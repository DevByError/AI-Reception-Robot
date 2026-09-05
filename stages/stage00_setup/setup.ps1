# Stage 00 - automated venv setup (Windows / dev machine)
# Run from anywhere: .\setup.ps1  (or ..\..\stages\stage00_setup\setup.ps1 from project root)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param([string]$Description, [scriptblock]$Command)
    Write-Host "-> $Description"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Description (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot

Write-Host "== Stage 00 setup (Windows) ==" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"

if (-Not (Test-Path ".venv")) {
    Invoke-Checked "Creating new venv at .venv ..." { python -m venv .venv }
} else {
    Write-Host "-> .venv already exists, skipping creation."
}

# Use 'python -m pip' (not pip.exe directly) so pip can safely upgrade itself on Windows.
Invoke-Checked "Upgrading pip..." { & ".\.venv\Scripts\python.exe" -m pip install --upgrade pip }
Invoke-Checked "Installing requirements..." {
    & ".\.venv\Scripts\python.exe" -m pip install -r "stages\stage00_setup\requirements.txt"
}

Write-Host ""
Write-Host "Done - venv ready and requirements installed." -ForegroundColor Green
Write-Host "To activate the venv in any new terminal:"
Write-Host "   .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then run the hardware check:"
Write-Host "   python stages\stage00_setup\test_hardware.py"
