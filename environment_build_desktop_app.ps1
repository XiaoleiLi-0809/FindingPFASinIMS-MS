param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

& $Python -m pip install -r requirements.txt
& $Python -m unittest discover -s tests -v
& $Python -m PyInstaller --noconfirm --clean pfas_screening_app.spec

Write-Host ""
Write-Host "Executable created at:"
Write-Host (Join-Path $root "dist\PFAS_CCS_Screening\PFAS_CCS_Screening.exe")
