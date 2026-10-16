# Convenience entry point: build stable-retro, import the SF2 ROM, verify, and
# optionally start training.
#
#   powershell -ExecutionPolicy Bypass -File build_and_play.ps1
#   powershell -ExecutionPolicy Bypass -File build_and_play.ps1 -Timesteps 200000
#   powershell -ExecutionPolicy Bypass -File build_and_play.ps1 -SkipBuild
#
# All the real work lives in build_stable_retro.ps1, setup_rom.py,
# verify_setup.py and train_sf2.py.
#
[CmdletBinding()]
param(
    [int]$Timesteps = 0,
    [switch]$SkipBuild,
    [switch]$SkipRlDeps
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

$python = ""
foreach ($candidate in @(
        "C:\Users\HEC\miniconda3\python.exe",
        (Join-Path $env:LOCALAPPDATA "Python\bin\python.exe"))) {
    if (Test-Path $candidate) { $python = $candidate; break }
}
if (-not $python) {
    $cmd = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($cmd) { $python = $cmd.Source }
}
if (-not $python -or -not (Test-Path $python)) {
    Write-Host "[X] No Python interpreter found." -ForegroundColor Red
    exit 1
}

if (-not $SkipBuild) {
    $buildArgs = @()
    if (-not $SkipRlDeps) { $buildArgs += "-RlDeps" }
    & (Join-Path $root "build_stable_retro.ps1") @buildArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "`n=== Importing the Street Fighter II ROM ===" -ForegroundColor Cyan
& $python (Join-Path $root "setup_rom.py")

Write-Host "`n=== Verifying the setup ===" -ForegroundColor Cyan
& $python (Join-Path $root "verify_setup.py")

if ($Timesteps -gt 0) {
    Write-Host "`n=== Training for $Timesteps timesteps ===" -ForegroundColor Cyan
    & $python (Join-Path $root "train_sf2.py") --timesteps $Timesteps
}
else {
    Write-Host "`nTo train:  $python $root\train_sf2.py --timesteps 1000000" -ForegroundColor Green
}
