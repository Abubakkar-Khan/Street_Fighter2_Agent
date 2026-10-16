# Build & install stable-retro on Windows for the Street Fighter II RL setup.
#
# stable-retro publishes no Windows wheels, so the C++ core has to be compiled
# locally. This script installs the MSYS2 MinGW-w64 toolchain if needed, builds
# only the cores you ask for (default: just Genesis, which SF2 needs) and
# installs the Python package in editable mode.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File build_stable_retro.ps1
#   powershell -ExecutionPolicy Bypass -File build_stable_retro.ps1 -RlDeps
#   powershell -ExecutionPolicy Bypass -File build_stable_retro.ps1 -Cores "genesis;fbneo"
#
[CmdletBinding()]
param(
    [string]$Repo = (Join-Path $PSScriptRoot "stable-retro"),
    [string]$Python = "",
    [string]$Cores = "genesis",
    [switch]$RlDeps,
    [switch]$SkipInstall,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

function Section([string]$text) {
    Write-Host ""
    Write-Host ("=== " + $text + " " + ("=" * [Math]::Max(0, 58 - $text.Length))) -ForegroundColor Cyan
}

function Info([string]$text) { Write-Host "[*] $text" -ForegroundColor Yellow }
function Good([string]$text) { Write-Host "[OK] $text" -ForegroundColor Green }
function Bad([string]$text) { Write-Host "[X] $text" -ForegroundColor Red }

Section "1. Locating Python"
if (-not $Python) {
    foreach ($candidate in @(
            "C:\Users\HEC\miniconda3\python.exe",
            (Join-Path $env:LOCALAPPDATA "Python\bin\python.exe"))) {
        if (Test-Path $candidate) { $Python = $candidate; break }
    }
}
if (-not $Python) {
    $cmd = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($cmd) { $Python = $cmd.Source }
}
if (-not $Python -or -not (Test-Path $Python)) {
    Bad "No Python interpreter found. Pass -Python <path to python.exe>."
    exit 1
}
$pyVersion = & $Python -c "import sys;print('.'.join(map(str,sys.version_info[:3])))"
Good "python $pyVersion  ($Python)"

Section "2. MSYS2 MinGW-w64 toolchain"
$msys = "C:\msys64"
$gcc = Join-Path $msys "mingw64\bin\gcc.exe"
$gxx = Join-Path $msys "mingw64\bin\g++.exe"
$make = Join-Path $msys "usr\bin\make.exe"
$pacman = Join-Path $msys "usr\bin\pacman.exe"

if (-not (Test-Path $pacman)) {
    Bad "MSYS2 not found at $msys. Install it from https://www.msys2.org/ and re-run."
    exit 1
}
if (Test-Path (Join-Path $msys "var\lib\pacman\db.lck")) {
    Info "Removing stale pacman lock"
    Remove-Item -Force (Join-Path $msys "var\lib\pacman\db.lck") -ErrorAction SilentlyContinue
}
if (-not (Test-Path $gcc) -or -not (Test-Path $gxx)) {
    Info "Installing gcc/zlib/pkgconf into MSYS2 (this downloads a few hundred MB)..."
    & $pacman -Sy --noconfirm
    & $pacman -S --noconfirm --needed mingw-w64-x86_64-gcc mingw-w64-x86_64-zlib mingw-w64-x86_64-pkgconf
}
if (-not (Test-Path $make)) {
    Info "Installing MSYS2 make..."
    & $pacman -S --noconfirm --needed make
}
if (-not (Test-Path $gcc) -or -not (Test-Path $make)) {
    Bad "Toolchain still incomplete (gcc=$gcc, make=$make)"
    exit 1
}
Good "gcc : $((& $gcc --version | Select-Object -First 1))"
Good "make: $make"

# MSYS2 make and the MinGW compiler must come first on PATH.
$env:PATH = "$msys\mingw64\bin;$msys\usr\bin;" +
    (Split-Path $Python) + ";" +
    (Join-Path (Split-Path $Python) "Scripts") + ";" +
    (Join-Path (Split-Path $Python) "Library\bin") + ";" +
    $env:PATH
$env:CC = $gcc
$env:CXX = $gxx

Section "3. Building stable-retro ($Cores core(s))"
if (-not (Test-Path $Repo)) {
    Bad "stable-retro checkout not found at $Repo"
    exit 1
}

if ($Clean) {
    Info "Removing build leftovers..."
    foreach ($item in @("CMakeCache.txt", "CMakeFiles", "build")) {
        $path = Join-Path $Repo $item
        if (Test-Path $path) { Remove-Item -Recurse -Force $path -ErrorAction SilentlyContinue }
    }
}

# The generator and make program are pinned because setup.py's auto-detection
# picks MinGW Makefiles, which cannot run the POSIX makefiles of the vendored
# LuaJIT and libretro cores.
$env:STABLE_RETRO_CMAKE_ARGS = @(
    '-G "MSYS Makefiles"',
    "-DCMAKE_MAKE_PROGRAM=$($make -replace '\\', '/')",
    "-DZLIB_ROOT=$($msys -replace '\\', '/')/mingw64",
    "-DSTABLE_RETRO_CORES=$Cores",
    "-DBUILD_TESTS=OFF",
    "-DBUILD_CAPNP=OFF",
    "-DBUILD_N64=OFF"
) -join " "
Info "CMAKE_ARGS: $env:STABLE_RETRO_CMAKE_ARGS"

Info "Installing Python build/runtime requirements..."
& $Python -m pip install --disable-pip-version-check wheel setuptools `
    "gymnasium>=1.0.0" "pyglet>=1.5.27,<2" "farama-notifications>=0.0.1"
if ($LASTEXITCODE -ne 0) { Bad "pip install of requirements failed"; exit 1 }

if (-not $SkipInstall) {
    Info "Compiling the C++ core (this can take several minutes)..."
    Push-Location $Repo
    try {
        & $Python -m pip install -e . --no-build-isolation
        if ($LASTEXITCODE -ne 0) {
            Bad "stable-retro build failed. See the output above."
            exit 1
        }
    }
    finally {
        Pop-Location
    }
    Good "stable-retro installed"
}

Section "4. Import check"
& $Python -c @"
import stable_retro, os
print('stable_retro', stable_retro.__version__.strip(), 'at', os.path.dirname(stable_retro.__file__))
print('systems:', ', '.join(sorted(stable_retro.data.EMU_INFO)))
"@
if ($LASTEXITCODE -ne 0) {
    Bad "stable_retro failed to import"
    exit 1
}

if ($RlDeps) {
    Section "5. Reinforcement-learning dependencies"
    & $Python -m pip install --disable-pip-version-check stable-baselines3 tensorboard
    if ($LASTEXITCODE -ne 0) { Bad "stable-baselines3 install failed"; exit 1 }
    Good "stable-baselines3 installed"
}

Section "Done"
Write-Host "Next steps:"
Write-Host "  1) copy the SF2 Sega Genesis ROM into $PSScriptRoot"
Write-Host "  2) $Python $PSScriptRoot\setup_rom.py"
Write-Host "  3) $Python $PSScriptRoot\verify_setup.py"
Write-Host "  4) $Python $PSScriptRoot\train_sf2.py --timesteps 1000000"
