@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo   Launching Street Fighter II (Human Play Mode)
echo ========================================================
echo Controls:
echo   Arrow Keys : Move / Jump / Crouch
echo   Z, X, C    : A, B, C (punches/kicks)
echo   A, S, D    : X, Y, Z (punches/kicks)
echo   Enter      : START
echo   Esc        : Exit
echo ========================================================
"C:\Users\HEC\miniconda3\python.exe" play_sf2_human.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Script ended with exit code %ERRORLEVEL%.
    pause
)
