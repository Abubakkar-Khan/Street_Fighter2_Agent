@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo   Launching Street Fighter II Trained Model Viewer
echo ========================================================
"C:\Users\HEC\miniconda3\python.exe" watch_model_play.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Script ended with exit code %ERRORLEVEL%.
    pause
)
