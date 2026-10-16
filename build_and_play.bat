@echo off
rem Convenience launcher: build stable-retro, import the ROM, verify, train.
rem Pass through to the PowerShell script which does the real work.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_and_play.ps1" %*
echo.
pause
