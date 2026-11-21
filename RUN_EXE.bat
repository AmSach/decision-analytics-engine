@echo off
setlocal
title Launch DecisionGraph.exe

if not exist "%~dp0dist\DecisionGraph\DecisionGraph.exe" (
    echo [ERROR] DecisionGraph.exe not found at %~dp0dist\DecisionGraph\DecisionGraph.exe
    echo Run START_APP.bat to run via Python or rebuild with PyInstaller.
    pause
    exit /b 1
)

echo Launching Decision Graph Desktop Executable...
start "" "%~dp0dist\DecisionGraph\DecisionGraph.exe"
exit /b 0
