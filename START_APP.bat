@echo off
setlocal enabledelayedexpansion

title Decision Graph - Enterprise Node-Spreadsheet and Analytics Engine

echo ===============================================================================
echo                DECISION GRAPH DESKTOP ANALYTICS ENGINE
echo          Native Python Spreadsheet fused with Visual Node Pipeline
echo ===============================================================================
echo.

:: 1. Check if compiled standalone executable exists
if exist "dist\DecisionGraph\DecisionGraph.exe" (
    echo [OK] Launching compiled standalone executable: dist\DecisionGraph\DecisionGraph.exe
    start "" "dist\DecisionGraph\DecisionGraph.exe"
    exit /b 0
)

:: 2. Otherwise launch via Python environment
set "PYTHON_CMD="
if exist "E:\Python312\python.exe" (
    set "PYTHON_CMD=E:\Python312\python.exe"
) else (
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        set "PYTHON_CMD=python"
    ) else (
        where py >nul 2>nul
        if %errorlevel% equ 0 (
            set "PYTHON_CMD=py"
        )
    )
)

if "%PYTHON_CMD%"=="" (
    echo [ERROR] Neither DecisionGraph.exe nor Python 3.10+ was found.
    echo Please install Python or verify the build.
    echo.
    pause
    exit /b 1
)

echo [OK] Launching via Python: %PYTHON_CMD% run.py
echo.
"%PYTHON_CMD%" run.py

if %errorlevel% neq 0 (
    echo.
    echo [NOTICE] Application closed with code %errorlevel%.
    pause
)
