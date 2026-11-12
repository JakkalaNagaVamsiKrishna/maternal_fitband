@echo off
title Maternal FitBand — Live Demonstration
echo ======================================================================
echo   Maternal FitBand - Launching Live Demonstration
echo ======================================================================

:: Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10+ from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Create virtualenv if not present
if not exist .venv (
    echo [1/3] Creating Python virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo [2/3] Installing dependencies...
    pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

:: Start Web App
echo [3/3] Starting Maternal Health Cloud Web App...
echo.
echo ======================================================================
echo   * Web App URL: http://localhost:8080
echo   * Press Ctrl + C to stop the demo.
echo ======================================================================
start http://localhost:8080
python run_web_app.py

pause
