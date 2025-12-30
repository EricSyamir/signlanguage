@echo off
echo ========================================
echo SignBridge - Local Setup Script
echo ========================================
echo.
echo This script will set up your local development environment.
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.11 from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [1/4] Python found!
python --version
echo.

REM Navigate to python-backend
cd python-backend

REM Create virtual environment
echo [2/4] Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists. Skipping...
) else (
    python -m venv venv
    echo Virtual environment created!
)
echo.

REM Activate virtual environment
echo [3/4] Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet
echo.

REM Install dependencies
echo [4/4] Installing dependencies...
echo This may take a few minutes...
echo.
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To start the server, run: START_SERVER.bat
echo Or use: START_SERVER_QUICK.bat (if dependencies are already installed)
echo.
pause

