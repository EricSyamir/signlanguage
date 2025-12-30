@echo off
echo ========================================
echo SignBridge Sign Language Interpreter
echo Starting local server...
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.11 from https://www.python.org/
    pause
    exit /b 1
)

echo Python found!
python --version
echo.

REM Navigate to python-backend directory
cd python-backend

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created!
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install/update dependencies
echo Installing dependencies...
echo This may take a few minutes on first run...
pip install -r requirements.txt
echo.

REM Check if installation was successful
if errorlevel 1 (
    echo ERROR: Failed to install dependencies!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo ========================================
echo Starting server...
echo Server will be available at: http://localhost:8000
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Start the server
python server.py

REM If server stops, keep window open
echo.
echo Server stopped.
pause

