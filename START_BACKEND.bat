@echo off
echo ========================================
echo SignBridge Python Backend Starter
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed!
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo Python found!
echo.

REM Check if in correct directory
if not exist "python-backend\server.py" (
    echo ERROR: Please run this script from the SignLanguage directory
    pause
    exit /b 1
)

REM Check if dependencies are installed
pip show fastapi >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Python dependencies...
    cd python-backend
    pip install -r requirements.txt
    cd ..
    echo.
)

echo Starting Python backend server...
echo Server will run on http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

cd python-backend
python server.py

pause

