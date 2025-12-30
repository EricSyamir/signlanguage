@echo off
echo ========================================
echo SignBridge Sign Language Interpreter
echo Starting local server...
echo ========================================
echo.

REM Check if port 8000 is in use
netstat -ano | findstr :8000 >nul 2>&1
if not errorlevel 1 (
    echo WARNING: Port 8000 is already in use!
    echo.
    echo Options:
    echo 1. Kill the process using port 8000 (recommended)
    echo 2. Use a different port
    echo.
    choice /C 12 /N /M "Choose option (1 or 2): "
    
    if errorlevel 2 goto use_different_port
    if errorlevel 1 goto kill_port
)

goto start_server

:kill_port
echo.
echo Killing process on port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
    echo Killing PID: %%a
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 2 /nobreak >nul
goto start_server

:use_different_port
set /p PORT="Enter port number (default: 8001): "
if "%PORT%"=="" set PORT=8001
set PORT=%PORT%
goto start_server

:start_server
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
python -m pip install --upgrade pip >nul 2>&1
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
if defined PORT (
    echo Server will be available at: http://localhost:%PORT%
    set PORT=%PORT%
) else (
    echo Server will be available at: http://localhost:8000
)
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Start the server
if defined PORT (
    python server.py
) else (
    python server.py
)

REM If server stops, keep window open
echo.
echo Server stopped.
pause
