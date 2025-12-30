@echo off
REM Quick start - assumes dependencies are already installed
echo ========================================
echo SignBridge - Quick Start
echo ========================================
echo.

REM Check if port 8000 is in use
netstat -ano | findstr :8000 >nul 2>&1
if not errorlevel 1 (
    echo WARNING: Port 8000 is already in use!
    echo Run KILL_PORT_8000.bat to free the port, or press Ctrl+C to cancel.
    pause
    exit /b 1
)

cd python-backend

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Start server
python server.py

pause
