@echo off
REM Quick start - assumes dependencies are already installed
echo ========================================
echo SignBridge - Quick Start
echo ========================================
echo.

cd python-backend

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Start server
python server.py

pause

