@echo off
echo ========================================
echo Sign Language Model Training
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
if not exist "set_hand_histogram.py" (
    echo ERROR: Please run this script from python-backend/training directory
    echo Current directory: %CD%
    pause
    exit /b 1
)

echo ========================================
echo TRAINING PIPELINE
echo ========================================
echo.
echo This will guide you through training:
echo 1. Setting hand histogram
echo 2. Capturing gestures (you'll do this manually)
echo 3. Loading images
echo 4. Training the model
echo.
echo ========================================
echo.

python train_all.py

pause

