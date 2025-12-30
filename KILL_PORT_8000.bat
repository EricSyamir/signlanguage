@echo off
echo ========================================
echo Kill Process on Port 8000
echo ========================================
echo.

REM Find process using port 8000
echo Checking for processes on port 8000...
netstat -ano | findstr :8000 >nul 2>&1

if errorlevel 1 (
    echo Port 8000 is free - no process found.
    pause
    exit /b 0
)

echo Found process(es) using port 8000:
netstat -ano | findstr :8000

echo.
echo Killing processes on port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
    echo Killing PID: %%a
    taskkill /PID %%a /F >nul 2>&1
)

echo.
echo Done! Port 8000 should now be free.
echo.
pause

