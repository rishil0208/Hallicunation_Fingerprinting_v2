@echo off

echo ===================================================
echo   Hallucination Fingerprinting Gate (HFG) Launcher
echo ===================================================
echo.

REM 1. Check Python installation
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not found in your PATH.
    echo Please install Python 3.10 or higher from https://www.python.org/
    pause
    exit /b 1
)

REM 2. Activate virtual environment if present
if exist ".venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment .venv ...
    call .venv\Scripts\activate.bat
)

REM 3. Open browser and start application
echo.
echo [*] Launching dashboard at http://localhost:8000 ...
start http://localhost:8000
echo [*] Starting FastAPI application server on http://localhost:8000
echo [Press Ctrl+C to stop the server]
echo.

python -m uvicorn backend.app.api.routes:app --host 127.0.0.1 --port 8000 --reload

if errorlevel 1 (
    echo.
    echo [ERROR] Server stopped unexpectedly.
    pause
)
