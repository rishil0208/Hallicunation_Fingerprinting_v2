@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo   Hallucination Fingerprinting Gate (HFG) Launcher
echo ===================================================
echo.

:: 1. Check Python installation
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not found in your PATH.
    echo Please install Python 3.10 or higher from https://www.python.org/
    pause
    exit /b 1
)

:: 2. Check or create virtual environment
if not exist ".venv" (
    echo [*] Creating virtual environment (.venv)...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: 3. Activate virtual environment
call .venv\Scripts\activate.bat

:: 4. Check if dependencies are installed
python -c "import fastapi, pyreason, spacy" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [*] Installing dependencies (first-time setup)...
    python -m pip install --upgrade pip
    pip install -e .
    echo [*] Downloading SpaCy English model...
    python -m spacy download en_core_web_sm
)

:: 5. Copy .env if not present
if not exist ".env" (
    if exist ".env.example" (
        echo [*] Initializing .env configuration...
        copy .env.example .env >nul
    )
)

:: 6. Check frontend build
if not exist "frontend\dist\index.html" (
    echo [!] Built frontend not found in frontend\dist.
    where npm >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        echo [*] Building frontend static assets...
        cd frontend
        call npm install
        call npm run build
        cd ..
    ) else (
        echo [WARNING] Node/npm not found. Serving without prebuilt UI.
    )
)

:: 7. Launch browser
echo.
echo [*] Opening application in your default browser...
start http://localhost:8000
echo [*] Starting FastAPI application server on http://localhost:8000 ...
echo [Press Ctrl+C to stop the server]
echo.

uvicorn backend.app.api.routes:app --host 127.0.0.1 --port 8000
pause
