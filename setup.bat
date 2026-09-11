@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo   HFG Windows Setup Utility
echo ===================================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+.
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [*] Creating .venv...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo [*] Upgrading pip...
python -m pip install --upgrade pip

echo [*] Installing project dependencies...
pip install -e ".[dev]"

echo [*] Installing SpaCy language model...
python -m spacy download en_core_web_sm

if exist "frontend" (
    where npm >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        echo [*] Installing frontend dependencies and building UI...
        cd frontend
        call npm install
        call npm run build
        cd ..
    )
)

if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env
        echo [*] Copied .env.example to .env
    )
)

echo.
echo [SUCCESS] Setup complete! You can now launch the app using run.bat.
pause
