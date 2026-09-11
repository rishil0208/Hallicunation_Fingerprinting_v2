@echo off
echo ===================================================
echo   Starting Frontend Development Server (Vite)
echo ===================================================
echo.

where node >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Node.js is not installed or not in PATH.
    echo Please install Node.js (LTS version) from: https://nodejs.org/
    pause
    exit /b 1
)

if not exist "node_modules" (
    echo [*] Installing npm dependencies...
    call npm install
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] npm install failed.
        pause
        exit /b 1
    )
)

echo.
echo [*] Starting Vite React Dev Server...
echo [*] Open http://localhost:5173 in your browser
echo.
call npm run dev
pause
