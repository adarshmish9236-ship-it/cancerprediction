@echo off
REM ============================================================================
REM CANCERWISE AI — Windows Unified Application Launcher
REM 1. Runs pre-flight diagnostics
REM 2. Launches FastAPI Backend in dedicated window
REM 3. Launches Streamlit Frontend in dedicated window
REM ============================================================================

setlocal
set PROJECT_DIR=%~dp0..
cd /d "%PROJECT_DIR%"
set PYTHONPATH=%PROJECT_DIR%

REM Auto-detect Python interpreter
set "PYTHON_EXE="
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python -c "import sys" >nul 2>&1
    if %ERRORLEVEL% EQU 0 set "PYTHON_EXE=python"
)
if not defined PYTHON_EXE (
    if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" (
        set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
        set "PATH=%LOCALAPPDATA%\Programs\Python\Python314;%LOCALAPPDATA%\Programs\Python\Python314\Scripts;%PATH%"
    ) else if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
        set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        set "PATH=%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%PATH%"
    ) else if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
        set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        set "PATH=%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;%PATH%"
    ) else (
        set "PYTHON_EXE=python"
    )
)

echo ============================================================================
echo   CANCERWISE AI — Universal Startup Orchestrator
echo ============================================================================

echo [STEP 1/3] Running pre-flight system diagnostics...
"%PYTHON_EXE%" scripts/preflight_check.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Pre-flight verification failed. Aborting startup.
    pause
    exit /b %ERRORLEVEL%
)

echo [STEP 2/3] Spawning FastAPI Backend on http://localhost:8000 ...
start "CANCERWISE AI — Backend API" cmd /k "cd /d "%PROJECT_DIR%" && set PYTHONPATH=%PROJECT_DIR% && "%PYTHON_EXE%" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

echo [STEP 3/3] Spawning Streamlit Frontend on http://localhost:8501 ...
timeout /t 3 /nobreak > nul
start "CANCERWISE AI — Frontend Dashboard" cmd /k "cd /d "%PROJECT_DIR%" && set PYTHONPATH=%PROJECT_DIR% && "%PYTHON_EXE%" -m streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0"

echo.
echo ============================================================================
echo   APPLICATION SUCCESSFULLY LAUNCHED!
echo   - Backend Documentation: http://localhost:8000/docs
echo   - Frontend Portal:        http://localhost:8501
echo ============================================================================
echo.
