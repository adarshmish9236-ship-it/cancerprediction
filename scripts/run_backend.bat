@echo off
REM ============================================================================
REM CANCERWISE AI — Windows Backend Launcher
REM Starts FastAPI production ASGI server with Uvicorn
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
    )
)
if not defined PYTHON_EXE set "PYTHON_EXE=python"

echo [CANCERWISE AI] Starting FastAPI Backend on port 8000...
"%PYTHON_EXE%" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
pause
