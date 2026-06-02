@echo off
REM ─────────────────────────────────────────────────────────────
REM  E-14 Anomaly Detector — lanzador Windows
REM  Uso: go.bat --dataset dataset\ --pdfs pdfs\
REM  Todos los args extra se pasan directamente a run.py
REM ─────────────────────────────────────────────────────────────

setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no encontrado en PATH.
    echo         Descarga Python 3.10+ desde https://python.org
    pause
    exit /b 1
)

if not exist venv (
    echo   Creando entorno virtual...
    python -m venv venv
)

call venv\Scripts\activate.bat

python run.py --skip_install %*

if errorlevel 1 (
    echo.
    echo [ERROR] El proceso termino con errores.
    pause
    exit /b 1
)
