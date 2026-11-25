#!/usr/bin/env bash
# ============================================================================
# CANCERWISE AI — Linux/macOS Backend Launcher
# Starts FastAPI production ASGI server with Uvicorn
# ============================================================================

set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

PYTHON_BIN="$(command -v python3 || command -v python)"
echo "[CANCERWISE AI] Starting FastAPI Backend on port 8000..."
exec "$PYTHON_BIN" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
