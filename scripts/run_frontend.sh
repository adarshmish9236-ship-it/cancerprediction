#!/usr/bin/env bash
# ============================================================================
# CANCERWISE AI — Linux/macOS Frontend Launcher
# Starts Streamlit Scientific Dashboard
# ============================================================================

set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

PYTHON_BIN="$(command -v python3 || command -v python)"
echo "[CANCERWISE AI] Starting Streamlit Frontend Dashboard on port 8501..."
exec "$PYTHON_BIN" -m streamlit run "${PROJECT_DIR}/frontend/app.py" --server.port 8501 --server.address 0.0.0.0
