#!/usr/bin/env bash
# ============================================================================
# CANCERWISE AI — Linux/macOS Unified Application Launcher
# 1. Runs pre-flight diagnostics
# 2. Starts FastAPI Backend in background
# 3. Starts Streamlit Frontend in foreground
# 4. Cleans up processes on SIGINT/SIGTERM
# ============================================================================

set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
cd "${PROJECT_DIR}"

echo "============================================================================"
echo "  CANCERWISE AI — Universal Startup Orchestrator"
echo "============================================================================"

# Pre-flight check
echo "[STEP 1/3] Running pre-flight system diagnostics..."
python3 "${PROJECT_DIR}/scripts/preflight_check.py"

# Function to kill background processes on exit
cleanup() {
    echo ""
    echo "[CANCERWISE AI] Shutting down services..."
    if [ -n "$BACKEND_PID" ]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Start backend
echo "[STEP 2/3] Launching FastAPI Backend on http://localhost:8000 ..."
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to become healthy
echo "Waiting for backend API to initialize..."
for i in {1..15}; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo "Backend is operational!"
        break
    fi
    sleep 1
done

# Start frontend
echo "[STEP 3/3] Launching Streamlit Frontend on http://localhost:8501 ..."
echo "Access Frontend: http://localhost:8501"
echo "Access OpenAPI:  http://localhost:8000/docs"
streamlit run "${PROJECT_DIR}/frontend/app.py" --server.port 8501 --server.address 0.0.0.0
