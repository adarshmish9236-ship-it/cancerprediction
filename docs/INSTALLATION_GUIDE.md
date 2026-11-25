# CANCERWISE AI — Comprehensive Installation & Deployment Guide

## 1. System Requirements & Hardware Compatibility

| Component | Minimum Specification | Recommended Production / Research |
|-----------|-----------------------|-----------------------------------|
| **Operating System** | Windows 10/11 (64-bit), Ubuntu 20.04/22.04 LTS, Debian 12, macOS 13+ | Ubuntu 22.04 LTS or Windows 11 Enterprise |
| **Python Runtime** | Python 3.11, 3.12, 3.14 (Verified: Python 3.14.0) | Python 3.12 LTS / Python 3.14 |
| **CPU** | Dual-Core x86_64 or Apple Silicon ARM64 | 8+ Logical Cores (Intel Core i7/i9 or AMD Ryzen 7/9) |
| **RAM** | 8 GB System Memory | 16 GB - 32 GB DDR4/DDR5 |
| **Disk Storage** | 10 GB Free Storage (SSD) | 50 GB NVMe SSD (for MLflow runs & models) |
| **GPU Acceleration** | Not required (CPU mode supported) | NVIDIA RTX 3060+ / T4 / A10G with CUDA 12.x |

---

## 2. Platform 1: Local Windows Deployment

### Step 1: Clone Repository
```powershell
git clone https://github.com/cancerwise-ai/cancerwise-ai.git
cd cancerwise-ai
```

### Step 2: Set Up Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate environment
.venv\Scripts\Activate.ps1
# If PowerShell policy blocks execution, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 3: Install Dependencies
```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# For CUDA GPU support (optional, if NVIDIA GPU present):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Configure Environment Variables
```powershell
copy .env.example .env
# Edit .env to customize settings if necessary
```

### Step 5: Run Pre-Flight Diagnostics
```powershell
python scripts/preflight_check.py
```

### Step 6: Launch Applications
**1-Click Launch (Both Services):**
```powershell
.\scripts\run_all.bat
```

**Manual Launch (Separate Consoles):**
```powershell
# Console 1 — FastAPI Backend:
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Console 2 — Streamlit Frontend:
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
```

---

## 3. Platform 2: Local Linux / macOS Deployment

### Step 1: System Packages
```bash
# Ubuntu / Debian:
sudo apt-get update && sudo apt-get install -y python3-dev python3-venv build-essential curl git
```

### Step 2: Virtual Environment & Packages
```bash
git clone https://github.com/cancerwise-ai/cancerwise-ai.git
cd cancerwise-ai

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
cp .env.example .env
```

### Step 3: Pre-Flight Verification & Execution
```bash
# Run diagnostics
python3 scripts/preflight_check.py

# Grant executable permissions to scripts
chmod +x scripts/*.sh

# Launch both backend and frontend with process isolation
./scripts/run_all.sh
```

---

## 4. Platform 3: Docker & Docker Compose Deployment

The platform provides a production-grade multi-stage `Dockerfile` and `docker-compose.yml` to orchestrate isolated containers for the backend API and frontend portal.

### Production Multi-Container Launch:
```bash
# Build and launch in detached mode
docker compose up --build -d

# Verify container health status
docker compose ps
# Output:
# cancerwise_backend    Up (healthy)   0.0.0.0:8000->8000/tcp
# cancerwise_frontend   Up (healthy)   0.0.0.0:8501->8501/tcp

# View consolidated real-time logs
docker compose logs -f

# Gracefully terminate containers
docker compose down
```

### Persistent Volume Mapping:
- `cancerwise_data`: Holds uploaded datasets and processed Parquet tables.
- `cancerwise_models`: Holds serialized model pipelines and manifests.
- `cancerwise_experiments`: Holds MLflow SQLite database and run artifacts.
- `cancerwise_logs`: Holds structured application audit logs.

### Development Mode with Live Code Mounting:
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

---

## 5. Platform 4: Streamlit Community Cloud Hosting

To deploy the frontend on Streamlit Community Cloud:

1. **Repository Structure:** Ensure `frontend/app.py` is identified as the main entry point.
2. **Requirements:** Commit `requirements.txt` to the repository root.
3. **Environment Secrets:** In the Streamlit Cloud Dashboard, configure Secrets:
   ```toml
   API_BASE_URL = "https://your-backend-api-url.com/api/v1"
   PROJECT_ENV = "production"
   ```
4. **Standalone Fallback:** If hosting without a backend API instance, the frontend automatically falls back to local service execution using its built-in `APIClient` bridge.

---

## 6. Platform 5: Production Cloud Deployment

### 6.1. AWS (Elastic Container Service / App Runner)
- **Backend Service:** Deploy `cancerwise_backend` container to AWS App Runner or ECS Fargate (2 vCPU, 4GB RAM) pointing health check to `/health`.
- **Frontend Service:** Deploy `cancerwise_frontend` container to AWS App Runner (1 vCPU, 2GB RAM).
- **Storage:** Mount AWS EFS for persistent model registry and MLflow artifacts.

### 6.2. Google Cloud Platform (Cloud Run)
- Build container via Cloud Build:
  ```bash
  gcloud builds submit --tag gcr.io/PROJECT_ID/cancerwise-backend
  ```
- Deploy serverless instance:
  ```bash
  gcloud run deploy cancerwise-backend \
    --image gcr.io/PROJECT_ID/cancerwise-backend \
    --platform managed \
    --port 8000 \
    --memory 4Gi \
    --cpu 2
  ```

### 6.3. Microsoft Azure (Azure Container Apps)
- Provision Azure Container App Environment:
  ```bash
  az containerapp create \
    --name cancerwise-backend \
    --resource-group cancerwise-rg \
    --image cancerwise-backend:1.0.0 \
    --target-port 8000 \
    --ingress 'external'
  ```

---

## 7. Verification & Health Monitoring

Verify connectivity after deployment:

```bash
# 1. Root health check
curl -I http://localhost:8000/health
# HTTP/1.1 200 OK

# 2. Versioned API health check
curl http://localhost:8000/api/v1/health

# 3. System resource telemetry
curl http://localhost:8000/api/v1/system/resources

# 4. Streamlit frontend health check
curl -I http://localhost:8501/_stcore/health
# HTTP/1.1 200 OK
```

---

## 8. Troubleshooting & FAQ

### Issue: Port 8000 or 8501 already in use
- **Windows:** `netstat -ano | findstr :8000` then `taskkill /PID <PID> /F`
- **Linux:** `lsof -i :8000` then `kill -9 <PID>`

### Issue: PdhAddEnglishCounterW failed on Windows
- System telemetry uses a safe fallback if Windows performance counters are disabled in sandboxed environments.

### Issue: MLflow database locked
- If SQLite reporting locked during multi-threaded tests, ensure `MLFLOW_ALLOW_FILE_STORE=true` is set in `.env`.
