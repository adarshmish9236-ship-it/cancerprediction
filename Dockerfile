# ==============================================================================
# CANCERWISE AI — Production Multi-Stage Container Dockerfile
# Builds secure, minimal, production-ready container image for CANCERWISE AI
# ==============================================================================

# Stage 1: Build & Dependencies
FROM python:3.12-slim-bookworm AS builder

WORKDIR /build

# Install system compilation dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel/setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch CPU build first to avoid huge CUDA binary overhead in production container
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2: Final Runtime Image
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# Install minimal runtime shared libraries and curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Create non-root system user for security compliance (Principle of Least Privilege)
RUN groupadd -r cancerwise && useradd -r -g cancerwise -d /app -s /sbin/nologin cancerwise

# Copy project source code
COPY --chown=cancerwise:cancerwise . /app

# Ensure directories exist with proper permissions
RUN mkdir -p /app/data/raw /app/data/processed /app/models/trained /app/models/registry \
             /app/models/governance /app/experiments/mlruns /app/logs /app/reports \
    && chown -R cancerwise:cancerwise /app

# Switch to non-root user
USER cancerwise

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose FastAPI backend (8000) and Streamlit frontend (8501)
EXPOSE 8000 8501

# Default health check targeting FastAPI /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command launches backend API (override in docker-compose)
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
