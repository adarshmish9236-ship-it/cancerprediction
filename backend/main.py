"""
CANCERWISE AI — FastAPI Application Entry Point
Production-ready backend architecture for cancer risk prediction, intelligent data mining,
and clinically-validated Explainable AI.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

# Ensure project root on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.logging_setup import setup_logging, get_logger
from backend.core.config import get_config

# Initialize logging before anything else
setup_logging()
logger = get_logger("api.main")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.middleware.error_handler import register_exception_handlers
from backend.middleware.request_limits import RequestSizeLimitMiddleware
from backend.services.system_service import SystemService

config = get_config()
_system_service = SystemService()


# ---------------------------------------------------------------------------
# Lifespan Context Manager (Modern FastAPI lifecycle)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("CANCERWISE AI API starting up — version %s (Phase 9)", config.api.version)
    yield
    logger.info("CANCERWISE AI API shutting down gracefully.")


app = FastAPI(
    title=config.api.title,
    description=(
        "### CANCERWISE AI — Clinical Decision Support & Data Mining Backend\n\n"
        "Provides clinical data ingestion, automated quality wrangling, exploratory data mining, "
        "supervised classification, advanced learning paradigms (semi-supervised, multitask, multiclass, "
        "one-class, continuous regression), and Explainable AI (SHAP, LIME, PDP/ICE, calibration).\n\n"
        "**Governance Directive:** Model-estimated probabilities and predictions are clinical decision-support "
        "tools and do NOT constitute a standalone definitive medical diagnosis."
    ),
    version=config.api.version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Centralized Error Handlers
# ---------------------------------------------------------------------------
register_exception_handlers(app)

# ---------------------------------------------------------------------------
# Middleware (CORS)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Root & Health Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"])
def health_check() -> JSONResponse:
    """Basic root health check endpoint for orchestrators and load balancers."""
    return JSONResponse(_system_service.get_health())


@app.get(f"{config.api.prefix}/health", tags=["System"])
def api_health_check() -> JSONResponse:
    """Versioned API health check."""
    data = _system_service.get_health()
    data["api_prefix"] = config.api.prefix
    return JSONResponse(data)


@app.get("/", tags=["Root"])
def root() -> dict:
    """Platform landing metadata and API documentation shortcuts."""

    return {
        "name": "CANCERWISE AI",
        "version": config.api.version,
        "phase": "Phase 9 — FastAPI Backend and Service Integration",
        "description": "Advanced Cancer Prediction and Intelligent Data Mining Platform",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
        },
        "endpoints": {
            "health": "/health",
            "datasets": f"{config.api.prefix}/datasets",
            "wrangling": f"{config.api.prefix}/wrangling",
            "eda": f"{config.api.prefix}/eda",
            "features": f"{config.api.prefix}/features",
            "models": f"{config.api.prefix}/models",
            "advanced": f"{config.api.prefix}/advanced",
            "scale": f"{config.api.prefix}/scale",
            "explainability": f"{config.api.prefix}/explainability",
            "system": f"{config.api.prefix}/system",
            "jobs": f"{config.api.prefix}/jobs",
        },
    }


# ---------------------------------------------------------------------------
# API Routers
# ---------------------------------------------------------------------------
from backend.api.datasets import router as datasets_router
from backend.api.wrangling import router as wrangling_router
from backend.api.eda import router as eda_router
from backend.api.features import router as features_router
from backend.api.models import router as models_router
from backend.api.advanced import router as advanced_router
from backend.api.large_scale import router as scale_router
from backend.api.explainability import router as explainability_router
from backend.api.system import router as system_router
from backend.api.jobs import router as jobs_router

app.include_router(datasets_router, prefix=config.api.prefix)
app.include_router(wrangling_router, prefix=config.api.prefix)
app.include_router(eda_router, prefix=config.api.prefix)
app.include_router(features_router, prefix=config.api.prefix)
app.include_router(models_router, prefix=config.api.prefix)
app.include_router(advanced_router, prefix=config.api.prefix)
app.include_router(scale_router, prefix=config.api.prefix)
app.include_router(explainability_router, prefix=config.api.prefix)
app.include_router(system_router, prefix=config.api.prefix)
app.include_router(jobs_router, prefix=config.api.prefix)


def main() -> None:
    """CLI launcher for CANCERWISE AI FastAPI backend."""
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=config.api.host,
        port=int(config.api.port),
        reload=bool(config.api.reload),
    )


if __name__ == "__main__":
    main()
