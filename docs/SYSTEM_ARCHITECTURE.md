# CANCERWISE AI — System Architecture Specification

## 1. Executive Architectural Blueprint

CANCERWISE AI is engineered as a high-performance, modular, multi-tier software system designed for medical data mining, oncological machine learning, and clinically governed explainability.

The platform follows clean architecture principles, strictly decoupling the Presentation Tier (Streamlit), Application/API Routing Tier (FastAPI), Domain Service Tier (17 independent business services), Core Machine Learning Engine, and Artifact Storage Layer.

```
+-----------------------------------------------------------------------------------+
|                            PRESENTATION TIER (Streamlit)                          |
|  - Universal Theme Engine (Light / Dark Navy)    - 12 Scientific UI Components    |
|  - Dynamic Breadcrumbs & Module Nav              - 14 Interactive Feature Pages   |
+-----------------------------------------+-----------------------------------------+
                                          |
                      REST HTTP / Direct Service Bridge
                                          |
+-----------------------------------------v-----------------------------------------+
|                              API ROUTER TIER (FastAPI)                            |
|  - Versioned API Catalog (/api/v1/*)             - Pydantic v2 Request Validation |
|  - Centralized RFC Error Envelopes               - Safe File Upload & RBAC Guards |
+-----------------------------------------+-----------------------------------------+
                                          |
                        Pure Delegation (Zero In-Route ML)
                                          |
+-----------------------------------------v-----------------------------------------+
|                        INDEPENDENT SERVICE LAYER (17 Services)                    |
|  - Dataset Management    - Wrangling Service       - EDA & Hypothesis Testing     |
|  - Feature Engineering   - Feature Selection       - Supervised Classifiers       |
|  - Semi-Supervised Lab   - Multitask PyTorch       - Multiclass Subtypes          |
|  - One-Class Novelty     - Continuous Regression   - Large-Scale Polars/SGD       |
|  - SHAP / LIME XAI       - Evaluation & Calib      - Governed Prediction Service  |
|  - Asynchronous Job Manager (ThreadPoolExecutor)   - System Telemetry & Health    |
+-----------------------------------------+-----------------------------------------+
                                          |
                  Scientific Computation & Model Persistence
                                          |
+-----------------------------------------v-----------------------------------------+
|                     CORE COMPUTATION & DATA PERSISTENCE LAYER                     |
|  - Scikit-Learn / PyTorch / LightGBM             - Polars & PyArrow Columnar IO   |
|  - SHAP Tree/Linear & LIME Surrogate Engines     - Pandera Data Contract Engine   |
|  - Model Registry (Serialized Pipelines)         - MLflow Experiment Lineage (DB) |
|  - Immutable JSONL Transformation Logs           - Parquet / Raw File Store       |
+-----------------------------------------------------------------------------------+
```

---

## 2. Layer-by-Layer Architectural Decomposition

### 2.1. Presentation Layer (Frontend Dashboard)
- **Framework:** Streamlit (v1.32+)
- **Architecture Pattern:** Multi-page architecture (`frontend/pages/`) with shared components (`frontend/components/`).
- **Resilience Feature:** Features a dual-mode communication bridge (`frontend/utils/api_client.py`). When the FastAPI backend is online, requests are dispatched over asynchronous HTTP REST. If the backend is unreachable or operating in standalone local mode, the client seamlessly falls back to invoking domain service classes directly, guaranteeing zero downtime during academic demonstrations.
- **Visual Design:** High-contrast scientific dark palette (`#0A0F1D`) and clinical light palette (`#F8FAFC`), custom-crafted interactive Plotly visualizations, responsive stat cards, and dynamic breadcrumbs.

### 2.2. API Routing Layer (FastAPI Backend)
- **Framework:** FastAPI (v0.110+) running atop Uvicorn ASGI.
- **Routing Protocol:** Versioned prefix under `/api/v1` organized across 10 specialized routers:
  1. `datasets`: Upload, retrieval, profiling, preview, deletion.
  2. `wrangling`: Quality profiling, imputation, deduplication, scaling, transformation history.
  3. `eda`: Hypothesis tests, Plotly visual JSON specs, HTML report compilation.
  4. `features`: Morphometric feature derivation, Borda consensus ranking, PCA/UMAP reduction.
  5. `models`: Supervised training, model registry querying, performance evaluation, inference.
  6. `advanced`: Semi-supervised, multitask, multiclass, one-class, regression, and kernel experiments.
  7. `large_scale`: Polars benchmarking, memory downcasting, out-of-core SGD learning.
  8. `explainability`: Global and local SHAP explanations, LIME surrogate rules, PDP curves.
  9. `system`: Real-time CPU, RAM, CUDA GPU telemetry, and subsystem health checks.
  10. `jobs`: Background execution telemetry, log polling, cancellation tokens.
- **Error Handling Envelope:** Centralized middleware translates domain exceptions (`DatasetNotFoundError`, `ModelNotFoundError`, `FeatureMismatchError`) into standardized RFC-compliant JSON envelopes with specific error codes, paths, and timestamps.
- **Security & Upload Guards:** Request size limit enforced at 100MB, file extension whitelisting (`.csv`, `.tsv`, `.xlsx`, `.parquet`, `.json`), path traversal neutralization via filename sanitization, and Role-Based Access Control (`UserRole`: `VIEWER`, `RESEARCHER`, `CLINICIAN`, `ADMIN`).

### 2.3. Domain Service Layer (17 Independent Services)
- Every business operation is encapsulated within a dedicated service class in `backend/services/`.
- Controllers in `backend/api/` execute **no** machine learning, statistical math, or file I/O directly; they merely validate the Pydantic schema, call the service method, and return the response schema.
- **Thread-Safe Background Execution:** Long-running workflows (model training, hyperparameter optimization, report compilation) are managed by `JobManager` using a `ThreadPoolExecutor` (4 worker threads). Tasks report granular progress (0.0 to 1.0) and streaming log messages, and observe cooperative cancellation tokens.

### 2.4. Core Machine Learning & Data Mining Engine
- **Preprocessing & Wrangling:** `DataQualityAnalyzer`, `MissingValueHandler`, `DuplicateManager`, `OutlierDetector`, `DataTransformer`, and `DataValidationEngine`.
- **Feature Store & Selection:** `DerivedFeatureGenerator`, `ConsensusFeatureSelector`, and `DimensionalityReducer`.
- **Classification & Advanced Learning:** `ModelTrainer`, `ModelRegistry`, `ModelSerializer`, `SemiSupervisedEngine`, `SharedMultitaskNet` (PyTorch), `MulticlassEngine`, `OneClassNoveltyEngine`, `ContinuousRegressionEngine`, and `MercerKernelEngine`.
- **Explainability & Validation:** `SHAPExplainer`, `LIMEExplainer`, `ModelValidator`, `CalibrationEngine`, `RobustnessAnalyzer`, `ModelGovernanceService` (MLflow), and `ResearchReporter`.

### 2.5. Data Persistence & Lineage Layer
- **Raw Data (`data/raw/`):** Immutable raw files preserving original biomedical acquisition.
- **Processed Data (`data/processed/`):** Cleaned, memory-optimized columnar Parquet files with SHA-256 integrity checksums.
- **Model Registry (`models/trained/`, `models/registry/`):** Serialized pipelines (fitted preprocessor + model artifact) accompanied by JSON lineage manifests specifying training dataset hash, algorithm parameters, and feature schemas.
- **MLflow Tracking (`experiments/mlruns/`):** SQLite database recording runs, metrics, parameters, and artifact references.

---

## 3. Asynchronous Concurrency & Background Job Architecture

```
Client (Frontend / External API)
       |
       | POST /api/v1/models/train (async_job=True)
       v
+-------------------------------+
|  FastAPI Route Controller     |
+---------------+---------------+
                |
                | Delegates to SupervisedLearningService
                v
+---------------+---------------+
|       JobManager              | <--- Singleton Thread Pool (max_workers=4)
+---------------+---------------+
                |
                +---> Spawns worker thread (cw_job_worker_N)
                |     - Instantiates JobInfo (UUID: job_<hex>)
                |     - Sets status: RUNNING
                |     - Passes JobContext (thread-safe lock & cancel_event)
                |
                +---> Returns immediate 202 ACCEPTED with JobInfo
                                |
Client polls GET /api/v1/jobs/{job_id} <---------------+
- Receives: progress (0.0 -> 1.0), status, logs, elapsed time
```

---

## 4. Clinical Governance & Inference Safety Architecture

In strict compliance with healthcare software engineering directives, inference is mediated by `PredictionService`:

1. **Registry Verification:** Rejects unapproved or unmanifested model IDs (`ModelNotFoundError`).
2. **Schema & Ordering Verification:** Validates incoming JSON features against the exact feature names and ordering recorded in the model manifest (`FeatureMismatchError`).
3. **Pipeline Transformation:** Automatically routes raw patient inputs through the fitted transformation pipeline embedded inside the model bundle.
4. **Calibrated Probability Output:** Emits calibrated probabilities alongside binary class designations.
5. **Governance Notice Injection:** Injects immutable clinical decision-support notices explicitly stating that outputs are computational decision support and do not constitute medical diagnosis.
