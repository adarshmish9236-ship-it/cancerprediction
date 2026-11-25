# CANCERWISE AI — Project Structure & Component Inventory

## 1. Directory Tree Overview

```
CANCERWISE_AI/
├── .env.example                     # Environment variables configuration template
├── .gitignore                       # Git exclusion rules (secrets, models, logs, cache)
├── Dockerfile                       # Multi-stage production container specification
├── docker-compose.yml               # Multi-service production orchestration (API + UI)
├── docker-compose.dev.yml           # Development overlay with host volume mounting
├── pyproject.toml                   # Project metadata, build system, and tool configs
├── requirements.txt                 # Pinned runtime and scientific dependencies
├── README.md                        # Primary project documentation & quickstart
├── PROJECT_ROADMAP.md               # 12-Phase engineering roadmap & milestone tracker
│
├── backend/                         # FastAPI Backend Application Tier
│   ├── __init__.py                  # Package initialization & logging bootstrap
│   ├── main.py                      # FastAPI application entry point, lifespan, & router mounting
│   ├── api/                         # Versioned API route controllers (/api/v1/*)
│   │   ├── __init__.py
│   │   ├── advanced.py              # Advanced learning endpoints (semi-supervised, multitask, etc.)
│   │   ├── datasets.py              # Dataset ingestion, metadata, upload, deletion endpoints
│   │   ├── eda.py                   # Exploratory analysis, visual chart specs, HTML report endpoints
│   │   ├── explainability.py        # SHAP, LIME, PDP/ICE, calibration, and governance endpoints
│   │   ├── features.py              # Feature engineering, selection, and dimensionality reduction
│   │   ├── jobs.py                  # Asynchronous job monitoring and cancellation endpoints
│   │   ├── large_scale.py           # Polars streaming, downcasting, and out-of-core learning
│   │   ├── models.py                # Supervised model training, evaluation, and prediction
│   │   ├── system.py                # Hardware resources and operational status endpoints
│   │   └── wrangling.py             # Data quality hygiene, cleaning, scaling, and audit history
│   ├── core/                        # Core system utilities & configuration
│   │   ├── __init__.py
│   │   ├── config.py                # Pydantic Settings & YAML hierarchical configuration loader
│   │   ├── exceptions.py            # Centralized exception hierarchy (CancerwiseException)
│   │   └── logging_setup.py         # Standardized structured logging with rotation
│   ├── middleware/                  # HTTP interceptors & security layers
│   │   ├── __init__.py
│   │   ├── auth.py                  # API Key & JWT Bearer authentication with RBAC
│   │   ├── error_handler.py         # Centralized exception handling & RFC JSON envelopes
│   │   └── request_limits.py        # 100MB payload limit & path traversal defense
│   ├── schemas/                     # Pydantic request/response data contracts
│   │   ├── __init__.py
│   │   ├── advanced_learning.py     # Schemas for semi-supervised, multitask, multiclass, novelty
│   │   ├── dataset.py               # Dataset metadata, validation reports, and previews
│   │   ├── eda.py                   # Univariate, bivariate, multivariate, and hypothesis tests
│   │   ├── explainability.py        # SHAP, LIME, calibration, and governance schemas
│   │   ├── feature_engineering.py   # Derived features, selection rankings, dimensionality schemas
│   │   ├── large_scale.py           # Telemetry, downcasting, streaming benchmark schemas
│   │   ├── modeling.py              # Model training configs, metrics, and comparisons
│   │   ├── phase9_schemas.py        # Unified Section 2 request & response schemas
│   │   └── preprocessing.py         # Data quality reports, outlier reports, audit entries
│   └── services/                    # 17 Independent Domain Service Classes
│       ├── __init__.py              # Unified service exports
│       ├── data_validation_service.py      # Pandera contract validation service
│       ├── data_wrangling_service.py       # Imputation, deduplication, and winsorization
│       ├── dataset_ingestion_service.py    # Raw acquisition & multi-modal connectors
│       ├── dataset_management_service.py   # Registry querying, profiling, artifact deletion
│       ├── eda_service.py                  # Statistical mining & visual chart compilation
│       ├── explainability_service.py       # SHAP global/local & LIME surrogate service
│       ├── feature_engineering_service.py  # Cytological feature derivation service
│       ├── feature_selection_service.py    # Consensus Borda ranking & PCA/UMAP
│       ├── job_service.py                  # Thread-safe async JobManager singleton
│       ├── large_scale_service.py          # Memory optimization & out-of-core learning
│       ├── model_evaluation_service.py     # Clinical diagnostic metrics & calibration
│       ├── multiclass_service.py           # Molecular cancer subtype classification
│       ├── multitask_service.py            # PyTorch joint classification & regression
│       ├── one_class_service.py            # Normative benign cytology novelty detection
│       ├── prediction_service.py           # Governed prediction with feature validation
│       ├── regression_service.py           # Continuous biomarker regression service
│       ├── semi_supervised_service.py      # Label propagation in low-annotation regimes
│       ├── supervised_learning_service.py  # 16 supervised classification algorithms
│       └── system_service.py               # Host hardware telemetry & health service
│
├── configs/                         # Declarative System Configurations
│   ├── config.yaml                  # Base system parameters, paths, and hyperparameters
│   ├── development.yaml             # Development overrides (hot-reloading, debug logs)
│   └── production.yaml              # Production settings (multi-worker, locked security)
│
├── data/                            # Physical Data Storage Layers (Separation of Concerns)
│   ├── raw/                         # Immutable raw biological acquisition files
│   ├── processed/                   # Cleaned, optimized Parquet files with checksums
│   ├── interim/                     # Temporary staging for multi-modal ingestion
│   └── metadata/                    # JSON data dictionaries & dataset manifests
│
├── data_mining/                     # Core Machine Learning & Data Mining Modules
│   ├── classification/              # Supervised classification engines, trainer, & registry
│   ├── eda/                         # EDA engine, hypothesis testing, Plotly visualizer
│   ├── evaluation/                  # 12 clinical metrics, bootstrap CIs, comparison engine
│   ├── experiments/                 # Advanced experiment manager & research factory
│   └── regression/                  # Continuous biomarker regression engine
│
├── preprocessing/                   # Data Wrangling & Preprocessing Engine
│   ├── data_validator.py            # Pandera dataframe schema validation rules
│   ├── duplicate_manager.py         # Exact and cosine duplicate detection
│   ├── missing_values.py            # Median, mean, KNN, MICE imputation strategies
│   ├── outlier_detector.py          # IQR, Z-score, Isolation Forest, winsorization
│   ├── quality_analyzer.py          # Automated data cleanliness scoring (0-100)
│   ├── transformation_logger.py     # JSONL immutable audit history logger
│   └── transformers.py              # Reversible feature scalers & encoders
│
├── feature_engineering/             # Cytological Domain Synthesis & Selection
│   ├── domain_features.py           # Nuclear irregularity, sphericity, circularity
│   ├── feature_pipeline.py          # End-to-end scikit-learn feature pipeline
│   ├── feature_selector.py          # Filter, wrapper, embedded, and Borda selector
│   └── dimensionality.py            # PCA, Kernel PCA, and UMAP reducers
│
├── explainability/                  # Explainable AI (XAI) Frameworks
│   ├── shap_explainer.py            # TreeExplainer, LinearExplainer, PDP & ICE curves
│   └── lime_explainer.py            # LIME tabular surrogate sparse linear rules
│
├── evaluation/                      # Clinical Validation, Calibration & Lineage
│   ├── calibration_engine.py        # Reliability curves, ECE, Platt & Isotonic calibration
│   ├── governance_service.py        # MLflow lineage, model fingerprinting, dependencies
│   ├── model_validator.py           # 12 diagnostic metrics & bootstrap confidence intervals
│   ├── research_reporter.py         # Automated Markdown & HTML research report generator
│   └── robustness_analyzer.py       # Gaussian noise, MCAR missingness stress tester
│
├── frontend/                        # Streamlit Scientific Analytics Dashboard
│   ├── app.py                       # Main frontend portal & executive summary
│   ├── components/                  # 12 Reusable Scientific UI Cards & Containers
│   │   ├── __init__.py
│   │   ├── cards.py                 # MetricCard, DatasetUploader, DataPreview, etc.
│   │   ├── header.py                # Dynamic breadcrumbs, status badge, disclaimer
│   │   ├── sidebar.py               # 14-page navigation sidebar & cohort selector
│   │   └── theme.py                 # Scientific dark/light CSS injection engine
│   ├── pages/                       # 14 Independent Scientific Workspaces
│   │   ├── 01_Overview.py           # Platform architecture & executive KPIs
│   │   ├── 02_Dataset_Ingestion.py  # Cohort manager & multi-format uploader
│   │   ├── 03_Data_Wrangling.py     # Quality scoring, imputation & outlier studio
│   │   ├── 04_Exploratory_Data_Analysis.py # Univariate, bivariate, & hypothesis tests
│   │   ├── 05_Feature_Engineering.py# Cytological feature synthesis & Borda rank
│   │   ├── 06_Model_Training.py     # 16-algorithm supervised learning laboratory
│   │   ├── 07_Advanced_Learning_Lab.py # Semi-supervised, multitask, multiclass, novelty
│   │   ├── 08_Scalable_Processing.py# Large-scale Polars benchmarks & out-of-core SGD
│   │   ├── 09_Explainability_and_Validation.py # SHAP, LIME, PDP, calibration
│   │   ├── 10_Cancer_Prediction.py  # Governed patient inference & risk classification
│   │   ├── 11_Model_Governance.py   # MLflow lineage & SHA-256 audit manifest viewer
│   │   ├── 12_Research_Reports.py   # HTML/PDF publication-ready report downloads
│   │   ├── 13_API_Console.py        # Interactive REST API explorer & request tester
│   │   └── 14_System_Monitoring.py  # Host hardware telemetry & process RAM monitor
│   └── utils/                       # Frontend utilities
│       ├── __init__.py
│       └── api_client.py            # Dual-mode HTTP REST & fallback service bridge
│
├── models/                          # Serialized Model Artifacts & Lineage
│   ├── trained/                     # Pickled/joblib model pipelines (.joblib)
│   ├── registry/                    # Model JSON manifests (features, hyperparameters)
│   └── governance/                  # MLflow audit records and SHA-256 fingerprints
│
├── experiments/                     # MLflow Tracking & Run Storage
│   └── mlruns/                      # SQLite-backed experiment tracking database & runs
│
├── logs/                            # Rotating Structured Execution Logs
│   └── cancerwise.log               # Application, service, and security access logs
│
├── reports/                         # Compiled Scientific Research Artifacts
│   ├── eda/                         # Automated Exploratory Data Analysis HTML reports
│   └── evaluation/                  # Clinical Model Validation HTML & Markdown reports
│
├── scripts/                         # DevOps, Pre-Flight, & Startup Utilities
│   ├── preflight_check.py           # Pre-flight environment & dependency diagnostic
│   ├── run_all.bat                  # Windows 1-click launcher (preflight + API + UI)
│   ├── run_all.sh                   # Linux/macOS 1-click launcher (with trap cleanup)
│   ├── run_backend.bat              # Windows FastAPI backend launcher
│   ├── run_backend.sh               # Linux/macOS FastAPI backend launcher
│   ├── run_frontend.bat             # Windows Streamlit frontend launcher
│   └── run_frontend.sh              # Linux/macOS Streamlit frontend launcher
│
└── tests/                           # Comprehensive Automated Verification Suite
    ├── test_phase0.py               # Foundation & logging scaffold tests
    ├── test_phase1.py               # Data Ingestion Engine (23 tests)
    ├── test_phase2.py               # Data Wrangling & Preprocessing Engine (36 tests)
    ├── test_phase3.py               # Exploratory Data Analysis & Statistics tests
    ├── test_phase4.py               # Feature Engineering & Selection tests
    ├── test_phase5.py               # Supervised Learning & Prediction tests
    ├── test_phase6.py               # Advanced Learning Lab tests
    ├── test_phase7.py               # Large-Scale Learning & Optimization tests
    ├── test_phase8.py               # Explainable AI & Calibration Engine tests
    ├── test_phase9.py               # FastAPI Backend & 17 Service Integrations (40 tests)
    └── test_phase10.py              # Modern Frontend & Component Integration (26 tests)
```
