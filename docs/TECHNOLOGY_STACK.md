# CANCERWISE AI — Technology Stack & Architectural Justification

## 1. Core Runtime & Language Environment

| Component | Verified Version | Production Target | Architectural Justification |
|-----------|------------------|-------------------|-----------------------------|
| **Python** | `3.14.0` | `3.12 LTS` / `3.14` | Universal scientific language; support for native async/await, modern type hint annotations (PEP 585/604), and high-performance C-extensions. |
| **pip** | `26.3` | `>= 24.0` | Standard, reproducible wheel dependency resolution. |
| **Uvicorn** | `0.54.0` | `>= 0.29.0` | Lightning-fast ASGI web server implementation built atop `uvloop` and `httptools`. |

---

## 2. Comprehensive Dependency Stack

### 2.1. Machine Learning & Statistical Computing
| Library | Version | Technical Justification |
|---------|---------|-------------------------|
| **scikit-learn** | `1.9.1` | Unified estimator API (`fit`/`transform`/`predict`), ColumnTransformers, calibration curves, bootstrap metrics, and cross-validation pipelines. |
| **imbalanced-learn** | `0.14.2` | In-fold SMOTE, ADASYN, and random over/under sampling preventing class imbalance bias without test set data leakage. |
| **PyTorch (`torch`)** | `2.15.0+cpu` | PyTorch autograd engine, tensor computing, customizable deep Multi-Layer Perceptrons, and joint multitask neural networks. |
| **xgboost** | `3.4.1` | Extreme Gradient Boosting providing high tree depth, regularization ($L_1 / L_2$), and fast inference. |
| **lightgbm** | `4.7.0` | Leaf-wise tree growth gradient boosting, histogram binning, and low-latency predictions. |
| **catboost** | `1.2.10` | Symmetric decision trees, native handling of categorical features, and robustness to overfitting. |
| **scipy** | `1.18.1` | Statistical hypothesis testing (`mannwhitneyu`, `f_oneway`, `kruskal`, `chisquare`, `shapiro`, `levene`). |
| **numpy** | `2.5.1` | High-dimensional array computation and linear algebra. |

### 2.2. High-Throughput & Columnar Data Processing
| Library | Version | Technical Justification |
|---------|---------|-------------------------|
| **polars** | `2.1.0` | Rust-backed columnar DataFrame library; multi-threaded execution, memory-mapped streaming, and lazy execution queries. |
| **pandas** | `3.1.1` | Rich tabular manipulation, timestamp manipulation, and integration with scikit-learn. |
| **pyarrow** | `15.0.0` | Cross-language Apache Arrow columnar memory format, Parquet reading/writing, and zero-copy data exchange. |
| **pandera** | `0.20.0` | Statistical data contract validation and DataFrame schema enforcement. |

### 2.3. Explainable AI & Model Governance
| Library | Version | Technical Justification |
|---------|---------|-------------------------|
| **shap** | `0.54.1` | Game-theoretic cooperative Shapley values (`TreeExplainer`, `LinearExplainer`, `KernelExplainer`, and `Explainer`). |
| **lime** | `0.2.0.1` | Local Interpretable Model-agnostic Explanations with sparse linear surrogate approximations and fidelity metrics. |
| **mlflow** | `3.18.0` | End-to-end experiment lineage tracking, model registry, artifact storage, and reproducibility auditing. |

### 2.4. Backend API & Web Tier
| Library | Version | Technical Justification |
|---------|---------|-------------------------|
| **fastapi** | `0.136.0` | Asynchronous REST microframework with automatic OpenAPI / Swagger generation, dependency injection, and Pydantic validation. |
| **pydantic** | `2.14.3` | Ultra-fast Rust-compiled data validation and strict serialization contracts. |
| **python-multipart** | `0.0.9` | Secure streaming parsing of multi-part form data and file uploads. |
| **psutil** | `7.2.2` | Process and host hardware telemetry (CPU, physical RAM, process RSS memory). |

### 2.5. Frontend Presentation & Visualization
| Library | Version | Technical Justification |
|---------|---------|-------------------------|
| **streamlit** | `1.67.0` | Rapid, interactive data science application framework with live reactive state management. |
| **plotly** | `7.1.0` | Interactive JavaScript-rendered charting (scatter, heatmaps, waterfall, ROC/PR curves). |
| **matplotlib / seaborn** | `3.11.2 / 0.13.2` | Publication-ready vector graphics and publication figure synthesis. |

---

## 3. Concurrency & Concurrency Models

```
FastAPI Server (Main Process)
 ├── Event Loop (Asyncio)
 │    ├── Request Routing & Pydantic Validation
 │    ├── Lightweight HTTP Request Handlers
 │    └── Health Checks & Telemetry (< 5ms response)
 │
 └── ThreadPoolExecutor (JobManager Worker Pool)
      ├── Worker 1: Model Training (Scikit-Learn / PyTorch)
      ├── Worker 2: SHAP Global Feature Value Computation
      ├── Worker 3: Large-Scale Polars Columnar Processing
      └── Worker 4: Automated HTML Report Synthesis
```

- **Thread-Safety:** Background tasks are isolated from the main async loop via `JobManager`, ensuring the FastAPI server remains 100% responsive under heavy training workloads.
- **Memory Boundary:** Large-scale learning module downcasts double-precision floats to single-precision (`float32`), reducing in-memory working sets by 50%.
