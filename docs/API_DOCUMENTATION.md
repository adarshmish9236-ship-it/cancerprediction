# CANCERWISE AI — API Documentation
## Version: v1.0.0 | Base Path: `/api/v1`

---

> [!IMPORTANT]
> **Clinical & Research Decision-Support Disclaimer:**
> The CANCERWISE AI REST API is designed strictly for academic, research, and clinical decision-support demonstration. It is not an FDA/CE-cleared diagnostic device.

---

## 1. Overview & Architecture

The CANCERWISE AI backend provides a modular, asynchronous RESTful API built on **FastAPI** and **Pydantic v2**. The API is versioned under `/api/v1` and connects directly to the Phase 1 through Phase 8 analytical engines.

### Key Capabilities
- **Authentication**: Bearer Token / API Key header (`X-API-Key` or `Authorization: Bearer <token>`).
- **Input Validation**: Strict Pydantic schema validation for request payloads and query parameters.
- **Asynchronous Execution**: Long-running model training and large dataset wrangling run in an internal background thread pool, tracked via unique `job_id` UUIDs.
- **Standardized Error Handling**: Centralized exception handlers returning RFC 7807 compliant error envelopes.

---

## 2. Authentication & Headers

Requests targeting protected endpoints require an API key in either of the following formats:

```http
Authorization: Bearer <API_KEY>
```
or
```http
X-API-Key: <API_KEY>
```

Default development API Key: `dev_cancerwise_api_key_2026` (configurable via `API_KEY` in `.env`).

---

## 3. Standard Response Format

### Success Response Envelope
```json
{
  "status": "success",
  "data": { ... },
  "message": "Operation completed successfully.",
  "timestamp": "2026-11-22T13:30:00.000Z"
}
```

### Error Response Envelope
```json
{
  "status": "error",
  "error_code": "RESOURCE_NOT_FOUND",
  "message": "Dataset with ID 'uci_wdbc' not found.",
  "details": {},
  "timestamp": "2026-11-22T13:30:00.000Z"
}
```

---

## 4. System & Health Endpoints

### 4.1 Health Check
- **Route**: `GET /health`
- **Auth**: None (Public)
- **Description**: Lightweight health probe for Kubernetes liveness and Docker healthcheck.

```bash
curl -X GET http://localhost:8000/health
```

**Response (200 OK)**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "production",
  "uptime_seconds": 1284.5
}
```

### 4.2 System Status
- **Route**: `GET /api/v1/system/status`
- **Auth**: Required
- **Description**: Health status of core analytical subsystems.

```bash
curl -X GET http://localhost:8000/api/v1/system/status \
  -H "X-API-Key: dev_cancerwise_api_key_2026"
```

**Response (200 OK)**:
```json
{
  "status": "operational",
  "services": {
    "dataset_engine": "online",
    "ml_registry": "online",
    "explainability_engine": "online",
    "job_queue": "idle"
  },
  "timestamp": "2026-11-22T13:30:00Z"
}
```

### 4.3 Hardware & System Resources
- **Route**: `GET /api/v1/system/resources`
- **Auth**: Required
- **Description**: Real-time CPU, RAM, and disk utilization telemetry.

```bash
curl -X GET http://localhost:8000/api/v1/system/resources \
  -H "X-API-Key: dev_cancerwise_api_key_2026"
```

---

## 5. Dataset Management APIs

### 5.1 Upload Dataset
- **Route**: `POST /api/v1/datasets/upload`
- **Content-Type**: `multipart/form-data`
- **Auth**: Required
- **Parameters**:
  - `file`: CSV or Parquet binary file (max size: 100 MB).
  - `dataset_id`: Custom alphanumeric identifier.
  - `dataset_name`: Friendly display name.
  - `description`: Optional clinical metadata.

```bash
curl -X POST http://localhost:8000/api/v1/datasets/upload \
  -H "X-API-Key: dev_cancerwise_api_key_2026" \
  -F "file=@data/raw/wdbc.data" \
  -F "dataset_id=custom_cohort_01" \
  -F "dataset_name=Custom Oncology Cohort"
```

### 5.2 List Datasets
- **Route**: `GET /api/v1/datasets`
- **Auth**: Required
- **Description**: Enumerate registered datasets and storage paths.

### 5.3 Get Dataset Details
- **Route**: `GET /api/v1/datasets/{dataset_id}`
- **Auth**: Required

### 5.4 Get Dataset Profile & Quality Summary
- **Route**: `GET /api/v1/datasets/{dataset_id}/profile`
- **Auth**: Required
- **Description**: Returns sample count, feature count, missingness, and column data types.

### 5.5 Delete Dataset
- **Route**: `DELETE /api/v1/datasets/{dataset_id}`
- **Auth**: Required

---

## 6. Data Wrangling & Quality APIs

### 6.1 Analyze Data Quality
- **Route**: `POST /api/v1/wrangling/analyze`
- **Auth**: Required
- **Payload**:
```json
{
  "dataset_id": "uci_wdbc"
}
```
**Response**:
Returns hygiene score (0-100), duplicate count, missing value counts, and outlier summary.

### 6.2 Clean Dataset
- **Route**: `POST /api/v1/wrangling/clean`
- **Auth**: Required
- **Payload**:
```json
{
  "dataset_id": "uci_wdbc",
  "imputation_strategy": "median",
  "remove_duplicates": true,
  "outlier_handling": "clip"
}
```

### 6.3 Transform Features
- **Route**: `POST /api/v1/wrangling/transform`
- **Auth**: Required
- **Payload**:
```json
{
  "dataset_id": "uci_wdbc",
  "scaling_strategy": "standard",
  "skewness_strategy": "yeo_johnson"
}
```

### 6.4 Wrangling Audit Trail
- **Route**: `GET /api/v1/wrangling/history?dataset_id=uci_wdbc`
- **Auth**: Required

---

## 7. Exploratory Data Analysis (EDA) APIs

### 7.1 Statistical Summary Analysis
- **Route**: `POST /api/v1/eda/analyze`
- **Auth**: Required
- **Payload**:
```json
{
  "dataset_id": "uci_wdbc",
  "include_correlations": true,
  "include_distributions": true
}
```

### 7.2 Get Visualizations
- **Route**: `GET /api/v1/eda/visualizations?dataset_id=uci_wdbc&plot_type=correlation_matrix`
- **Auth**: Required
- **Supported Plot Types**: `correlation_matrix`, `pairplot`, `pca_projection`, `class_distribution`.

### 7.3 Generate Full EDA Report
- **Route**: `GET /api/v1/eda/report?dataset_id=uci_wdbc&format=html`
- **Auth**: Required

---

## 8. Feature Engineering APIs

### 8.1 Engineer Domain Cytological Features
- **Route**: `POST /api/v1/features/engineer`
- **Auth**: Required
- **Payload**:
```json
{
  "dataset_id": "uci_wdbc",
  "enable_cytological_ratios": true,
  "enable_polynomial_interactions": false
}
```

### 8.2 Select Key Predictive Features
- **Route**: `POST /api/v1/features/select`
- **Auth**: Required
- **Payload**:
```json
{
  "dataset_id": "uci_wdbc",
  "method": "mutual_info",
  "k_features": 15
}
```

### 8.3 Dimensionality Reduction (PCA / t-SNE)
- **Route**: `POST /api/v1/features/reduce`
- **Auth**: Required
- **Payload**:
```json
{
  "dataset_id": "uci_wdbc",
  "method": "pca",
  "n_components": 5
}
```

---

## 9. Supervised Machine Learning APIs

### 9.1 List Registered Models
- **Route**: `GET /api/v1/models`
- **Auth**: Required

### 9.2 Train Supervised Model
- **Route**: `POST /api/v1/models/train`
- **Auth**: Required
- **Payload**:
```json
{
  "dataset_id": "uci_wdbc",
  "model_type": "logistic_regression",
  "hyperparameters": {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 1000
  },
  "cross_validation_folds": 5,
  "enable_smote": false
}
```
**Response**:
```json
{
  "status": "success",
  "model_id": "uci_wdbc_logistic_regression_57cbbb98",
  "training_duration_seconds": 0.42,
  "metrics": {
    "accuracy": 0.9824,
    "sensitivity": 0.9655,
    "specificity": 0.9928,
    "roc_auc": 0.9958,
    "f1_score": 0.9767
  }
}
```

### 9.3 Get Model Artifact Details
- **Route**: `GET /api/v1/models/{model_id}`
- **Auth**: Required

### 9.4 Comprehensive Clinical Model Evaluation
- **Route**: `GET /api/v1/models/{model_id}/evaluate`
- **Auth**: Required
- **Returns**: 12 clinical metrics, confusion matrix, ROC curve coordinates, and calibration curves.

### 9.5 Model Inference / Prediction
- **Route**: `POST /api/v1/models/{model_id}/predict`
- **Auth**: Required
- **Payload**:
```json
{
  "features": {
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    "perimeter_mean": 122.8,
    "area_mean": 1001.0,
    "smoothness_mean": 0.1184,
    "compactness_mean": 0.2776,
    "concavity_mean": 0.3001,
    "concave_points_mean": 0.1471,
    "symmetry_mean": 0.2419,
    "fractal_dimension_mean": 0.07871,
    "radius_se": 1.095,
    "texture_se": 0.9053,
    "perimeter_se": 8.589,
    "area_se": 153.4,
    "smoothness_se": 0.006399,
    "compactness_se": 0.04904,
    "concavity_se": 0.05373,
    "concave_points_se": 0.01587,
    "symmetry_se": 0.03003,
    "fractal_dimension_se": 0.006193,
    "radius_worst": 25.38,
    "texture_worst": 17.33,
    "perimeter_worst": 184.6,
    "area_worst": 2019.0,
    "smoothness_worst": 0.1622,
    "compactness_worst": 0.6656,
    "concavity_worst": 0.7119,
    "concave_points_worst": 0.2654,
    "symmetry_worst": 0.4601,
    "fractal_dimension_worst": 0.1189
  }
}
```
**Response**:
```json
{
  "prediction_id": "pred_9f81a742",
  "model_id": "uci_wdbc_logistic_regression_57cbbb98",
  "prediction": "Malignant",
  "probability_malignant": 0.9942,
  "probability_benign": 0.0058,
  "confidence_level": "High",
  "clinical_warning": "Educational prototype — not for direct clinical triage."
}
```

---

## 10. Advanced Learning Paradigms

- `POST /api/v1/advanced/semi-supervised/train`: Label Propagation & Label Spreading on partially labeled datasets.
- `POST /api/v1/advanced/multitask/train`: Multi-task classification of malignancy + nuclear grade simultaneously.
- `POST /api/v1/advanced/multiclass/train`: Multi-class histopathological classification (Benign, In Situ, Invasive, Metastatic).
- `POST /api/v1/advanced/one-class/train`: Novelty and anomaly detection using One-Class SVM and Isolation Forest.
- `POST /api/v1/advanced/regression/train`: Tumor size and mitotic index progression regression.
- `POST /api/v1/advanced/kernel/train`: Kernel Ridge and RBF Support Vector Machines.

---

## 11. Explainable AI (XAI) APIs

### 11.1 SHAP Attribution
- **Route**: `POST /api/v1/explainability/shap`
- **Auth**: Required
- **Payload**:
```json
{
  "model_id": "uci_wdbc_random_forest_ee63920a",
  "sample_index": 0,
  "explainer_type": "tree"
}
```
**Returns**: Base value, Shapley feature attributions, and top 5 risk-contributing cytological metrics.

### 11.2 LIME Local Surrogate Explanation
- **Route**: `POST /api/v1/explainability/lime`
- **Auth**: Required
- **Payload**:
```json
{
  "model_id": "uci_wdbc_logistic_regression_57cbbb98",
  "sample_index": 0,
  "num_features": 10
}
```

### 11.3 Retrieve Model Global Feature Importance
- **Route**: `GET /api/v1/explainability/{model_id}`
- **Auth**: Required

---

## 12. Asynchronous Job Management APIs

### 12.1 List Background Jobs
- **Route**: `GET /api/v1/jobs`
- **Auth**: Required

### 12.2 Get Job Status & Progress
- **Route**: `GET /api/v1/jobs/{job_id}`
- **Auth**: Required
- **Response**:
```json
{
  "job_id": "7df44a2b-b6d1-4475-8ce9-930438ec67cb",
  "status": "COMPLETED",
  "task_type": "MODEL_TRAINING",
  "progress_pct": 100.0,
  "created_at": "2026-11-22T13:25:00Z",
  "completed_at": "2026-11-22T13:25:02Z",
  "error": null,
  "result_summary": {
    "model_id": "uci_wdbc_random_forest_1f4b4fa3",
    "roc_auc": 0.9982
  }
}
```

### 12.3 Cancel / Delete Job
- **Route**: `DELETE /api/v1/jobs/{job_id}`
- **Auth**: Required
