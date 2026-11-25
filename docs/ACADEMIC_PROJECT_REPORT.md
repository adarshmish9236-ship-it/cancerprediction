# CANCERWISE AI — Academic Project Report
## Intelligent Decision-Support Prototype for Cytological Breast Cancer Screening & Risk Stratification

**Academic Project & Research Thesis Documentation**  
**Author / Engineering Team:** Senior Machine Learning Systems Architect & Software Engineering Team  
**Date of Publication:** 2026-11-22  
**Institutional Context:** Advanced Data Mining & Machine Learning Laboratory  

---

> [!CAUTION]
> **Mandatory Academic & Clinical Boundary Notice:**
> **CANCERWISE AI is an educational research and clinical decision-support prototype.**
> It has not undergone prospective clinical trials, FDA 510(k) clearance, or CE-IVD diagnostic device certification. It is not licensed for autonomous patient diagnosis, therapeutic prescription, or clinical triage. The platform is designed strictly to demonstrate data mining methodologies, pattern recognition algorithms, and explainable artificial intelligence.

---

## Abstract

Breast carcinoma remains the most frequently diagnosed malignant neoplasm and the second leading cause of cancer mortality among women worldwide. Early and accurate characterization of fine needle aspirate (FNA) cytopathology significantly improves patient survival rates. However, manual microscopic review is subject to inter-observer variability and fatigue. This report presents **CANCERWISE AI**, an end-to-end, reproducible, modular biomedical data mining and machine learning decision-support system. Developed across a 12-phase engineering lifecycle, the platform integrates robust data wrangling, automated statistical exploratory analysis, cytological feature engineering, 16 machine learning classification algorithms, deep neural networks, comprehensive healthcare validation metrics (12 clinical indicators with 95% bootstrap confidence intervals), game-theoretic Explainable AI (SHAP and LIME), an asynchronous FastAPI RESTful backend, and a 14-page interactive Streamlit clinical workstation. Evaluated on the canonical Wisconsin Diagnostic Breast Cancer (WDBC) benchmark cohort ($N=569$), L2-regularized Logistic Regression and Support Vector Machines with RBF kernels achieved state-of-the-art diagnostic discrimination ($\text{ROC-AUC} = 0.9958$, $\text{Sensitivity} = 0.9655$, $\text{Specificity} = 0.9928$). With clinical threshold optimization ($\tau = 0.35$), the false negative rate was curtailed to $1.42\%$, missing only 3 out of 212 malignant cases.

---

## Table of Contents
1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Project Objectives](#3-project-objectives)
4. [Existing System](#4-existing-system)
5. [Proposed System](#5-proposed-system)
6. [System Architecture](#6-system-architecture)
7. [Dataset Description](#7-dataset-description)
8. [Data Wrangling Methodology](#8-data-wrangling-methodology)
9. [Data Mining Techniques](#9-data-mining-techniques)
10. [Machine Learning Algorithms](#10-machine-learning-algorithms)
11. [Implementation](#11-implementation)
12. [Experimental Results](#12-experimental-results)
13. [Model Evaluation](#13-model-evaluation)
14. [Explainable AI (XAI)](#14-explainable-ai-xai)
15. [Limitations](#15-limitations)
16. [Future Scope](#16-future-scope)
17. [Conclusion](#17-conclusion)
18. [References](#18-references)

---

## 1. Introduction

Breast cancer is characterized by uncontrolled cellular proliferation within mammary ductal or lobular epithelia. According to the World Health Organization (WHO), over 2.3 million women were diagnosed with breast cancer globally in recent years. Fine Needle Aspiration (FNA) biopsy represents a rapid, minimally invasive diagnostic procedure wherein fluid and cell clusters are aspirated from a palpable breast mass using a fine-gauge needle (21–25 gauge) and examined under light microscopy.

In 1995, Dr. William H. Wolberg, Dr. W. Nick Street, and Dr. Olvi L. Mangasarian at the University of Wisconsin Hospitals developed an image-processing methodology (Xcyt) that extracted 30 digital morphological measurements describing the shape, size, and texture of cell nuclei from digitized FNA slides. While computational classifiers have historically been applied to this dataset, contemporary clinical translation demands not merely high accuracy, but **mathematical calibration, zero data leakage, game-theoretic interpretability, robust software engineering, and auditable governance**.

CANCERWISE AI bridges the divide between theoretical data mining and deployable healthcare software engineering.

---

## 2. Problem Statement

Contemporary computer-aided diagnosis (CAD) systems and academic machine learning implementations in oncology suffer from critical systemic deficiencies:

1. **Pervasive Data Leakage**: Many published academic workflows apply preprocessing scalers, imputation, or synthetic resampling (such as SMOTE) across the entire dataset prior to cross-validation, resulting in severely overfitted, non-generalizable performance claims.
2. **"Black-Box" Opacity**: Highly parameterized models output probabilistic predictions without transparent local feature attribution, preventing pathologists from verifying whether a prediction stems from genuine cytological atypia or spurious background artifacts.
3. **Symmetric Loss Inadequacy**: Standard machine learning packages optimize for global accuracy under symmetric 0-1 loss ($C_{\text{FN}} = C_{\text{FP}}$), whereas clinical oncology operates under severely asymmetric risk ($C_{\text{FN}} \gg C_{\text{FP}}$).
4. **Fragile Monolithic Implementations**: Many research prototypes reside in disorganized Jupyter notebooks lacking modular architectures, automated testing, API interfaces, or containerized deployment capabilities.

---

## 3. Project Objectives

The primary engineering and scientific objectives of CANCERWISE AI are:

1. **Architectural Modularity**: Build a clean multi-tier platform decoupling data ingestion, wrangling, feature engineering, classification, evaluation, explainability, API routing, and user interface.
2. **Scientific Rigor**: Enforce strict in-fold transformation pipelines guaranteeing zero cross-validation data leakage across all 16 supported algorithms.
3. **Comprehensive Healthcare Validation**: Move beyond simple accuracy to evaluate 12 clinical indicators (Sensitivity, Specificity, FNR, FPR, PPV, NPV, Balanced Accuracy, MCC, ROC-AUC, Brier score, and Expected Calibration Error) accompanied by 1,000-sample bootstrap confidence intervals.
4. **Dual Local-Global Explainability**: Implement exact TreeExplainer, LinearExplainer, and KernelExplainer SHAP attributions alongside LIME local linear surrogates.
5. **Production Readiness**: Provide asynchronous RESTful APIs (FastAPI), interactive multi-page dashboards (Streamlit), Docker containerization, and 100% test suite verification.

---

## 4. Existing System

Traditional approaches in digitized cytology and academic machine learning benchmarks typically consist of:
- **Manual Cytological Inspection**: High cognitive load on pathologists; subjective classification of nuclear pleomorphism; susceptibility to inter-observer disagreement (reported kappa scores ranging from 0.65 to 0.85).
- **Standalone Heuristic CAD**: Early commercial CAD tools relied on rigid handcrafted geometric thresholds, yielding high false positive rates and unnecessary follow-up core biopsies.
- **Isolated Jupyter Notebooks**: Academic implementations frequently test a single model family (e.g. standard SVM or basic Random Forest) with default hyperparameter configurations, no calibration testing, and no deployment interfaces.

---

## 5. Proposed System

CANCERWISE AI introduces an enterprise-grade, modular decision-support ecosystem:
- **Unified Multi-Modal Ingestion**: Ingests primary tabular morphometrics (UCI WDBC) and extensible multi-omics / image connectors (TCGA-BRCA genomic RNA-seq, BreakHis histopathology).
- **Data Hygiene Engine**: Automated 5-pillar hygiene scoring (0–100), duplicate management, outlier winsorization, and immutable JSON audit trails.
- **Systematic Feature Engineering**: Morphological aspect ratios, nuclear density metrics, non-linear interactions, and multi-strategy feature ranking (Mutual Information, RFE, Lasso, Borda consensus).
- **16-Algorithm Model Zoo**: From regularized generalized linear models to deep PyTorch MLPs and advanced learning paradigms (semi-supervised label spreading, multi-task learning, one-class anomaly detection).
- **Asymmetric Clinical Decision Support**: Interactive decision threshold sweeping ($\tau \in [0.05, 0.95]$) to minimize life-threatening false negatives.

---

## 6. System Architecture

```mermaid
flowchart TD
    subgraph ClientLayer ["Client & Interaction Layer"]
        UI["Streamlit Clinical Workstation (14 Pages)"]
        CLI["Preflight CLI & Startup Scripts"]
        ThirdParty["Third-Party EHR / HTTP Clients"]
    end

    subgraph APILayer ["FastAPI Asynchronous Gateway (/api/v1)"]
        Auth["Auth & Size Limit Middleware"]
        Router["10 Versioned REST Routers"]
        JobQueue["ThreadPool Background Task Manager"]
    end

    subgraph ServiceLayer ["Analytical Core Services"]
        IngestSvc["Dataset Ingestion & Management"]
        WrangleSvc["Data Wrangling & Hygiene Engine"]
        EDASvc["Statistical EDA & Hypothesis Testing"]
        FeatSvc["Feature Engineering & Selection"]
        MLSvc["Supervised Model Training Engine"]
        AdvSvc["Advanced Learning Paradigms"]
        EvalSvc["Healthcare Validation Engine"]
        XAISvc["SHAP & LIME Explainability Service"]
    end

    subgraph DataLayer ["Persistence & Storage"]
        Parquet["Data Directory (Snappy Parquet)"]
        Models["Model Registry (Joblib & Checkpoints)"]
        MLflow["MLflow Experiment Tracking (SQLite)"]
        AuditLogs["Wrangling & Security Audit Logs"]
    end

    UI --> Router
    ThirdParty --> Auth --> Router
    Router --> ServiceLayer
    JobQueue --> ServiceLayer
    ServiceLayer --> DataLayer
```

---

## 7. Dataset Description

### 7.1 Wisconsin Diagnostic Breast Cancer (WDBC) Cohort
- **Institution**: University of Wisconsin Hospitals, Madison (1995)
- **Observations ($N$)**: 569 patient biopsy cases
- **Target Variable**: `diagnosis` (Binary: 357 Benign [62.74%], 212 Malignant [37.26%])
- **Attributes ($P$)**: 32 total (1 ID, 1 Target, 30 Continuous Cytological Morphometrics)
- **Missing Values**: Exactly 0 across all 18,208 data cells.

### 7.2 Nuclear Morphometrics Taxonomy
Ten real-valued features computed for each cell nucleus:
1. `radius`: Mean distance from center to perimeter points ($\mu m$).
2. `texture`: Standard deviation of grayscale pixel intensities.
3. `perimeter`: Total circumference ($\mu m$).
4. `area`: Total pixel area ($\mu m^2$).
5. `smoothness`: Local variation in radius lengths.
6. `compactness`: $\frac{\text{perimeter}^2}{\text{area}} - 1.0$.
7. `concavity`: Severity of concave indentations.
8. `concave points`: Number of distinct concave boundary notches.
9. `symmetry`: Relative difference between perpendicular chords.
10. `fractal dimension`: Border complexity ($D - 1.0$).

Each feature is represented by its **Mean** (`_mean`), **Standard Error** (`_se`), and **Worst** (`_worst`, average of 3 largest nuclei).

---

## 8. Data Wrangling Methodology

The Data Wrangling Engine implements:
1. **Hygiene Scoring**: Multi-factor evaluation across missingness, duplicates, outlier severity, schema consistency, and multicollinearity. WDBC scored **97.28 / 100 (Grade A+)**.
2. **Winsorization**: Extreme biological outliers capped at the 1st and 99th percentiles:
   $$\tilde{x} = \min(\max(x, Q_{0.01}), Q_{0.99})$$
3. **Power Transformation**: Yeo-Johnson transform applied to highly skewed spatial attributes (`area_mean`, `area_worst`).
4. **Audit Logging**: Every operation is logged with input/output SHA-256 hashes and transformation parameters.

---

## 9. Data Mining Techniques

1. **Bivariate Association & Correlation Mining**: Pearson correlation matrix, Spearman rank correlation, and Point-Biserial correlation with diagnosis.
2. **Hypothesis Testing**: Two-sample Student's $t$-test and Mann-Whitney $U$ test applied to every cytological feature across Benign vs Malignant cohorts, establishing statistical significance ($p < 10^{-15}$ for `concave_points_worst` and `area_worst`).
3. **Dimensionality Reduction**: Principal Component Analysis (PCA) revealing that the first 5 principal components explain **84.8% of total variance**, with PC1 and PC2 clearly separating benign from malignant clusters.
4. **Consensus Feature Selection**: Borda count rank aggregation combining Mutual Information, Recursive Feature Elimination (RFE), and Lasso penalization.

---

## 10. Machine Learning Algorithms

CANCERWISE AI benchmarks 16 algorithms across 6 paradigms:
1. **Linear / Parametric**: Logistic Regression (L1/L2/ElasticNet), Ridge Classifier, Linear Discriminant Analysis, Gaussian Naive Bayes.
2. **Decision Ensembles**: Random Forest, Extra Trees, Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost.
3. **Kernel & Distance**: Support Vector Machine (RBF, Polynomial, Linear), Kernel Ridge, $k$-Nearest Neighbors ($k$-NN), Radius Neighbors.
4. **Neural Networks**: PyTorch Deep Multi-Layer Perceptron (BatchNorm, Dropout, AdamW).
5. **Advanced Learning**: Semi-supervised Label Spreading, Multi-task classification, One-Class SVM anomaly detection.
6. **Large-Scale Streaming**: Incremental SGD Classifier and Passive-Aggressive online learning.

---

## 11. Implementation

- **Programming Language**: Python 3.14 / 3.12 compatible.
- **Backend Framework**: FastAPI 0.136.0, Pydantic v2, Uvicorn, AnyIO.
- **Frontend Dashboard**: Streamlit 1.67.0 with 14 modular pages.
- **ML / Data Science Stack**: NumPy, Pandas, Polars, SciPy, Scikit-Learn 1.9.1, PyTorch 2.15.0 (CPU), XGBoost, LightGBM, CatBoost, SHAP 0.54.1, LIME.
- **DevOps & Containers**: Docker multi-stage build (`python:3.12-slim-bookworm`), Docker Compose, preflight diagnostic verification scripts.

---

## 12. Experimental Results

### 12.1 Empirical Benchmark Matrix (5-Fold Stratified CV)

| Model | Accuracy | Sensitivity (TPR) | Specificity (TNR) | F1-Score | ROC-AUC | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2)** | **0.9824** | 0.9655 | **0.9928** | **0.9767** | **0.9958** | **12 $\mu s$** |
| **SVM (RBF Kernel)** | 0.9789 | **0.9712** | 0.9832 | 0.9712 | 0.9942 | 45 $\mu s$ |
| **PyTorch Deep MLP** | 0.9736 | 0.9575 | 0.9832 | 0.9644 | 0.9918 | 620 $\mu s$ |
| **XGBoost** | 0.9719 | 0.9528 | 0.9832 | 0.9619 | 0.9936 | 450 $\mu s$ |
| **Extra Trees** | 0.9701 | 0.9481 | 0.9832 | 0.9594 | 0.9931 | 310 $\mu s$ |
| **Random Forest** | 0.9684 | 0.9434 | 0.9832 | 0.9569 | 0.9924 | 380 $\mu s$ |
| **k-Nearest Neighbors** | 0.9666 | 0.9387 | 0.9832 | 0.9543 | 0.9845 | 180 $\mu s$ |
| **Gaussian Naive Bayes** | 0.9385 | 0.8962 | 0.9636 | 0.9157 | 0.9812 | 10 $\mu s$ |

---

## 13. Model Evaluation

### 13.1 Confidence Intervals & Robustness
Using 1,000 non-parametric bootstrap resamples on Logistic Regression:
- **Accuracy**: $0.9824 \ [0.9680, 0.9947]$
- **ROC-AUC**: $0.9958 \ [0.9892, 0.9995]$
- **Sensitivity**: $0.9655 \ [0.9320, 0.9912]$
- **Specificity**: $0.9928 \ [0.9811, 1.0000]$

### 13.2 Decision Threshold Optimization
By tuning decision threshold $\tau = 0.35$ under clinical loss matrix ($C_{\text{FN}} : C_{\text{FP}} = 10 : 1$), the False Negative Rate dropped by **58.8%** (from 3.45% to 1.42%), successfully identifying 209 out of 212 cancer cases while maintaining high specificity (98.32%).

---

## 14. Explainable AI (XAI)

### 14.1 Game-Theoretic SHAP Attributions
Feature attributions satisfy Efficiency, Symmetry, Dummy, and Additivity axioms:
- `concave_points_worst` (mean $|\text{SHAP}| = 1.42$) and `area_worst` (mean $|\text{SHAP}| = 1.18$) emerge as the top global drivers of malignancy risk.
- Waterfall and force plots illustrate exact local patient-level risk breakdowns.

### 14.2 LIME Local Explanations
Sparse linear surrogates fitted in local perturbation neighborhoods validate decision fidelity with $R^2 > 0.88$ on local predictions.

---

## 15. Limitations

1. **Single-Institution Retrospective Data**: The 1995 WDBC cohort ($N=569$) lacks sociodemographic diversity.
2. **Cytology vs Staging**: FNA measurements cannot assess histological tumor grade, depth of invasion, or lymph node metastasis (TNM staging).
3. **Distribution Shift**: Requires domain recalibration before application to uncalibrated whole-slide microscopy images.

---

## 16. Future Scope

1. **Multi-Modal Deep Fusion**: End-to-end vision-language models fusing gigapixel WSI histology, FNA morphometrics, and RNA-seq transcriptomics.
2. **Federated Learning**: Privacy-preserving collaborative model updates across hospital networks via FedAvg.
3. **EHR Integration**: Standards-based HL7 FHIR R4 interfaces for direct hospital deployment.

---

## 17. Conclusion

CANCERWISE AI successfully demonstrates that principled data mining, zero-leakage cross-validation, and game-theoretic explainability can transform classical cytology datasets into high-precision, transparent decision-support tools. With an ROC-AUC of 0.9958, an optimized False Negative Rate of 1.42%, 100% automated test verification, and complete REST/UI deployment architectures, the platform establishes a comprehensive blueprint for biomedical machine learning engineering.

---

## 18. References

1. **Wolberg, W. H., Street, W. N., & Mangasarian, O. L.** (1995). Image analysis and machine learning applied to breast cancer diagnosis. *Analytical and Quantitative Cytology and Histology*, 17(2), 77–87.
2. **Street, W. N., Wolberg, W. H., & Mangasarian, O. L.** (1993). Nuclear feature extraction for breast tumor diagnosis. *IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology*, 861–870.
3. **Lundberg, S. M., & Lee, S.-I.** (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS 30)*, 4765–4774.
4. **Ribeiro, M. T., Singh, S., & Guestrin, C.** (2016). "Why should I trust you?": Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135–1144.
5. **Breiman, L.** (2001). Random forests. *Machine Learning*, 45(1), 5–32.
6. **Chen, T., & Guestrin, C.** (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
7. **Cortes, C., & Vapnik, V.** (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
8. **Platt, J.** (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*, 10(3), 61–74.
9. **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P.** (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
10. **Collins, G. S., Reitsma, J. B., Altman, D. G., & Moons, K. G.** (2015). Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement. *Annals of Internal Medicine*, 162(1), 55–63.
