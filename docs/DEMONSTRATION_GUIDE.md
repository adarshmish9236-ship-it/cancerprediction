# CANCERWISE AI — Final Project Demonstration & Defense Guide

---

> [!NOTE]
> **Purpose of this Guide:**
> This document provides a step-by-step walkthrough for faculty evaluators, viva panels, research audiences, and demonstration review. It guides you through the preflight verification, system startup, and a sequential tour through all 14 frontend pages and backend REST APIs.

---

## 1. Quick Launch Sequence

### Step 1: Execute Preflight Environment Diagnostics
Run the automated pre-flight checker to verify Python version, required packages, directory write access, dataset presence, and port availability:

```bash
python scripts/preflight_check.py
```
*Expected Output*: `[SUCCESS] All preflight system checks passed (0 warnings, 0 errors).`

### Step 2: Launch Platform Services
Execute the single-command startup script:

**On Windows:**
```cmd
scripts\run_all.bat
```

**On Linux / macOS:**
```bash
bash scripts/run_all.sh
```

**Or Launch Independently in Two Terminals:**
- **Terminal 1 (Backend API)**:
  ```bash
  python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
  ```
- **Terminal 2 (Frontend Dashboard)**:
  ```bash
  python -m streamlit run frontend/app.py --server.port 8501
  ```

### Step 3: Open Web Interfaces
- **Streamlit Clinical Workstation**: [http://localhost:8501](http://localhost:8501)
- **FastAPI OpenAPI Interactive Swagger**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **FastAPI Health Probe**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 2. Step-by-Step Viva & Demonstration Script

### Act 1: Platform Overview & System Health (Page 1)
1. Navigate to **Home / Overview** on the sidebar.
2. Highlight the **Clinical Decision Support Notice** prominently displayed at the top, emphasizing that the system is an educational and research prototype.
3. Review the system telemetry cards: active Python runtime (3.14/3.12), memory status, and core analytical service health (Dataset Engine, ML Registry, Explainability Engine all `ONLINE`).

### Act 2: Dataset Ingestion & Profile Inspection (Pages 2 & 3)
1. Navigate to **Dataset Ingestion**:
   - Demonstrate the primary Wisconsin Diagnostic Breast Cancer (WDBC) cohort ingestion button.
   - Show the multi-cohort connectors: TCGA-BRCA genomic RNA-seq connector and BreakHis histopathology connector.
2. Navigate to **Dataset Explorer**:
   - Select `uci_wdbc`.
   - Inspect the first 5 records, data types, physical units table ($\mu m, \mu m^2$, grayscale variance), and the class distribution chart (357 Benign vs 212 Malignant).
   - Point out the cryptographic SHA-256 hash displayed in the metadata summary.

### Act 3: Data Quality & Wrangling Engine (Page 4)
1. Navigate to **Data Wrangling**:
   - Review the **Composite Data Hygiene Score** (**97.28 / 100 — Grade A+**).
   - Display the five component scores: Missingness (100), Duplicates (100), Outliers (92.4), Schema (100), Distribution (94).
   - Demonstrate the interactive cleaning controls: median imputation, outlier winsorization (clipping to 1st and 99th percentiles), and Yeo-Johnson skewness transformation.
   - Show the immutable JSON audit trail logging every operation with before-and-after checksums.

### Act 4: Exploratory Data Analysis & Statistical Mining (Page 5)
1. Navigate to **Exploratory Data Analysis**:
   - Examine the interactive Plotly **Correlation Heatmap** showing strong collinearity among radius, perimeter, and area.
   - Switch to **PCA 2D Projection**: observe the clear geometric clustering and hyperplanar separability between Benign (blue) and Malignant (red) cases.
   - Show the **Automated Statistical Hypothesis Testing** table: highlight that `concave_points_worst` and `area_worst` exhibit $p < 10^{-15}$ under two-sample $t$-tests and Mann-Whitney $U$ tests.

### Act 5: Feature Engineering & Selection (Page 6)
1. Navigate to **Feature Engineering**:
   - Demonstrate the generation of domain-specific cytological features:
     - Area-to-Perimeter Disproportion Index: $\frac{\text{area}}{\text{perimeter}^2}$
     - Nuclear Density / Chromatin Condensation: $\text{texture} \times \text{smoothness}$
     - Boundary Irregularity Ratio: $\text{concavity} \times \text{fractal\_dimension}$
   - Review the **Consensus Feature Ranking** table (Borda count aggregating Mutual Information, RFE, and Lasso), confirming `concave_points_worst` as the rank #1 predictive biomarker.

### Act 6: Model Training & Algorithmic Comparison (Pages 7 & 8)
1. Navigate to **Model Training**:
   - Showcase the 16 available classification algorithms.
   - Train an L2-regularized Logistic Regression model and a Random Forest ensemble using 5-fold stratified cross-validation.
   - Show that background asynchronous execution allows user navigation without freezing the UI.
2. Navigate to **Model Comparison**:
   - Present the multi-algorithm benchmark leaderboard.
   - Explain why L2 Logistic Regression ($\text{ROC-AUC} = 0.9958$) and SVM RBF ($\text{ROC-AUC} = 0.9942$) outperform deep neural networks on this sample size ($N=569$).

### Act 7: Clinical Evaluation & Threshold Optimization (Page 9)
1. Navigate to **Model Evaluation**:
   - Select the trained Logistic Regression model.
   - Review the **12 Clinical Metrics** table, noting Sensitivity ($96.55\%$), Specificity ($99.28\%$), and Brier Score ($0.0162$).
   - Show the **1,000-sample Bootstrap Confidence Intervals** confirming statistical robustness.
   - Demonstrate the **Interactive Decision Threshold Slider**:
     - Move the slider from default $\tau = 0.50$ down to clinical screening threshold $\tau = 0.35$.
     - Observe the confusion matrix update in real time: False Negatives drop from 7 down to 3 (**58.8% reduction in missed cancers**).

### Act 8: Explainable AI & Patient Risk Attribution (Page 10)
1. Navigate to **Explainability (XAI)**:
   - Select patient sample #0.
   - View the **SHAP Waterfall Plot**: observe how high `concave_points_worst` ($0.265$) and high `area_worst` ($2019.0\,\mu m^2$) push the log-odds strongly toward malignancy ($+3.42$ log-odds).
   - View the **LIME Local Surrogate Plot**: confirm that local linear rules corroborate the SHAP attribution.
   - Point out the **Non-Causal Disclaimer**, explaining the difference between statistical discrimination and biological carcinogenesis.

### Act 9: Interactive Clinical Prediction (Page 11)
1. Navigate to **Prediction**:
   - Load sample biopsy values or tweak individual cytological sliders.
   - Click **Run Clinical Inference**.
   - Review the structured result: Predicted Class (`Malignant`), Probability ($99.4\%$), Confidence Interval, and Clinical Decision-Support Warning.

### Act 10: Advanced Paradigms & System Health (Pages 12, 13 & 14)
1. Navigate to **Advanced Learning**:
   - Demonstrate Semi-Supervised learning on partially labeled data (Label Spreading).
   - Show One-Class Anomaly Detection (Isolation Forest) identifying atypical cell clusters.
2. Navigate to **Experiment Tracking**:
   - Review logged MLflow runs, hyperparameter tags, and historical validation artifacts.
3. Show the **API Interactive Swagger Documentation** in a separate browser tab (`http://localhost:8000/docs`), proving that every UI action corresponds to a clean, authenticated, documented REST API endpoint.
