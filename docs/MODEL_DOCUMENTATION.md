# CANCERWISE AI — Machine Learning Model Documentation

---

> [!IMPORTANT]
> **Clinical Model Governance & Intended Use:**
> The algorithms and neural architectures documented herein are intended strictly for clinical decision support, oncological risk stratification, and educational pattern recognition. Predictions should be interpreted by board-certified pathologists and oncologists alongside comprehensive histological biopsies and immunohistochemical assays.

---

## 1. Algorithmic Architecture Overview

CANCERWISE AI incorporates 16 diverse machine learning algorithms spanning 6 algorithmic families, enabling empirical comparison between interpretable linear baselines, non-linear ensembles, kernel machines, and deep neural networks.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CANCERWISE AI Model Registry                          │
├─────────────────────┬─────────────────────┬─────────────────────────────────┤
│ Linear & Parametric │ Decision Ensembles  │ Kernel & Instance-Based         │
├─────────────────────┼─────────────────────┼─────────────────────────────────┤
│ Logistic Regression │ Random Forest       │ Support Vector Machines (SVM)   │
│ Ridge Classifier    │ Extra Trees         │ Kernel Ridge Classification     │
│ Linear Discriminant │ Gradient Boosting   │ k-Nearest Neighbors (k-NN)      │
│ Gaussian Naive Bayes│ AdaBoost Classifier │ Radius Neighbors Classifier     │
│                     │ XGBoost             │                                 │
│                     │ LightGBM            │                                 │
│                     │ CatBoost            │                                 │
├─────────────────────┴─────────────────────┴─────────────────────────────────┤
│ Neural Architectures & Advanced Paradigms                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ • PyTorch Deep Multi-Layer Perceptron (BatchNorm + Dropout + AdamW)         │
│ • Semi-Supervised Learning (Label Propagation & Label Spreading)            │
│ • Multi-Task Learning (Joint Malignancy & Nuclear Grade Classification)     │
│ • One-Class Anomaly Detection (One-Class SVM & Isolation Forest)            │
│ • Incremental SGD & Passive-Aggressive Online Learning                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. In-Depth Algorithmic Specifications

### 2.1 Logistic Regression with ElasticNet Regularization
- **Mathematical Formulation:**
  $$P(Y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
- **Objective Function:**
  $$\min_{\mathbf{w}, b} -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln \sigma(\mathbf{w}^T \mathbf{x}_i + b) + (1 - y_i) \ln (1 - \sigma(\mathbf{w}^T \mathbf{x}_i + b)) \right] + \lambda_1 \|\mathbf{w}\|_1 + \frac{\lambda_2}{2} \|\mathbf{w}\|_2^2$$
- **Clinical Utility:** Direct odds-ratio interpretability. Feature weights translate immediately into multiplicative odds of malignancy:
  $$\text{Odds Ratio} = \exp(w_j)$$

### 2.2 Random Forest Classifier
- **Architecture:** Ensemble of $B = 100$ decorrelated classification trees using bootstrap aggregation (bagging) and randomized feature subspace selection:
  $$m = \lfloor \sqrt{p} \rfloor \approx \sqrt{30} \approx 5 \text{ features per split}$$
- **Split Criterion:** Gini Impurity:
  $$I_G(t) = 1 - \sum_{k=1}^K p(k \mid t)^2$$
- **Variance Reduction:** Reduces variance by factor $\frac{1}{B} + \frac{B - 1}{B} \rho$, where $\rho$ is the inter-tree correlation.

### 2.3 Extreme Gradient Boosting (XGBoost)
- **Objective Function:**
  $$\mathcal{L}^{(t)} = \sum_{i=1}^N \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
  where $g_i = \partial_{\hat{y}^{(t-1)}} \ell(y_i, \hat{y}^{(t-1)})$ and $h_i = \partial_{\hat{y}^{(t-1)}}^2 \ell(y_i, \hat{y}^{(t-1)})$.
- **Clinical Advantage:** Efficient handling of non-linear cytological interactions (e.g., perimeter-to-area disproportion).

### 2.4 Support Vector Machine with Radial Basis Function (RBF) Kernel
- **Dual Optimization Formulation:**
  $$\max_{\boldsymbol{\alpha}} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j) \quad \text{s.t.} \quad 0 \le \alpha_i \le C, \quad \sum_{i=1}^N \alpha_i y_i = 0$$
- **RBF Kernel Function:**
  $$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$
- **Effective Hyperparameters:** $C = 1.0, \gamma = \text{scale} = \frac{1}{30 \times \text{Var}(X)}$.

### 2.5 PyTorch Deep Multi-Layer Perceptron (MLP)
- **Layer Architecture:**
  - Input Layer: 30 standardized cytological inputs
  - Dense Block 1: Linear(30, 64) $\to$ BatchNorm1d(64) $\to$ ReLU $\to$ Dropout(0.3)
  - Dense Block 2: Linear(64, 32) $\to$ BatchNorm1d(32) $\to$ ReLU $\to$ Dropout(0.2)
  - Dense Block 3: Linear(32, 16) $\to$ ReLU
  - Output Layer: Linear(16, 2) $\to$ LogSoftmax
- **Optimization:** AdamW optimizer ($\text{lr} = 0.001$, weight decay $= 10^{-4}$), Cross-Entropy loss with label smoothing (0.05) to prevent overconfident probabilities on ambiguous FNA samples.

---

## 3. Model Persistence & Internal Registry

All trained model pipelines are serialized using `joblib` with LZ4/zlib compression into `models/trained/`. Accompanying JSON metadata files (`models/registry/`) store:
- Model UUID (`{dataset_id}_{algorithm}_{short_hash}`)
- Architecture name & scikit-learn / PyTorch version
- Preprocessing steps embedded in the pipeline
- Exact feature column names and input order
- 5-fold cross-validation metrics
- Calibration curves and threshold operating points
