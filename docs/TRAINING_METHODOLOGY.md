# CANCERWISE AI — Training Methodology & Cross-Validation Protocol

---

> [!IMPORTANT]
> **Zero Data Leakage Guarantee:**
> In high-stakes medical machine learning, preprocessing or resampling outside the cross-validation loop inflates reported performance and yields catastrophic real-world failure. CANCERWISE AI enforces strict **in-fold transformation pipelines**: all scalers, imputers, selectors, and SMOTE resamplers are fitted strictly on the $K-1$ training folds during each split.

---

## 1. Cross-Validation Framework

CANCERWISE AI utilizes **$K$-Fold Stratified Cross-Validation** ($K=5$ default, configurable to $K=10$):

```
Dataset (569 samples: 357 Benign, 212 Malignant)
│
├── Fold 1: [Train: Folds 2,3,4,5 (80%)] ───► In-Fold Fit [Scaler, SMOTE, Model] ───► Eval Fold 1 (20%)
├── Fold 2: [Train: Folds 1,3,4,5 (80%)] ───► In-Fold Fit [Scaler, SMOTE, Model] ───► Eval Fold 2 (20%)
├── Fold 3: [Train: Folds 1,2,4,5 (80%)] ───► In-Fold Fit [Scaler, SMOTE, Model] ───► Eval Fold 3 (20%)
├── Fold 4: [Train: Folds 1,2,3,5 (80%)] ───► In-Fold Fit [Scaler, SMOTE, Model] ───► Eval Fold 4 (20%)
└── Fold 5: [Train: Folds 1,2,3,4 (80%)] ───► In-Fold Fit [Scaler, SMOTE, Model] ───► Eval Fold 5 (20%)
```

- **Stratification**: Preserves the exact 62.7% : 37.3% Benign-to-Malignant ratio within every single fold.
- **Repeatability**: Global pseudo-random number seed (`random_state=42`) is set across NumPy, Python random, PyTorch, and scikit-learn.

---

## 2. Imbalanced Class Handling & In-Fold Resampling

The WDBC cohort exhibits a class imbalance ratio of $\approx 1.68 : 1$ (357 Benign vs 212 Malignant). While mild compared to screening populations, unweighted models tend to favor specificity over sensitivity.

CANCERWISE AI provides three in-fold mitigation strategies:
1. **Class Weighting (`class_weight='balanced'`):**
   $$w_c = \frac{N}{2 \times N_c}$$
   Adjusts the loss gradient inversely proportional to class frequency without synthesizing artificial samples.
2. **Synthetic Minority Over-sampling Technique (SMOTE):**
   Synthesizes convex combinations of $k$-nearest minority neighbors:
   $$\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda (\mathbf{x}_{zi} - \mathbf{x}_i), \quad \lambda \sim U(0, 1)$$
   Executed strictly on training folds via `imblearn.pipeline.Pipeline`.
3. **Random Under-Sampling (RUS):**
   Downsamples majority benign class for rapid large-scale prototyping.

---

## 3. Hyperparameter Optimization Framework

CANCERWISE AI supports three optimization engines (`data_mining/classification/tuning.py`):

### 3.1 Stratified Grid Search
- Exhaustive evaluation over predefined parameter grids.
- Optimized for deterministic baselines (e.g. Logistic Regression regularizer $C \in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]$).

### 3.2 Randomized Search with Halving
- Samples fixed budgets from continuous distributions (e.g., $C \sim \text{LogUniform}(10^{-4}, 10^3)$).
- Reduces search time by up to 70% while discovering competitive configurations.

### 3.3 Bayesian Optimization (Optuna Integration)
- Uses Tree-structured Parzen Estimator (TPE) to model $P(\theta \mid \text{score})$ and intelligently sample regions with maximum Expected Improvement (EI).
- Applied to deep PyTorch MLP and XGBoost ensembles (tuning learning rate, tree depth, gamma, and dropout rates).

---

## 4. Deep Learning Convergence & Regularization

For the PyTorch Multi-Layer Perceptron:
- **Optimization**: AdamW with cosine annealing learning rate schedule:
  $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)$$
- **Early Stopping**: Monitored on validation loss with patience of 15 epochs.
- **Batch Normalization & Dropout**: Mitigates internal covariate shift and prevents co-adaptation of hidden cytological representations.
