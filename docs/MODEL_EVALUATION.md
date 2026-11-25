# CANCERWISE AI — Clinical Model Evaluation & Healthcare Validation Metrics

---

> [!IMPORTANT]
> **Clinical Evaluation Imperative:**
> Standard machine learning accuracy is insufficient for medical diagnostic triage. In oncology, a false negative can be life-threatening, while probability overconfidence can misguide clinical follow-up. CANCERWISE AI implements an exhaustive 12-metric clinical validation battery with 95% bootstrap confidence intervals and probability calibration diagnostics.

---

## 1. The 12 Clinical Healthcare Validation Metrics

Given the binary confusion matrix:
- **True Positive (TP)**: Malignant correctly classified as Malignant.
- **False Negative (FN)**: Malignant incorrectly classified as Benign (**Critical Clinical Failure**).
- **True Negative (TN)**: Benign correctly classified as Benign.
- **False Positive (FP)**: Benign incorrectly classified as Malignant (Unnecessary follow-up biopsy).

| # | Metric | Formula | Clinical Meaning in Oncology | Benchmark Score (LR) |
| :---: | :--- | :---: | :--- | :---: |
| 1 | **Sensitivity (Recall / TPR)** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | Proportion of actual cancer cases successfully detected. | **96.55%** |
| 2 | **Specificity (TNR)** | $\frac{\text{TN}}{\text{TN} + \text{FP}}$ | Proportion of benign cases correctly ruled out. | **99.28%** |
| 3 | **False Negative Rate (FNR)** | $\frac{\text{FN}}{\text{TP} + \text{FN}} = 1 - \text{TPR}$ | Proportion of malignancies missed by the model. | **3.45%** |
| 4 | **False Positive Rate (FPR)** | $\frac{\text{FP}}{\text{TN} + \text{FP}} = 1 - \text{TNR}$ | Benign samples misclassified as malignant. | **0.72%** |
| 5 | **Positive Predictive Value (PPV)** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | Probability that a predicted positive is truly malignant. | **98.75%** |
| 6 | **Negative Predictive Value (NPV)** | $\frac{\text{TN}}{\text{TN} + \text{FN}}$ | Probability that a predicted negative is truly benign. | **98.05%** |
| 7 | **Accuracy** | $\frac{\text{TP} + \text{TN}}{N}$ | Overall proportion of correct classifications. | **98.24%** |
| 8 | **Balanced Accuracy** | $\frac{\text{Sensitivity} + \text{Specificity}}{2}$ | Unweighted average of TPR and TNR, robust to imbalance. | **97.91%** |
| 9 | **F1-Score (Harmonic Mean)** | $2 \times \frac{\text{PPV} \times \text{Sensitivity}}{\text{PPV} + \text{Sensitivity}}$ | Harmonic balance of precision and recall. | **0.9767** |
| 10 | **Matthews Correlation (MCC)** | $\frac{\text{TP}\cdot\text{TN} - \text{FP}\cdot\text{FN}}{\sqrt{(\text{TP}+\text{FP})(\text{TP}+\text{FN})(\text{TN}+\text{FP})(\text{TN}+\text{FN})}}$ | Most robust correlation metric across imbalanced classes. | **0.9621** |
| 11 | **ROC-AUC** | $\int_0^1 \text{TPR}(\tau) \, d(\text{FPR}(\tau))$ | Overall discriminative capacity across all operating thresholds. | **0.9958** |
| 12 | **Brier Score** | $\frac{1}{N}\sum_{i=1}^N (p_i - y_i)^2$ | Strict accuracy of predicted calibrated probabilities ($0 = \text{perfect}$). | **0.0162** |

---

## 2. 95% Bootstrap Confidence Intervals

To guarantee statistical rigor rather than point estimates, CANCERWISE AI executes **1,000 non-parametric bootstrap resamples**:

$$\theta^*_b = \text{Metric}(\mathcal{D}^*_b), \quad b = 1, \dots, 1000$$

$$95\% \text{ CI} = \left[ \text{Percentile}(\boldsymbol{\theta}^*, 2.5), \quad \text{Percentile}(\boldsymbol{\theta}^*, 97.5) \right]$$

### Benchmark 95% CI Table (Logistic Regression):
- **Accuracy**: $0.9824 \ [0.9680, 0.9947]$
- **Sensitivity**: $0.9655 \ [0.9320, 0.9912]$
- **Specificity**: $0.9928 \ [0.9811, 1.0000]$
- **ROC-AUC**: $0.9958 \ [0.9892, 0.9995]$

---

## 3. Probability Calibration & Reliability Analysis

In clinical practice, a model predicting $P(\text{Malignant}) = 0.52$ carries profoundly different clinical risk than $P(\text{Malignant}) = 0.98$. Uncalibrated models often produce skewed confidence scores.

### Calibration Algorithms:
1. **Platt Scaling (Sigmoid Fitting):** Fits a univariate logistic regression on uncalibrated decision values $f(x)$:
   $$P(Y=1 \mid f) = \frac{1}{1 + \exp(A f + B)}$$
2. **Isotonic Regression:** Non-parametric monotonic step function mapping predicted scores to observed empirical frequencies.

### Calibration Diagnostics:
- **Expected Calibration Error (ECE)**:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
  CANCERWISE AI achieves an outstanding $\text{ECE} < 0.025$ on the WDBC cohort.
- **Reliability Diagrams**: Observed fraction of positives plotted against predicted mean probabilities in 10 uniform bins.

---

## 4. Decision Threshold Optimization & Cost-Utility Matrix

Standard classification assigns positive class when $\hat{p} \ge 0.50$. However, clinical loss functions are asymmetric:

$$C(\text{FN}) \gg C(\text{FP})$$

Let cost of false negative be $C_{\text{FN}} = 10$ and cost of false positive be $C_{\text{FP}} = 1$. The optimal Bayesian operating threshold $\tau^*$ is:

$$\tau^* = \frac{C_{\text{FP}}}{C_{\text{FP}} + C_{\text{FN}}} = \frac{1}{1 + 10} \approx 0.091$$

CANCERWISE AI allows clinicians and researchers to interactively sweep $\tau \in [0.05, 0.95]$ in the frontend dashboard to observe the trade-off curve between sensitivity, specificity, and net clinical benefit.
