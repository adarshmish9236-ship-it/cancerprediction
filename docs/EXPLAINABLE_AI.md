# CANCERWISE AI — Explainable Artificial Intelligence (XAI) Documentation

---

> [!CAUTION]
> **Clinical Non-Causality & Interpretability Notice:**
> Feature importance and Shapley values indicate **statistical correlation and diagnostic discriminative utility** within the WDBC fine needle aspirate dataset. They do NOT establish biological cause-and-effect or molecular carcinogenesis. Predictions must be evaluated in conjunction with cellular immunohistochemistry and full histological examination.

---

## 1. Game-Theoretic SHAP (SHapley Additive exPlanations)

CANCERWISE AI implements SHAP based on cooperative game theory (Lundberg & Lee, 2017), allocating each feature's marginal contribution to the prediction deviation from the base expectation:

$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$

### 1.1 Mathematical Axioms Satisfied:
1. **Efficiency**: $\sum_{i=1}^M \phi_i(x) = f(x) - \mathbb{E}[f(X)]$. The sum of all feature attributions equals the difference between the model output and the dataset base rate.
2. **Symmetry**: If $f(S \cup \{i\}) = f(S \cup \{j\})$ for all subsets $S$, then $\phi_i = \phi_j$.
3. **Dummy (Null Player)**: If $f(S \cup \{i\}) = f(S)$ for all $S$, then $\phi_i = 0$.
4. **Additivity**: For ensemble models $f = \sum w_k f_k$, $\phi_i(f) = \sum w_k \phi_i(f_k)$.

### 1.2 Specialized Explainer Implementations:
- **TreeExplainer**: Exact polynomial-time attribution $\mathcal{O}(T L D^2)$ for Random Forest, Extra Trees, and XGBoost models.
- **LinearExplainer**: Analytical covariance-aware attribution for regularized Logistic Regression.
- **KernelExplainer**: Model-agnostic weighted linear surrogate sampling for black-box neural networks and support vector machines.

---

## 2. LIME (Local Interpretable Model-Agnostic Explanations)

For instance-level clinician transparency, CANCERWISE AI integrates LIME (Ribeiro et al., 2016):

$$\xi(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)$$

Where:
- $f(x)$: Complex black-box predictive model.
- $g$: Sparse linear surrogate model ($g(z') = w_g^T z'$).
- $\pi_x(z) = \exp(-D(x, z)^2 / \sigma^2)$: Exponential kernel weighting perturbations by proximity to the patient sample.
- $\Omega(g)$: Complexity penalty enforcing sparsity (e.g. top $K=5$ features).

---

## 3. Global Interpretability & Consensus Cytological Ranking

CANCERWISE AI combines multiple feature importance methodologies into an aggregated **Borda Count Consensus Ranking**:
1. Mean Absolute SHAP values: $\frac{1}{N} \sum_{k=1}^N |\phi_i(x_k)|$
2. Permutation Feature Importance (loss increase upon feature shuffling)
3. Mutual Information Score ($I(X; Y)$)
4. Random Forest Mean Decrease in Impurity (Gini importance)

### Consensus Top Cytological Differentiating Biomarkers:

| Rank | Feature | Physical Meaning | Clinical Interpretation in Breast FNA | SHAP Impact Direction |
| :---: | :--- | :--- | :--- | :--- |
| **1** | `concave_points_worst` | Severity of contour notches in 3 largest nuclei | Discontinuous irregular nuclear boundaries indicative of severe cellular atypia. | High values $\to$ Strong Malignant Risk |
| **2** | `area_worst` | Maximum nuclear volume / area ($\mu m^2$) | Anisokaryosis and gigantism of aggressive malignant clones. | High values $\to$ Strong Malignant Risk |
| **3** | `perimeter_worst` | Maximum nuclear circumference ($\mu m$) | Directly collinear with nuclear volume expansion. | High values $\to$ Strong Malignant Risk |
| **4** | `radius_worst` | Maximum nuclear radius ($\mu m$) | Hallmark of malignant cytological nuclear enlargement. | High values $\to$ Strong Malignant Risk |
| **5** | `texture_worst` | Highest chromatin clumpiness / grayscale std | Hyperchromasia; coarse, clumped, unevenly distributed chromatin within the nucleus. | High values $\to$ Strong Malignant Risk |

---

## 4. Partial Dependence & Individual Conditional Expectation

- **Partial Dependence Plots (PDP)**: Visualizes the marginal effect of a cytological feature on the predicted probability across the entire cohort:
  $$\hat{f}_S(x_S) = \frac{1}{N}\sum_{i=1}^N \hat{f}(x_S, x_{C, i})$$
- **Individual Conditional Expectation (ICE)**: Displays patient-specific curves, revealing feature interactions and heterogeneous response profiles that aggregate PDPs might mask.
