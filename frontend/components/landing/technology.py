"""
CANCERWISE AI — Machine Learning Technology Component (Section 6)
Showcases genuinely implemented scientific libraries, algorithms, and computational architectures.
"""

from __future__ import annotations

import streamlit as st


def render_technology(is_dark: bool = True) -> None:
    """Renders Section 6: Machine Learning Technology."""
    st.markdown('<div id="technology"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="lp-section">
            <div style="text-align: center; max-width: 800px; margin: 0 auto 2.5rem auto;">
                <div class="lp-eyebrow">✦ COMPUTATIONAL ENGINE</div>
                <h2 class="lp-heading">Powered by Modern Machine Learning</h2>
                <p class="lp-subheading lp-subheading-center">
                    Built upon an industry-standard open-source scientific stack, CANCERWISE AI integrates 
                    high-performance tabular modeling, deep neural networks, and mathematically grounded explainability frameworks.
                </p>
            </div>

            <!-- Core Technologies Stack Grid -->
            <div style="background: var(--lp-surface); border: 1px solid var(--lp-border); border-radius: 14px; padding: 1.5rem; margin-bottom: 1.5rem;">
                <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--lp-healthcare-teal); margin-bottom: 1rem;">
                    Core Scientific &amp; Engineering Stack
                </div>
                <div class="lp-tech-badges">
                    <div class="lp-tech-badge">🐍 Python 3.14 (Free-Threaded)</div>
                    <div class="lp-tech-badge">⚙️ Scikit-learn 1.6+</div>
                    <div class="lp-tech-badge">🔥 PyTorch 2.6+ (CUDA/CPU)</div>
                    <div class="lp-tech-badge">🚀 XGBoost 2.0+</div>
                    <div class="lp-tech-badge">⚡ LightGBM 4.3+</div>
                    <div class="lp-tech-badge">🐱 CatBoost 1.2+</div>
                    <div class="lp-tech-badge">🔬 SHAP (Tree &amp; Kernel)</div>
                    <div class="lp-tech-badge">🔍 LIME Tabular Explainer</div>
                    <div class="lp-tech-badge">🐼 Pandas &amp; NumPy</div>
                    <div class="lp-tech-badge">🐻‍❄️ Polars (Fast Parquet IO)</div>
                    <div class="lp-tech-badge">⚡ FastAPI (REST OpenAPI v1)</div>
                    <div class="lp-tech-badge">📊 Plotly Interactive Engine</div>
                </div>
            </div>

            <!-- Algorithm Taxonomies: Classification vs Advanced Learning -->
            <div class="lp-tech-grid">
                <!-- Group 1: Supervised Classification -->
                <div class="lp-tech-card">
                    <div class="lp-tech-card-title">
                        <span>🤖</span> Supervised Classification Suite
                    </div>
                    <p style="font-size: 0.86rem; color: var(--lp-text-secondary); line-height: 1.5; margin-bottom: 1.25rem;">
                        Rigorous clinical classifiers benchmarked with stratified 5-fold cross-validation and hyperparameter grid searches.
                    </p>
                    <div class="lp-tech-badges">
                        <div class="lp-tech-badge">🌲 Random Forest (100 Trees)</div>
                        <div class="lp-tech-badge">⚡ Gradient Boosting (XGB / LGBM)</div>
                        <div class="lp-tech-badge">📈 Logistic Regression (ElasticNet)</div>
                        <div class="lp-tech-badge">🎯 Support Vector Machines (RBF / Linear)</div>
                        <div class="lp-tech-badge">🌿 Decision Trees (Cost-Complexity Pruned)</div>
                        <div class="lp-tech-badge">📍 K-Nearest Neighbors (Euclidean / Manhattan)</div>
                        <div class="lp-tech-badge">📊 Gaussian Naive Bayes</div>
                        <div class="lp-tech-badge">🧠 PyTorch Deep Multi-Layer Perceptron</div>
                    </div>
                </div>

                <!-- Group 2: Advanced Learning Paradigms -->
                <div class="lp-tech-card">
                    <div class="lp-tech-card-title">
                        <span>🧬</span> Advanced Learning &amp; Data Mining
                    </div>
                    <p style="font-size: 0.86rem; color: var(--lp-text-secondary); line-height: 1.5; margin-bottom: 1.25rem;">
                        Specialized paradigms for label-scarce cohorts, outlier detection, and high-dimensional oncology data.
                    </p>
                    <div class="lp-tech-badges">
                        <div class="lp-tech-badge">🏷️ Semi-Supervised: Label Propagation</div>
                        <div class="lp-tech-badge">🕸️ Semi-Supervised: Label Spreading</div>
                        <div class="lp-tech-badge">🌲 Anomaly Detection: Isolation Forest</div>
                        <div class="lp-tech-badge">⭕ Anomaly Detection: One-Class SVM</div>
                        <div class="lp-tech-badge">📐 PCA Dimensionality Reduction</div>
                        <div class="lp-tech-badge">🎯 Recursive Feature Elimination (RFE)</div>
                        <div class="lp-tech-badge">⚖️ Platt Scaling &amp; Isotonic Calibration</div>
                        <div class="lp-tech-badge">⚡ Large-Scale Batch Gradient Descents</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
