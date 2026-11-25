"""
CANCERWISE AI — Page 8: Supervised Learning Lab
Configurable multi-paradigm classifier training, stratified cross-validation,
hyperparameter tuning, and serialization.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix

from frontend.components import (
    render_header,
    render_sidebar,
    MetricCard,
    AlgorithmConfigurator,
    ConfusionMatrixView,
    ChartContainer,
    PALETTE,
)
from frontend.utils.api_client import APIClient
from data_mining.classification.serializer import ModelSerializer

try:
    st.set_page_config(page_title="Supervised Learning | CANCERWISE AI", page_icon="🤖", layout="wide")
except Exception:
    pass

render_sidebar("Supervised Learning")
api = APIClient()
serializer = ModelSerializer()

render_header(
    title="Supervised Machine Learning & Classification Laboratory",
    subtitle="Configure, tune, cross-validate, and evaluate multi-paradigm oncology models without data leakage.",
    breadcrumbs=["CANCERWISE AI", "Machine Learning", "Supervised Learning"],
)

selected_ds = st.session_state.get("selected_dataset", "uci_wdbc")
df = api.load_dataset(selected_ds)

if df is None:
    st.error("No dataset available.")
    st.stop()

target_col = "diagnosis" if "diagnosis" in df.columns else df.columns[-1]
y_raw = df[target_col]
mapping = {"m": 1, "b": 0, "malignant": 1, "benign": 0, "1": 1, "0": 0, 1: 1, 0: 0}
y = pd.Series([mapping.get(str(v).strip().lower(), mapping.get(v, 0)) for v in y_raw], dtype=int)
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col and c != "id"]
X = df[num_cols].fillna(df[num_cols].median())

# Multi-Step Configuration Layout
col_cfg, col_train = st.columns([1.2, 1.8])

with col_cfg:
    st.markdown("##### 1. Algorithm Architecture")
    algo_category = st.selectbox(
        "Algorithm Family",
        ["Ensemble Methods", "Linear Classifiers", "Kernel & Support Vector", "Boosting Algorithms", "Instance-Based (KNN)"],
    )

    if algo_category == "Ensemble Methods":
        algo_choice = st.selectbox("Algorithm", ["Random Forest Classifier", "Extra Trees Classifier"])
    elif algo_category == "Linear Classifiers":
        algo_choice = st.selectbox("Algorithm", ["Logistic Regression (L2)", "SGDClassifier (Linear)"])
    elif algo_category == "Kernel & Support Vector":
        algo_choice = st.selectbox("Algorithm", ["Support Vector Machine (SVM)"])
    elif algo_category == "Boosting Algorithms":
        algo_choice = st.selectbox("Algorithm", ["Gradient Boosting Classifier", "XGBoost Classifier"])
    else:
        algo_choice = st.selectbox("Algorithm", ["K-Nearest Neighbors (KNN)"])

    hparams = AlgorithmConfigurator.render(algo_choice)

    st.markdown("##### 2. Cross-Validation Protocol")
    cv_folds = st.slider("Stratified CV Folds", 2, 10, 5)
    test_split = st.slider("Holdout Test Size (%)", 10, 40, 20) / 100.0

    train_btn = st.button("🚀 Train & Validate Model", type="primary", use_container_width=True)

with col_train:
    st.markdown("##### 3. Diagnostic Performance Evaluation")

    if train_btn:
        with st.spinner(f"Executing {cv_folds}-Fold Cross-Validation for {algo_choice}..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_split, random_state=42, stratify=y
            )

            # Instantiate model
            if "Random Forest" in algo_choice:
                clf = RandomForestClassifier(**hparams, random_state=42)
            elif "Extra Trees" in algo_choice:
                clf = ExtraTreesClassifier(**hparams, random_state=42)
            elif "Logistic" in algo_choice:
                clf = LogisticRegression(**hparams, random_state=42)
            elif "SVM" in algo_choice:
                clf = SVC(**hparams, probability=True, random_state=42)
            elif "Gradient" in algo_choice or "XGBoost" in algo_choice:
                clf = GradientBoostingClassifier(**hparams, random_state=42)
            else:
                clf = KNeighborsClassifier(n_neighbors=5)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            sens = recall_score(y_test, y_pred, zero_division=0)
            prec = precision_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_prob)

            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")

            st.success(f"Successfully trained {algo_choice}! Mean CV AUC: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                MetricCard.render("ROC-AUC", f"{auc:.4f}", delta=f"CV: {np.mean(cv_scores):.3f}", icon="📈")
            with m2:
                MetricCard.render("Sensitivity (Recall)", f"{sens*100:.1f}%", delta="True Positive", icon="🩺")
            with m3:
                MetricCard.render("Precision (PPV)", f"{prec*100:.1f}%", delta="Positive Pred", icon="🎯")
            with m4:
                MetricCard.render("F1-Score", f"{f1:.4f}", delta="Harmonic Mean", icon="⚖️")

            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            cm = confusion_matrix(y_test, y_pred)
            ConfusionMatrixView.render(cm, title=f"Confusion Matrix ({algo_choice})")
    else:
        st.info("Configure model hyperparameters on the left and click **Train & Validate Model** to execute training.")
