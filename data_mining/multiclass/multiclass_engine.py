"""
CANCERWISE AI — Multiclass Classification Engine
Deploys One-vs-Rest, One-vs-One, Multinomial Logistic Regression, Multiclass SVM,
Random Forest, and Gradient Boosting across legitimate multi-subtype oncology cohorts.

GOVERNANCE DIRECTIVE:
Requires legitimate multiclass datasets (e.g. molecular tumor subtypes).
Binary WDBC labels are NEVER artificially converted into multiclass stages.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from backend.schemas.advanced_learning import (
    MulticlassStrategy,
    MulticlassClassMetrics,
    MulticlassExperimentResult,
)

logger = logging.getLogger("cancerwise.multiclass.engine")


class MulticlassEngine:
    """
    Evaluates multi-category classification strategies for oncology subtype discovery.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def run_multiclass_experiment(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        class_names: Optional[List[str]] = None,
        strategy: Union[MulticlassStrategy, str] = MulticlassStrategy.MULTINOMIAL_LOGISTIC,
        dataset_name: str = "Cancer Molecular Subtypes",
        test_size: float = 0.20,
    ) -> MulticlassExperimentResult:
        """
        Fits multiclass model and generates granular per-class and aggregated metrics.
        """
        if isinstance(strategy, str):
            strategy = MulticlassStrategy(strategy)

        start_time = time.perf_counter()

        # Check genuine multiclass nature
        unique_classes = np.unique(y)
        if len(unique_classes) < 3:
            raise ValueError(
                f"Multiclass engine requires at least 3 distinct classes, but found {len(unique_classes)}. "
                "Per clinical governance, binary WDBC data must not be artificially converted to multiclass."
            )

        if class_names is None:
            class_names = [f"Class_{c}" for c in unique_classes]

        # Stratified Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model instantiation
        if strategy == MulticlassStrategy.ONE_VS_REST:
            base = LogisticRegression(max_iter=1000, random_state=self.random_state)
            model = OneVsRestClassifier(base)
        elif strategy == MulticlassStrategy.ONE_VS_ONE:
            base = LogisticRegression(max_iter=1000, random_state=self.random_state)
            model = OneVsOneClassifier(base)
        elif strategy == MulticlassStrategy.MULTINOMIAL_LOGISTIC:
            model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=self.random_state)
        elif strategy == MulticlassStrategy.MULTICLASS_SVM:
            model = SVC(kernel="rbf", decision_function_shape="ovr", probability=True, random_state=self.random_state)
        elif strategy == MulticlassStrategy.RANDOM_FOREST:
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        elif strategy == MulticlassStrategy.GRADIENT_BOOSTING:
            model = GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown multiclass strategy: {strategy}")

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        micro_f1 = float(f1_score(y_test, y_pred, average="micro", zero_division=0))
        weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        # Multiclass ROC-AUC (OvR)
        multiclass_auc = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test_scaled)
                multiclass_auc = float(roc_auc_score(y_test, y_prob, multi_class="ovr"))
            except Exception:
                multiclass_auc = None

        # Per-class metrics
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        per_class: List[MulticlassClassMetrics] = []

        for i, c in enumerate(unique_classes):
            c_str = str(c)
            name = class_names[i] if i < len(class_names) else f"Subtype_{c}"
            if c_str in report_dict:
                per_class.append(
                    MulticlassClassMetrics(
                        class_name=name,
                        precision=round(float(report_dict[c_str]["precision"]), 4),
                        recall=round(float(report_dict[c_str]["recall"]), 4),
                        f1_score=round(float(report_dict[c_str]["f1-score"]), 4),
                        support=int(report_dict[c_str]["support"]),
                    )
                )

        cm = confusion_matrix(y_test, y_pred).tolist()
        elapsed = round(time.perf_counter() - start_time, 3)

        return MulticlassExperimentResult(
            strategy=strategy,
            dataset_name=dataset_name,
            class_names=class_names,
            macro_f1=round(macro_f1, 4),
            micro_f1=round(micro_f1, 4),
            weighted_f1=round(weighted_f1, 4),
            accuracy=round(acc, 4),
            multiclass_roc_auc=round(multiclass_auc, 4) if multiclass_auc is not None else None,
            per_class_metrics=per_class,
            confusion_matrix=cm,
            execution_time_seconds=elapsed,
        )
