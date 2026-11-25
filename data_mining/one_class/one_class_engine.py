"""
CANCERWISE AI — One-Class Classification & Novelty Detection Engine
Implements One-Class SVM, Isolation Forest, and Local Outlier Factor (Novelty Mode).
Profiles normative healthy/benign baseline cytology and isolates atypical morphological deviations.

GOVERNANCE DIRECTIVE:
Anomaly and novelty scoring measures mathematical departure from baseline distributions.
It MUST NOT be interpreted or presented as a definitive clinical cancer diagnosis.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score

from backend.schemas.advanced_learning import (
    OneClassMethod,
    OneClassExperimentResult,
)

logger = logging.getLogger("cancerwise.one_class.engine")


class OneClassEngine:
    """
    Novelty detection and outlier isolation engine.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def run_novelty_experiment(
        self,
        X_normal: pd.DataFrame,
        X_test_normal: pd.DataFrame,
        X_test_atypical: pd.DataFrame,
        method: Union[OneClassMethod, str] = OneClassMethod.ISOLATION_FOREST,
        nu: float = 0.05,
        dataset_name: str = "UCI WDBC Cytology Novelty Baseline",
    ) -> OneClassExperimentResult:
        """
        Trains model purely on normal baseline (e.g. Benign cytology) and evaluates
        novelty detection performance on mixed holdout (Normal vs Atypical).
        """
        if isinstance(method, str):
            method = OneClassMethod(method)

        start_time = time.perf_counter()

        scaler = StandardScaler()
        X_norm_train = scaler.fit_transform(X_normal)
        X_test_norm = scaler.transform(X_test_normal)
        X_test_atyp = scaler.transform(X_test_atypical)

        # Model fitting on strictly normal baseline
        if method == OneClassMethod.ONE_CLASS_SVM:
            model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
        elif method == OneClassMethod.ISOLATION_FOREST:
            model = IsolationForest(contamination=nu, random_state=self.random_state, n_jobs=-1)
        elif method == OneClassMethod.LOCAL_OUTLIER_FACTOR:
            model = LocalOutlierFactor(novelty=True, contamination=nu, n_neighbors=20)
        else:
            raise ValueError(f"Unknown one-class method: {method}")

        model.fit(X_norm_train)

        # Evaluation on holdout test sets
        # Standard convention: +1 = inlier (normal), -1 = outlier (atypical)
        pred_norm = model.predict(X_test_norm)
        pred_atyp = model.predict(X_test_atyp)

        # Score function: decision_function (higher = more normal, lower = more anomalous)
        scores_norm = model.decision_function(X_test_norm)
        scores_atyp = model.decision_function(X_test_atyp)

        # Ground truth: 1 for normal, -1 for atypical
        y_true = np.concatenate([np.ones(len(pred_norm)), -np.ones(len(pred_atyp))])
        y_pred = np.concatenate([pred_norm, pred_atyp])

        # Anomaly detection metrics (treating -1 as the target event)
        y_true_binary_outlier = (y_true == -1).astype(int)
        y_pred_binary_outlier = (y_pred == -1).astype(int)

        atyp_recall = float(recall_score(y_true_binary_outlier, y_pred_binary_outlier, zero_division=0))
        norm_recall = float(np.mean(pred_norm == 1))
        f1_novelty = float(f1_score(y_true_binary_outlier, y_pred_binary_outlier, zero_division=0))

        score_summary = {
            "mean_score_normal": round(float(np.mean(scores_norm)), 4),
            "std_score_normal": round(float(np.std(scores_norm)), 4),
            "mean_score_atypical": round(float(np.mean(scores_atyp)), 4),
            "std_score_atypical": round(float(np.std(scores_atyp)), 4),
            "score_separation_gap": round(float(np.mean(scores_norm) - np.mean(scores_atyp)), 4),
        }

        elapsed = round(time.perf_counter() - start_time, 3)

        return OneClassExperimentResult(
            method=method,
            dataset_name=dataset_name,
            normal_training_samples=len(X_normal),
            test_normal_samples=len(X_test_normal),
            test_atypical_samples=len(X_test_atypical),
            decision_threshold=0.0,
            anomaly_detection_rate=round(atyp_recall, 4),
            normal_identification_rate=round(norm_recall, 4),
            f1_novelty=round(f1_novelty, 4),
            score_summary=score_summary,
            execution_time_seconds=elapsed,
        )
