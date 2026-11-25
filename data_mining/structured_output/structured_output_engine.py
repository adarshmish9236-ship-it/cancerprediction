"""
CANCERWISE AI — Structured Output Learning Engine
Implements Multi-Output and Classifier Chains for structured multi-label oncology biomarkers.
Models correlated co-expression patterns across receptor panels (e.g., ER, PR, HER2, Ki-67).

GOVERNANCE DIRECTIVE:
Activated exclusively when legitimate structured target vectors (e.g., multi-label panels) are present.
Ordinary binary classification is NEVER misrepresented as structured-output prediction.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    hamming_loss,
    accuracy_score,
    f1_score,
)

from backend.schemas.advanced_learning import (
    StructuredOutputExperimentResult,
)

logger = logging.getLogger("cancerwise.structured_output.engine")


class StructuredOutputEngine:
    """
    Evaluates structured multi-target prediction models accounting for inter-label dependencies.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def run_structured_experiment(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        structure_type: str = "classifier_chains",
        dataset_name: str = "Synthetic Multi-Label Receptor Status Panel",
        test_size: float = 0.20,
    ) -> StructuredOutputExperimentResult:
        """
        Fits structured prediction model and computes Hamming Loss, Subset Accuracy, and F1 metrics.
        """
        start_time = time.perf_counter()

        # Clinical governance check: verify multi-dimensional structured target
        if Y.ndim != 2 or Y.shape[1] < 2:
            raise ValueError(
                f"Structured output engine requires a multi-target matrix (columns >= 2), but found shape {Y.shape}. "
                "Per clinical governance, ordinary single-target classification must not be misrepresented as structured learning."
            )

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=self.random_state
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        base_estimator = LogisticRegression(max_iter=500, random_state=self.random_state)

        if structure_type == "classifier_chains":
            model = ClassifierChain(estimator=base_estimator, order="random", random_state=self.random_state)
        else:
            model = MultiOutputClassifier(estimator=base_estimator, n_jobs=-1)

        model.fit(X_train_scaled, Y_train)
        Y_pred = model.predict(X_test_scaled)

        # Metrics for structured prediction
        h_loss = float(hamming_loss(Y_test, Y_pred))
        # Exact match subset accuracy
        subset_acc = float(accuracy_score(Y_test, Y_pred))
        micro_f1 = float(f1_score(Y_test, Y_pred, average="micro", zero_division=0))
        macro_f1 = float(f1_score(Y_test, Y_pred, average="macro", zero_division=0))

        elapsed = round(time.perf_counter() - start_time, 3)

        governance = (
            "Research Benchmark Notice: Structured multi-target receptor prediction evaluated on a correlated "
            "biomarker panel (ER, PR, HER2, Ki-67). Models joint dependencies rather than isolated binary tasks."
        )

        return StructuredOutputExperimentResult(
            dataset_name=dataset_name,
            structure_type=structure_type,
            output_dimensions=Y.shape[1],
            target_labels=list(Y.columns),
            hamming_loss=round(h_loss, 4),
            subset_accuracy=round(subset_acc, 4),
            micro_f1=round(micro_f1, 4),
            macro_f1=round(macro_f1, 4),
            execution_time_seconds=elapsed,
            governance_statement=governance,
        )
