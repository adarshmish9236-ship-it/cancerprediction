"""
CANCERWISE AI — Incremental & Out-of-Core Learning Engine
Implements stream-based online learning via partial_fit for memory-constrained streaming data.
Supports SGDClassifier, SGDRegressor, MiniBatchKMeans, and PassiveAggressiveClassifier
with incremental batch-wise StandardScaler updates.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, SGDRegressor, PassiveAggressiveClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss

from backend.schemas.large_scale import (
    IncrementalAlgorithm,
    IncrementalChunkStep,
    IncrementalLearningResult,
)
from data_mining.large_scale.performance_monitor import PerformanceMonitor

logger = logging.getLogger("cancerwise.large_scale.incremental")


class IncrementalLearningEngine:
    """
    Executes streaming out-of-core learning over data chunks using partial_fit.
    """

    def __init__(self, random_state: int = 42, monitor: Optional[PerformanceMonitor] = None):
        self.random_state = random_state
        self.monitor = monitor or PerformanceMonitor()

    @staticmethod
    def get_compatible_algorithms() -> List[Dict[str, Any]]:
        """Documents scikit-learn algorithms supporting incremental training."""
        return [
            {
                "algorithm": "SGDClassifier",
                "paradigm": "Linear Classification",
                "supported_losses": ["log_loss", "hinge", "modified_huber"],
                "partial_fit_support": True,
                "notes": "Streams stochastic gradient descent updates with convex loss.",
            },
            {
                "algorithm": "SGDRegressor",
                "paradigm": "Continuous Regression",
                "supported_losses": ["squared_error", "huber", "epsilon_insensitive"],
                "partial_fit_support": True,
                "notes": "Incremental linear regression for unbounded data streams.",
            },
            {
                "algorithm": "MiniBatchKMeans",
                "paradigm": "Unsupervised Clustering",
                "supported_losses": ["inertia"],
                "partial_fit_support": True,
                "notes": "Incremental centroid convergence using mini-batches.",
            },
            {
                "algorithm": "PassiveAggressiveClassifier",
                "paradigm": "Online Margin Learning",
                "supported_losses": ["hinge"],
                "partial_fit_support": True,
                "notes": "Aggressive weight updates for streaming biomedical classification.",
            },
            {
                "algorithm": "StandardScaler (copy=False)",
                "paradigm": "Preprocessing",
                "supported_losses": ["N/A"],
                "partial_fit_support": True,
                "notes": "Welford's algorithm for incremental mean and variance estimation.",
            },
        ]

    def train_incremental_classification(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        algorithm: IncrementalAlgorithm = IncrementalAlgorithm.SGD_CLASSIFIER,
        chunk_size: int = 64,
        test_size: float = 0.20,
        dataset_name: str = "UCI WDBC",
    ) -> IncrementalLearningResult:
        """
        Streams through training chunks using partial_fit, tracking running test performance.
        """
        start_time = time.perf_counter()
        start_rss = self.monitor.process.memory_info().rss / (1024.0 * 1024.0)

        # 1. Clean numeric features and binary target
        X_num = X.select_dtypes(include=[np.number]).dropna(axis=1)
        if not pd.api.types.is_numeric_dtype(y):
            mapping = {"M": 1, "B": 0, "malignant": 1, "benign": 0}
            y_arr = y.map(mapping).to_numpy() if all(k in mapping for k in y.unique()) else pd.factorize(y)[0]
        else:
            y_arr = y.to_numpy().astype(int)

        classes = np.array([0, 1])

        # 2. Holdout Test Set (kept untouched, used to evaluate generalization after each chunk)
        X_train, X_test, y_train, y_test = train_test_split(
            X_num, y_arr, test_size=test_size, stratify=y_arr, random_state=self.random_state
        )

        # 3. Incremental Scaler & Model Instantiation
        incremental_scaler = StandardScaler(copy=False)

        if algorithm == IncrementalAlgorithm.SGD_CLASSIFIER:
            model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, random_state=self.random_state)
        elif algorithm == IncrementalAlgorithm.PASSIVE_AGGRESSIVE:
            model = PassiveAggressiveClassifier(loss="hinge", C=1.0, random_state=self.random_state)
        else:
            model = SGDClassifier(loss="log_loss", penalty="l2", random_state=self.random_state)

        # 4. Stream training in chunks
        n_train = len(X_train)
        num_chunks = int(np.ceil(n_train / chunk_size))
        steps: List[IncrementalChunkStep] = []
        cumulative_records = 0

        # Prime scaler on the first chunk so test set can be progressively scaled
        first_chunk = X_train.iloc[: min(chunk_size, n_train)].to_numpy(dtype=np.float64)
        incremental_scaler.partial_fit(first_chunk)

        for chunk_idx in range(num_chunks):
            chunk_start = time.perf_counter()
            start_i = chunk_idx * chunk_size
            end_i = min(start_i + chunk_size, n_train)

            X_chunk = X_train.iloc[start_i:end_i].to_numpy(dtype=np.float64)
            y_chunk = y_train[start_i:end_i]

            # Incremental scaling update
            incremental_scaler.partial_fit(X_chunk)
            X_chunk_scaled = incremental_scaler.transform(X_chunk)

            # Incremental model update
            model.partial_fit(X_chunk_scaled, y_chunk, classes=classes)

            chunk_elapsed_ms = (time.perf_counter() - chunk_start) * 1000.0
            cumulative_records += len(X_chunk)

            # Evaluate on holdout test set using current scaler & model
            X_test_scaled = incremental_scaler.transform(X_test.to_numpy(dtype=np.float64))
            preds_test = model.predict(X_test_scaled)

            acc_test = float(accuracy_score(y_test, preds_test))

            # Probability calculation if available
            loss_val = 0.0
            auc_test = acc_test
            if hasattr(model, "predict_proba"):
                try:
                    probs_test = model.predict_proba(X_test_scaled)
                    auc_test = float(roc_auc_score(y_test, probs_test[:, 1]))
                    loss_val = float(log_loss(y_test, probs_test))
                except Exception:
                    pass
            elif hasattr(model, "decision_function"):
                try:
                    scores = model.decision_function(X_test_scaled)
                    probs = 1.0 / (1.0 + np.exp(-scores))
                    auc_test = float(roc_auc_score(y_test, probs))
                except Exception:
                    pass

            steps.append(
                IncrementalChunkStep(
                    chunk_index=chunk_idx + 1,
                    records_in_chunk=len(X_chunk),
                    cumulative_records=cumulative_records,
                    chunk_training_time_ms=round(chunk_elapsed_ms, 2),
                    running_test_accuracy=round(acc_test, 4),
                    running_test_roc_auc=round(auc_test, 4),
                    running_loss=round(loss_val, 4),
                )
            )

        total_elapsed = round(time.perf_counter() - start_time, 4)
        end_rss = self.monitor.process.memory_info().rss / (1024.0 * 1024.0)
        peak_rss = max(end_rss - start_rss, 1.0)

        # Final test evaluation
        X_test_scaled = incremental_scaler.transform(X_test.to_numpy(dtype=np.float64))
        final_preds = model.predict(X_test_scaled)
        final_acc = float(accuracy_score(y_test, final_preds))
        final_f1 = float(f1_score(y_test, final_preds, zero_division=0))

        final_auc = final_acc
        if hasattr(model, "predict_proba"):
            try:
                final_auc = float(roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]))
            except Exception:
                pass

        logger.info(
            "Incremental learning completed: %d chunks (%d samples) | Test Acc: %.4f | AUC: %.4f | Time: %.3fs",
            num_chunks, cumulative_records, final_acc, final_auc, total_elapsed,
        )

        return IncrementalLearningResult(
            algorithm=algorithm,
            dataset_name=dataset_name,
            total_chunks=num_chunks,
            total_records_streamed=cumulative_records,
            chunk_size=chunk_size,
            total_training_time_seconds=total_elapsed,
            final_test_accuracy=round(final_acc, 4),
            final_test_roc_auc=round(final_auc, 4),
            final_test_f1=round(final_f1, 4),
            streaming_steps=steps,
            supports_partial_fit=True,
            memory_peak_mb=round(peak_rss, 2),
        )
