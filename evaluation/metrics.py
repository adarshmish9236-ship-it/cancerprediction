"""
CANCERWISE AI — Model Evaluation Utilities
Standardized metrics for classification tasks in cancer prediction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    classification_report, confusion_matrix,
)

logger = logging.getLogger("cancerwise.evaluation")


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = "binary",
    labels: Optional[list] = None,
) -> dict:
    """
    Compute a comprehensive set of classification metrics.

    Args:
        y_true:  Ground truth labels.
        y_pred:  Predicted labels.
        y_proba: Predicted probabilities (optional, for ROC-AUC and PR-AUC).
        average: Averaging strategy for multi-class ('binary','macro','weighted').
        labels:  Class label names for the report.

    Returns:
        Dictionary of metric name → value.
    """
    metrics: dict = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    if y_proba is not None:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                proba = y_proba[:, 1]
            else:
                proba = y_proba
            metrics["roc_auc"] = float(roc_auc_score(y_true, proba, average=average if average != "binary" else None))
            metrics["pr_auc"] = float(average_precision_score(y_true, proba, average=average if average != "binary" else None))
        except Exception as e:
            logger.warning("Could not compute ROC/PR AUC: %s", e)

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    metrics["classification_report"] = classification_report(y_true, y_pred, target_names=labels, zero_division=0)

    logger.info(
        "Evaluation | accuracy=%.4f | f1=%.4f | mcc=%.4f | roc_auc=%s",
        metrics["accuracy"], metrics["f1"], metrics["mcc"],
        f'{metrics.get("roc_auc", "N/A"):.4f}' if "roc_auc" in metrics else "N/A"
    )
    return metrics


def log_metrics(metrics: dict, prefix: str = "") -> None:
    """Log metrics dict to the cancerwise logger."""
    for name, val in metrics.items():
        if isinstance(val, float):
            logger.info("%s%s = %.6f", prefix, name, val)
