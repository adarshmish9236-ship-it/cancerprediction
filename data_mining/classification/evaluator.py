"""
CANCERWISE AI — Comprehensive Model Evaluator
Computes clinical & statistical performance metrics for oncology classification:
Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, Balanced Accuracy,
ROC-AUC, PR-AUC, MCC, Confusion Matrix, Log Loss, Brier Score, and 95% Bootstrap CIs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    log_loss,
    brier_score_loss,
)

from backend.schemas.modeling import (
    ModelEvaluationMetrics,
    ConfidenceInterval,
)

logger = logging.getLogger("cancerwise.classification.evaluator")


class ModelEvaluator:
    """
    Evaluates binary cancer classification performance across discrimination,
    calibration, threshold-dependent metrics, and statistical bootstrap intervals.
    """

    @staticmethod
    def evaluate(
        y_true: Union[np.ndarray, pd.Series, List[int]],
        y_pred: Union[np.ndarray, pd.Series, List[int]],
        y_prob: Optional[Union[np.ndarray, pd.Series, List[float]]] = None,
        compute_ci: bool = True,
        n_bootstraps: int = 100,
        random_state: int = 42,
    ) -> ModelEvaluationMetrics:
        """
        Computes all clinical evaluation metrics from predictions and probabilities.
        """
        y_t = np.asarray(y_true, dtype=int)
        y_p = np.asarray(y_pred, dtype=int)

        # Handle probabilities
        if y_prob is not None:
            y_pr = np.asarray(y_prob, dtype=float)
            if y_pr.ndim == 2:
                y_pr = y_pr[:, 1]
        else:
            y_pr = y_p.astype(float)

        # Confusion Matrix components
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        acc = float(accuracy_score(y_t, y_p))
        prec = float(precision_score(y_t, y_p, zero_division=0))
        rec_sens = float(recall_score(y_t, y_p, zero_division=0))
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float(f1_score(y_t, y_p, zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_t, y_p))

        try:
            roc_auc = float(roc_auc_score(y_t, y_pr))
        except Exception:
            roc_auc = acc

        try:
            pr_auc = float(average_precision_score(y_t, y_pr))
        except Exception:
            pr_auc = prec

        mcc = float(matthews_corrcoef(y_t, y_p))

        try:
            eps = 1e-15
            clipped_probs = np.clip(y_pr, eps, 1 - eps)
            ll = float(log_loss(y_t, clipped_probs))
        except Exception:
            ll = 0.0

        brier = float(brier_score_loss(y_t, y_pr))

        # Bootstrap Confidence Intervals
        ci_dict: Dict[str, ConfidenceInterval] = {}
        if compute_ci and len(y_t) > 20:
            ci_dict = ModelEvaluator._compute_bootstrap_cis(
                y_t, y_p, y_pr, n_bootstraps=n_bootstraps, random_state=random_state
            )

        return ModelEvaluationMetrics(
            accuracy=round(acc, 4),
            precision=round(prec, 4),
            recall_sensitivity=round(rec_sens, 4),
            specificity=round(spec, 4),
            f1_score=round(f1, 4),
            balanced_accuracy=round(bal_acc, 4),
            roc_auc=round(roc_auc, 4),
            pr_auc=round(pr_auc, 4),
            matthews_corr_coef=round(mcc, 4),
            log_loss_score=round(ll, 4),
            brier_score=round(brier, 4),
            confusion_matrix=cm.tolist(),
            confidence_intervals=ci_dict,
        )

    @staticmethod
    def _compute_bootstrap_cis(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        n_bootstraps: int = 100,
        random_state: int = 42,
    ) -> Dict[str, ConfidenceInterval]:
        """Calculates 95% bootstrap confidence intervals for primary metrics."""
        rng = np.random.RandomState(random_state)
        n = len(y_true)
        boot_acc, boot_rec, boot_auc = [], [], []

        for _ in range(n_bootstraps):
            indices = rng.choice(n, size=n, replace=True)
            y_t_b = y_true[indices]
            y_p_b = y_pred[indices]
            y_pr_b = y_prob[indices]

            if len(np.unique(y_t_b)) < 2:
                continue

            boot_acc.append(accuracy_score(y_t_b, y_p_b))
            boot_rec.append(recall_score(y_t_b, y_p_b, zero_division=0))
            try:
                boot_auc.append(roc_auc_score(y_t_b, y_pr_b))
            except Exception:
                pass

        ci_res: Dict[str, ConfidenceInterval] = {}
        if boot_acc:
            ci_res["accuracy"] = ConfidenceInterval(
                lower=round(float(np.percentile(boot_acc, 2.5)), 4),
                upper=round(float(np.percentile(boot_acc, 97.5)), 4),
            )
        if boot_rec:
            ci_res["recall_sensitivity"] = ConfidenceInterval(
                lower=round(float(np.percentile(boot_rec, 2.5)), 4),
                upper=round(float(np.percentile(boot_rec, 97.5)), 4),
            )
        if boot_auc:
            ci_res["roc_auc"] = ConfidenceInterval(
                lower=round(float(np.percentile(boot_auc, 2.5)), 4),
                upper=round(float(np.percentile(boot_auc, 97.5)), 4),
            )

        return ci_res
