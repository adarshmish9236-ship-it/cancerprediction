"""
CANCERWISE AI — Probability Calibration & Reliability Engine
Calculates reliability curves, Brier Score, and Expected Calibration Error (ECE).
Implements Platt Scaling (Sigmoid) and Isotonic Regression calibration.

GOVERNANCE DIRECTIVE:
Probability calibration ensures risk estimates reflect true empirical frequencies.
Decision thresholds must never be selected using test labels.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from backend.schemas.explainability import (
    CalibrationMethod,
    CalibrationPoint,
    CalibrationResult,
)

logger = logging.getLogger("cancerwise.evaluation.calibration")


class CalibrationEngine:
    """
    Evaluates probabilistic reliability and applies post-hoc calibration.
    """

    @staticmethod
    def calculate_ece(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Computes Expected Calibration Error (ECE):
        ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
        """
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_total = len(y_true)

        for i in range(n_bins):
            bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if i == n_bins - 1:
                bin_mask = bin_mask | (y_prob == bin_edges[i + 1])

            bin_count = np.sum(bin_mask)
            if bin_count > 0:
                bin_acc = np.mean(y_true[bin_mask])
                bin_conf = np.mean(y_prob[bin_mask])
                ece += (bin_count / n_total) * abs(bin_acc - bin_conf)

        return float(ece)

    def evaluate_calibration(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        method: CalibrationMethod = CalibrationMethod.UNCALIBRATED,
        n_bins: int = 10,
    ) -> CalibrationResult:
        """
        Generates reliability curve points, Brier score, and Expected Calibration Error.
        """
        y_t = np.asarray(y_true, dtype=int)
        y_pr = np.asarray(y_prob, dtype=float)

        prob_true, prob_pred = calibration_curve(y_t, y_pr, n_bins=n_bins, strategy="uniform")

        brier = float(brier_score_loss(y_t, y_pr))
        ece = self.calculate_ece(y_t, y_pr, n_bins=n_bins)

        points: List[CalibrationPoint] = []
        for frac, mean_p in zip(prob_true, prob_pred):
            points.append(
                CalibrationPoint(
                    mean_predicted_probability=round(float(mean_p), 4),
                    fraction_of_positives=round(float(frac), 4),
                    sample_count=int(np.sum((y_pr >= mean_p - 0.05) & (y_pr <= mean_p + 0.05))),
                )
            )

        return CalibrationResult(
            method=method,
            brier_score=round(brier, 4),
            expected_calibration_error_ece=round(ece, 4),
            curve_points=points,
        )

    def fit_calibrator(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        method: CalibrationMethod = CalibrationMethod.PLATT_SIGMOID,
        cv: int = 3,
    ) -> Any:
        """
        Fits cross-validated CalibratedClassifierCV using Platt Scaling or Isotonic Regression.
        """
        method_str = "sigmoid" if method == CalibrationMethod.PLATT_SIGMOID else "isotonic"
        calibrator = CalibratedClassifierCV(
            estimator=model,
            method=method_str,
            cv=cv,
        )
        calibrator.fit(X_train, y_train)
        return calibrator
