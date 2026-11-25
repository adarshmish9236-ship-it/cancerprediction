"""
CANCERWISE AI — Computational Robustness & Perturbation Stress Testing Engine
Evaluates model stability against artificial Gaussian noise, Missing Completely at Random (MCAR)
data degradation, distribution shifts, and adversarial boundary perturbations.

GOVERNANCE DIRECTIVE:
Robustness stress tests measure computational stability against artificial perturbations.
They DO NOT prove real-world clinical generalizability or diagnostic immunity to hospital demographic shifts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.impute import SimpleImputer

from backend.schemas.explainability import (
    RobustnessDegradationStep,
    RobustnessAuditReport,
)

logger = logging.getLogger("cancerwise.evaluation.robustness")


class RobustnessAnalyzer:
    """
    Evaluates algorithmic resilience to input degradation, noise, and distributional shifts.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def stress_test_noise(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        noise_levels: Optional[List[float]] = None,
    ) -> List[RobustnessDegradationStep]:
        """
        Injects feature-scaled Gaussian noise N(0, (sigma * std)^2) across test samples.
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50]

        rng = np.random.RandomState(self.random_state)
        X_mat = X_test.values.astype(float)
        y_vec = np.asarray(y_test, dtype=int)
        std_devs = np.std(X_mat, axis=0)
        # Avoid division or scaling by zero for constant features
        std_devs[std_devs == 0.0] = 1.0

        # Baseline metrics (sigma = 0.0)
        baseline_pred = model.predict(X_mat)
        baseline_acc = float(accuracy_score(y_vec, baseline_pred))

        steps: List[RobustnessDegradationStep] = []

        for sigma in noise_levels:
            if sigma == 0.0:
                X_perturbed = X_mat
            else:
                noise = rng.normal(loc=0.0, scale=sigma, size=X_mat.shape) * std_devs
                X_perturbed = X_mat + noise

            preds = model.predict(X_perturbed)
            acc = float(accuracy_score(y_vec, preds))
            f1 = float(f1_score(y_vec, preds, zero_division=0))

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_perturbed)
                p_pos = probs[:, 1] if probs.ndim == 2 else probs
                try:
                    auc = float(roc_auc_score(y_vec, p_pos))
                except Exception:
                    auc = acc
            else:
                auc = acc

            drop = float(((baseline_acc - acc) / baseline_acc) * 100.0) if baseline_acc > 0 else 0.0

            steps.append(
                RobustnessDegradationStep(
                    perturbation_level=round(sigma, 3),
                    accuracy=round(acc, 4),
                    roc_auc=round(auc, 4),
                    f1_score=round(f1, 4),
                    performance_drop_percent=round(max(drop, 0.0), 2),
                )
            )

        return steps

    def stress_test_missingness(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        missing_rates: Optional[List[float]] = None,
    ) -> List[RobustnessDegradationStep]:
        """
        Simulates Missing Completely at Random (MCAR) data degradation and measures post-imputation stability.
        """
        if missing_rates is None:
            missing_rates = [0.0, 0.05, 0.10, 0.20, 0.30]

        rng = np.random.RandomState(self.random_state)
        X_mat = X_test.values.astype(float)
        y_vec = np.asarray(y_test, dtype=int)

        # Baseline metrics
        baseline_pred = model.predict(X_mat)
        baseline_acc = float(accuracy_score(y_vec, baseline_pred))

        steps: List[RobustnessDegradationStep] = []
        imputer = SimpleImputer(strategy="median")
        imputer.fit(X_mat)

        for rate in missing_rates:
            if rate == 0.0:
                X_imputed = X_mat
            else:
                mask = rng.rand(*X_mat.shape) < rate
                X_corrupted = X_mat.copy()
                X_corrupted[mask] = np.nan
                # Fill missing with median
                X_imputed = imputer.transform(X_corrupted)

            preds = model.predict(X_imputed)
            acc = float(accuracy_score(y_vec, preds))
            f1 = float(f1_score(y_vec, preds, zero_division=0))

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_imputed)
                p_pos = probs[:, 1] if probs.ndim == 2 else probs
                try:
                    auc = float(roc_auc_score(y_vec, p_pos))
                except Exception:
                    auc = acc
            else:
                auc = acc

            drop = float(((baseline_acc - acc) / baseline_acc) * 100.0) if baseline_acc > 0 else 0.0

            steps.append(
                RobustnessDegradationStep(
                    perturbation_level=round(rate, 2),
                    accuracy=round(acc, 4),
                    roc_auc=round(auc, 4),
                    f1_score=round(f1, 4),
                    performance_drop_percent=round(max(drop, 0.0), 2),
                )
            )

        return steps

    def evaluate_distribution_shift_gap(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        shift_magnitude: float = 0.5,
    ) -> float:
        """
        Simulates an empirical covariate distribution shift (mean shift of shift_magnitude * std).
        Returns the drop in accuracy.
        """
        X_mat = X_test.values.astype(float)
        y_vec = np.asarray(y_test, dtype=int)
        std_devs = np.std(X_mat, axis=0)

        base_pred = model.predict(X_mat)
        base_acc = float(accuracy_score(y_vec, base_pred))

        # Apply systematic positive mean shift
        X_shifted = X_mat + (shift_magnitude * std_devs)
        shifted_pred = model.predict(X_shifted)
        shifted_acc = float(accuracy_score(y_vec, shifted_pred))

        gap = max(0.0, base_acc - shifted_acc)
        return round(float(gap), 4)

    def evaluate_boundary_sensitivity(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        delta: float = 0.05,
    ) -> float:
        """
        Evaluates the fraction of borderline predictions (probability in [0.4, 0.6])
        whose classification flips when perturbed by delta.
        """
        X_mat = X_test.values.astype(float)
        y_vec = np.asarray(y_test, dtype=int)

        if not hasattr(model, "predict_proba"):
            return 0.0

        probs = model.predict_proba(X_mat)
        p_pos = probs[:, 1] if probs.ndim == 2 else probs

        borderline_mask = (p_pos >= 0.40) & (p_pos <= 0.60)
        n_borderline = int(np.sum(borderline_mask))

        if n_borderline == 0:
            return 0.0

        X_borderline = X_mat[borderline_mask]
        preds_orig = (p_pos[borderline_mask] >= 0.5).astype(int)

        rng = np.random.RandomState(self.random_state)
        noise = rng.normal(0, delta, size=X_borderline.shape)
        X_perturbed = X_borderline + noise

        probs_pert = model.predict_proba(X_perturbed)
        p_pert = probs_pert[:, 1] if probs_pert.ndim == 2 else probs_pert
        preds_pert = (p_pert >= 0.5).astype(int)

        flipped = np.sum(preds_orig != preds_pert)
        fraction_flipped = float(flipped / n_borderline)
        return round(fraction_flipped, 4)

    def audit_robustness(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_id: str = "model",
    ) -> RobustnessAuditReport:
        """
        Executes a comprehensive robustness audit spanning noise, missingness,
        distribution shift, and decision boundary sensitivity.
        """
        noise_steps = self.stress_test_noise(model, X_test, y_test)
        missing_steps = self.stress_test_missingness(model, X_test, y_test)
        shift_gap = self.evaluate_distribution_shift_gap(model, X_test, y_test)
        boundary_sens = self.evaluate_boundary_sensitivity(model, X_test, y_test)

        max_noise_drop = max([s.performance_drop_percent for s in noise_steps if s.perturbation_level <= 0.20], default=0.0)
        max_missing_drop = max([s.performance_drop_percent for s in missing_steps if s.perturbation_level <= 0.20], default=0.0)

        if max_noise_drop < 10.0 and max_missing_drop < 15.0 and shift_gap < 0.15:
            resilience = "High Resilience"
        elif max_noise_drop < 25.0 and max_missing_drop < 35.0:
            resilience = "Moderate Resilience"
        else:
            resilience = "Sensitive"

        return RobustnessAuditReport(
            model_id=model_id,
            noise_sensitivity=noise_steps,
            missing_value_sensitivity=missing_steps,
            distribution_shift_gap=shift_gap,
            adversarial_boundary_sensitivity=boundary_sens,
            resilience_rating=resilience,
        )
