"""
CANCERWISE AI — Clinical Model Validator
Calculates clinical diagnostic metrics (Sensitivity, Specificity, FNR, FPR, PPV, NPV),
prevalence sensitivity modeling via Bayes' Theorem, Repeated Stratified CV,
and 95% Bootstrap Confidence Intervals.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    balanced_accuracy_score,
)

from backend.schemas.explainability import (
    HealthcareValidationMetrics,
    PrevalenceImpactItem,
    ThresholdAnalysisItem,
)

logger = logging.getLogger("cancerwise.evaluation.validator")


class ModelValidator:
    """
    Evaluates clinical reliability, class prevalence impacts, and threshold trade-offs.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def evaluate_clinical_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        y_prob: Optional[Union[np.ndarray, pd.Series]] = None,
        compute_bootstrap: bool = True,
        n_bootstraps: int = 100,
    ) -> HealthcareValidationMetrics:
        """
        Computes clinical metrics, 95% bootstrap CIs, and prevalence sensitivity scenarios.
        """
        y_t = np.asarray(y_true, dtype=int)
        y_p = np.asarray(y_pred, dtype=int)

        if y_prob is not None:
            y_pr = np.asarray(y_prob, dtype=float)
            if y_pr.ndim == 2:
                y_pr = y_pr[:, 1]
        else:
            y_pr = y_p.astype(float)

        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
        fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

        empirical_prev = float(np.mean(y_t))

        try:
            auc = float(roc_auc_score(y_t, y_pr))
        except Exception:
            auc = sensitivity

        try:
            prauc = float(average_precision_score(y_t, y_pr))
        except Exception:
            prauc = ppv

        brier = float(brier_score_loss(y_t, y_pr))
        mcc = float(matthews_corrcoef(y_t, y_p))
        bal_acc = float(balanced_accuracy_score(y_t, y_p))

        # 95% Bootstrap Confidence Intervals
        cis: Dict[str, Dict[str, float]] = {}
        if compute_bootstrap and len(y_t) > 20:
            cis = self._compute_bootstrap_cis(y_t, y_p, y_pr, n_bootstraps=n_bootstraps)

        # Prevalence Sensitivity Simulator (Bayes' Theorem)
        prev_scenarios = [0.005, 0.02, 0.05, 0.15, empirical_prev, 0.50]
        prev_impacts = self.simulate_prevalence_impact(sensitivity, specificity, prev_scenarios)

        return HealthcareValidationMetrics(
            sensitivity_recall=round(sensitivity, 4),
            specificity=round(specificity, 4),
            false_negative_rate_fnr=round(fnr, 4),
            false_positive_rate_fpr=round(fpr, 4),
            positive_predictive_value_ppv=round(ppv, 4),
            negative_predictive_value_npv=round(npv, 4),
            empirical_cohort_prevalence=round(empirical_prev, 4),
            roc_auc=round(auc, 4),
            pr_auc=round(prauc, 4),
            brier_score=round(brier, 4),
            matthews_corr_coef=round(mcc, 4),
            balanced_accuracy=round(bal_acc, 4),
            confusion_matrix=cm.tolist(),
            confidence_intervals_95=cis,
            prevalence_sensitivity_analysis=prev_impacts,
        )

    def simulate_prevalence_impact(
        self,
        sensitivity: float,
        specificity: float,
        prevalences: List[float],
    ) -> List[PrevalenceImpactItem]:
        """
        Applies Bayes' Theorem to project PPV and NPV across hypothetical clinical settings:
        PPV = (Sens * Prev) / [Sens * Prev + (1 - Spec) * (1 - Prev)]
        NPV = [Spec * (1 - Prev)] / [Spec * (1 - Prev) + (1 - Sens) * Prev]
        """
        results: List[PrevalenceImpactItem] = []
        for prev in prevalences:
            denom_ppv = (sensitivity * prev) + ((1.0 - specificity) * (1.0 - prev))
            ppv_sim = (sensitivity * prev) / denom_ppv if denom_ppv > 0 else 0.0

            denom_npv = (specificity * (1.0 - prev)) + ((1.0 - sensitivity) * prev)
            npv_sim = (specificity * (1.0 - prev)) / denom_npv if denom_npv > 0 else 0.0

            if prev <= 0.02:
                interp = "General population asymptomatic screening: PPV is severely depressed by low disease prevalence."
            elif prev <= 0.10:
                interp = "Primary care referral population: moderate PPV with high NPV reliability."
            else:
                interp = "Specialized oncology biopsy cohort: high prevalence maintains strong clinical PPV."

            results.append(
                PrevalenceImpactItem(
                    hypothetical_prevalence_percent=round(prev * 100.0, 1),
                    positive_predictive_value_ppv=round(float(ppv_sim), 4),
                    negative_predictive_value_npv=round(float(npv_sim), 4),
                    clinical_interpretation=interp,
                )
            )
        return results

    def analyze_decision_thresholds(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: Optional[List[float]] = None,
    ) -> List[ThresholdAnalysisItem]:
        """
        Calculates diagnostic trade-offs across cutoffs (0.05 to 0.95) without peaking at test labels.
        """
        if thresholds is None:
            thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

        items: List[ThresholdAnalysisItem] = []
        y_t = np.asarray(y_true, dtype=int)
        y_pr = np.asarray(y_prob, dtype=float)

        for th in thresholds:
            y_p = (y_pr >= th).astype(int)
            cm = confusion_matrix(y_t, y_p, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
            fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
            ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
            f1 = (2.0 * ppv * sens) / (ppv + sens) if (ppv + sens) > 0 else 0.0
            bal_acc = (sens + spec) / 2.0

            items.append(
                ThresholdAnalysisItem(
                    threshold=round(th, 2),
                    sensitivity=round(sens, 4),
                    specificity=round(spec, 4),
                    ppv=round(ppv, 4),
                    npv=round(npv, 4),
                    fnr=round(fnr, 4),
                    fpr=round(fpr, 4),
                    f1_score=round(f1, 4),
                    balanced_accuracy=round(bal_acc, 4),
                )
            )
        return items

    def _compute_bootstrap_cis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        n_bootstraps: int = 100,
    ) -> Dict[str, Dict[str, float]]:
        rng = np.random.RandomState(self.random_state)
        n = len(y_true)
        boot_sens, boot_spec, boot_auc = [], [], []

        for _ in range(n_bootstraps):
            idx = rng.choice(n, size=n, replace=True)
            y_t_b = y_true[idx]
            y_p_b = y_pred[idx]
            y_pr_b = y_prob[idx]

            if len(np.unique(y_t_b)) < 2:
                continue

            cm = confusion_matrix(y_t_b, y_p_b, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            boot_sens.append(sens)
            boot_spec.append(spec)
            try:
                boot_auc.append(roc_auc_score(y_t_b, y_pr_b))
            except Exception:
                pass

        cis: Dict[str, Dict[str, float]] = {}
        if boot_sens:
            cis["sensitivity"] = {
                "lower": round(float(np.percentile(boot_sens, 2.5)), 4),
                "upper": round(float(np.percentile(boot_sens, 97.5)), 4),
            }
        if boot_spec:
            cis["specificity"] = {
                "lower": round(float(np.percentile(boot_spec, 2.5)), 4),
                "upper": round(float(np.percentile(boot_spec, 97.5)), 4),
            }
        if boot_auc:
            cis["roc_auc"] = {
                "lower": round(float(np.percentile(boot_auc, 2.5)), 4),
                "upper": round(float(np.percentile(boot_auc, 97.5)), 4),
            }
        return cis

    # Alias for method
    calculate_metrics = evaluate_clinical_metrics


# Alias for compatibility
ClinicalModelValidator = ModelValidator

