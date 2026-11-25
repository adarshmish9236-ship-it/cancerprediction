"""
CANCERWISE AI — One-Class Novelty Classification Service
Coordinates one-class novelty and malignancy deviation detection on normative benign baselines
(Isolation Forest, One-Class SVM, Local Outlier Factor).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from backend.core.logging_setup import get_logger
from backend.schemas.advanced_learning import (
    OneClassMethod,
    OneClassExperimentResult,
)
from backend.services.job_service import get_job_manager, JobInfo
from data_mining.experiments.advanced_experiment_manager import AdvancedExperimentManager

logger = get_logger("services.one_class")


class OneClassClassificationService:
    """
    Dedicated service for anomaly and novelty detection on normative benign cytology.
    """

    def __init__(self, manager: Optional[AdvancedExperimentManager] = None):
        self.manager = manager or AdvancedExperimentManager()
        self.job_manager = get_job_manager()

    def train_one_class(
        self,
        method: OneClassMethod = OneClassMethod.ISOLATION_FOREST,
        contamination: float = 0.05,
        async_job: bool = False,
    ) -> OneClassExperimentResult | JobInfo:
        if async_job:
            def _worker(ctx):
                ctx.set_progress(0.2, f"Fitting one-class novelty model ({method.value})...")
                res = self.manager.run_one_class(method=method)
                ctx.set_progress(1.0, "One-class novelty model complete.")
                return res

            return self.job_manager.submit(
                job_type=f"one_class_{method.value}",
                target_fn=_worker,
            )

        return self.manager.run_one_class(method=method)
