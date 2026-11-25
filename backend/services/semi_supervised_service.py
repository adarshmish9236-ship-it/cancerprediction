"""
CANCERWISE AI — Semi-Supervised Learning Service
Coordinates label propagation, label spreading, and low-annotation regimes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.core.logging_setup import get_logger
from backend.schemas.advanced_learning import (
    SemiSupervisedMethod,
    SemiSupervisedExperimentResult,
)
from backend.services.job_service import get_job_manager, JobInfo
from data_mining.experiments.advanced_experiment_manager import AdvancedExperimentManager

logger = get_logger("services.semi_supervised")


class SemiSupervisedLearningService:
    """
    Dedicated service for semi-supervised transductive and inductive oncology modeling.
    """

    def __init__(self, manager: Optional[AdvancedExperimentManager] = None):
        self.manager = manager or AdvancedExperimentManager()
        self.job_manager = get_job_manager()

    def train_semi_supervised(
        self,
        method: SemiSupervisedMethod = SemiSupervisedMethod.LABEL_SPREADING,
        labeled_fractions: Optional[List[float]] = None,
        async_job: bool = False,
    ) -> SemiSupervisedExperimentResult | JobInfo:
        if async_job:
            def _worker(ctx):
                ctx.set_progress(0.2, f"Starting {method.value} semi-supervised learning...")
                res = self.manager.run_semi_supervised(method=method, proportions=labeled_fractions)
                ctx.set_progress(1.0, "Completed semi-supervised evaluation.")
                return res

            return self.job_manager.submit(
                job_type=f"semi_supervised_{method.value}",
                target_fn=_worker,
            )

        return self.manager.run_semi_supervised(method=method, proportions=labeled_fractions)


