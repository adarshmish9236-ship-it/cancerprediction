"""
CANCERWISE AI — Multiclass Classification Service
Coordinates molecular subtype classification (Luminal A, Luminal B, HER2, Triple-Negative)
across One-vs-Rest, One-vs-One, and Multinomial strategies.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from backend.core.logging_setup import get_logger
from backend.schemas.advanced_learning import (
    MulticlassStrategy,
    MulticlassExperimentResult,
)
from backend.services.job_service import get_job_manager, JobInfo
from data_mining.experiments.advanced_experiment_manager import AdvancedExperimentManager

logger = get_logger("services.multiclass")


class MulticlassClassificationService:
    """
    Dedicated service for multiclass molecular subtype prediction.
    """

    def __init__(self, manager: Optional[AdvancedExperimentManager] = None):
        self.manager = manager or AdvancedExperimentManager()
        self.job_manager = get_job_manager()

    def train_multiclass(
        self,
        strategy: MulticlassStrategy = MulticlassStrategy.MULTINOMIAL_LOGISTIC,
        async_job: bool = False,
    ) -> MulticlassExperimentResult | JobInfo:
        if async_job:
            def _worker(ctx):
                ctx.set_progress(0.2, f"Training multiclass model ({strategy.value})...")
                res = self.manager.run_multiclass(strategy=strategy)
                ctx.set_progress(1.0, "Multiclass training complete.")
                return res

            return self.job_manager.submit(
                job_type=f"multiclass_{strategy.value}",
                target_fn=_worker,
            )

        return self.manager.run_multiclass(strategy=strategy)
