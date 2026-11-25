"""
CANCERWISE AI — Multitask Learning Service
Coordinates PyTorch shared-representation multitask neural network training
for joint binary diagnosis classification and continuous biomarker regression.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from backend.core.logging_setup import get_logger
from backend.schemas.advanced_learning import MultitaskExperimentResult
from backend.services.job_service import get_job_manager, JobInfo
from data_mining.experiments.advanced_experiment_manager import AdvancedExperimentManager

logger = get_logger("services.multitask")


class MultitaskLearningService:
    """
    Dedicated service for PyTorch shared-representation multitask training.
    """

    def __init__(self, manager: Optional[AdvancedExperimentManager] = None):
        self.manager = manager or AdvancedExperimentManager()
        self.job_manager = get_job_manager()

    def train_multitask(
        self,
        epochs: int = 40,
        learning_rate: float = 0.005,
        async_job: bool = False,
    ) -> MultitaskExperimentResult | JobInfo:
        if async_job:
            def _worker(ctx):
                ctx.set_progress(0.1, f"Training multitask network for {epochs} epochs...")
                res = self.manager.run_multitask(epochs=epochs)
                ctx.set_progress(1.0, "Multitask network converged.")
                return res

            return self.job_manager.submit(
                job_type="multitask_training",
                target_fn=_worker,
            )

        return self.manager.run_multitask(epochs=epochs)


