"""
CANCERWISE AI — Regression Learning Service
Coordinates continuous biomarker predictive modeling (Ridge, Lasso, ElasticNet, SVR, Random Forest).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from backend.core.logging_setup import get_logger
from backend.schemas.advanced_learning import (
    RegressionMethod,
    RegressionExperimentResult,
)
from backend.services.job_service import get_job_manager, JobInfo
from data_mining.experiments.advanced_experiment_manager import AdvancedExperimentManager

logger = get_logger("services.regression")


class RegressionLearningService:
    """
    Dedicated service for continuous biological biomarker prediction.
    """

    def __init__(self, manager: Optional[AdvancedExperimentManager] = None):
        self.manager = manager or AdvancedExperimentManager()
        self.job_manager = get_job_manager()

    def train_regression(
        self,
        method: RegressionMethod = RegressionMethod.RIDGE,
        async_job: bool = False,
    ) -> RegressionExperimentResult | JobInfo:
        if async_job:
            def _worker(ctx):
                ctx.set_progress(0.2, f"Fitting continuous regression model ({method.value})...")
                res = self.manager.run_regression(method=method)
                ctx.set_progress(1.0, "Regression training complete.")
                return res

            return self.job_manager.submit(
                job_type=f"regression_{method.value}",
                target_fn=_worker,
            )

        return self.manager.run_regression(method=method)
