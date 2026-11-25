"""
CANCERWISE AI — Model Evaluation & Clinical Validation Service
Coordinates 12 diagnostic performance metrics, 95% bootstrap confidence intervals,
probability calibration curves (ECE), decision thresholds, robustness audits,
model governance lineage, and publication-ready research reports.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from backend.core.exceptions import ModelNotFoundError
from backend.core.logging_setup import get_logger
from backend.schemas.explainability import (
    CalibrationMethod,
    CalibrationResult,
    HealthcareValidationMetrics,
    ThresholdAnalysisItem,
    RobustnessAuditReport,
    ModelGovernanceRecord,
    ResearchReportMetadata,
)
from backend.services.explainability_service import ExplainabilityService
from evaluation.model_validator import ModelValidator
from evaluation.calibration_engine import CalibrationEngine
from evaluation.robustness_analyzer import RobustnessAnalyzer
from evaluation.governance_service import ModelGovernanceService
from evaluation.research_reporter import ResearchReporter

logger = get_logger("services.model_evaluation")


class ModelEvaluationService:
    """
    Dedicated service for comprehensive clinical-grade model evaluation and governance.
    """

    def __init__(
        self,
        xai_service: Optional[ExplainabilityService] = None,
        validator: Optional[ModelValidator] = None,
        calibration_engine: Optional[CalibrationEngine] = None,
        robustness_analyzer: Optional[RobustnessAnalyzer] = None,
        governance_service: Optional[ModelGovernanceService] = None,
        research_reporter: Optional[ResearchReporter] = None,
    ):
        self.xai_service = xai_service or ExplainabilityService()
        self.validator = validator or ModelValidator()
        self.calibration_engine = calibration_engine or CalibrationEngine()
        self.robustness_analyzer = robustness_analyzer or RobustnessAnalyzer()
        self.governance_service = governance_service or ModelGovernanceService()
        self.research_reporter = research_reporter or ResearchReporter()

    def evaluate_model(
        self,
        model_id: str,
        compute_bootstrap: bool = True,
    ) -> HealthcareValidationMetrics:
        """Evaluates model with 12 clinical & statistical metrics + 95% bootstrap CIs."""
        model, X_train, y_train, X_test, y_test = self.xai_service.get_or_load_model_and_data(model_id)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        return self.validator.evaluate_clinical_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            compute_bootstrap=compute_bootstrap,
            n_bootstraps=60,
        )

    def evaluate_calibration(
        self,
        model_id: str,
        method: CalibrationMethod = CalibrationMethod.PLATT_SIGMOID,
    ) -> CalibrationResult:
        """Evaluates Expected Calibration Error (ECE) and fits calibrator."""
        model, X_train, y_train, X_test, y_test = self.xai_service.get_or_load_model_and_data(model_id)
        if method != CalibrationMethod.UNCALIBRATED:
            calibrator = self.calibration_engine.fit_calibrator(
                model=model,
                X_train=X_train.values,
                y_train=y_train.values,
                method=method,
            )
            y_prob = calibrator.predict_proba(X_test.values)[:, 1]
        else:
            y_prob = model.predict_proba(X_test.values)[:, 1]

        return self.calibration_engine.evaluate_calibration(
            y_true=y_test.values,
            y_prob=y_prob,
            method=method,
        )

    def evaluate_decision_thresholds(
        self,
        model_id: str,
    ) -> List[ThresholdAnalysisItem]:
        """Calculates sensitivity/specificity trade-offs across cutoffs (0.10 to 0.90)."""
        model, X_train, y_train, X_test, y_test = self.xai_service.get_or_load_model_and_data(model_id)
        y_prob = model.predict_proba(X_test.values)[:, 1]
        return self.validator.analyze_decision_thresholds(y_true=y_test.values, y_prob=y_prob)

    def audit_robustness(
        self,
        model_id: str,
    ) -> RobustnessAuditReport:
        """Stress tests model stability under Gaussian noise and MCAR data degradation."""
        model, X_train, y_train, X_test, y_test = self.xai_service.get_or_load_model_and_data(model_id)
        return self.robustness_analyzer.audit_robustness(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_id=model_id,
        )

    def get_governance_record(self, model_id: str) -> ModelGovernanceRecord:
        """Retrieves or registers an audit manifest with software lineage."""
        rec = self.governance_service.get_governance_record(model_id)
        if not rec:
            model, X_tr, y_tr, X_te, y_te = self.xai_service.get_or_load_model_and_data(model_id)
            y_pred = model.predict(X_te)
            rec = self.governance_service.register_governance_record(
                model_id=model_id,
                algorithm=type(model).__name__,
                selected_features=X_tr.columns.tolist(),
                hyperparameters=getattr(model, "get_params", lambda: {})(),
                validation_metrics={"accuracy": float(np.mean(y_pred == y_te))},
            )
        return rec

    def list_governance_records(self) -> List[Dict[str, Any]]:
        """Lists all governed model audit manifests."""
        return self.governance_service.list_governed_models()

    def generate_research_report(self, model_id: str) -> ResearchReportMetadata:
        """Generates comprehensive Markdown and HTML clinical research validation reports."""
        model, X_train, y_train, X_test, y_test = self.xai_service.get_or_load_model_and_data(model_id)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test.values)[:, 1]

        metrics = self.validator.evaluate_clinical_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            compute_bootstrap=False,
        )
        calib = self.calibration_engine.evaluate_calibration(
            y_true=y_test.values,
            y_prob=y_prob,
        )
        shap_summary = self.xai_service.explain_global_shap(model_id, max_samples=40)
        robust = self.robustness_analyzer.audit_robustness(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_id=model_id,
        )

        return self.research_reporter.generate_validation_report(
            model_id=model_id,
            algorithm=type(model).__name__,
            dataset_name="Wisconsin Diagnostic Breast Cancer (WDBC)",
            clinical_metrics=metrics,
            calibration_result=calib,
            shap_summary=shap_summary,
            robustness_report=robust,
        )
