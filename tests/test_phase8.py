"""
CANCERWISE AI — Phase 8 Test Suite
Exhaustive unit and integration tests for Explainable AI, Clinical Model Validation,
Probability Calibration, Robustness Audits, Model Governance, and Research Reporting.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas.explainability import (
    CalibrationMethod,
    HealthcareValidationMetrics,
    SHAPExplanationSummary,
    LIMEExplanationResult,
    RobustnessAuditReport,
)
from explainability.shap_explainer import SHAPExplainer
from explainability.lime_explainer import LIMEExplainer
from evaluation.model_validator import ModelValidator
from evaluation.calibration_engine import CalibrationEngine
from evaluation.robustness_analyzer import RobustnessAnalyzer
from evaluation.governance_service import ModelGovernanceService
from evaluation.research_reporter import ResearchReporter


@pytest.fixture(scope="module")
def synthetic_cancer_data():
    """Generates synthetic diagnostic features resembling WDBC."""
    rng = np.random.RandomState(42)
    n = 100
    p = 10
    feature_names = [f"biomarker_{i}" for i in range(p)]
    X = pd.DataFrame(rng.randn(n, p), columns=feature_names)
    # Log-odds
    logits = 1.2 * X["biomarker_0"] - 0.8 * X["biomarker_1"] + rng.randn(n) * 0.2
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = pd.Series((prob > 0.5).astype(int), name="diagnosis")
    return X, y


@pytest.fixture(scope="module")
def trained_models(synthetic_cancer_data):
    X, y = synthetic_cancer_data
    rf = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42)
    rf.fit(X, y)
    lr = LogisticRegression(random_state=42)
    lr.fit(X, y)
    return {"rf": rf, "lr": lr}


class TestSHAPExplainer:
    def test_tree_and_linear_shap(self, synthetic_cancer_data, trained_models):
        X, y = synthetic_cancer_data
        explainer = SHAPExplainer(random_state=42)

        # Tree model
        summary_rf = explainer.explain_global(trained_models["rf"], X, max_samples=40, model_id="test_rf")
        assert isinstance(summary_rf, SHAPExplanationSummary)
        assert len(summary_rf.global_importances) == X.shape[1]
        assert summary_rf.global_importances[0].rank == 1
        assert summary_rf.global_importances[0].mean_abs_shap >= summary_rf.global_importances[-1].mean_abs_shap

        # Linear model
        summary_lr = explainer.explain_global(trained_models["lr"], X, max_samples=40, model_id="test_lr")
        assert isinstance(summary_lr, SHAPExplanationSummary)
        assert len(summary_lr.global_importances) == X.shape[1]

    def test_local_shap_waterfall(self, synthetic_cancer_data, trained_models):
        X, y = synthetic_cancer_data
        explainer = SHAPExplainer(random_state=42)
        local_exp = explainer.explain_instance(
            model=trained_models["rf"],
            instance=X.iloc[0],
            X_background=X.iloc[:25],
            sample_index=0,
            top_k=5,
        )
        assert len(local_exp.contributions) == 5
        assert 0.0 <= local_exp.predicted_probability <= 1.0
        assert local_exp.predicted_label in ["Benign", "Malignant"]
        assert "NOT direct biological etiology" in local_exp.governance_notice

    def test_partial_dependence_and_ice(self, synthetic_cancer_data, trained_models):
        X, y = synthetic_cancer_data
        explainer = SHAPExplainer(random_state=42)
        pdp_res = explainer.compute_partial_dependence(
            model=trained_models["rf"],
            X_data=X,
            feature_name="biomarker_0",
            grid_resolution=15,
        )
        assert pdp_res.feature_name == "biomarker_0"
        assert len(pdp_res.grid_values) > 0
        assert len(pdp_res.average_partial_dependence) == len(pdp_res.grid_values)
        assert len(pdp_res.ice_curves) > 0


class TestLIMEExplainer:
    def test_lime_instance_explanation(self, synthetic_cancer_data, trained_models):
        X, y = synthetic_cancer_data
        lime = LIMEExplainer(random_state=42)
        res = lime.explain_instance(
            model=trained_models["rf"],
            instance=X.iloc[2],
            X_train=X,
            num_features=5,
            sample_index=2,
            model_id="test_rf",
        )
        assert isinstance(res, LIMEExplanationResult)
        assert len(res.rules) == 5
        assert res.predicted_label in ["Benign", "Malignant"]
        assert res.probability_benign + res.probability_malignant == pytest.approx(1.0, abs=1e-2)
        assert isinstance(res.explanation_fidelity_score, float)


class TestClinicalModelValidator:
    def test_clinical_metrics_and_bootstraps(self, synthetic_cancer_data, trained_models):
        X, y = synthetic_cancer_data
        validator = ModelValidator(random_state=42)
        rf = trained_models["rf"]
        y_pred = rf.predict(X)
        y_prob = rf.predict_proba(X)[:, 1]

        metrics = validator.evaluate_clinical_metrics(
            y_true=y,
            y_pred=y_pred,
            y_prob=y_prob,
            compute_bootstrap=True,
            n_bootstraps=50,
        )
        assert isinstance(metrics, HealthcareValidationMetrics)
        assert 0.0 <= metrics.sensitivity_recall <= 1.0
        assert 0.0 <= metrics.specificity <= 1.0
        assert metrics.false_negative_rate_fnr == pytest.approx(1.0 - metrics.sensitivity_recall, abs=1e-3)
        assert metrics.false_positive_rate_fpr == pytest.approx(1.0 - metrics.specificity, abs=1e-3)
        assert "sensitivity" in metrics.confidence_intervals_95
        assert "specificity" in metrics.confidence_intervals_95

    def test_bayes_prevalence_sensitivity(self):
        validator = ModelValidator(random_state=42)
        # 95% sensitivity, 95% specificity
        sens, spec = 0.95, 0.95
        scenarios = [0.01, 0.05, 0.10, 0.35, 0.50]
        results = validator.simulate_prevalence_impact(sens, spec, scenarios)

        assert len(results) == 5
        # Low prevalence (1%) -> PPV should be much lower than high prevalence (50%)
        low_prev_ppv = results[0].positive_predictive_value_ppv
        high_prev_ppv = results[-1].positive_predictive_value_ppv
        assert low_prev_ppv < 0.25
        assert high_prev_ppv >= 0.90
        assert results[0].negative_predictive_value_npv > 0.99

    def test_decision_thresholds(self, synthetic_cancer_data, trained_models):
        X, y = synthetic_cancer_data
        validator = ModelValidator(random_state=42)
        y_prob = trained_models["rf"].predict_proba(X)[:, 1]
        thresholds = [0.2, 0.5, 0.8]
        items = validator.analyze_decision_thresholds(y.values, y_prob, thresholds=thresholds)

        assert len(items) == 3
        # Higher threshold yields equal or higher specificity, and lower or equal sensitivity
        assert items[0].sensitivity >= items[2].sensitivity
        assert items[0].specificity <= items[2].specificity


class TestCalibrationEngine:
    def test_ece_and_reliability_curve(self, synthetic_cancer_data, trained_models):
        X, y = synthetic_cancer_data
        engine = CalibrationEngine()
        y_prob = trained_models["rf"].predict_proba(X)[:, 1]

        ece = engine.calculate_ece(y.values, y_prob, n_bins=5)
        assert 0.0 <= ece <= 1.0

        calib_res = engine.evaluate_calibration(y.values, y_prob, n_bins=5)
        assert calib_res.method == CalibrationMethod.UNCALIBRATED
        assert calib_res.brier_score >= 0.0
        assert len(calib_res.curve_points) > 0

    def test_fit_calibrators(self, synthetic_cancer_data, trained_models):
        X, y = synthetic_cancer_data
        engine = CalibrationEngine()
        model = trained_models["lr"]

        platt_cal = engine.fit_calibrator(model, X.values, y.values, method=CalibrationMethod.PLATT_SIGMOID, cv=2)
        assert hasattr(platt_cal, "predict_proba")
        probs_platt = platt_cal.predict_proba(X.values)
        assert probs_platt.shape == (len(X), 2)

        iso_cal = engine.fit_calibrator(model, X.values, y.values, method=CalibrationMethod.ISOTONIC, cv=2)
        assert hasattr(iso_cal, "predict_proba")


class TestRobustnessAnalyzer:
    def test_robustness_stress_audit(self, synthetic_cancer_data, trained_models):
        X, y = synthetic_cancer_data
        analyzer = RobustnessAnalyzer(random_state=42)
        model = trained_models["rf"]

        report = analyzer.audit_robustness(model, X, y, model_id="rf_audit")
        assert isinstance(report, RobustnessAuditReport)
        assert len(report.noise_sensitivity) >= 5
        assert len(report.missing_value_sensitivity) >= 4
        assert report.resilience_rating in ["High Resilience", "Moderate Resilience", "Sensitive"]
        assert 0.0 <= report.distribution_shift_gap <= 1.0
        assert 0.0 <= report.adversarial_boundary_sensitivity <= 1.0
        assert "DO NOT prove real-world clinical generalizability" in report.medical_disclaimer


class TestModelGovernanceAndReporting:
    def test_governance_service(self, tmp_path):
        service = ModelGovernanceService(governance_dir=tmp_path / "gov", mlflow_tracking_dir=tmp_path / "mlruns")
        record = service.register_governance_record(
            model_id="test_model_v1",
            algorithm="RandomForestClassifier",
            selected_features=["f1", "f2"],
            hyperparameters={"n_estimators": 50},
            validation_metrics={"sensitivity": 0.96, "specificity": 0.94},
            dataset_version="wdbc_v1.0",
        )
        assert record.model_id == "test_model_v1"
        assert record.feature_count == 2
        assert "python" in record.software_dependencies

        loaded = service.get_governance_record("test_model_v1")
        assert loaded is not None
        assert loaded.algorithm == "RandomForestClassifier"

        all_models = service.list_governed_models()
        assert len(all_models) == 1

    def test_research_reporter(self, tmp_path, synthetic_cancer_data, trained_models):
        X, y = synthetic_cancer_data
        validator = ModelValidator(random_state=42)
        metrics = validator.evaluate_clinical_metrics(y, trained_models["rf"].predict(X), trained_models["rf"].predict_proba(X)[:, 1])

        reporter = ResearchReporter(reports_dir=tmp_path / "reports")
        meta = reporter.generate_validation_report(
            model_id="test_report_model",
            algorithm="RandomForestClassifier",
            dataset_name="Synthetic Diagnostic Cohort",
            clinical_metrics=metrics,
        )
        assert Path(meta.report_path_html).exists()
        assert Path(meta.report_path_md).exists()

        md_content = Path(meta.report_path_md).read_text(encoding="utf-8")
        assert "Prevalence Sensitivity & Bayes' Theorem Analysis" in md_content
        assert "Regulatory & Ethical Governance Notice" in md_content


from backend.api.explainability import (
    list_governance_records,
    get_governance_record,
    calculate_shap_local,
    calculate_lime_local,
    evaluate_clinical_metrics as api_evaluate_clinical_metrics,
    evaluate_calibration as api_evaluate_calibration,
    evaluate_threshold_tradeoffs,
    audit_robustness,
    generate_research_report,
    LocalExplanationRequest,
    CalibrationRequest,
)


class TestExplainabilityAPI:
    def test_api_governance_endpoints(self):
        records = list_governance_records()
        assert isinstance(records, list)

        rec = get_governance_record("test_api_rf")
        assert rec.model_id == "test_api_rf"
        assert "python" in rec.software_dependencies

    def test_api_clinical_metrics(self):
        metrics = api_evaluate_clinical_metrics("baseline_rf", compute_bootstrap=False)
        assert 0.0 <= metrics.sensitivity_recall <= 1.0
        assert 0.0 <= metrics.specificity <= 1.0
        assert len(metrics.prevalence_sensitivity_analysis) > 0

    def test_api_thresholds(self):
        items = evaluate_threshold_tradeoffs("baseline_rf")
        assert isinstance(items, list)
        assert len(items) > 0
        assert items[0].sensitivity >= items[-1].sensitivity

    def test_api_calibration(self):
        req = CalibrationRequest(method=CalibrationMethod.PLATT_SIGMOID)
        calib = api_evaluate_calibration("baseline_rf", req)
        assert calib.expected_calibration_error_ece >= 0.0
        assert len(calib.curve_points) > 0

    def test_api_shap_local(self):
        req = LocalExplanationRequest(sample_index=0, num_features=4)
        exp = calculate_shap_local("baseline_rf", req)
        assert len(exp.contributions) <= 4
        assert exp.predicted_label in ["Benign", "Malignant"]

    def test_api_lime_local(self):
        req = LocalExplanationRequest(sample_index=0, num_features=4)
        lime_res = calculate_lime_local("baseline_rf", req)
        assert len(lime_res.rules) == 4
        assert lime_res.predicted_label in ["Benign", "Malignant"]

    def test_api_robustness_audit(self):
        audit = audit_robustness("baseline_rf")
        assert audit.resilience_rating in ["High Resilience", "Moderate Resilience", "Sensitive"]
        assert len(audit.noise_sensitivity) > 0

    def test_api_generate_report(self):
        rep = generate_research_report("baseline_rf")
        assert Path(rep.report_path_html).exists()
        assert Path(rep.report_path_md).exists()
