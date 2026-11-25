"""
CANCERWISE AI — Phase 11 End-to-End (E2E) Test Suite
Executes and independently verifies a complete 12-step cancer research workflow:
Step 1: Upload / Load WDBC
Step 2: Inspect dataset metadata
Step 3: Generate data quality report
Step 4: Apply preprocessing
Step 5: Perform EDA
Step 6: Perform feature selection
Step 7: Train multiple classification models
Step 8: Evaluate models
Step 9: Select a model using predefined validation criterion
Step 10: Generate SHAP explanations
Step 11: Run predictions using approved model
Step 12: Export reports
"""

from __future__ import annotations

import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from backend.services.dataset_management_service import DatasetManagementService
from backend.services.dataset_ingestion_service import DatasetIngestionService
from preprocessing.quality_analyzer import DataQualityAnalyzer
from preprocessing.transformers import DataTransformer
from data_mining.eda.eda_engine import EDAEngine
from feature_engineering.selection_engine import FeatureSelectionEngine
from data_mining.classification.trainer import ModelTrainer
from data_mining.classification.prediction_service import PredictionService
from evaluation.model_validator import ClinicalModelValidator
from explainability.shap_explainer import SHAPExplainer
from evaluation.research_reporter import ResearchReporter
from backend.schemas.modeling import AlgorithmType, ModelTrainingConfig


class TestEndToEndResearchWorkflow:
    def test_complete_12_step_research_workflow(self):
        # -------------------------------------------------------------------
        # Step 1: Upload / Load WDBC
        # -------------------------------------------------------------------
        mgmt = DatasetManagementService()
        try:
            df = mgmt.get_dataframe("uci_wdbc")
        except Exception:
            ingest = DatasetIngestionService()
            ingest.ingest_wdbc()
            df = mgmt.get_dataframe("uci_wdbc")
        assert len(df) == 569
        assert "diagnosis" in df.columns

        # -------------------------------------------------------------------
        # Step 2: Inspect dataset metadata
        # -------------------------------------------------------------------
        meta = mgmt.get_metadata("uci_wdbc")
        assert meta.dataset_id == "uci_wdbc"
        assert meta.records_count == 569
        profile = mgmt.get_profile("uci_wdbc")
        assert profile["total_features"] >= 31

        # -------------------------------------------------------------------
        # Step 3: Generate data quality report
        # -------------------------------------------------------------------
        analyzer = DataQualityAnalyzer()
        quality_report = analyzer.analyze(df, dataset_name="uci_wdbc", target_column="diagnosis")
        assert quality_report.overall_quality_score >= 90.0

        # -------------------------------------------------------------------
        # Step 4: Apply preprocessing
        # -------------------------------------------------------------------
        num_cols = df.select_dtypes(include=[np.number]).columns
        transformer = DataTransformer(strategy="standard")
        scaled_df = transformer.fit_transform(df[num_cols])
        scaled_df["diagnosis"] = df["diagnosis"]
        assert scaled_df.shape[0] == 569

        # -------------------------------------------------------------------
        # Step 5: Perform EDA
        # -------------------------------------------------------------------
        eda_engine = EDAEngine()
        overview = eda_engine.get_overview(scaled_df, target_column="diagnosis")
        assert overview.total_records == 569
        hypo_results = eda_engine.run_statistical_tests(df, target_column="diagnosis")
        assert len(hypo_results) > 0
        significant_tests = [r for r in hypo_results if r.is_significant]
        assert len(significant_tests) >= 10

        # -------------------------------------------------------------------
        # Step 6: Perform feature selection
        # -------------------------------------------------------------------
        y_raw = df["diagnosis"]
        y = y_raw.map({"M": 1, "B": 0, "Malignant": 1, "Benign": 0}).astype(int)
        X = df.drop(columns=["diagnosis", "id"], errors="ignore").select_dtypes(include=[np.number]).fillna(0.0)

        selector = FeatureSelectionEngine()
        selected_features, rankings = selector.select_features(X, y, method="mutual_info", k=10)
        assert len(selected_features) == 10
        X_selected = X[selected_features]

        # -------------------------------------------------------------------
        # Step 7: Train multiple classification models
        # -------------------------------------------------------------------
        trainer = ModelTrainer()
        algorithms_to_train = [
            AlgorithmType.LOGISTIC_REGRESSION,
            AlgorithmType.RANDOM_FOREST,
            AlgorithmType.GRADIENT_BOOSTING,
        ]
        trained_results = []
        for alg in algorithms_to_train:
            config = ModelTrainingConfig(
                algorithm=alg,
                dataset_id="uci_wdbc",
                target_column="diagnosis",
                cv_folds=3,
            )
            pipe, res = trainer.train(X_selected, y, config)
            trained_results.append((alg, pipe, res))
            assert res.test_metrics.roc_auc > 0.90

        # -------------------------------------------------------------------
        # Step 8: Evaluate models
        # -------------------------------------------------------------------
        validator = ClinicalModelValidator(random_state=42)
        evaluation_summaries = []
        for alg, pipe, res in trained_results:
            probs = pipe.predict_proba(X_selected)[:, 1]
            preds = pipe.predict(X_selected)
            metrics = validator.evaluate_clinical_metrics(y.values, preds, probs)
            evaluation_summaries.append({
                "algorithm": alg.value,
                "pipeline": pipe,
                "result": res,
                "metrics": metrics,
                "roc_auc": metrics.roc_auc,
                "sensitivity": metrics.sensitivity,
                "specificity": metrics.specificity,
            })
            assert metrics.roc_auc > 0.90

        # -------------------------------------------------------------------
        # Step 9: Select a model using predefined validation criterion
        # Criterion: Highest ROC-AUC among models with Sensitivity >= 0.90
        # -------------------------------------------------------------------
        qualifying_models = [m for m in evaluation_summaries if m["sensitivity"] >= 0.90]
        assert len(qualifying_models) >= 1
        best_model_entry = max(qualifying_models, key=lambda m: m["roc_auc"])
        selected_model_id = best_model_entry["result"].model_id
        selected_pipeline = best_model_entry["pipeline"]
        assert selected_model_id is not None
        assert best_model_entry["roc_auc"] > 0.95

        # -------------------------------------------------------------------
        # Step 10: Generate SHAP explanations
        # -------------------------------------------------------------------
        explainer = SHAPExplainer(random_state=42)
        global_summary = explainer.explain_global(selected_pipeline, X_selected, max_samples=30, model_id=selected_model_id)
        assert len(global_summary.global_importances) == 10
        local_exp = explainer.explain_instance(selected_pipeline, X_selected, sample_index=0, model_id=selected_model_id)
        assert local_exp.waterfall_features is not None

        # -------------------------------------------------------------------
        # Step 11: Run predictions using approved model
        # -------------------------------------------------------------------
        pred_svc = PredictionService()
        sample_patient = X_selected.iloc[0].to_dict()
        pred_res = pred_svc.predict(selected_model_id, sample_patient)
        assert len(pred_res.predictions) == 1
        p0 = pred_res.predictions[0]
        assert p0.predicted_label in ["Benign", "Malignant"]
        assert 0.0 <= p0.probability_malignant <= 1.0

        # -------------------------------------------------------------------
        # Step 12: Export reports
        # -------------------------------------------------------------------
        reporter = ResearchReporter()
        report_meta = reporter.generate_validation_report(
            model_id=selected_model_id,
            algorithm=best_model_entry["algorithm"],
            dataset_name="uci_wdbc",
            clinical_metrics=best_model_entry["metrics"],
        )
        assert Path(report_meta.report_path_html).exists()
        assert Path(report_meta.report_path_md).exists()
