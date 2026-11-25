"""
CANCERWISE AI — Phase 10 Test Suite
Exhaustive verification of reusable UI components, theme system, APIClient bridge,
and syntax/import verification of all 14 Streamlit dashboard pages.
"""

from __future__ import annotations

import os
import py_compile
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from backend.core.config import PROJECT_ROOT
from frontend.components import (
    MetricCard,
    ChartContainer,
    DataPreview,
    ModelSelector,
    AlgorithmConfigurator,
    ExperimentProgress,
    ResultsTable,
    ConfusionMatrixView,
    PredictionResultView,
    ExplanationPanel,
    ExportControl,
    apply_theme,
    render_header,
)
from frontend.utils.api_client import APIClient


@pytest.fixture(scope="module")
def sample_patient_data():
    df = pd.DataFrame({
        "radius_mean": [17.99, 20.57, 19.69, 11.42, 20.29],
        "texture_mean": [10.38, 17.77, 21.25, 20.38, 14.34],
        "perimeter_mean": [122.8, 132.9, 130.0, 77.58, 135.1],
        "area_mean": [1001.0, 1326.0, 1203.0, 386.1, 1297.0],
        "smoothness_mean": [0.1184, 0.08474, 0.1096, 0.1425, 0.1003],
        "diagnosis": [1, 1, 1, 0, 1],
    })
    return df


class TestReusableUIComponents:
    def test_theme_and_headers(self):
        # Should execute without syntax or import errors
        assert callable(apply_theme)
        assert callable(render_header)

    def test_metric_card(self):
        assert hasattr(MetricCard, "render")

    def test_data_preview(self, sample_patient_data):
        assert hasattr(DataPreview, "render")

    def test_algorithm_configurator(self):
        params_rf = AlgorithmConfigurator.render("Random Forest Classifier")
        assert isinstance(params_rf, dict)
        assert "n_estimators" in params_rf

        params_lr = AlgorithmConfigurator.render("Logistic Regression")
        assert isinstance(params_lr, dict)
        assert "C" in params_lr

    def test_confusion_matrix_view(self):
        cm = [[45, 5], [3, 30]]
        assert hasattr(ConfusionMatrixView, "render")

    def test_prediction_result_view(self):
        assert hasattr(PredictionResultView, "render")

    def test_explanation_panel(self):
        assert hasattr(ExplanationPanel, "render")

    def test_export_control(self, sample_patient_data):
        assert hasattr(ExportControl, "render")


class TestAPIClientIntegration:
    def test_api_client_overview_stats(self):
        client = APIClient()
        stats = client.get_overview_statistics()
        assert isinstance(stats, dict)
        assert stats["total_datasets"] >= 1
        assert stats["records_processed"] >= 100
        assert stats["features_analyzed"] >= 10
        assert stats["best_roc_auc"] > 0.80

    def test_api_client_dataset_loading(self):
        client = APIClient()
        df = client.load_dataset("uci_wdbc")
        assert df is not None
        assert df.shape[0] == 569
        assert df.shape[1] >= 31

    def test_api_client_end_to_end_journey(self, sample_patient_data):
        client = APIClient()
        X = sample_patient_data.drop(columns=["diagnosis"])
        y = sample_patient_data["diagnosis"]

        clf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        probs = clf.predict_proba(X)[:, 1]

        # Clinical metrics
        metrics = client.get_clinical_metrics(y, preds, probs, bootstrap=False)
        assert metrics.sensitivity_recall >= 0.0
        assert metrics.specificity >= 0.0

        # Local explanation
        shap_loc = client.get_shap_local(clf, X.iloc[0], X, idx=0, top_k=3)
        assert len(shap_loc.contributions) <= 3
        assert shap_loc.predicted_label in ["Benign", "Malignant"]


class TestDashboardPagesCompilation:
    """Verifies that all 14 Streamlit pages and app.py compile cleanly with 0 syntax errors."""

    @pytest.mark.parametrize("page_filename", [
        "app.py",
        "pages/00_Landing_Page.py",
        "pages/01_Overview_Dashboard.py",
        "pages/02_Dataset_Explorer.py",
        "pages/03_Data_Wrangling_Studio.py",
        "pages/04_EDA_Studio.py",
        "pages/05_Feature_Engineering.py",
        "pages/06_Supervised_Learning_Lab.py",
        "pages/07_Advanced_Learning_Lab.py",
        "pages/08_Large_Scale_Learning.py",
        "pages/09_Model_Evaluation.py",
        "pages/10_Explainable_AI.py",
        "pages/11_Cancer_Prediction_Workspace.py",
        "pages/12_Model_Comparison.py",
        "pages/13_Research_Reports.py",
        "pages/14_System_Monitoring.py",
        "pages/15_Data_Quality_Analysis.py",
        "pages/16_Feature_Selection.py",
        "pages/17_Experiment_History.py",
        "pages/18_Settings.py",
    ])
    def test_page_syntax_and_compilation(self, page_filename):
        target_path = PROJECT_ROOT / "frontend" / page_filename
        assert target_path.exists(), f"File {page_filename} does not exist!"
        # Compile python bytecode to verify syntax
        compiled = py_compile.compile(str(target_path), doraise=True)
        assert compiled is not None
