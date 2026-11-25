"""
CANCERWISE AI — Phase 6 Comprehensive Test Suite
Tests Semi-Supervised, Multitask (PyTorch), Multiclass Subtypes,
One-Class Novelty, Continuous Regression, Kernel Learning, Structured Output,
Advanced Experiment Manager, and REST Endpoints.
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backend.schemas.advanced_learning import (
    SemiSupervisedMethod,
    MulticlassStrategy,
    OneClassMethod,
    RegressionMethod,
    KernelType,
    SemiSupervisedExperimentResult,
    MultitaskExperimentResult,
    MulticlassExperimentResult,
    OneClassExperimentResult,
    RegressionExperimentResult,
    KernelLearningExperimentResult,
    StructuredOutputExperimentResult,
    AdvancedLabComparisonSummary,
)
from data_mining.experiments.research_data_factory import ResearchDataFactory
from data_mining.semi_supervised.semi_supervised_engine import SemiSupervisedEngine
from data_mining.multitask.multitask_engine import MultitaskEngine
from data_mining.multiclass.multiclass_engine import MulticlassEngine
from data_mining.one_class.one_class_engine import OneClassEngine
from data_mining.regression.regression_engine import RegressionEngine
from data_mining.kernel_learning.kernel_learning_engine import KernelLearningEngine
from data_mining.structured_output.structured_output_engine import StructuredOutputEngine
from data_mining.experiments.advanced_experiment_manager import AdvancedExperimentManager
from backend.api.advanced import (
    run_semi_supervised_experiment,
    run_multitask_experiment,
    run_multiclass_experiment,
    run_one_class_novelty,
    run_regression_experiment,
    run_kernel_benchmark,
    run_structured_output_experiment,
    run_full_advanced_suite,
)


@pytest.fixture
def factory():
    return ResearchDataFactory()


# ---------------------------------------------------------------------------
# 1. Semi-Supervised Learning Tests
# ---------------------------------------------------------------------------
class TestSemiSupervisedEngine:
    def test_semi_supervised_label_spreading(self, factory):
        X, y = factory.get_wdbc_binary_data()
        engine = SemiSupervisedEngine(random_state=42)
        res = engine.run_experiment(
            X, y,
            method=SemiSupervisedMethod.LABEL_SPREADING,
            proportions=[0.20, 0.50],
            dataset_name="UCI WDBC",
        )

        assert isinstance(res, SemiSupervisedExperimentResult)
        assert len(res.proportion_evaluations) == 2
        for prop in res.proportion_evaluations:
            assert 0.0 <= prop.test_accuracy <= 1.0
            assert 0.0 <= prop.test_roc_auc <= 1.0
            assert 0.0 <= prop.test_f1 <= 1.0
            assert prop.num_labeled + prop.num_unlabeled > 0

    def test_confidence_pseudo_labeling(self, factory):
        X, y = factory.get_wdbc_binary_data()
        engine = SemiSupervisedEngine(random_state=42)
        res = engine.run_experiment(
            X, y,
            method=SemiSupervisedMethod.CONFIDENCE_PSEUDO_LABELING,
            proportions=[0.30],
            confidence_threshold=0.80,
        )
        assert len(res.proportion_evaluations) == 1
        assert res.proportion_evaluations[0].pseudo_labeled_count >= 0


# ---------------------------------------------------------------------------
# 2. Multitask Learning (PyTorch) Tests
# ---------------------------------------------------------------------------
class TestMultitaskEngine:
    def test_multitask_dual_task_training(self, factory):
        X, targets, task_types = factory.get_multitask_data(n_samples=120, random_state=42)
        engine = MultitaskEngine(random_state=42)
        res = engine.train_multitask(
            X, targets, task_types, epochs=15, batch_size=32, is_synthetic_benchmark=True
        )

        assert isinstance(res, MultitaskExperimentResult)
        assert res.is_synthetic_benchmark is True
        assert len(res.tasks) == 2
        assert res.combined_test_loss >= 0.0
        assert res.early_stopped_epoch > 0


# ---------------------------------------------------------------------------
# 3. Multiclass Subtype Tests
# ---------------------------------------------------------------------------
class TestMulticlassEngine:
    def test_multiclass_strategies(self, factory):
        X, y, class_names = factory.get_multiclass_subtype_data(n_samples=150, random_state=42)
        engine = MulticlassEngine(random_state=42)

        for strat in [MulticlassStrategy.MULTINOMIAL_LOGISTIC, MulticlassStrategy.ONE_VS_REST, MulticlassStrategy.RANDOM_FOREST]:
            res = engine.run_multiclass_experiment(X, y, class_names=class_names, strategy=strat)
            assert isinstance(res, MulticlassExperimentResult)
            assert 0.0 <= res.macro_f1 <= 1.0
            assert 0.0 <= res.accuracy <= 1.0
            assert len(res.per_class_metrics) == 4
            assert len(res.confusion_matrix) == 4

    def test_multiclass_rejects_binary_data(self, factory):
        X, y = factory.get_wdbc_binary_data()
        engine = MulticlassEngine(random_state=42)
        with pytest.raises(ValueError, match="Multiclass engine requires at least 3 distinct classes"):
            engine.run_multiclass_experiment(X, y)


# ---------------------------------------------------------------------------
# 4. One-Class Novelty Detection Tests
# ---------------------------------------------------------------------------
class TestOneClassEngine:
    def test_isolation_forest_and_svm_novelty(self, factory):
        X, y = factory.get_wdbc_binary_data()
        benign_mask = (y == 0)
        X_benign = X[benign_mask]
        X_malignant = X[~benign_mask]

        X_train_norm = X_benign.iloc[:150]
        X_test_norm = X_benign.iloc[150:]
        X_test_atyp = X_malignant

        engine = OneClassEngine(random_state=42)
        for method in [OneClassMethod.ISOLATION_FOREST, OneClassMethod.ONE_CLASS_SVM]:
            res = engine.run_novelty_experiment(X_train_norm, X_test_norm, X_test_atyp, method=method)
            assert isinstance(res, OneClassExperimentResult)
            assert 0.0 <= res.anomaly_detection_rate <= 1.0
            assert 0.0 <= res.normal_identification_rate <= 1.0
            assert "score_separation_gap" in res.score_summary


# ---------------------------------------------------------------------------
# 5. Continuous Regression Tests
# ---------------------------------------------------------------------------
class TestRegressionEngine:
    def test_regression_algorithms(self, factory):
        X, y, target_name = factory.get_continuous_regression_data(n_samples=120, random_state=42)
        engine = RegressionEngine(random_state=42)

        for method in [RegressionMethod.RIDGE, RegressionMethod.LASSO, RegressionMethod.RANDOM_FOREST_REGRESSOR]:
            res = engine.run_regression_experiment(X, y, method=method, target_name=target_name)
            assert isinstance(res, RegressionExperimentResult)
            assert res.mse >= 0.0
            assert res.rmse >= 0.0
            assert res.mae >= 0.0

    def test_regression_rejects_discrete_labels(self, factory):
        X, y = factory.get_wdbc_binary_data()
        engine = RegressionEngine(random_state=42)
        with pytest.raises(ValueError, match="Continuous regression engine requires a genuine continuous target"):
            engine.run_regression_experiment(X, y)


# ---------------------------------------------------------------------------
# 6. Kernel Learning Tests
# ---------------------------------------------------------------------------
class TestKernelLearningEngine:
    def test_kernel_benchmarks(self, factory):
        X, y = factory.get_wdbc_binary_data()
        engine = KernelLearningEngine(random_state=42)
        res = engine.benchmark_kernels(X, y)

        assert isinstance(res, KernelLearningExperimentResult)
        assert len(res.benchmarks) == 4
        assert res.top_kernel in [KernelType.LINEAR, KernelType.RBF, KernelType.POLYNOMIAL, KernelType.SIGMOID]
        assert "grid_evaluations" in res.sensitivity_analysis


# ---------------------------------------------------------------------------
# 7. Structured Output Tests
# ---------------------------------------------------------------------------
class TestStructuredOutputEngine:
    def test_multilabel_receptor_panel(self, factory):
        X, Y, labels = factory.get_structured_multilabel_data(n_samples=120, random_state=42)
        engine = StructuredOutputEngine(random_state=42)
        res = engine.run_structured_experiment(X, Y)

        assert isinstance(res, StructuredOutputExperimentResult)
        assert res.output_dimensions == 4
        assert 0.0 <= res.hamming_loss <= 1.0
        assert 0.0 <= res.subset_accuracy <= 1.0
        assert 0.0 <= res.micro_f1 <= 1.0


# ---------------------------------------------------------------------------
# 8. Advanced Experiment Manager & REST API Tests
# ---------------------------------------------------------------------------
class TestAdvancedExperimentManagerAndAPI:
    def test_manager_full_suite(self):
        manager = AdvancedExperimentManager(random_state=42)
        # Verify individual runner methods work
        ss_res = manager.run_semi_supervised(proportions=[0.20])
        assert ss_res.method == SemiSupervisedMethod.LABEL_SPREADING

        mc_res = manager.run_multiclass()
        assert mc_res.strategy == MulticlassStrategy.MULTINOMIAL_LOGISTIC

    def test_api_endpoints(self):
        # Semi-supervised API
        ss_api = run_semi_supervised_experiment(method=SemiSupervisedMethod.LABEL_SPREADING)
        assert ss_api.fully_supervised_baseline_auc > 0.85

        # Multiclass API
        mc_api = run_multiclass_experiment(strategy=MulticlassStrategy.MULTINOMIAL_LOGISTIC)
        assert mc_api.accuracy > 0.70

        # One-Class Novelty API
        oc_api = run_one_class_novelty(method=OneClassMethod.ISOLATION_FOREST)
        assert oc_api.normal_identification_rate > 0.80

        # Regression API
        reg_api = run_regression_experiment(method=RegressionMethod.RIDGE)
        assert reg_api.rmse > 0.0
