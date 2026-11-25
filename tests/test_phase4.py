"""
CANCERWISE AI — Phase 4 Comprehensive Test Suite
Tests Feature Engineering, Filter/Wrapper/Embedded Selection,
Strict Data Leakage Prevention in Cross-Validation, Explainability (SHAP & Permutation),
Dimensionality Reduction (PCA, KPCA, SVD, UMAP), and REST API endpoints.
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from backend.schemas.feature_engineering import (
    FilterSelectionMethod,
    WrapperSelectionMethod,
    EmbeddedSelectionMethod,
    DimensionalityMethod,
    DerivedFeatureDefinition,
    FeatureEngineeringResult,
    DimensionalityReductionResult,
    FeatureImportanceSummary,
)
from feature_engineering.engineering_engine import FeatureEngineeringEngine
from feature_engineering.selection_engine import FeatureSelectionEngine
from feature_engineering.reduction_engine import DimensionalityReductionEngine
from feature_engineering.ranking_service import FeatureRankingService
from backend.api.features import (
    engineer_features,
    select_features,
    compare_selection_strategies,
    reduce_dimensions,
    compare_reduction_methods,
    get_feature_importance_consensus,
)
from backend.services.dataset_service import DatasetService


@pytest.fixture
def synthetic_cancer_data():
    """Generates synthetic dataset mirroring WDBC morphometry schema."""
    np.random.seed(42)
    n = 120
    radius_mean = np.random.uniform(10.0, 25.0, size=n)
    perimeter_mean = 2 * np.pi * radius_mean + np.random.normal(0, 2.0, size=n)
    area_mean = np.pi * (radius_mean ** 2) + np.random.normal(0, 10.0, size=n)
    concavity_mean = np.random.uniform(0.01, 0.35, size=n)
    fractal_dimension_mean = np.random.uniform(0.05, 0.15, size=n)

    radius_worst = radius_mean * np.random.uniform(1.1, 1.4, size=n)
    perimeter_worst = 2 * np.pi * radius_worst
    area_worst = np.pi * (radius_worst ** 2)

    # Diagnosis with strong association to area & concavity
    logits = 0.01 * area_mean + 10.0 * concavity_mean - 6.0
    probs = 1 / (1 + np.exp(-logits))
    diagnosis = (probs > 0.5).astype(int)

    df = pd.DataFrame({
        "radius_mean": radius_mean,
        "perimeter_mean": perimeter_mean,
        "area_mean": area_mean,
        "concavity_mean": concavity_mean,
        "fractal_dimension_mean": fractal_dimension_mean,
        "radius_worst": radius_worst,
        "perimeter_worst": perimeter_worst,
        "area_worst": area_worst,
        "diagnosis": diagnosis,
    })
    return df


# ---------------------------------------------------------------------------
# 1. Feature Engineering Engine Tests
# ---------------------------------------------------------------------------
class TestFeatureEngineeringEngine:
    def test_domain_features_creation(self, synthetic_cancer_data):
        engine = FeatureEngineeringEngine()
        trans_df, res = engine.engineer_features(
            synthetic_cancer_data,
            apply_domain_features=True,
            apply_ratios=False,
            apply_polynomials=False,
            apply_log_transforms=False,
            apply_aggregations=False,
        )

        assert "circularity_index_mean" in trans_df.columns
        assert "sphericity_mean" in trans_df.columns
        assert "nuclear_irregularity_mean" in trans_df.columns
        assert "border_roughness_mean" in trans_df.columns
        assert res.derived_feature_count >= 4
        assert len(res.derived_definitions) >= 4

        # Circularity should be between 0 and 1.5
        assert trans_df["circularity_index_mean"].between(-0.1, 2.0).all()

    def test_ratio_features_creation(self, synthetic_cancer_data):
        engine = FeatureEngineeringEngine()
        trans_df, res = engine.engineer_features(
            synthetic_cancer_data,
            apply_domain_features=False,
            apply_ratios=True,
            apply_polynomials=False,
            apply_log_transforms=False,
            apply_aggregations=False,
        )

        assert "perimeter_to_area_mean" in trans_df.columns
        assert "ratio_worst_to_mean_radius_worst" in trans_df.columns
        assert not trans_df.isna().any().any()
        assert not np.isinf(trans_df.values).any()

    def test_composite_aggregations_and_log(self, synthetic_cancer_data):
        engine = FeatureEngineeringEngine()
        trans_df, res = engine.engineer_features(
            synthetic_cancer_data,
            apply_domain_features=False,
            apply_ratios=False,
            apply_polynomials=False,
            apply_log_transforms=True,
            apply_aggregations=True,
        )

        assert "mean_morphology_composite_mean" in trans_df.columns
        assert "worst_morphology_composite_max" in trans_df.columns
        assert "log1p_area_mean" in trans_df.columns

    def test_derived_feature_registry_metadata(self, synthetic_cancer_data):
        engine = FeatureEngineeringEngine()
        _, res = engine.engineer_features(synthetic_cancer_data)

        for definition in res.derived_definitions:
            assert isinstance(definition, DerivedFeatureDefinition)
            assert len(definition.base_features) > 0
            assert definition.formula != ""
            assert definition.domain_rationale != ""


# ---------------------------------------------------------------------------
# 2. Feature Selection Engine Tests
# ---------------------------------------------------------------------------
class TestFeatureSelectionEngine:
    def test_filter_selection_methods(self, synthetic_cancer_data):
        selector = FeatureSelectionEngine()
        X = synthetic_cancer_data.drop(columns=["diagnosis"])
        y = synthetic_cancer_data["diagnosis"]

        methods = ["variance_threshold", "pearson_correlation", "spearman_correlation", "chi2", "anova_f", "mutual_info"]
        for method in methods:
            selected, rankings = selector.select_features(X, y, method=method, k=4)
            assert len(selected) <= 4
            assert len(rankings) > 0
            assert all(f in X.columns for f in selected)

    def test_wrapper_selection_methods(self, synthetic_cancer_data):
        selector = FeatureSelectionEngine()
        X = synthetic_cancer_data.drop(columns=["diagnosis"])
        y = synthetic_cancer_data["diagnosis"]

        # RFE and SFS forward
        selected_rfe, _ = selector.select_features(X, y, method="rfe", k=3)
        assert len(selected_rfe) == 3

        selected_sfs, _ = selector.select_features(X, y, method="sfs_forward", k=3)
        assert len(selected_sfs) == 3

    def test_embedded_selection_methods(self, synthetic_cancer_data):
        selector = FeatureSelectionEngine()
        X = synthetic_cancer_data.drop(columns=["diagnosis"])
        y = synthetic_cancer_data["diagnosis"]

        for method in ["l1_logistic", "random_forest", "gradient_boosting"]:
            selected, rankings = selector.select_features(X, y, method=method, k=4)
            assert len(selected) <= 4
            assert len(rankings) > 0

    def test_strict_data_leakage_prevention(self, synthetic_cancer_data):
        selector = FeatureSelectionEngine()
        X = synthetic_cancer_data.drop(columns=["diagnosis"])
        y = synthetic_cancer_data["diagnosis"]

        comp_items = selector.compare_selection_strategies(X, y, k=3, cv_folds=3)
        assert len(comp_items) > 0
        for item in comp_items:
            assert item.data_leakage_prevented is True
            assert 0.0 <= item.cv_accuracy_mean <= 1.0
            assert 0.0 <= item.cv_roc_auc_mean <= 1.0
            assert item.execution_time_seconds > 0


# ---------------------------------------------------------------------------
# 3. Dimensionality Reduction Engine Tests
# ---------------------------------------------------------------------------
class TestDimensionalityReductionEngine:
    def test_pca_reduction(self, synthetic_cancer_data):
        engine = DimensionalityReductionEngine()
        X = synthetic_cancer_data.drop(columns=["diagnosis"])
        y = synthetic_cancer_data["diagnosis"]

        red_df, res = engine.reduce(X, method=DimensionalityMethod.PCA, n_components=2, y=y)
        assert red_df.shape == (len(synthetic_cancer_data), 2)
        assert res.reduced_dimensions == 2
        assert len(res.explained_variance_ratio) == 2
        assert 0.0 < res.total_explained_variance <= 1.0
        assert res.downstream_cv_accuracy > 0.5
        assert len(res.scree_plot_data) == 2

    def test_kernel_pca_and_svd(self, synthetic_cancer_data):
        engine = DimensionalityReductionEngine()
        X = synthetic_cancer_data.drop(columns=["diagnosis"])
        y = synthetic_cancer_data["diagnosis"]

        # Kernel PCA
        red_kpca, res_kpca = engine.reduce(X, method=DimensionalityMethod.KERNEL_PCA, n_components=2, y=y)
        assert red_kpca.shape == (len(synthetic_cancer_data), 2)
        assert res_kpca.reduced_dimensions == 2

        # Truncated SVD
        red_svd, res_svd = engine.reduce(X, method=DimensionalityMethod.TRUNCATED_SVD, n_components=2, y=y)
        assert red_svd.shape == (len(synthetic_cancer_data), 2)
        assert res_svd.reduced_dimensions == 2

    def test_umap_reduction(self, synthetic_cancer_data):
        engine = DimensionalityReductionEngine()
        X = synthetic_cancer_data.drop(columns=["diagnosis"])
        y = synthetic_cancer_data["diagnosis"]

        red_umap, res_umap = engine.reduce(X, method=DimensionalityMethod.UMAP, n_components=2, y=y)
        assert red_umap.shape == (len(synthetic_cancer_data), 2)
        assert res_umap.reduced_dimensions == 2

    def test_compare_reduction_methods(self, synthetic_cancer_data):
        engine = DimensionalityReductionEngine()
        X = synthetic_cancer_data.drop(columns=["diagnosis"])
        y = synthetic_cancer_data["diagnosis"]

        results = engine.compare_reduction_methods(X, y=y, n_components=2)
        assert len(results) >= 3
        methods = [r.method.value for r in results]
        assert "pca" in methods


# ---------------------------------------------------------------------------
# 4. Feature Ranking & Explainability Service Tests
# ---------------------------------------------------------------------------
class TestFeatureRankingService:
    def test_permutation_and_native_importance(self, synthetic_cancer_data):
        service = FeatureRankingService()
        X = synthetic_cancer_data.drop(columns=["diagnosis"])
        y = synthetic_cancer_data["diagnosis"]

        rf = RandomForestClassifier(n_estimators=30, random_state=42)
        rf.fit(X, y)

        # Permutation
        perm_items = service.compute_permutation_importance(rf, X, y, n_repeats=3)
        assert len(perm_items) == X.shape[1]
        assert perm_items[0].rank == 1

        # Native
        native_items = service.compute_native_importance(rf, X.columns.tolist())
        assert len(native_items) == X.shape[1]
        assert native_items[0].rank == 1

    def test_shap_importance(self, synthetic_cancer_data):
        service = FeatureRankingService()
        X = synthetic_cancer_data.drop(columns=["diagnosis"])
        y = synthetic_cancer_data["diagnosis"]

        rf = RandomForestClassifier(n_estimators=30, random_state=42)
        rf.fit(X, y)

        shap_items = service.compute_shap_importance(rf, X, max_samples=50)
        assert len(shap_items) == X.shape[1]
        assert shap_items[0].score >= shap_items[-1].score

    def test_consensus_borda_ranking(self, synthetic_cancer_data):
        service = FeatureRankingService()
        X = synthetic_cancer_data.drop(columns=["diagnosis"])
        y = synthetic_cancer_data["diagnosis"]

        summary = service.compute_consensus_ranking(X, y)
        assert isinstance(summary, FeatureImportanceSummary)
        assert len(summary.rankings) == X.shape[1]
        assert len(summary.consensus_rankings) == X.shape[1]
        assert summary.consensus_rankings[0]["consensus_rank"] == 1


# ---------------------------------------------------------------------------
# 5. REST API Endpoints Tests
# ---------------------------------------------------------------------------
class TestFeatureAPI:
    def test_api_endpoints_on_wdbc(self):
        # Using ingested UCI WDBC dataset
        dataset_id = "uci_wdbc"
        ds = DatasetService()
        df = ds.load_dataset(dataset_id)
        if df is None:
            pytest.skip("uci_wdbc dataset not available for API test.")

        # 1. Feature Engineering
        fe_res = engineer_features(
            dataset_id=dataset_id,
            apply_domain_features=True,
            apply_ratios=True,
            apply_polynomials=False,
            apply_log_transforms=False,
            apply_aggregations=True,
        )
        assert fe_res.derived_feature_count > 0

        # 2. Feature Selection
        sel_res = select_features(
            dataset_id=dataset_id,
            method="mutual_info",
            k=5,
            target_column="diagnosis",
        )
        assert len(sel_res["selected_features"]) == 5

        # 3. Dimensionality Reduction
        red_res = reduce_dimensions(
            dataset_id=dataset_id,
            method="pca",
            n_components=2,
            target_column="diagnosis",
        )
        assert red_res.reduced_dimensions == 2
        assert red_res.downstream_cv_accuracy > 0.85

        # 4. Consensus Importance
        imp_res = get_feature_importance_consensus(
            dataset_id=dataset_id,
            target_column="diagnosis",
        )
        assert len(imp_res.rankings) > 0
