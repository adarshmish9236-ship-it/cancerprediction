"""
CANCERWISE AI — Phase 3 Test Suite: Exploratory Data Analysis & Statistical Mining
Tests:
- EDAEngine: dataset overview, univariate distributions, bivariate correlations, multivariate PCA.
- StatisticalMiningEngine: Pearson, Spearman, Kendall, Mann-Whitney U, ANOVA, Kruskal-Wallis, Chi-square, Mutual Information.
- Test Assumptions: Shapiro-Wilk normality, Levene homogeneity of variance.
- Multiple-Testing Correction: Benjamini-Hochberg False Discovery Rate (FDR).
- Cancer Morphometrics (WDBC): Benign vs Malignant effect sizes, Cohen's d, collinear clusters, non-causal disclaimer.
- VisualizationService: Plotly figure generation for histograms, violins, boxplots, scatter, heatmaps, PCA, clusters.
- EDAReporter: Automated HTML report synthesis and persistence.
- FastAPI EDA Endpoints: overview, univariate, bivariate, multivariate, tests, cancer-findings, generate-report.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data_mining.eda.eda_engine import EDAEngine
from data_mining.eda.statistical_tests import StatisticalMiningEngine
from data_mining.eda.visualization_service import VisualizationService
from data_mining.eda.eda_reporter import EDAReporter
from backend.api.eda import (
    get_dataset_overview,
    get_feature_univariate,
    get_feature_bivariate,
    get_multivariate_summary,
    run_statistical_hypothesis_tests,
    get_cancer_morphometrics_findings,
    generate_eda_report,
)


@pytest.fixture
def sample_cancer_df():
    np.random.seed(42)
    n = 100
    benign_rad = np.random.normal(loc=12.0, scale=1.5, size=60)
    malig_rad = np.random.normal(loc=17.0, scale=2.0, size=40)
    radius = np.concatenate([benign_rad, malig_rad])
    perimeter = radius * 6.28 + np.random.normal(0, 0.5, size=n)
    area = np.pi * (radius ** 2)
    diagnosis = ["Benign"] * 60 + ["Malignant"] * 40
    stage = ["Early"] * 50 + ["Advanced"] * 50

    return pd.DataFrame({
        "id": range(1, n + 1),
        "radius_mean": radius,
        "perimeter_mean": perimeter,
        "area_mean": area,
        "diagnosis": diagnosis,
        "stage": stage,
    })


# ===========================================================================
# 1. Dataset Overview Tests
# ===========================================================================
class TestDatasetOverview:

    def test_overview_metrics(self, sample_cancer_df):
        engine = EDAEngine()
        overview = engine.get_overview(sample_cancer_df, target_column="diagnosis")

        assert overview.total_records == 100
        assert overview.total_features == 6
        assert overview.numerical_features_count >= 3
        assert overview.categorical_features_count >= 2
        assert overview.missing_value_percentage == 0.0
        assert overview.duplicate_records_count == 0
        assert overview.target_distribution["Benign"] == 60
        assert overview.target_distribution["Malignant"] == 40
        assert overview.memory_utilization_bytes > 0


# ===========================================================================
# 2. Univariate Distribution Tests
# ===========================================================================
class TestUnivariateAnalysis:

    def test_numeric_univariate_stats(self, sample_cancer_df):
        engine = EDAEngine()
        stats = engine.get_univariate_profile(sample_cancer_df, "radius_mean")

        assert stats.feature_name == "radius_mean"
        assert stats.mean is not None and stats.mean > 10.0
        assert stats.median is not None
        assert stats.iqr is not None
        assert stats.normality_test_name in ("Shapiro-Wilk", "D'Agostino-Pearson")
        assert stats.normality_p_value is not None

    def test_categorical_univariate_stats(self, sample_cancer_df):
        engine = EDAEngine()
        stats = engine.get_univariate_profile(sample_cancer_df, "stage")

        assert stats.feature_name == "stage"
        assert "Early" in stats.top_categories
        assert stats.top_categories["Early"] == 50


# ===========================================================================
# 3. Bivariate Relationship Tests
# ===========================================================================
class TestBivariateAnalysis:

    def test_correlations_and_strength(self, sample_cancer_df):
        engine = EDAEngine()
        biv = engine.get_bivariate_profile(sample_cancer_df, "radius_mean", "perimeter_mean", target_column="diagnosis")

        assert biv.pearson_r is not None and biv.pearson_r > 0.95
        assert biv.spearman_rho is not None and biv.spearman_rho > 0.95
        assert biv.kendall_tau is not None and biv.kendall_tau > 0.80
        assert biv.relationship_strength == "Very Strong"
        assert "Benign" in biv.group_statistics


# ===========================================================================
# 4. Multivariate Analysis Tests
# ===========================================================================
class TestMultivariateAnalysis:

    def test_pca_and_correlation_matrix(self, sample_cancer_df):
        engine = EDAEngine()
        multi = engine.get_multivariate_profile(sample_cancer_df, target_column="diagnosis")

        assert len(multi.features_included) >= 3
        assert len(multi.pca_explained_variance_ratio) >= 2
        assert multi.pca_total_variance_explained > 0.80
        assert len(multi.top_correlated_pairs) > 0


# ===========================================================================
# 5. Statistical Mining & Multiple Testing Tests
# ===========================================================================
class TestStatisticalMining:

    def test_two_group_comparison_and_assumptions(self, sample_cancer_df):
        engine = StatisticalMiningEngine(alpha=0.05)
        res = engine.compare_two_groups(
            series=sample_cancer_df["radius_mean"],
            groups=sample_cancer_df["diagnosis"],
            feature_name="radius_mean",
        )

        assert res.test_name == "Mann-Whitney U"
        assert res.is_significant is True
        assert res.p_value is not None and res.p_value < 1e-4
        assert res.effect_size is not None  # Rank-biserial correlation
        assert len(res.assumptions_checked) >= 2

    def test_multi_group_anova_and_kruskal(self, sample_cancer_df):
        engine = StatisticalMiningEngine(alpha=0.05)
        res = engine.compare_multiple_groups(
            series=sample_cancer_df["radius_mean"],
            groups=sample_cancer_df["diagnosis"],
            feature_name="radius_mean",
        )

        assert res.statistic > 0
        assert res.is_significant is True

    def test_chi_square_categorical(self, sample_cancer_df):
        engine = StatisticalMiningEngine()
        res = engine.test_categorical_association(
            cat1=sample_cancer_df["diagnosis"],
            cat2=sample_cancer_df["stage"],
        )
        assert res.test_name == "Chi-Square Test"
        assert res.effect_size_name == "Cramér's V"

    def test_mutual_information(self, sample_cancer_df):
        mi_scores = StatisticalMiningEngine.calculate_mutual_information(
            sample_cancer_df,
            target_column="diagnosis",
        )
        assert "radius_mean" in mi_scores
        assert mi_scores["radius_mean"] > 0.10

    def test_multiple_testing_fdr_correction(self, sample_cancer_df):
        engine = EDAEngine()
        results = engine.run_statistical_tests(sample_cancer_df, target_column="diagnosis")

        assert len(results) >= 3
        for r in results:
            assert r.p_value_corrected is not None
            assert r.correction_method == "FDR_BH"
            assert r.p_value_corrected >= r.p_value - 1e-18  # Adjusted p-values cannot be smaller


# ===========================================================================
# 6. WDBC Cytology Morphometrics Tests
# ===========================================================================
class TestWDBCCytologyFindings:

    def test_wdbc_findings(self, sample_cancer_df):
        engine = EDAEngine()
        findings = engine.analyze_cancer_wdbc(sample_cancer_df, target_column="diagnosis")

        assert findings.total_samples == 100
        assert findings.benign_count == 60
        assert findings.malignant_count == 40
        assert len(findings.top_differentiating_features) >= 3
        # Malignant mean should be higher than benign
        assert findings.top_differentiating_features[0].malignant_mean > findings.top_differentiating_features[0].benign_mean
        assert findings.top_differentiating_features[0].cohens_d > 1.0
        assert "DO NOT establish biological etiology" in findings.non_causal_disclaimer


# ===========================================================================
# 7. Visualization Service Tests
# ===========================================================================
class TestVisualizationService:

    def test_univariate_figures(self, sample_cancer_df):
        fig_hist = VisualizationService.create_histogram(sample_cancer_df, "radius_mean", target_column="diagnosis")
        assert isinstance(fig_hist, go.Figure)

        fig_violin = VisualizationService.create_violin_plot(sample_cancer_df, "radius_mean", target_column="diagnosis")
        assert isinstance(fig_violin, go.Figure)

        fig_box = VisualizationService.create_box_plot(sample_cancer_df, "radius_mean", target_column="diagnosis")
        assert isinstance(fig_box, go.Figure)

    def test_bivariate_and_multivariate_figures(self, sample_cancer_df):
        fig_scatter = VisualizationService.create_scatter_plot(sample_cancer_df, "radius_mean", "area_mean", target_column="diagnosis")
        assert isinstance(fig_scatter, go.Figure)

        fig_heat = VisualizationService.create_correlation_heatmap(sample_cancer_df, columns=["radius_mean", "area_mean"])
        assert isinstance(fig_heat, go.Figure)

        fig_pca, exp_var = VisualizationService.create_pca_projection(sample_cancer_df, target_column="diagnosis", n_components=2)
        assert isinstance(fig_pca, go.Figure)
        assert len(exp_var) == 2


# ===========================================================================
# 8. Automated EDA Report Generation Tests
# ===========================================================================
class TestEDAReporter:

    def test_html_report_generation(self, sample_cancer_df, tmp_path: Path):
        reporter = EDAReporter(output_dir=tmp_path)
        report_path, _ = reporter.generate_html_report(
            sample_cancer_df,
            dataset_id="test_dataset",
            dataset_name="Test Cancer Study",
            target_column="diagnosis",
        )

        p = Path(report_path)
        assert p.exists()
        assert p.stat().st_size > 1000  # Report contains rich HTML content
        content = p.read_text(encoding="utf-8")
        assert "CANCERWISE AI — Comprehensive EDA & Statistical Mining Report" in content
        assert "Top Diagnostic Differentiating Features" in content
        assert "Clinical Governance Notice" in content


# ===========================================================================
# 9. FastAPI EDA Endpoints Tests
# ===========================================================================
class TestEDAEndpoints:

    def test_api_overview(self):
        ov = get_dataset_overview("uci_wdbc", target_column="diagnosis")
        assert ov.total_records == 569
        assert ov.total_features == 32
        assert ov.target_distribution["Benign"] == 357

    def test_api_univariate(self):
        uni = get_feature_univariate("uci_wdbc", "radius_mean")
        assert uni.mean is not None and uni.mean > 10.0

    def test_api_bivariate(self):
        biv = get_feature_bivariate("uci_wdbc", feature_x="radius_mean", feature_y="perimeter_mean", target_column="diagnosis")
        assert biv.pearson_r > 0.95

    def test_api_multivariate(self):
        multi = get_multivariate_summary("uci_wdbc", target_column="diagnosis")
        assert len(multi.pca_explained_variance_ratio) >= 2

    def test_api_hypothesis_tests(self):
        tests = run_statistical_hypothesis_tests("uci_wdbc", target_column="diagnosis")
        assert len(tests) >= 30
        assert any(t.is_significant for t in tests)

    def test_api_cancer_findings(self):
        findings = get_cancer_morphometrics_findings("uci_wdbc", target_column="diagnosis")
        assert findings.total_samples == 569
        assert len(findings.top_differentiating_features) >= 10
        assert "DO NOT establish biological etiology" in findings.non_causal_disclaimer

    def test_api_generate_report(self):
        res = generate_eda_report("uci_wdbc", target_column="diagnosis")
        assert res["status"] == "SUCCESS"
        assert os.path.exists(res["report_path"])
