"""
CANCERWISE AI — Phase 2 Test Suite: Advanced Data Wrangling and Preprocessing Engine
Tests:
- DataQualityAnalyzer: missingness, duplicates, constant columns, near-zero variance,
  outliers, high-cardinality, inconsistent dtypes, potential leakage, quality score, non-clinical disclaimer.
- MissingValueHandler: mean, median, mode, constant, KNN, iterative, indicators, strategy comparison, target protection.
- DuplicateManager: exact duplicates, duplicate identifiers, similar records, audit report, clinical preservation rule.
- OutlierDetector: IQR, Z-score, Modified Z-score, Isolation Forest, LOF, biological extremes disclaimer, winsorization.
- DataTransformer: 9 scalers/encoders, mathematical safety, parameter preservation, inverse_transform.
- PreprocessingPipeline: ColumnTransformer, JSON/YAML recipe export/import, Polars interoperability, CancerwisePreprocessor.
- TransformationLogger: operation tracking, JSONL persistence, history retrieval.
- DataValidationEngine: Pandera schema validation, missingness, constraints, target validity.
- FastAPI Wrangling Endpoints: quality-profile, compare-imputation, audit-duplicates, detect-outliers, validate-schema, history.
"""

from __future__ import annotations

import io
import json
import pytest
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

from backend.schemas.preprocessing import (
    ImputationStrategy,
    OutlierMethod,
    ScalingStrategy,
    EncodingStrategy,
)
from preprocessing.quality_analyzer import DataQualityAnalyzer
from preprocessing.missing_values import MissingValueHandler
from preprocessing.duplicate_manager import DuplicateManager
from preprocessing.outlier_detector import OutlierDetector
from preprocessing.transformers import DataTransformer, Log1pTransformer
from preprocessing.pipeline import PreprocessingPipeline, CancerwisePreprocessor
from preprocessing.transformation_logger import TransformationLogger
from preprocessing.data_validator import DataValidationEngine
from backend.api.wrangling import (
    get_quality_profile,
    compare_imputation_strategies,
    audit_dataset_duplicates,
    detect_dataset_outliers,
    validate_dataset_schema,
    get_transformation_history,
)


# ===========================================================================
# 1. DataQualityAnalyzer Tests
# ===========================================================================
class TestDataQualityAnalyzer:

    def test_clean_dataset_quality_score(self):
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_b": [10.0, 20.0, 30.0, 40.0, 50.0],
            "diagnosis": ["Benign", "Malignant", "Benign", "Malignant", "Benign"],
        })
        analyzer = DataQualityAnalyzer(target_column="diagnosis")
        report = analyzer.analyze(df, dataset_name="clean_test")

        assert report.overall_quality_score >= 90.0
        assert report.duplicate_rows_count == 0
        assert report.duplicate_identifiers_count == 0
        assert len(report.constant_columns) == 0
        assert len(report.potential_leakage_columns) == 0
        assert "NOT a clinical quality" in report.clinical_disclaimer

    def test_detects_missingness_and_penalizes_score(self):
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "feat_a": [1.0, np.nan, np.nan, np.nan, 5.0],  # 60% missing -> critical
            "diagnosis": ["Benign", "Malignant", "Benign", "Malignant", "Benign"],
        })
        analyzer = DataQualityAnalyzer(target_column="diagnosis")
        report = analyzer.analyze(df)

        assert report.overall_quality_score < 90.0
        assert report.missing_summary["total_missing_cells"] == 3
        assert any(i.issue_type == "missing_values" and i.severity == "critical" for i in report.issues)

    def test_detects_duplicate_rows_and_duplicate_ids(self):
        df = pd.DataFrame({
            "id": [101, 101, 103],  # Duplicate ID
            "val": [5.0, 5.0, 6.0],  # Rows 0 & 1 are exact duplicates
        })
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze(df)

        assert report.duplicate_rows_count >= 1
        assert report.duplicate_identifiers_count >= 1
        assert any(i.issue_type == "duplicate_identifiers" for i in report.issues)

    def test_detects_constant_and_near_zero_variance(self):
        df = pd.DataFrame({
            "const_col": [42.0, 42.0, 42.0, 42.0],
            "nzv_col": [1.0, 1.0, 1.0, 1.0000001],
            "normal_col": [1.0, 2.0, 3.0, 4.0],
        })
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze(df)

        assert "const_col" in report.constant_columns
        assert "nzv_col" in report.near_zero_variance_columns

    def test_detects_potential_data_leakage(self):
        df = pd.DataFrame({
            "feat_normal": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_leak": ["Benign", "Malignant", "Benign", "Malignant", "Benign"],  # Identical to target
            "diagnosis": ["Benign", "Malignant", "Benign", "Malignant", "Benign"],
        })
        analyzer = DataQualityAnalyzer(target_column="diagnosis")
        report = analyzer.analyze(df)

        assert "feat_leak" in report.potential_leakage_columns
        assert report.overall_quality_score <= 75.0


# ===========================================================================
# 2. MissingValueHandler Tests
# ===========================================================================
class TestMissingValueHandler:

    def test_median_imputation(self):
        df = pd.DataFrame({
            "f1": [10.0, 20.0, np.nan, 40.0],  # Median of 10, 20, 40 is 20
            "f2": [1.0, 2.0, 3.0, 4.0],
        })
        handler = MissingValueHandler(strategy=ImputationStrategy.MEDIAN)
        df_imputed = handler.fit_transform(df)

        assert df_imputed["f1"].isna().sum() == 0
        assert df_imputed.loc[2, "f1"] == 20.0

    def test_mean_and_constant_imputation(self):
        df = pd.DataFrame({
            "val": [10.0, 20.0, 30.0, np.nan],
        })
        # Mean
        h_mean = MissingValueHandler(strategy=ImputationStrategy.MEAN)
        df_mean = h_mean.fit_transform(df)
        assert df_mean.loc[3, "val"] == 20.0

        # Constant
        h_const = MissingValueHandler(strategy=ImputationStrategy.CONSTANT, fill_value=-999.0)
        df_const = h_const.fit_transform(df)
        assert df_const.loc[3, "val"] == -999.0

    def test_knn_imputation(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 10.0, 11.0],
            "y": [1.0, np.nan, 10.0, 11.0],
        })
        handler = MissingValueHandler(strategy=ImputationStrategy.KNN, columns=["x", "y"], knn_neighbors=1)
        df_knn = handler.fit_transform(df)
        assert df_knn["y"].isna().sum() == 0
        assert abs(df_knn.loc[1, "y"] - 1.0) < 1.0

    def test_missingness_indicator_generation(self):
        df = pd.DataFrame({
            "val": [1.0, np.nan, 3.0],
        })
        handler = MissingValueHandler(strategy=ImputationStrategy.MEDIAN, add_indicator=True)
        df_out = handler.fit_transform(df)

        assert "val_isna" in df_out.columns
        assert df_out["val_isna"].tolist() == [0, 1, 0]

    def test_target_imputation_forbidden(self):
        df = pd.DataFrame({
            "feature": [1.0, 2.0, 3.0],
            "diagnosis": ["Benign", np.nan, "Malignant"],
        })
        # If user explicitly requests imputing target, must raise ValueError
        handler = MissingValueHandler(
            strategy=ImputationStrategy.MODE,
            columns=["diagnosis"],
            target_column="diagnosis",
        )
        with pytest.raises(ValueError, match="cannot be imputed"):
            handler.fit(df)

    def test_compare_strategies(self):
        df = pd.DataFrame({
            "f1": [1.0, 2.0, np.nan, 4.0, 5.0],
            "diagnosis": ["Benign", "Malignant", "Benign", "Malignant", "Benign"],
        })
        results = MissingValueHandler.compare_strategies(df, target_column="diagnosis")
        assert len(results) >= 2
        strategies = {r.strategy for r in results}
        assert "mean" in strategies
        assert "median" in strategies


# ===========================================================================
# 3. DuplicateManager Tests
# ===========================================================================
class TestDuplicateManager:

    def test_audit_duplicates(self):
        df = pd.DataFrame({
            "patient_id": ["P1", "P2", "P2", "P3"],
            "feat1": [1.0, 2.0, 2.0, 3.0],
            "feat2": [4.0, 5.0, 5.0, 6.0],
        })
        manager = DuplicateManager()
        audit = manager.audit_duplicates(df, id_column="patient_id")

        assert audit.total_records == 4
        assert audit.exact_duplicates_count == 2  # rows 1 and 2 are exact dups
        assert audit.duplicate_identifier_count == 2  # P2 appears twice
        assert "distinct clinical patients" in audit.clinical_preservation_warning

    def test_protection_against_deleting_distinct_patients(self):
        df = pd.DataFrame({
            "patient_id": ["P101", "P102"],  # Two distinct patients
            "feat": [5.0, 5.0],  # Same measurements
        })
        manager = DuplicateManager()
        # Attempting to drop duplicates based only on features without confirmation must fail
        with pytest.raises(PermissionError, match="Refusing to remove duplicates"):
            manager.remove_duplicates(df, subset=["feat"], id_column="patient_id", confirm_clinical_purging=False)

        # With confirmation, it succeeds
        df_cleaned, rem = manager.remove_duplicates(df, subset=["feat"], id_column="patient_id", confirm_clinical_purging=True)
        assert rem == 1
        assert len(df_cleaned) == 1


# ===========================================================================
# 4. OutlierDetector Tests
# ===========================================================================
class TestOutlierDetector:

    @pytest.fixture
    def outlier_df(self):
        np.random.seed(42)
        normal_vals = np.random.normal(loc=10.0, scale=1.0, size=100)
        outliers = [100.0, 150.0]  # Obvious extreme values
        all_vals = np.concatenate([normal_vals, outliers])
        return pd.DataFrame({"marker": all_vals, "diagnosis": ["Benign"] * 102})

    def test_iqr_detection(self, outlier_df):
        detector = OutlierDetector(method=OutlierMethod.IQR, iqr_multiplier=1.5)
        report = detector.detect(outlier_df, target_column="diagnosis")

        assert report.total_outliers >= 2
        assert "marker" in report.column_outlier_counts
        assert "Do not purge without clinical review" in report.biological_extreme_disclaimer

    def test_z_score_detection(self, outlier_df):
        detector = OutlierDetector(method=OutlierMethod.Z_SCORE, z_threshold=3.0)
        report = detector.detect(outlier_df, target_column="diagnosis")
        assert report.total_outliers >= 2

    def test_modified_z_score_detection(self, outlier_df):
        detector = OutlierDetector(method=OutlierMethod.MODIFIED_Z_SCORE, modified_z_threshold=3.5)
        report = detector.detect(outlier_df, target_column="diagnosis")
        assert report.total_outliers >= 2

    def test_isolation_forest_and_lof(self, outlier_df):
        det_if = OutlierDetector(method=OutlierMethod.ISOLATION_FOREST, contamination=0.05)
        rep_if = det_if.detect(outlier_df, target_column="diagnosis")
        assert rep_if.total_outliers > 0

        det_lof = OutlierDetector(method=OutlierMethod.LOCAL_OUTLIER_FACTOR, contamination=0.05)
        rep_lof = det_lof.detect(outlier_df, target_column="diagnosis")
        assert rep_lof.total_outliers > 0

    def test_winsorization_clips_bounds(self, outlier_df):
        detector = OutlierDetector()
        df_clipped = detector.clip_outliers(outlier_df, lower_percentile=0.01, upper_percentile=0.95, target_column="diagnosis")

        assert df_clipped["marker"].max() < 100.0
        assert len(df_clipped) == len(outlier_df)  # Zero samples deleted

    def test_refusal_to_delete_outliers_without_clinical_override(self, outlier_df):
        detector = OutlierDetector()
        rep = detector.detect(outlier_df, target_column="diagnosis")
        with pytest.raises(PermissionError, match="Refusing to delete outliers without clinical confirmation"):
            detector.remove_outliers(outlier_df, rep.outlier_indices, confirm_biological_override=False)


# ===========================================================================
# 5. DataTransformer Tests
# ===========================================================================
class TestDataTransformer:

    @pytest.fixture
    def numeric_df(self):
        return pd.DataFrame({
            "a": [10.0, 20.0, 30.0, 40.0, 50.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

    def test_standard_scaler(self, numeric_df):
        t = DataTransformer(strategy=ScalingStrategy.STANDARD)
        df_scaled = t.fit_transform(numeric_df)

        assert abs(df_scaled["a"].mean()) < 1e-6
        assert abs(df_scaled["a"].std(ddof=0) - 1.0) < 1e-6
        assert "mean" in t.parameters

        # Invert
        df_recovered = t.inverse_transform(df_scaled)
        assert np.allclose(df_recovered["a"], numeric_df["a"])

    def test_minmax_scaler(self, numeric_df):
        t = DataTransformer(strategy=ScalingStrategy.MINMAX, feature_range=(0, 1))
        df_scaled = t.fit_transform(numeric_df)

        assert df_scaled["a"].min() == 0.0
        assert df_scaled["a"].max() == 1.0

        # Invert
        df_recovered = t.inverse_transform(df_scaled)
        assert np.allclose(df_recovered["a"], numeric_df["a"])

    def test_power_and_quantile_transformers(self, numeric_df):
        # Power Yeo-Johnson
        t_pow = DataTransformer(strategy=ScalingStrategy.POWER_YEO_JOHNSON)
        df_pow = t_pow.fit_transform(numeric_df)
        assert len(df_pow) == 5

        # Quantile Uniform
        t_q = DataTransformer(strategy=ScalingStrategy.QUANTILE_UNIFORM, n_quantiles=5)
        df_q = t_q.fit_transform(numeric_df)
        assert df_q["a"].max() <= 1.0

    def test_log1p_transformer_safety(self):
        # Test safety on negative numbers
        df_neg = pd.DataFrame({"vals": [-5.0, -2.0, 0.0, 3.0, 10.0]})
        t_log = DataTransformer(strategy=ScalingStrategy.LOG1P)
        df_log = t_log.fit_transform(df_neg)

        assert not df_log["vals"].isna().any()
        assert not np.isinf(df_log["vals"]).any()

        # Invert
        df_rec = t_log.inverse_transform(df_log)
        assert np.allclose(df_rec["vals"], df_neg["vals"], atol=1e-5)

    def test_onehot_and_ordinal_encoding(self):
        df_cat = pd.DataFrame({
            "stage": ["I", "II", "III", "I"],
            "grade": ["Low", "High", "Low", "Medium"],
        })
        # OneHot
        t_oh = DataTransformer(strategy=EncodingStrategy.ONEHOT)
        df_oh = t_oh.fit_transform(df_cat)
        assert any(c.startswith("stage_") for c in df_oh.columns)
        assert len(df_oh) == 4

        # Ordinal
        t_ord = DataTransformer(strategy=EncodingStrategy.ORDINAL)
        df_ord = t_ord.fit_transform(df_cat)
        assert np.issubdtype(df_ord["stage"].dtype, np.number)


# ===========================================================================
# 6. PreprocessingPipeline & Backwards Compatibility
# ===========================================================================
class TestPreprocessingPipeline:

    @pytest.fixture
    def mixed_df(self):
        return pd.DataFrame({
            "num1": [1.0, 2.0, np.nan, 4.0, 5.0],
            "num2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "cat1": ["Low", "High", "Medium", "Low", "High"],
            "diagnosis": ["Benign", "Malignant", "Benign", "Malignant", "Benign"],
        })

    def test_pipeline_fit_transform_and_inversion(self, mixed_df):
        pipeline = PreprocessingPipeline(
            numeric_features=["num1", "num2"],
            categorical_features=["cat1"],
            imputation_strategy="median",
            scaling_strategy="standard",
            encoding_strategy="ordinal",
            target_column="diagnosis",
        )
        df_trans = pipeline.fit_transform(mixed_df)

        assert df_trans["num1"].isna().sum() == 0
        assert "diagnosis" in df_trans.columns  # Target preserved untouched

        # Invert
        df_inv = pipeline.inverse_transform(df_trans)
        assert np.allclose(df_inv["num2"], mixed_df["num2"])

    def test_recipe_export_and_import(self, mixed_df, tmp_path: Path):
        pipeline = PreprocessingPipeline(
            numeric_features=["num1", "num2"],
            categorical_features=["cat1"],
            imputation_strategy="median",
            scaling_strategy="standard",
            encoding_strategy="onehot",
            target_column="diagnosis",
        )
        pipeline.fit(mixed_df)

        recipe_path = tmp_path / "recipe.yaml"
        pipeline.save_recipe(recipe_path)
        assert recipe_path.exists()

        # Load recipe
        loaded_pipeline = PreprocessingPipeline.load_recipe(recipe_path)
        assert loaded_pipeline.scaling_strategy == "standard"
        assert loaded_pipeline.target_column == "diagnosis"

    def test_polars_interoperability(self, mixed_df):
        pl_df = pl.from_pandas(mixed_df)
        pipeline = PreprocessingPipeline(
            numeric_features=["num1", "num2"],
            target_column="diagnosis",
        )
        df_out = pipeline.fit_transform(pl_df)
        assert isinstance(df_out, pd.DataFrame)
        assert df_out["num1"].isna().sum() == 0

    def test_phase0_preprocessor_backwards_compatibility(self, mixed_df):
        preprocessor = CancerwisePreprocessor(
            numeric_features=["num1", "num2"],
            categorical_features=["cat1"],
            missing_strategy="median",
            scaling_strategy="standard",
        )
        arr = preprocessor.fit_transform(mixed_df)
        assert isinstance(arr, np.ndarray)
        assert arr.shape[0] == 5


# ===========================================================================
# 7. TransformationLogger Tests
# ===========================================================================
class TestTransformationLogger:

    def test_logging_and_retrieval(self, tmp_path: Path):
        log_file = tmp_path / "test_transformations.jsonl"
        t_logger = TransformationLogger(log_path=log_file)

        entry = t_logger.log_operation(
            operation_name="StandardScaler",
            parameters={"mean": [1.0], "scale": [2.0]},
            input_shape=[569, 32],
            output_shape=[569, 32],
            records_affected=569,
            columns_affected=["radius_mean"],
        )

        assert entry.operation_id is not None
        assert entry.execution_status == "SUCCESS"
        assert log_file.exists()

        history = t_logger.get_history()
        assert len(history) == 1
        assert history[0].operation_name == "StandardScaler"


# ===========================================================================
# 8. DataValidationEngine Tests
# ===========================================================================
class TestDataValidationEngine:

    def test_valid_dataset(self):
        df = pd.DataFrame({
            "diagnosis": ["Benign", "Malignant"] * 10,
            "radius_mean": [14.0] * 20,
        })
        validator = DataValidationEngine(
            target_column="diagnosis",
            min_records=10,
            non_negative_columns=["radius_mean"],
        )
        rep = validator.validate(df)
        assert rep.is_valid is True
        assert len(rep.errors) == 0

    def test_rejects_missing_target_and_negative_dimensions(self):
        df = pd.DataFrame({
            "diagnosis": ["Benign", np.nan] * 5,  # Missing target labels!
            "radius_mean": [-5.0] * 10,           # Negative radius!
        })
        validator = DataValidationEngine(
            target_column="diagnosis",
            non_negative_columns=["radius_mean"],
        )
        rep = validator.validate(df)
        assert rep.is_valid is False
        assert any("Target column 'diagnosis' contains" in e for e in rep.errors)
        assert any("Numerical constraint violated" in e for e in rep.errors)


# ===========================================================================
# 9. FastAPI Wrangling Endpoints Tests
# ===========================================================================
class TestWranglingAPI:

    def test_api_quality_profile(self):
        report = get_quality_profile("uci_wdbc", target_column="diagnosis")
        assert report.total_records == 569
        assert report.total_columns == 32
        assert report.overall_quality_score >= 90.0
        assert "NOT a clinical quality" in report.clinical_disclaimer

    def test_api_audit_duplicates(self):
        audit = audit_dataset_duplicates("uci_wdbc")
        assert audit.total_records == 569
        assert audit.exact_duplicates_count == 0

    def test_api_detect_outliers(self):
        report = detect_dataset_outliers("uci_wdbc", method=OutlierMethod.IQR)
        assert report.total_outliers > 0
        assert "Do not purge without clinical review" in report.biological_extreme_disclaimer

    def test_api_validate_schema(self):
        rep = validate_dataset_schema("uci_wdbc", target_column="diagnosis")
        assert rep.is_valid is True

    def test_api_transformation_history(self):
        history = get_transformation_history(limit=10)
        assert isinstance(history, list)
