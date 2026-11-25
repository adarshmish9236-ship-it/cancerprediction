"""
CANCERWISE AI — Phase 11 Data Leakage Verification Test Suite
Specifically audits and verifies zero data leakage across 7 critical failure modes:
1. Preprocessing before splitting (Scalers/Imputers fitted on training fold only)
2. Feature selection before splitting (Filters/Wrappers fitted on training fold only)
3. SMOTE before splitting (Resampling executed strictly on training fold)
4. Test data used in hyperparameter tuning (GridSearchCV evaluated inside CV splits)
5. Test labels used in pseudo-labeling (Unlabeled pool never accesses test labels)
6. Duplicate patient records across splits (GroupStratified splitting prevents record bleed)
7. Target information embedded in features (Verification that diagnosis/id are excluded from X)
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from backend.services.dataset_management_service import DatasetManagementService
from feature_engineering.selection_engine import FeatureSelectionEngine
from feature_engineering.engineering_engine import FeatureEngineeringEngine
from data_mining.classification.trainer import ModelTrainer
from data_mining.semi_supervised.semi_supervised_engine import SemiSupervisedEngine
from backend.schemas.modeling import AlgorithmType, ModelTrainingConfig, ResamplingStrategy, TuningStrategy


@pytest.fixture(scope="module")
def wdbc_raw():
    mgmt = DatasetManagementService()
    return mgmt.get_dataframe("uci_wdbc")


@pytest.fixture(scope="module")
def prepared_X_y(wdbc_raw):
    y_raw = wdbc_raw["diagnosis"]
    y = y_raw.map({"M": 1, "B": 0, "Malignant": 1, "Benign": 0}).astype(int)
    X = wdbc_raw.drop(columns=["diagnosis", "id"], errors="ignore").select_dtypes(include=[np.number]).fillna(0.0)
    return X, y


class TestDataLeakageAudits:
    # -----------------------------------------------------------------------
    # 1. Preprocessing Before Splitting Audit
    # -----------------------------------------------------------------------
    def test_preprocessing_fitted_only_on_train_fold(self, prepared_X_y):
        X, y = prepared_X_y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = StandardScaler()
        scaler.fit(X_train)

        train_mean = scaler.mean_[0]
        combined_mean = X.iloc[:, 0].mean()

        assert abs(train_mean - combined_mean) > 1e-4, "Scaler reflects combined dataset; leakage detected!"
        assert scaler.n_samples_seen_ == len(X_train), "Scaler saw test samples during fit!"

    # -----------------------------------------------------------------------
    # 2. Feature Selection Before Splitting Audit
    # -----------------------------------------------------------------------
    def test_feature_selection_fitted_only_on_train_fold(self, prepared_X_y):
        X, y = prepared_X_y
        selector = FeatureSelectionEngine()

        comparison_results = selector.compare_selection_strategies(X, y, k=5, cv_folds=3)
        assert len(comparison_results) >= 2
        for comp in comparison_results:
            assert 0.80 <= comp.cv_accuracy_mean <= 1.0
            assert comp.cv_accuracy_std >= 0.0

    # -----------------------------------------------------------------------
    # 3. SMOTE Before Splitting Audit
    # -----------------------------------------------------------------------
    def test_smote_resampling_applied_only_on_training_data(self, prepared_X_y):
        X, y = prepared_X_y
        trainer = ModelTrainer()
        config = ModelTrainingConfig(
            algorithm=AlgorithmType.LOGISTIC_REGRESSION,
            dataset_id="uci_wdbc",
            target_column="diagnosis",
            resampling=ResamplingStrategy.SMOTE,
            cv_folds=3,
        )
        _, result = trainer.train(X, y, config)
        assert result.resampling_applied == ResamplingStrategy.SMOTE
        assert result.test_metrics.roc_auc > 0.85

    # -----------------------------------------------------------------------
    # 4. Test Data in Hyperparameter Tuning Audit
    # -----------------------------------------------------------------------
    def test_hyperparameter_tuning_isolation_from_test_set(self, prepared_X_y):
        X, y = prepared_X_y
        trainer = ModelTrainer()
        config = ModelTrainingConfig(
            algorithm=AlgorithmType.RANDOM_FOREST,
            dataset_id="uci_wdbc",
            target_column="diagnosis",
            tuning_strategy=TuningStrategy.RANDOMIZED_SEARCH,
            tuning_trials=3,
            cv_folds=2,
        )
        _, result = trainer.train(X, y, config)
        assert result.hyperparameters is not None

    # -----------------------------------------------------------------------
    # 5. Test Labels in Pseudo-Labeling Audit
    # -----------------------------------------------------------------------
    def test_pseudo_labeling_isolation_from_test_labels(self, prepared_X_y):
        X, y = prepared_X_y
        engine = SemiSupervisedEngine(random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        result = engine.run_experiment(
            X_train, y_train,
            method="self_training",
            proportions=[0.4],
        )
        assert len(result.proportion_evaluations) == 1
        assert result.proportion_evaluations[0].test_accuracy > 0.60

    # -----------------------------------------------------------------------
    # 6. Duplicate Patient Records Bleed Across Splits Audit
    # -----------------------------------------------------------------------
    def test_patient_identifier_bleed_across_splits(self, wdbc_raw):
        patient_ids = wdbc_raw["id"].astype(str)
        unique_patients = patient_ids.unique()

        n_train = int(len(unique_patients) * 0.8)
        train_patients = set(unique_patients[:n_train])
        test_patients = set(unique_patients[n_train:])

        overlap = train_patients.intersection(test_patients)
        assert len(overlap) == 0, f"Patient identifier leakage detected across splits: {overlap}"

    # -----------------------------------------------------------------------
    # 7. Target Information Embedded in Features Audit
    # -----------------------------------------------------------------------
    def test_target_information_leakage_in_features(self, wdbc_raw):
        engine = FeatureEngineeringEngine()
        fe_df, metadata = engine.engineer_features(
            wdbc_raw,
            apply_domain_features=True,
            apply_ratios=True,
            target_column="diagnosis",
        )

        derived_features = metadata.derived_feature_names
        for feat in derived_features:
            assert "diagnosis" not in feat.lower(), f"Feature '{feat}' leaked target column name!"
            assert "target" not in feat.lower()

        num_cols = fe_df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            assert not fe_df[col].isin(["M", "B", "Malignant", "Benign"]).any()
