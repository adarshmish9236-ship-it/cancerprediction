"""
CANCERWISE AI — Research Data Factory
Provides genuine and research-grade synthetic multi-modal cancer datasets
for advanced data mining experiments with strict provenance labeling.

GOVERNANCE DIRECTIVE:
1. Real patient diagnostic labels (WDBC) are used strictly where appropriate.
2. Binary cancer labels are NEVER artificially converted into multiclass stages or continuous regression targets.
3. Multitask, multiclass subtypes, and structured output benchmarks are clearly identified as synthetic research benchmarks.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from backend.services.dataset_service import DatasetService

logger = logging.getLogger("cancerwise.experiments.data_factory")


class ResearchDataFactory:
    """
    Supplies validated datasets for advanced machine learning modules.
    """

    def __init__(self):
        self.dataset_service = DatasetService()

    def get_wdbc_binary_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Loads genuine UCI WDBC dataset for binary classification, semi-supervised, and novelty detection."""
        df = self.dataset_service.load_dataset("uci_wdbc")
        if df is None:
            raise FileNotFoundError("Primary UCI WDBC dataset not found.")

        target_col = "diagnosis"
        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
        y = (df[target_col] == "M").astype(int) if "M" in df[target_col].values else pd.factorize(df[target_col])[0]
        return X, pd.Series(y, index=X.index, name="diagnosis")

    def get_multitask_data(
        self,
        n_samples: int = 400,
        n_features: int = 12,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, str]]:
        """
        Generates synthetic research benchmark dataset with dual clinical tasks:
        Task 1: Primary Malignancy (binary classification)
        Task 2: Proliferation Index (continuous regression, 0.0 - 100.0)
        """
        rng = np.random.RandomState(random_state)
        feature_names = [
            "nuclear_area", "perimeter_ratio", "concavity_index", "chromatin_density",
            "mitotic_frequency", "membrane_ruffling", "cytoplasmic_ratio", "texture_heterogeneity",
            "expression_marker_a", "expression_marker_b", "microvessel_density", "stromal_fraction"
        ]

        X_mat = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))

        # Latent biological drivers
        latent_malignancy = (
            1.2 * X_mat[:, 0] + 0.8 * X_mat[:, 2] + 1.5 * X_mat[:, 4] - 0.5 * X_mat[:, 6] + rng.normal(0, 0.5, n_samples)
        )
        task1_probs = 1.0 / (1.0 + np.exp(-latent_malignancy))
        task1_binary = (task1_probs > 0.5).astype(int)

        # Proliferation Index (continuous, correlated with mitotic frequency and nuclear area)
        task2_continuous = (
            25.0 + 15.0 * X_mat[:, 0] + 20.0 * X_mat[:, 4] + 8.0 * X_mat[:, 7] + rng.normal(0, 3.0, n_samples)
        )
        task2_continuous = np.clip(task2_continuous, 0.0, 100.0)

        df_X = pd.DataFrame(X_mat, columns=feature_names)
        targets = {
            "task_malignancy": task1_binary,
            "task_proliferation_index": task2_continuous,
        }
        task_types = {
            "task_malignancy": "binary_classification",
            "task_proliferation_index": "regression",
        }
        return df_X, targets, task_types

    def get_multiclass_subtype_data(
        self,
        n_samples: int = 400,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Generates synthetic breast cancer molecular intrinsic subtypes benchmark:
        Classes: Luminal A, Luminal B, HER2-Enriched, Basal-like Triple-Negative
        """
        class_names = ["Luminal_A", "Luminal_B", "HER2_Enriched", "Basal_Like"]
        X, y = make_classification(
            n_samples=n_samples,
            n_features=14,
            n_informative=10,
            n_classes=4,
            n_clusters_per_class=1,
            weights=[0.40, 0.25, 0.15, 0.20],
            random_state=random_state,
        )
        cols = [f"biomarker_signature_{i+1}" for i in range(14)]
        df_X = pd.DataFrame(X, columns=cols)
        y_series = pd.Series(y, name="molecular_subtype")
        return df_X, y_series, class_names

    def get_continuous_regression_data(
        self,
        n_samples: int = 350,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.Series, str]:
        """
        Generates synthetic continuous tumor burden / cellularity density benchmark.
        """
        X, y = make_regression(
            n_samples=n_samples,
            n_features=10,
            n_informative=7,
            noise=8.0,
            random_state=random_state,
        )
        # Shift and scale to represent tumor cellularity percentage (10% to 95%)
        y_scaled = 50.0 + 10.0 * (y - np.mean(y)) / (np.std(y) + 1e-8)
        y_scaled = np.clip(y_scaled, 5.0, 98.0)

        cols = [f"morphometry_dim_{i+1}" for i in range(10)]
        df_X = pd.DataFrame(X, columns=cols)
        y_series = pd.Series(y_scaled, name="tumor_cellularity_percent")
        return df_X, y_series, "tumor_cellularity_percent"

    def get_structured_multilabel_data(
        self,
        n_samples: int = 320,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Generates structured multi-label receptor panel benchmark:
        Labels: ER_Positive, PR_Positive, HER2_Amplified, Ki67_Elevated
        """
        rng = np.random.RandomState(random_state)
        feature_names = [f"histochemical_intensity_{i+1}" for i in range(10)]
        X = rng.normal(0, 1, size=(n_samples, 10))

        # Biological correlations: ER and PR are positively correlated
        latent_hormone = 1.5 * X[:, 0] + 0.8 * X[:, 1] + rng.normal(0, 0.4, n_samples)
        er_prob = 1.0 / (1.0 + np.exp(-latent_hormone))
        pr_prob = 1.0 / (1.0 + np.exp(-(latent_hormone * 0.8 + rng.normal(0, 0.3, n_samples))))

        her2_prob = 1.0 / (1.0 + np.exp(-(1.2 * X[:, 3] - 0.5 * X[:, 0] + rng.normal(0, 0.4, n_samples))))
        ki67_prob = 1.0 / (1.0 + np.exp(-(1.1 * X[:, 4] + 0.7 * X[:, 3] + rng.normal(0, 0.4, n_samples))))

        y_labels = pd.DataFrame({
            "ER_Positive": (er_prob > 0.5).astype(int),
            "PR_Positive": (pr_prob > 0.5).astype(int),
            "HER2_Amplified": (her2_prob > 0.5).astype(int),
            "Ki67_Elevated": (ki67_prob > 0.5).astype(int),
        })

        return pd.DataFrame(X, columns=feature_names), y_labels, list(y_labels.columns)
