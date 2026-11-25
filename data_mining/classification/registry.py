"""
CANCERWISE AI — Model Registry & Algorithm Factory
Unified catalog of supported classification algorithms across linear, tree, kernel,
instance, probabilistic, boosting, and deep neural network paradigms.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from backend.schemas.modeling import AlgorithmType
from data_mining.classification.base import PyTorchMLPClassifier

logger = logging.getLogger("cancerwise.classification.registry")


class AlgorithmMetadata:
    """Descriptor for an algorithm family and its parameter spaces."""

    def __init__(
        self,
        algorithm_type: AlgorithmType,
        display_name: str,
        category: str,
        factory: Callable[..., BaseEstimator],
        default_params: Dict[str, Any],
        search_grid: Dict[str, List[Any]],
    ):
        self.algorithm_type = algorithm_type
        self.display_name = display_name
        self.category = category
        self.factory = factory
        self.default_params = default_params
        self.search_grid = search_grid


class ModelRegistry:
    """
    Central registry of classification algorithms for CANCERWISE AI.
    """

    def __init__(self):
        self._registry: Dict[AlgorithmType, AlgorithmMetadata] = {}
        self._register_all_algorithms()

    def _register_all_algorithms(self) -> None:
        # 1. Linear Models
        self._register(
            AlgorithmType.LOGISTIC_REGRESSION,
            "Logistic Regression",
            "Linear",
            lambda params, seed: LogisticRegression(
                max_iter=params.get("max_iter", 1000),
                C=params.get("C", 1.0),
                class_weight=params.get("class_weight", None),
                random_state=seed,
            ),
            {"C": 1.0, "max_iter": 1000},
            {"C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        )
        self._register(
            AlgorithmType.SGD,
            "Stochastic Gradient Descent (SGD)",
            "Linear",
            lambda params, seed: SGDClassifier(
                loss="log_loss",
                alpha=params.get("alpha", 0.0001),
                penalty=params.get("penalty", "l2"),
                max_iter=1000,
                random_state=seed,
            ),
            {"alpha": 0.0001, "penalty": "l2"},
            {"alpha": [1e-5, 1e-4, 1e-3, 1e-2], "penalty": ["l2", "l1", "elasticnet"]},
        )

        # 2. Tree-Based Models
        self._register(
            AlgorithmType.DECISION_TREE,
            "Decision Tree Classifier",
            "Tree",
            lambda params, seed: DecisionTreeClassifier(
                max_depth=params.get("max_depth", None),
                min_samples_split=params.get("min_samples_split", 2),
                criterion=params.get("criterion", "gini"),
                random_state=seed,
            ),
            {"max_depth": 6, "min_samples_split": 2, "criterion": "gini"},
            {"max_depth": [3, 5, 8, None], "min_samples_split": [2, 5, 10]},
        )
        self._register(
            AlgorithmType.RANDOM_FOREST,
            "Random Forest Classifier",
            "Tree",
            lambda params, seed: RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                min_samples_split=params.get("min_samples_split", 2),
                random_state=seed,
                n_jobs=-1,
            ),
            {"n_estimators": 100, "max_depth": 8, "min_samples_split": 2},
            {"n_estimators": [50, 100, 200], "max_depth": [4, 8, 12, None]},
        )
        self._register(
            AlgorithmType.EXTRA_TREES,
            "Extra Trees Classifier",
            "Tree",
            lambda params, seed: ExtraTreesClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                random_state=seed,
                n_jobs=-1,
            ),
            {"n_estimators": 100, "max_depth": 8},
            {"n_estimators": [50, 100, 200], "max_depth": [4, 8, None]},
        )
        self._register(
            AlgorithmType.GRADIENT_BOOSTING,
            "Gradient Boosting Classifier",
            "Tree",
            lambda params, seed: GradientBoostingClassifier(
                n_estimators=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.1),
                max_depth=params.get("max_depth", 3),
                random_state=seed,
            ),
            {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
            {"n_estimators": [50, 100, 150], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5]},
        )

        # 3. Kernel-Based Models
        self._register(
            AlgorithmType.LINEAR_SVM,
            "Support Vector Machine (Linear)",
            "Kernel",
            lambda params, seed: SVC(
                kernel="linear",
                C=params.get("C", 1.0),
                probability=True,
                random_state=seed,
            ),
            {"C": 1.0},
            {"C": [0.1, 1.0, 10.0]},
        )
        self._register(
            AlgorithmType.RBF_SVM,
            "Support Vector Machine (RBF Kernel)",
            "Kernel",
            lambda params, seed: SVC(
                kernel="rbf",
                C=params.get("C", 1.0),
                gamma=params.get("gamma", "scale"),
                probability=True,
                random_state=seed,
            ),
            {"C": 1.0, "gamma": "scale"},
            {"C": [0.1, 1.0, 10.0, 50.0], "gamma": ["scale", "auto", 0.01, 0.1]},
        )
        self._register(
            AlgorithmType.POLY_SVM,
            "Support Vector Machine (Polynomial Kernel)",
            "Kernel",
            lambda params, seed: SVC(
                kernel="poly",
                degree=params.get("degree", 3),
                C=params.get("C", 1.0),
                probability=True,
                random_state=seed,
            ),
            {"C": 1.0, "degree": 3},
            {"C": [0.1, 1.0, 10.0], "degree": [2, 3]},
        )

        # 4. Instance-Based Models
        self._register(
            AlgorithmType.KNN,
            "K-Nearest Neighbors",
            "Instance",
            lambda params, seed: KNeighborsClassifier(
                n_neighbors=params.get("n_neighbors", 5),
                weights=params.get("weights", "uniform"),
                metric=params.get("metric", "minkowski"),
            ),
            {"n_neighbors": 5, "weights": "uniform"},
            {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"]},
        )

        # 5. Probabilistic Models
        self._register(
            AlgorithmType.GAUSSIAN_NB,
            "Gaussian Naive Bayes",
            "Probabilistic",
            lambda params, seed: GaussianNB(
                var_smoothing=params.get("var_smoothing", 1e-9),
            ),
            {"var_smoothing": 1e-9},
            {"var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]},
        )

        # 6. Gradient Boosting Engines
        self._register(
            AlgorithmType.XGBOOST,
            "XGBoost Classifier",
            "Boosting",
            lambda params, seed: XGBClassifier(
                n_estimators=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.1),
                max_depth=params.get("max_depth", 4),
                eval_metric="logloss",
                random_state=seed,
            ),
            {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4},
            {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7]},
        )
        self._register(
            AlgorithmType.LIGHTGBM,
            "LightGBM Classifier",
            "Boosting",
            lambda params, seed: LGBMClassifier(
                n_estimators=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.1),
                max_depth=params.get("max_depth", -1),
                num_leaves=params.get("num_leaves", 31),
                verbose=-1,
                random_state=seed,
            ),
            {"n_estimators": 100, "learning_rate": 0.1, "max_depth": -1, "num_leaves": 31},
            {"n_estimators": [50, 100], "learning_rate": [0.03, 0.1], "num_leaves": [15, 31]},
        )
        self._register(
            AlgorithmType.CATBOOST,
            "CatBoost Classifier",
            "Boosting",
            lambda params, seed: CatBoostClassifier(
                iterations=params.get("iterations", 100),
                learning_rate=params.get("learning_rate", 0.1),
                depth=params.get("depth", 4),
                verbose=0,
                random_state=seed,
            ),
            {"iterations": 100, "learning_rate": 0.1, "depth": 4},
            {"iterations": [50, 100], "learning_rate": [0.03, 0.1], "depth": [4, 6]},
        )

        # 7. Neural Networks
        self._register(
            AlgorithmType.MLP_SKLEARN,
            "Multi-Layer Perceptron (Scikit-Learn)",
            "Neural Network",
            lambda params, seed: MLPClassifier(
                hidden_layer_sizes=params.get("hidden_layer_sizes", (64, 32)),
                alpha=params.get("alpha", 0.0001),
                learning_rate_init=params.get("learning_rate_init", 0.001),
                max_iter=500,
                random_state=seed,
            ),
            {"hidden_layer_sizes": (64, 32), "alpha": 0.0001, "learning_rate_init": 0.001},
            {"hidden_layer_sizes": [(32,), (64, 32), (100,)], "alpha": [1e-4, 1e-3]},
        )
        self._register(
            AlgorithmType.MLP_PYTORCH,
            "Deep MLP (PyTorch BCE & BatchNorm)",
            "Neural Network",
            lambda params, seed: PyTorchMLPClassifier(
                hidden_dim1=params.get("hidden_dim1", 64),
                hidden_dim2=params.get("hidden_dim2", 32),
                lr=params.get("lr", 0.003),
                epochs=params.get("epochs", 50),
                batch_size=params.get("batch_size", 32),
                dropout=params.get("dropout", 0.2),
                random_state=seed,
            ),
            {"hidden_dim1": 64, "hidden_dim2": 32, "lr": 0.003, "epochs": 50},
            {"hidden_dim1": [32, 64], "lr": [0.001, 0.005]},
        )

    def _register(
        self,
        alg_type: AlgorithmType,
        display_name: str,
        category: str,
        factory: Callable[..., BaseEstimator],
        default_params: Dict[str, Any],
        search_grid: Dict[str, List[Any]],
    ) -> None:
        self._registry[alg_type] = AlgorithmMetadata(
            algorithm_type=alg_type,
            display_name=display_name,
            category=category,
            factory=factory,
            default_params=default_params,
            search_grid=search_grid,
        )

    def get_algorithm_metadata(self, alg_type: Union[AlgorithmType, str]) -> AlgorithmMetadata:
        if isinstance(alg_type, str):
            alg_type = AlgorithmType(alg_type)
        if alg_type not in self._registry:
            raise KeyError(f"Algorithm '{alg_type}' not found in registry.")
        return self._registry[alg_type]

    def create_estimator(
        self,
        alg_type: Union[AlgorithmType, str],
        hyperparameters: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ) -> BaseEstimator:
        meta = self.get_algorithm_metadata(alg_type)
        params = {**meta.default_params, **(hyperparameters or {})}
        return meta.factory(params, random_state)

    def list_available_algorithms(self) -> List[Dict[str, Any]]:
        return [
            {
                "algorithm": k.value,
                "display_name": v.display_name,
                "category": v.category,
                "default_hyperparameters": v.default_params,
            }
            for k, v in self._registry.items()
        ]
