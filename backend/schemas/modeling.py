"""
CANCERWISE AI — Supervised Machine Learning & Modeling Schemas
Pydantic schemas for model configuration, training parameters, evaluation metrics,
model comparison benchmarks, prediction requests/responses, and artifact manifests.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime, timezone


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class AlgorithmType(str, Enum):
    # Linear
    LOGISTIC_REGRESSION = "logistic_regression"
    SGD = "sgd"
    # Tree
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    EXTRA_TREES = "extra_trees"
    GRADIENT_BOOSTING = "gradient_boosting"
    # Kernel
    LINEAR_SVM = "linear_svm"
    RBF_SVM = "rbf_svm"
    POLY_SVM = "poly_svm"
    # Instance
    KNN = "knn"
    # Probabilistic
    GAUSSIAN_NB = "gaussian_nb"
    # Boosting
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    # Neural
    MLP_SKLEARN = "mlp_sklearn"
    MLP_PYTORCH = "mlp_pytorch"


class ResamplingStrategy(str, Enum):
    NONE = "none"
    CLASS_WEIGHTS = "class_weights"
    RANDOM_OVERSAMPLING = "random_oversampling"
    RANDOM_UNDERSAMPLING = "random_undersampling"
    SMOTE = "smote"


class TuningStrategy(str, Enum):
    NONE = "none"
    GRID_SEARCH = "grid_search"
    RANDOMIZED_SEARCH = "randomized_search"
    OPTUNA = "optuna"


# ---------------------------------------------------------------------------
# Evaluation Metrics Schemas
# ---------------------------------------------------------------------------
class ConfidenceInterval(BaseModel):
    lower: float
    upper: float
    confidence_level: float = 0.95


class ModelEvaluationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall_sensitivity: float  # True Positive Rate
    specificity: float         # True Negative Rate
    f1_score: float
    balanced_accuracy: float
    roc_auc: float
    pr_auc: float
    matthews_corr_coef: float
    log_loss_score: float
    brier_score: float
    confusion_matrix: List[List[int]]  # [[TN, FP], [FN, TP]]
    confidence_intervals: Dict[str, ConfidenceInterval] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Training Request & Result Schemas
# ---------------------------------------------------------------------------
class ModelTrainingConfig(BaseModel):
    algorithm: AlgorithmType
    dataset_id: str
    target_column: str = "diagnosis"
    selected_features: Optional[List[str]] = None
    test_size: float = 0.20
    random_state: int = 42
    resampling: ResamplingStrategy = ResamplingStrategy.NONE
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    tuning_strategy: TuningStrategy = TuningStrategy.NONE
    tuning_trials: int = 15
    cv_folds: int = 5


class ModelTrainingResult(BaseModel):
    status: str = "SUCCESS"
    model_id: str
    model_name: str
    algorithm: AlgorithmType
    dataset_id: str
    target_column: str
    version: str
    created_at: str = Field(default_factory=_utc_now_str)
    selected_features: List[str]
    resampling_applied: ResamplingStrategy
    hyperparameters: Dict[str, Any]
    training_time_seconds: float
    inference_latency_ms: float
    artifact_path: str
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    test_metrics: ModelEvaluationMetrics

    @property
    def validation_metrics(self) -> ModelEvaluationMetrics:
        return self.test_metrics



# ---------------------------------------------------------------------------
# Model Comparison Schemas
# ---------------------------------------------------------------------------
class ModelComparisonItem(BaseModel):
    model_id: str
    model_name: str
    algorithm: AlgorithmType
    category: str
    test_accuracy: float
    test_precision: float
    test_recall_sensitivity: float
    test_specificity: float
    test_f1_score: float
    test_balanced_accuracy: float
    test_roc_auc: float
    test_pr_auc: float
    test_mcc: float
    test_brier_score: float
    training_time_seconds: float
    inference_latency_ms: float
    model_size_kb: float
    generalization_gap: float  # Train score minus Test score


class ModelComparisonSummary(BaseModel):
    ranking_metric: str
    ranked_models: List[ModelComparisonItem]
    top_recommended_model: str
    clinical_rationale: str


# ---------------------------------------------------------------------------
# Inference & Prediction Schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    features: Union[Dict[str, float], List[Dict[str, float]]]


class SinglePredictionResult(BaseModel):
    predicted_class: int
    predicted_label: str
    probability_benign: float
    probability_malignant: float
    confidence_score: float


class PredictionResponse(BaseModel):
    model_id: str
    model_version: str
    timestamp: str = Field(default_factory=_utc_now_str)
    predictions: List[SinglePredictionResult]
    latency_ms: float
    clinical_disclaimer: str = (
        "Medical Governance Notice: Estimated probabilities are statistical model approximations. "
        "They DO NOT constitute definitive clinical diagnosis or oncological staging certainty."
    )
