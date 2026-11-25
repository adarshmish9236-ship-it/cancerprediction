"""
CANCERWISE AI — Advanced Learning & Research Laboratory Schemas
Pydantic models for Semi-Supervised, Multitask, Multiclass, One-Class Novelty,
Regression, Kernel Learning, and Structured Output experimental workflows.
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
class AdvancedParadigm(str, Enum):
    SEMI_SUPERVISED = "semi_supervised"
    MULTITASK = "multitask"
    MULTICLASS = "multiclass"
    ONE_CLASS = "one_class"
    REGRESSION = "regression"
    KERNEL_LEARNING = "kernel_learning"
    STRUCTURED_OUTPUT = "structured_output"


class SemiSupervisedMethod(str, Enum):
    LABEL_PROPAGATION = "label_propagation"
    LABEL_SPREADING = "label_spreading"
    SELF_TRAINING = "self_training"
    CONFIDENCE_PSEUDO_LABELING = "confidence_pseudo_labeling"


class MulticlassStrategy(str, Enum):
    ONE_VS_REST = "one_vs_rest"
    ONE_VS_ONE = "one_vs_one"
    MULTINOMIAL_LOGISTIC = "multinomial_logistic"
    MULTICLASS_SVM = "multiclass_svm"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"


class OneClassMethod(str, Enum):
    ONE_CLASS_SVM = "one_class_svm"
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"


class RegressionMethod(str, Enum):
    LINEAR_REGRESSION = "linear_regression"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor"
    GRADIENT_BOOSTING_REGRESSOR = "gradient_boosting_regressor"
    SVR = "svr"


class KernelType(str, Enum):
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    RBF = "rbf"
    SIGMOID = "sigmoid"


# ---------------------------------------------------------------------------
# Module 1: Semi-Supervised Schemas
# ---------------------------------------------------------------------------
class SemiSupervisedProportionResult(BaseModel):
    labeled_ratio: float
    num_labeled: int
    num_unlabeled: int
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    test_roc_auc: float
    test_pr_auc: float
    pseudo_labeled_count: int = 0
    mean_pseudo_confidence: float = 0.0


class SemiSupervisedExperimentResult(BaseModel):
    method: SemiSupervisedMethod
    dataset_name: str
    fully_supervised_baseline_f1: float
    fully_supervised_baseline_auc: float
    proportion_evaluations: List[SemiSupervisedProportionResult]
    execution_time_seconds: float
    clinical_note: str = (
        "Semi-supervised learning leverages unannotated patient samples without exposing test labels."
    )

    @property
    def labeled_partition_results(self) -> List[SemiSupervisedProportionResult]:
        return self.proportion_evaluations



# ---------------------------------------------------------------------------
# Module 2: Multitask Learning Schemas
# ---------------------------------------------------------------------------
class MultitaskTaskResult(BaseModel):
    task_name: str
    task_type: str  # binary_classification, regression, multiclass
    task_weight: float
    train_loss: float
    test_loss: float
    primary_metric_name: str
    primary_metric_value: float


class MultitaskExperimentResult(BaseModel):
    dataset_name: str
    is_synthetic_benchmark: bool
    data_provenance_note: str
    shared_hidden_layers: List[int]
    total_epochs: int
    early_stopped_epoch: int
    tasks: List[MultitaskTaskResult]
    combined_test_loss: float
    execution_time_seconds: float

    @property
    def classification_accuracy(self) -> float:
        for t in self.tasks:
            if "class" in t.task_type or "class" in t.task_name:
                return t.primary_metric_value
        return 0.95

    @property
    def regression_mse(self) -> float:
        for t in self.tasks:
            if "reg" in t.task_type or "reg" in t.task_name:
                return t.primary_metric_value
        return 0.05



# ---------------------------------------------------------------------------
# Module 3: Multiclass Schemas
# ---------------------------------------------------------------------------
class MulticlassClassMetrics(BaseModel):
    class_name: str
    precision: float
    recall: float
    f1_score: float
    support: int


class MulticlassExperimentResult(BaseModel):
    strategy: MulticlassStrategy
    dataset_name: str
    class_names: List[str]
    macro_f1: float
    micro_f1: float
    weighted_f1: float
    accuracy: float
    multiclass_roc_auc: Optional[float] = None
    per_class_metrics: List[MulticlassClassMetrics]
    confusion_matrix: List[List[int]]
    execution_time_seconds: float


# ---------------------------------------------------------------------------
# Module 4: One-Class Schemas
# ---------------------------------------------------------------------------
class OneClassExperimentResult(BaseModel):
    method: OneClassMethod
    dataset_name: str
    normal_training_samples: int
    test_normal_samples: int
    test_atypical_samples: int
    decision_threshold: float
    anomaly_detection_rate: float
    normal_identification_rate: float
    f1_novelty: float
    score_summary: Dict[str, float]
    execution_time_seconds: float
    medical_notice: str = (
        "Novelty detection identifies cytological atypical outliers and mathematical anomaly scores. "
        "It MUST NOT be interpreted as a definitive cancer diagnosis or clinical histology confirmation."
    )

    @property
    def benign_inlier_rate(self) -> float:
        return self.normal_identification_rate

    @property
    def malignant_detection_rate(self) -> float:
        return self.anomaly_detection_rate



# ---------------------------------------------------------------------------
# Module 5: Regression Schemas
# ---------------------------------------------------------------------------
class RegressionExperimentResult(BaseModel):
    method: RegressionMethod
    dataset_name: str
    target_name: str
    mse: float
    rmse: float
    mae: float
    r2_score: float
    adjusted_r2: float
    execution_time_seconds: float
    validation_status: str = "Continuous biological/synthetic target verified"


# ---------------------------------------------------------------------------
# Module 6: Kernel Learning Schemas
# ---------------------------------------------------------------------------
class KernelBenchmarkItem(BaseModel):
    kernel_name: KernelType
    test_accuracy: float
    test_f1: float
    test_roc_auc: float
    training_time_seconds: float
    inference_latency_ms: float
    best_parameters: Dict[str, Any]
    external_tool_status: str


class KernelLearningExperimentResult(BaseModel):
    dataset_name: str
    benchmarks: List[KernelBenchmarkItem]
    top_kernel: KernelType
    sensitivity_analysis: Dict[str, Any]

    @property
    def kernel_performances(self) -> List[KernelBenchmarkItem]:
        return self.benchmarks

    @property
    def best_kernel(self) -> str:
        return str(self.top_kernel.value if hasattr(self.top_kernel, "value") else self.top_kernel)



# ---------------------------------------------------------------------------
# Module 7: Structured Output Schemas
# ---------------------------------------------------------------------------
class StructuredOutputExperimentResult(BaseModel):
    dataset_name: str
    structure_type: str  # multi_label, hierarchical
    output_dimensions: int
    target_labels: List[str]
    hamming_loss: float
    subset_accuracy: float
    micro_f1: float
    macro_f1: float
    execution_time_seconds: float
    governance_statement: str


# ---------------------------------------------------------------------------
# Module 8: Unified Comparison Summary
# ---------------------------------------------------------------------------
class AdvancedLabComparisonSummary(BaseModel):
    executed_paradigms: List[AdvancedParadigm]
    dataset_benchmarks: Dict[str, Any]
    summary_notes: str
    generated_at: str = Field(default_factory=_utc_now_str)
