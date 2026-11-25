"""
CANCERWISE AI — Frontend Components Package
"""

from frontend.components.theme import (
    apply_theme,
    get_theme_css,
    get_plotly_template,
    PALETTE,
    SPACING,
    TYPOGRAPHY,
)
from frontend.components.header import render_header
from frontend.components.sidebar import render_sidebar
from frontend.components.cards import (
    MetricCard,
    ChartContainer,
    DataPreview,
    DatasetUploader,
    ModelSelector,
    AlgorithmConfigurator,
    ExperimentProgress,
    ResultsTable,
    ConfusionMatrixView,
    PredictionResultView,
    ExplanationPanel,
    ExportControl,
)

__all__ = [
    "apply_theme",
    "get_theme_css",
    "get_plotly_template",
    "PALETTE",
    "SPACING",
    "TYPOGRAPHY",
    "render_header",
    "render_sidebar",
    "MetricCard",
    "ChartContainer",
    "DataPreview",
    "DatasetUploader",
    "ModelSelector",
    "AlgorithmConfigurator",
    "ExperimentProgress",
    "ResultsTable",
    "ConfusionMatrixView",
    "PredictionResultView",
    "ExplanationPanel",
    "ExportControl",
]
