"""Evaluation subpackage for lunar segmentation models.

Provides model-agnostic metrics and statistical comparison utilities
independently of trained models.
"""

from .metrics import (
    iou,
    dice_coefficient,
    precision,
    recall,
    f1_score,
    confusion_components,
    compute_all_metrics,
    confusion_components_vectorized,
    compute_all_metrics_vectorized,
)
from .protocols import SegmentationModel, SemanticModelAdapter
from .comparison import EvaluationResult, evaluate_model

__all__ = [
    # Metrics
    "iou",
    "dice_coefficient",
    "precision",
    "recall",
    "f1_score",
    "confusion_components",
    "compute_all_metrics",
    "confusion_components_vectorized",
    "compute_all_metrics_vectorized",
    # Protocols
    "SegmentationModel",
    "SemanticModelAdapter",
    # Comparison
    "EvaluationResult",
    "evaluate_model",
]
