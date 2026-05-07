"""Evaluation subpackage for lunar segmentation models.

Provides model-agnostic metrics, statistical comparison utilities,
and synthetic data generators for testing evaluation pipelines
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
    # Protocols
    "SegmentationModel",
    "SemanticModelAdapter",
    # Comparison
    "EvaluationResult",
    "evaluate_model",
]
