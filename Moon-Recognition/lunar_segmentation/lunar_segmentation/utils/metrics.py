"""Convenience re-exports from the evaluation subpackage.

This module exists to honour the package structure declared in the
project README (``utils/metrics.py``).  All implementation lives in
:mod:`lunar_segmentation.evaluation.metrics`.
"""

from ..evaluation.metrics import (
    iou,
    dice_coefficient,
    precision,
    recall,
    f1_score,
    confusion_components,
    compute_all_metrics,
    threshold_sweep,
    confusion_components_vectorized,
    compute_all_metrics_vectorized,
)

__all__ = [
    "iou",
    "dice_coefficient",
    "precision",
    "recall",
    "f1_score",
    "confusion_components",
    "compute_all_metrics",
    "threshold_sweep",
    "confusion_components_vectorized",
    "compute_all_metrics_vectorized",
]
