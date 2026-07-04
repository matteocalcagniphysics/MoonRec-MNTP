"""Pixel-level segmentation metrics for multi-label evaluation.

All functions operate on PyTorch tensors of shape ``(B, C, H, W)`` and
are fully differentiable-safe (decorated with ``@torch.no_grad``).
Results stay on the **same device** as the input tensors — no implicit
``.cpu()`` calls.

Typical usage
-------------
>>> from lunar_segmentation.evaluation.metrics import compute_all_metrics
>>> metrics = compute_all_metrics(logits, targets, from_logits=True)
>>> print(metrics["iou"])  # Tensor of shape (C,)
"""

from __future__ import annotations

import logging
from typing import Literal

import torch

logger = logging.getLogger(__name__)


# ======================================================================== #
#  Low-level helpers                                                        #
# ======================================================================== #

def _binarize(
    pred: torch.Tensor,
    from_logits: bool = True,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Convert predictions to binary masks.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions of shape ``(B, C, H, W)``.
    from_logits : bool
        If *True*, apply sigmoid before thresholding.
    threshold : float
        Decision boundary.

    Returns
    -------
    torch.Tensor
        Binary float tensor on the same device as *pred*.
    """
    if from_logits:
        pred = torch.sigmoid(pred)
    return (pred > threshold).float()


def _reduce(
    per_class: torch.Tensor,
    reduction: Literal["none", "mean", "per_class"],
) -> torch.Tensor:
    """Apply reduction strategy.

    Parameters
    ----------
    per_class : torch.Tensor
        Per-class metric of shape ``(C,)``.
    reduction : str
        ``"none"`` / ``"per_class"`` return as-is, ``"mean"`` averages.

    Returns
    -------
    torch.Tensor
        Scalar or ``(C,)`` tensor.
    """
    if reduction == "mean":
        return per_class.nanmean()
    return per_class  # "none" and "per_class" keep full shape


# ======================================================================== #
#  Confusion components                                                     #
# ======================================================================== #

@torch.no_grad()
def confusion_components(
    pred: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool = True,
    threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Compute per-class TP, FP, FN, TN pixel counts.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions ``(B, C, H, W)``.
    target : torch.Tensor
        Ground-truth binary masks ``(B, C, H, W)``.
    from_logits : bool
        Apply sigmoid to *pred* before thresholding.
    threshold : float
        Decision boundary for binarisation.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with keys ``"tp"``, ``"fp"``, ``"fn"``, ``"tn"``,
        each of shape ``(C,)`` summed over batch and spatial dims.
    """
    p = _binarize(pred, from_logits=from_logits, threshold=threshold)
    t = target.float()

    tp = (p * t).sum(dim=(0, 2, 3))
    fp = (p * (1.0 - t)).sum(dim=(0, 2, 3))
    fn = ((1.0 - p) * t).sum(dim=(0, 2, 3))
    tn = ((1.0 - p) * (1.0 - t)).sum(dim=(0, 2, 3))

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ======================================================================== #
#  Core metrics                                                             #
# ======================================================================== #

@torch.no_grad()
def iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool = True,
    threshold: float = 0.5,
    eps: float = 1e-6,
    reduction: Literal["none", "mean", "per_class"] = "per_class",
) -> torch.Tensor:
    """Intersection over Union (Jaccard Index).

    Parameters
    ----------
    pred : torch.Tensor
        Predictions ``(B, C, H, W)``.
    target : torch.Tensor
        Ground-truth binary masks ``(B, C, H, W)``.
    from_logits : bool
        Apply sigmoid to *pred* before thresholding.
    threshold : float
        Decision boundary for binarisation.
    eps : float
        Smoothing constant to avoid division by zero.
    reduction : str
        ``"per_class"`` → ``(C,)``, ``"mean"`` → scalar.

    Returns
    -------
    torch.Tensor
        IoU scores.
    """
    cc = confusion_components(pred, target, from_logits=from_logits, threshold=threshold)
    valid_mask = (cc["tp"] + cc["fp"] + cc["fn"]) > 0
    per_class = torch.where(
        valid_mask,
        (cc["tp"] + eps) / (cc["tp"] + cc["fp"] + cc["fn"] + eps),
        torch.tensor(float('nan'), device=pred.device)
    )
    return _reduce(per_class, reduction)


@torch.no_grad()
def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool = True,
    threshold: float = 0.5,
    eps: float = 1e-6,
    reduction: Literal["none", "mean", "per_class"] = "per_class",
) -> torch.Tensor:
    """Sørensen–Dice coefficient.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions ``(B, C, H, W)``.
    target : torch.Tensor
        Ground-truth binary masks ``(B, C, H, W)``.
    from_logits : bool
        Apply sigmoid to *pred* before thresholding.
    threshold : float
        Decision boundary.
    eps : float
        Smoothing constant.
    reduction : str
        ``"per_class"`` → ``(C,)``, ``"mean"`` → scalar.

    Returns
    -------
    torch.Tensor
        Dice scores.
    """
    cc = confusion_components(pred, target, from_logits=from_logits, threshold=threshold)
    valid_mask = (cc["tp"] + cc["fp"] + cc["fn"]) > 0
    per_class = torch.where(
        valid_mask,
        (2.0 * cc["tp"] + eps) / (2.0 * cc["tp"] + cc["fp"] + cc["fn"] + eps),
        torch.tensor(float('nan'), device=pred.device)
    )
    return _reduce(per_class, reduction)


@torch.no_grad()
def precision(
    pred: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool = True,
    threshold: float = 0.5,
    eps: float = 1e-6,
    reduction: Literal["none", "mean", "per_class"] = "per_class",
) -> torch.Tensor:
    """Precision = TP / (TP + FP).

    Parameters
    ----------
    pred, target, from_logits, threshold, eps, reduction
        See :func:`iou`.

    Returns
    -------
    torch.Tensor
        Precision scores.
    """
    cc = confusion_components(pred, target, from_logits=from_logits, threshold=threshold)
    valid_mask = (cc["tp"] + cc["fp"] + cc["fn"]) > 0
    per_class = torch.where(
        valid_mask,
        (cc["tp"] + eps) / (cc["tp"] + cc["fp"] + eps),
        torch.tensor(float('nan'), device=pred.device)
    )
    return _reduce(per_class, reduction)


@torch.no_grad()
def recall(
    pred: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool = True,
    threshold: float = 0.5,
    eps: float = 1e-6,
    reduction: Literal["none", "mean", "per_class"] = "per_class",
) -> torch.Tensor:
    """Recall (Sensitivity) = TP / (TP + FN).

    Parameters
    ----------
    pred, target, from_logits, threshold, eps, reduction
        See :func:`iou`.

    Returns
    -------
    torch.Tensor
        Recall scores.
    """
    cc = confusion_components(pred, target, from_logits=from_logits, threshold=threshold)
    valid_mask = (cc["tp"] + cc["fp"] + cc["fn"]) > 0
    per_class = torch.where(
        valid_mask,
        (cc["tp"] + eps) / (cc["tp"] + cc["fn"] + eps),
        torch.tensor(float('nan'), device=pred.device)
    )
    return _reduce(per_class, reduction)


@torch.no_grad()
def f1_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool = True,
    threshold: float = 0.5,
    eps: float = 1e-6,
    reduction: Literal["none", "mean", "per_class"] = "per_class",
) -> torch.Tensor:
    """F1-score (harmonic mean of Precision and Recall).

    Numerically equivalent to the Dice coefficient, but computed via
    the precision/recall decomposition for clarity.

    Parameters
    ----------
    pred, target, from_logits, threshold, eps, reduction
        See :func:`iou`.

    Returns
    -------
    torch.Tensor
        F1 scores.
    """
    p = precision(pred, target, from_logits=from_logits, threshold=threshold, eps=eps, reduction="per_class")
    r = recall(pred, target, from_logits=from_logits, threshold=threshold, eps=eps, reduction="per_class")
    per_class = (2.0 * p * r + eps) / (p + r + eps)
    return _reduce(per_class, reduction)


# ======================================================================== #
#  Threshold sensitivity                                                    #
# ======================================================================== #

@torch.no_grad()
def threshold_sweep(
    pred: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool = True,
    thresholds: tuple[float, ...] | None = None,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Sweep metrics over multiple thresholds.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions ``(B, C, H, W)``.
    target : torch.Tensor
        Ground-truth binary masks ``(B, C, H, W)``.
    from_logits : bool
        Apply sigmoid before thresholding.
    thresholds : tuple of float, optional
        Thresholds to evaluate. Defaults to ``(0.1, 0.2, ..., 0.9)``.
    eps : float
        Smoothing constant.

    Returns
    -------
    dict[str, torch.Tensor]
        ``"thresholds"`` → ``(T,)``,
        ``"iou"`` → ``(T, C)``,
        ``"dice"`` → ``(T, C)``.
    """
    if thresholds is None:
        thresholds = tuple(i / 10.0 for i in range(1, 10))

    # Apply sigmoid once if needed
    if from_logits:
        pred = torch.sigmoid(pred)

    iou_list: list[torch.Tensor] = []
    dice_list: list[torch.Tensor] = []

    for t in thresholds:
        iou_list.append(iou(pred, target, from_logits=False, threshold=t, eps=eps))
        dice_list.append(dice_coefficient(pred, target, from_logits=False, threshold=t, eps=eps))

    return {
        "thresholds": torch.tensor(thresholds, dtype=torch.float32),
        "iou": torch.stack(iou_list),      # (T, C)
        "dice": torch.stack(dice_list),     # (T, C)
    }


# ======================================================================== #
#  Vectorized metrics helpers                                               #
# ======================================================================== #

@torch.no_grad()
def confusion_components_vectorized(
    pred: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool = True,
    threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Compute per-sample, per-class TP, FP, FN, TN pixel counts.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions ``(B, C, H, W)``.
    target : torch.Tensor
        Ground-truth binary masks ``(B, C, H, W)``.
    from_logits : bool
        Apply sigmoid to *pred* before thresholding.
    threshold : float
        Decision boundary for binarisation.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with keys ``"tp"``, ``"fp"``, ``"fn"``, ``"tn"``,
        each of shape ``(B, C)`` summed over spatial dimensions only.
    """
    p = _binarize(pred, from_logits=from_logits, threshold=threshold)
    t = target.float()

    tp = (p * t).sum(dim=(2, 3))
    fp = (p * (1.0 - t)).sum(dim=(2, 3))
    fn = ((1.0 - p) * t).sum(dim=(2, 3))
    tn = ((1.0 - p) * (1.0 - t)).sum(dim=(2, 3))

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


@torch.no_grad()
def compute_all_metrics_vectorized(
    pred: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool = True,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Compute all pixel-level metrics for each sample in the batch in a single pass.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions ``(B, C, H, W)``.
    target : torch.Tensor
        Ground-truth binary masks ``(B, C, H, W)``.
    from_logits : bool
        Apply sigmoid to *pred* before thresholding.
    threshold : float
        Decision boundary.
    eps : float
        Smoothing constant.

    Returns
    -------
    dict[str, torch.Tensor]
        Keys: ``"iou"``, ``"dice"``, ``"precision"``, ``"recall"``,
        ``"f1"``, each of shape ``(B, C)``. Also includes
        ``"confusion"`` sub-dict with ``"tp"``, ``"fp"``, ``"fn"``,
        ``"tn"`` counts of shape ``(B, C)``.
    """
    cc = confusion_components_vectorized(pred, target, from_logits=from_logits, threshold=threshold)

    tp, fp, fn = cc["tp"], cc["fp"], cc["fn"]
    valid_mask = (tp + fp + fn) > 0

    iou_val = torch.where(valid_mask, (tp + eps) / (tp + fp + fn + eps), torch.tensor(float('nan'), device=pred.device))
    dice_val = torch.where(valid_mask, (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps), torch.tensor(float('nan'), device=pred.device))
    prec_val = torch.where(valid_mask, (tp + eps) / (tp + fp + eps), torch.tensor(float('nan'), device=pred.device))
    rec_val = torch.where(valid_mask, (tp + eps) / (tp + fn + eps), torch.tensor(float('nan'), device=pred.device))
    f1_val = torch.where(valid_mask, (2.0 * prec_val * rec_val + eps) / (prec_val + rec_val + eps), torch.tensor(float('nan'), device=pred.device))

    return {
        "iou": iou_val,
        "dice": dice_val,
        "precision": prec_val,
        "recall": rec_val,
        "f1": f1_val,
        "confusion": cc,
    }


# ======================================================================== #
#  Aggregate helper                                                         #
# ======================================================================== #

@torch.no_grad()
def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool = True,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Compute all pixel-level metrics in a single pass.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions ``(B, C, H, W)``.
    target : torch.Tensor
        Ground-truth binary masks ``(B, C, H, W)``.
    from_logits : bool
        Apply sigmoid to *pred* before thresholding.
    threshold : float
        Decision boundary.
    eps : float
        Smoothing constant.

    Returns
    -------
    dict[str, torch.Tensor]
        Keys: ``"iou"``, ``"dice"``, ``"precision"``, ``"recall"``,
        ``"f1"``, each of shape ``(C,)``.  Also includes
        ``"confusion"`` sub-dict with ``"tp"``, ``"fp"``, ``"fn"``,
        ``"tn"`` counts.
    """
    cc = confusion_components(pred, target, from_logits=from_logits, threshold=threshold)

    tp, fp, fn = cc["tp"], cc["fp"], cc["fn"]

    iou_val = (tp + eps) / (tp + fp + fn + eps)
    dice_val = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    prec_val = (tp + eps) / (tp + fp + eps)
    rec_val = (tp + eps) / (tp + fn + eps)
    f1_val = (2.0 * prec_val * rec_val + eps) / (prec_val + rec_val + eps)

    return {
        "iou": iou_val,
        "dice": dice_val,
        "precision": prec_val,
        "recall": rec_val,
        "f1": f1_val,
        "confusion": cc,
    }
