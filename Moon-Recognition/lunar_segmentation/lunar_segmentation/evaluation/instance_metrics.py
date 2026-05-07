"""Instance-level segmentation metrics (COCO-style AP/mAP).

This module provides stubs and a full implementation for computing
Average Precision at various IoU thresholds, following the COCO
evaluation protocol.  Designed for Mask R-CNN outputs in the standard
``torchvision.models.detection`` format.

Note
----
These functions are ready-to-use when Pasquale's Mask R-CNN produces
outputs.  The expected format is ``list[dict]`` where each dict contains
``"masks"`` (``(N, H, W)`` binary), ``"scores"`` (``(N,)``), and
``"labels"`` (``(N,)``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ======================================================================== #
#  Pairwise mask IoU                                                        #
# ======================================================================== #

@torch.no_grad()
def mask_pairwise_iou(
    masks_a: torch.Tensor,
    masks_b: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute pairwise IoU between two sets of binary masks.

    Parameters
    ----------
    masks_a : torch.Tensor
        ``(N, H, W)`` binary masks.
    masks_b : torch.Tensor
        ``(M, H, W)`` binary masks.
    eps : float
        Smoothing to avoid division by zero.

    Returns
    -------
    torch.Tensor
        ``(N, M)`` IoU matrix.
    """
    a = masks_a.flatten(1).float()  # (N, H*W)
    b = masks_b.flatten(1).float()  # (M, H*W)

    intersection = a @ b.T  # (N, M)
    area_a = a.sum(dim=1, keepdim=True)  # (N, 1)
    area_b = b.sum(dim=1, keepdim=True).T  # (1, M)
    union = area_a + area_b - intersection

    return (intersection + eps) / (union + eps)


# ======================================================================== #
#  Average Precision (single image, single class)                           #
# ======================================================================== #

def _match_predictions(
    pred_masks: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_masks: torch.Tensor,
    iou_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Greedy matching of predicted masks to GT.

    Parameters
    ----------
    pred_masks : torch.Tensor
        ``(N_pred, H, W)`` predicted binary masks.
    pred_scores : torch.Tensor
        ``(N_pred,)`` confidence scores.
    gt_masks : torch.Tensor
        ``(N_gt, H, W)`` ground-truth binary masks.
    iou_threshold : float
        Minimum IoU for a match.

    Returns
    -------
    tp_flags : np.ndarray
        ``(N_pred,)`` boolean array — True if prediction is a true positive.
    scores : np.ndarray
        ``(N_pred,)`` confidence scores (sorted descending).
    n_gt : int
        Number of ground-truth instances.
    """
    n_gt = gt_masks.shape[0] if gt_masks.numel() > 0 else 0

    if pred_masks.numel() == 0 or n_gt == 0:
        return (
            np.zeros(pred_masks.shape[0], dtype=bool),
            pred_scores.cpu().numpy() if pred_masks.numel() > 0 else np.array([]),
            n_gt,
        )

    # Sort predictions by descending confidence
    order = pred_scores.argsort(descending=True)
    pred_masks = pred_masks[order]
    scores_np = pred_scores[order].cpu().numpy()

    iou_matrix = mask_pairwise_iou(pred_masks, gt_masks)  # (N_pred, N_gt)
    iou_np = iou_matrix.cpu().numpy()

    tp_flags = np.zeros(len(pred_masks), dtype=bool)
    matched_gt: set[int] = set()

    for i in range(len(pred_masks)):
        best_gt = int(iou_np[i].argmax())
        best_iou = iou_np[i, best_gt]
        if best_iou >= iou_threshold and best_gt not in matched_gt:
            tp_flags[i] = True
            matched_gt.add(best_gt)

    return tp_flags, scores_np, n_gt


def _compute_ap_from_tp_fp(
    tp_flags: np.ndarray,
    n_gt: int,
) -> float:
    """Compute AP from cumulative TP/FP arrays (VOC/COCO style).

    Parameters
    ----------
    tp_flags : np.ndarray
        Boolean array indicating TP for each sorted detection.
    n_gt : int
        Total number of ground-truth instances.

    Returns
    -------
    float
        Average Precision value.
    """
    if n_gt == 0:
        return 0.0 if len(tp_flags) > 0 else 1.0

    tp_cumsum = np.cumsum(tp_flags).astype(np.float64)
    fp_cumsum = np.cumsum(~tp_flags).astype(np.float64)

    precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall_curve = tp_cumsum / n_gt

    # Append sentinel values for numerical stability
    precision_curve = np.concatenate(([1.0], precision_curve))
    recall_curve = np.concatenate(([0.0], recall_curve))

    # Make precision monotonically decreasing (right-to-left max)
    for i in range(len(precision_curve) - 2, -1, -1):
        precision_curve[i] = max(precision_curve[i], precision_curve[i + 1])

    # Find recall change-points
    recall_diff = np.diff(recall_curve)
    change_idx = np.where(recall_diff > 0)[0]

    ap = float(np.sum(recall_diff[change_idx] * precision_curve[change_idx + 1]))
    return ap


# ======================================================================== #
#  Public API                                                               #
# ======================================================================== #

@dataclass
class InstanceEvaluationResult:
    """Container for instance-level evaluation results.

    Attributes
    ----------
    ap_per_threshold : dict[float, float]
        AP at each IoU threshold.
    mean_ap : float
        Mean AP across all thresholds.
    n_predictions : int
        Total number of predicted instances.
    n_ground_truth : int
        Total number of ground-truth instances.
    """

    ap_per_threshold: dict[float, float] = field(default_factory=dict)
    mean_ap: float = 0.0
    n_predictions: int = 0
    n_ground_truth: int = 0


def average_precision(
    pred_masks: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_masks: torch.Tensor,
    iou_thresholds: tuple[float, ...] = (0.5, 0.75),
) -> InstanceEvaluationResult:
    """Compute Average Precision at given IoU thresholds.

    Parameters
    ----------
    pred_masks : torch.Tensor
        ``(N_pred, H, W)`` predicted binary masks.
    pred_scores : torch.Tensor
        ``(N_pred,)`` confidence scores.
    gt_masks : torch.Tensor
        ``(N_gt, H, W)`` ground-truth binary masks.
    iou_thresholds : tuple of float
        IoU thresholds at which to compute AP.

    Returns
    -------
    InstanceEvaluationResult
        Contains per-threshold AP and mean AP.
    """
    ap_dict: dict[float, float] = {}
    for t in iou_thresholds:
        tp_flags, _, n_gt = _match_predictions(
            pred_masks, pred_scores, gt_masks, iou_threshold=t,
        )
        ap_dict[t] = _compute_ap_from_tp_fp(tp_flags, n_gt)

    return InstanceEvaluationResult(
        ap_per_threshold=ap_dict,
        mean_ap=float(np.mean(list(ap_dict.values()))) if ap_dict else 0.0,
        n_predictions=int(pred_masks.shape[0]) if pred_masks.numel() > 0 else 0,
        n_ground_truth=int(gt_masks.shape[0]) if gt_masks.numel() > 0 else 0,
    )


def mean_average_precision(
    results: list[InstanceEvaluationResult],
) -> float:
    """Compute mean AP averaged over a list of per-image results.

    Parameters
    ----------
    results : list of InstanceEvaluationResult
        Per-image or per-class results.

    Returns
    -------
    float
        Global mean AP.
    """
    if not results:
        return 0.0
    return float(np.mean([r.mean_ap for r in results]))
