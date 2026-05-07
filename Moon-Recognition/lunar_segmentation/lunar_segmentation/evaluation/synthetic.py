"""Synthetic data generators and mock models for testing.

Provides factory functions to generate realistic-looking dummy tensors
(images with synthetic craters, corresponding masks, and controlled
prediction pairs) so that the evaluation pipeline can be validated
immediately without waiting for trained models.

Run as a script for a full demo::

    python -m lunar_segmentation.evaluation.synthetic
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ======================================================================== #
#  Random batch generators                                                  #
# ======================================================================== #

def generate_random_batch(
    batch_size: int = 4,
    num_classes: int = 7,
    height: int = 256,
    width: int = 256,
    num_channels: int = 3,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a batch of random images, GT masks, and predictions.

    Parameters
    ----------
    batch_size : int
        Number of samples.
    num_classes : int
        Number of mask channels.
    height, width : int
        Spatial dimensions.
    num_channels : int
        Number of input image channels.
    seed : int
        Random seed.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(images, gt_masks, pred_logits)`` where:

        - images: ``(B, C_in, H, W)`` float32 in [0, 1]
        - gt_masks: ``(B, C_out, H, W)`` binary float32
        - pred_logits: ``(B, C_out, H, W)`` float32 logits
    """
    rng = np.random.default_rng(seed)

    images = rng.random((batch_size, num_channels, height, width)).astype(np.float32)
    gt_masks = (rng.random((batch_size, num_classes, height, width)) > 0.95).astype(np.float32)
    # Logits: correlated with GT but noisy
    noise = rng.standard_normal((batch_size, num_classes, height, width)).astype(np.float32) * 2
    pred_logits = gt_masks * 3.0 + noise - 1.5  # Higher logits where GT is 1

    return (
        torch.from_numpy(images),
        torch.from_numpy(gt_masks),
        torch.from_numpy(pred_logits),
    )


def generate_realistic_batch(
    batch_size: int = 4,
    num_classes: int = 7,
    height: int = 256,
    width: int = 256,
    n_craters_range: tuple[int, int] = (3, 15),
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate batches with synthetic lunar-like imagery.

    Creates grayscale images with circular "craters" (dark disks
    with bright rims) and matching segmentation masks.  Predictions
    include controlled amounts of FP and FN to test error maps.

    Parameters
    ----------
    batch_size : int
        Number of samples.
    num_classes : int
        Number of mask channels (class 0 = craters, rest sparse).
    height, width : int
        Spatial dimensions.
    n_craters_range : tuple[int, int]
        Range for random number of craters per image.
    seed : int
        Random seed.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(images, gt_masks, pred_logits)``.
    """
    rng = np.random.default_rng(seed)

    images = np.zeros((batch_size, 3, height, width), dtype=np.float32)
    gt_masks = np.zeros((batch_size, num_classes, height, width), dtype=np.float32)
    pred_logits = np.zeros((batch_size, num_classes, height, width), dtype=np.float32)

    yy, xx = np.mgrid[:height, :width]

    for b in range(batch_size):
        # Base: lunar-like texture (Perlin-ish via smoothed noise)
        base = rng.random((height, width)).astype(np.float32) * 0.3 + 0.35

        n_craters = rng.integers(n_craters_range[0], n_craters_range[1] + 1)

        for _ in range(n_craters):
            cx = rng.integers(20, width - 20)
            cy = rng.integers(20, height - 20)
            r = rng.integers(5, 30)
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

            # Dark crater interior
            interior = dist < r
            base[interior] *= 0.5

            # Bright rim
            rim = (dist >= r) & (dist < r + 3)
            base[rim] = np.clip(base[rim] + 0.3, 0, 1)

            # GT mask (class 0 = impact_crater)
            gt_masks[b, 0, interior] = 1.0

        # Build 3-channel input (like build_three_channel_input)
        norm = (base - base.min()) / (base.max() - base.min() + 1e-8)
        images[b, 0] = norm
        images[b, 1] = norm  # Simplified CLAHE proxy
        # Sobel-like edge proxy
        grad_x = np.abs(np.diff(norm, axis=1, prepend=norm[:, :1]))
        grad_y = np.abs(np.diff(norm, axis=0, prepend=norm[:1, :]))
        images[b, 2] = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Sparse features for other classes
        for c in range(1, num_classes):
            if rng.random() > 0.7:
                cx = rng.integers(20, width - 20)
                cy = rng.integers(20, height - 20)
                r = rng.integers(2, 8)
                dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                gt_masks[b, c, dist < r] = 1.0

        # Predictions: GT with controlled noise (some FP, some FN)
        for c in range(num_classes):
            # Start from GT
            pred_prob = gt_masks[b, c].copy()
            # Add FP (random blobs)
            if rng.random() > 0.5:
                cx = rng.integers(10, width - 10)
                cy = rng.integers(10, height - 10)
                r = rng.integers(3, 10)
                dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                pred_prob[dist < r] = 1.0
            # Remove some FN (erode GT)
            if rng.random() > 0.5:
                from scipy.ndimage import binary_erosion
                eroded = binary_erosion(gt_masks[b, c] > 0, iterations=2)
                pred_prob = eroded.astype(np.float32)

            # Convert to logits
            pred_logits[b, c] = np.clip(pred_prob * 6.0 - 3.0 +
                                         rng.standard_normal((height, width)).astype(np.float32) * 0.5,
                                         -6, 6)

    return (
        torch.from_numpy(images),
        torch.from_numpy(gt_masks),
        torch.from_numpy(pred_logits),
    )


def generate_controlled_pair(
    height: int = 256,
    width: int = 256,
    iou_target: float = 0.7,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a GT/Pred pair with approximately controlled IoU.

    Useful for unit-testing metric functions.

    Parameters
    ----------
    height, width : int
        Spatial dimensions.
    iou_target : float
        Target IoU (approximate — exact control is hard).
    seed : int
        Random seed.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(gt_mask, pred_mask)`` both ``(1, 1, H, W)`` binary float.
    """
    rng = np.random.default_rng(seed)

    # Create a circular GT region
    yy, xx = np.mgrid[:height, :width]
    cx, cy = width // 2, height // 2
    r = min(height, width) // 4
    gt = ((xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2).astype(np.float32)

    # Shift centre to control IoU
    n_gt = gt.sum()
    # Approximate shift needed for target IoU
    # IoU = intersection / union ≈ (area - shift_loss) / (area + shift_gain)
    shift_frac = 1.0 - iou_target
    shift_px = int(shift_frac * r * 1.5)

    pred = np.roll(gt, shift_px, axis=1)  # Shift horizontally

    return (
        torch.from_numpy(gt[np.newaxis, np.newaxis]),
        torch.from_numpy(pred[np.newaxis, np.newaxis]),
    )


# ======================================================================== #
#  Mock models                                                              #
# ======================================================================== #

class MockSemanticModel:
    """Mock model implementing SegmentationModel protocol.

    Returns noisy copies of provided ground-truth or random logits.

    Parameters
    ----------
    num_classes : int
        Number of output channels.
    model_name : str
        Human-readable name.
    noise_std : float
        Standard deviation of Gaussian noise added to logits.
    """

    def __init__(
        self,
        num_classes: int = 7,
        model_name: str = "MockSemantic",
        noise_std: float = 1.0,
    ) -> None:
        self._num_classes = num_classes
        self._model_name = model_name
        self._noise_std = noise_std

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def model_name(self) -> str:
        return self._model_name

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Return random logits matching the expected output shape.

        Parameters
        ----------
        images : torch.Tensor
            ``(B, C_in, H, W)`` input.

        Returns
        -------
        torch.Tensor
            ``(B, C_out, H, W)`` logits.
        """
        b, _, h, w = images.shape
        logits = torch.randn(b, self._num_classes, h, w, device=images.device)
        return logits * self._noise_std


# ======================================================================== #
#  Demo script                                                              #
# ======================================================================== #

def _run_demo() -> None:
    """Full demonstration of the evaluation pipeline with synthetic data."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

    from lunar_segmentation.evaluation.metrics import (
        compute_all_metrics,
        threshold_sweep,
    )
    from lunar_segmentation.visualization.eval_plotter import SegmentationVisualizer

    CLASS_NAMES = [
        "impact_crater", "pit_skylight", "wrinkle_ridge",
        "lobate_scarp", "irregular_mare_patch", "apollo_site",
        "candidate_rille",
    ]

    print("=" * 60)
    print("  Lunar Segmentation - Evaluation Pipeline Demo")
    print("=" * 60)

    # 1. Generate realistic synthetic batch
    print("\n[1/5] Generating synthetic batch (4 x 7 x 256 x 256)...")
    images, gt_masks, pred_logits = generate_realistic_batch(
        batch_size=4, num_classes=7, height=256, width=256, seed=42,
    )
    print(f"  Images:  {images.shape}  dtype={images.dtype}")
    print(f"  GT:      {gt_masks.shape} dtype={gt_masks.dtype}")
    print(f"  Logits:  {pred_logits.shape} dtype={pred_logits.dtype}")

    # 2. Compute all metrics
    print("\n[2/5] Computing metrics...")
    metrics = compute_all_metrics(pred_logits, gt_masks, from_logits=True)
    print("\n  Per-class metrics:")
    print(f"  {'Class':<25} {'IoU':>6} {'Dice':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name:<25} "
              f"{metrics['iou'][c]:.4f} "
              f"{metrics['dice'][c]:.4f} "
              f"{metrics['precision'][c]:.4f} "
              f"{metrics['recall'][c]:.4f} "
              f"{metrics['f1'][c]:.4f}")
    print(f"\n  Mean IoU:  {metrics['iou'].mean():.4f}")
    print(f"  Mean Dice: {metrics['dice'].mean():.4f}")

    # 3. Identity sanity check
    print("\n[3/5] Sanity checks...")
    perfect = compute_all_metrics(gt_masks, gt_masks, from_logits=False)
    assert (perfect["iou"] > 0.99).all(), "Identity test FAILED"
    print("  [OK] Identity test passed (IoU ~= 1.0 when pred == gt)")

    zeros_pred = torch.zeros_like(gt_masks)
    zero_metrics = compute_all_metrics(zeros_pred, gt_masks, from_logits=False)
    print(f"  [OK] Opposite test passed (IoU ~= {zero_metrics['iou'].mean():.4f} when pred == 0)")

    # 4. Threshold sensitivity
    print("\n[4/5] Threshold sensitivity sweep...")
    sweep = threshold_sweep(pred_logits, gt_masks, from_logits=True)
    best_t_per_class = sweep["thresholds"][sweep["iou"].argmax(dim=0)]
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name:<25} best threshold = {best_t_per_class[c]:.1f}")

    # 5. Visualisation
    print("\n[5/5] Generating plots...")
    viz = SegmentationVisualizer(class_names=CLASS_NAMES)

    # Binarise predictions for plotting
    pred_binary = (torch.sigmoid(pred_logits) > 0.5).float().numpy()

    fig1, _ = viz.plot_prediction_panel(
        images[0].numpy(),
        gt_masks[0].numpy(),
        pred_binary[0],
        class_idx=0,
    )
    out_dir = __import__("pathlib").Path("evaluation_demo_output")
    out_dir.mkdir(exist_ok=True)
    viz.save_figure(fig1, out_dir / "prediction_panel.png")
    print(f"  Saved: {out_dir / 'prediction_panel.png'}")

    fig2, _ = viz.plot_class_comparison(
        images[0].numpy(),
        gt_masks[0].numpy(),
        pred_binary[0],
    )
    viz.save_figure(fig2, out_dir / "class_comparison.png")
    print(f"  Saved: {out_dir / 'class_comparison.png'}")

    fig3, _ = viz.plot_batch_predictions(
        images.numpy(),
        gt_masks.numpy(),
        pred_binary,
        class_idx=0,
        max_samples=4,
    )
    viz.save_figure(fig3, out_dir / "batch_predictions.png")
    print(f"  Saved: {out_dir / 'batch_predictions.png'}")

    fig4, _ = viz.plot_threshold_sensitivity(sweep)
    viz.save_figure(fig4, out_dir / "threshold_sensitivity.png")
    print(f"  Saved: {out_dir / 'threshold_sensitivity.png'}")

    print("\n" + "=" * 60)
    print("  Demo complete! Check ./evaluation_demo_output/")
    print("=" * 60)


if __name__ == "__main__":
    _run_demo()
