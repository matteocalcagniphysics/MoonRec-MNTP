"""Cross-model statistical comparison and evaluation orchestration.

Provides utilities to evaluate one or more models against a dataset,
aggregate per-sample metrics, compute bootstrap confidence intervals,
and run statistical significance tests (Wilcoxon signed-rank) between
model pairs.

Typical usage
-------------
>>> from lunar_segmentation.evaluation.comparison import evaluate_model, compare_models
>>> result_a = evaluate_model(adapter_a, val_loader, class_names)
>>> result_b = evaluate_model(adapter_b, val_loader, class_names)
>>> comparison = compare_models({"UNet": result_a, "Baseline": result_b})
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .metrics import compute_all_metrics

if TYPE_CHECKING:
    from .protocols import SegmentationModel

logger = logging.getLogger(__name__)


# ======================================================================== #
#  Data containers                                                          #
# ======================================================================== #

@dataclass
class EvaluationResult:
    """Container for per-sample evaluation results of a single model.

    Attributes
    ----------
    model_name : str
        Human-readable model identifier.
    class_names : list[str]
        Names of the output classes.
    per_sample_iou : np.ndarray
        ``(N_samples, C)`` per-sample, per-class IoU.
    per_sample_dice : np.ndarray
        ``(N_samples, C)`` per-sample, per-class Dice.
    per_sample_precision : np.ndarray
        ``(N_samples, C)`` per-sample, per-class Precision.
    per_sample_recall : np.ndarray
        ``(N_samples, C)`` per-sample, per-class Recall.
    per_sample_f1 : np.ndarray
        ``(N_samples, C)`` per-sample, per-class F1.
    """

    model_name: str
    class_names: list[str]
    per_sample_iou: np.ndarray = field(default_factory=lambda: np.array([]))
    per_sample_dice: np.ndarray = field(default_factory=lambda: np.array([]))
    per_sample_precision: np.ndarray = field(default_factory=lambda: np.array([]))
    per_sample_recall: np.ndarray = field(default_factory=lambda: np.array([]))
    per_sample_f1: np.ndarray = field(default_factory=lambda: np.array([]))

    # -- Computed summaries -------------------------------------------------

    def summary_df(self) -> pd.DataFrame:
        """Return a DataFrame with mean ± std per class per metric.

        Returns
        -------
        pd.DataFrame
            Columns: ``class``, ``metric``, ``mean``, ``std``.
        """
        rows: list[dict[str, object]] = []
        for metric_name, arr in [
            ("IoU", self.per_sample_iou),
            ("Dice", self.per_sample_dice),
            ("Precision", self.per_sample_precision),
            ("Recall", self.per_sample_recall),
            ("F1", self.per_sample_f1),
        ]:
            if arr.size == 0:
                continue
            for c, cname in enumerate(self.class_names):
                rows.append({
                    "model": self.model_name,
                    "class": cname,
                    "metric": metric_name,
                    "mean": float(np.mean(arr[:, c])),
                    "std": float(np.std(arr[:, c])),
                })
        return pd.DataFrame(rows)

    @property
    def mean_iou(self) -> float:
        """Macro-averaged IoU (mean over classes, then samples)."""
        if self.per_sample_iou.size == 0:
            return 0.0
        return float(np.mean(self.per_sample_iou))


# ======================================================================== #
#  Model evaluation                                                         #
# ======================================================================== #

@torch.no_grad()
def evaluate_model(
    model: SegmentationModel,
    dataloader: DataLoader,
    class_names: list[str],
    from_logits: bool = True,
    threshold: float = 0.5,
) -> EvaluationResult:
    """Evaluate a model on an entire dataset, collecting per-sample metrics.

    Parameters
    ----------
    model : SegmentationModel
        Any model satisfying the SegmentationModel protocol.
    dataloader : DataLoader
        Yields ``(images, masks)`` batches.
    class_names : list[str]
        Names for each output class channel.
    from_logits : bool
        Whether model output is logits (True) or probabilities (False).
    threshold : float
        Binarisation threshold.

    Returns
    -------
    EvaluationResult
        Per-sample metrics for the entire dataset.
    """
    all_iou: list[np.ndarray] = []
    all_dice: list[np.ndarray] = []
    all_prec: list[np.ndarray] = []
    all_rec: list[np.ndarray] = []
    all_f1: list[np.ndarray] = []

    for images, masks in dataloader:
        logits = model.predict(images)
        # Ensure masks match logits device
        masks = masks.to(logits.device).float()

        # Per-sample metrics: iterate each sample in the batch
        batch_size = images.shape[0]
        for i in range(batch_size):
            m = compute_all_metrics(
                logits[i : i + 1],
                masks[i : i + 1],
                from_logits=from_logits,
                threshold=threshold,
            )
            all_iou.append(m["iou"].cpu().numpy())
            all_dice.append(m["dice"].cpu().numpy())
            all_prec.append(m["precision"].cpu().numpy())
            all_rec.append(m["recall"].cpu().numpy())
            all_f1.append(m["f1"].cpu().numpy())

    return EvaluationResult(
        model_name=model.model_name,
        class_names=class_names,
        per_sample_iou=np.stack(all_iou) if all_iou else np.array([]),
        per_sample_dice=np.stack(all_dice) if all_dice else np.array([]),
        per_sample_precision=np.stack(all_prec) if all_prec else np.array([]),
        per_sample_recall=np.stack(all_rec) if all_rec else np.array([]),
        per_sample_f1=np.stack(all_f1) if all_f1 else np.array([]),
    )


# ======================================================================== #
#  Bootstrap confidence intervals                                           #
# ======================================================================== #

def bootstrap_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean.

    Parameters
    ----------
    values : np.ndarray
        1-D array of metric values.
    confidence : float
        Confidence level (e.g. 0.95 for 95% CI).
    n_resamples : int
        Number of bootstrap resamples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[float, float, float]
        ``(mean, ci_lower, ci_upper)``.
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0

    means = np.array([
        rng.choice(values, size=n, replace=True).mean()
        for _ in range(n_resamples)
    ])

    alpha = 1.0 - confidence
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(np.mean(values)), lo, hi


# ======================================================================== #
#  Cross-model comparison                                                   #
# ======================================================================== #

def compare_models(
    results: dict[str, EvaluationResult],
    confidence: float = 0.95,
    n_resamples: int = 1000,
) -> pd.DataFrame:
    """Compare multiple models with bootstrap CI per metric per class.

    Parameters
    ----------
    results : dict[str, EvaluationResult]
        Mapping from model name to evaluation results.
    confidence : float
        Confidence level for bootstrap CI.
    n_resamples : int
        Number of bootstrap resamples.

    Returns
    -------
    pd.DataFrame
        Columns: ``model``, ``class``, ``metric``, ``mean``,
        ``ci_lower``, ``ci_upper``.
    """
    rows: list[dict[str, object]] = []

    for model_name, result in results.items():
        for metric_name, arr in [
            ("IoU", result.per_sample_iou),
            ("Dice", result.per_sample_dice),
            ("Precision", result.per_sample_precision),
            ("Recall", result.per_sample_recall),
            ("F1", result.per_sample_f1),
        ]:
            if arr.size == 0:
                continue
            for c, cname in enumerate(result.class_names):
                mean, ci_lo, ci_hi = bootstrap_confidence_interval(
                    arr[:, c],
                    confidence=confidence,
                    n_resamples=n_resamples,
                )
                rows.append({
                    "model": model_name,
                    "class": cname,
                    "metric": metric_name,
                    "mean": mean,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                })

    return pd.DataFrame(rows)


def significance_test(
    result_a: EvaluationResult,
    result_b: EvaluationResult,
    metric: str = "iou",
) -> pd.DataFrame:
    """Wilcoxon signed-rank test between two models per class.

    Tests the null hypothesis that the paired per-sample metric
    distributions have equal medians.

    Parameters
    ----------
    result_a, result_b : EvaluationResult
        Results from two models evaluated on the **same** dataset
        (same order, same samples).
    metric : str
        One of ``"iou"``, ``"dice"``, ``"precision"``, ``"recall"``, ``"f1"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``class``, ``statistic``, ``p_value``, ``significant_005``.
    """
    from scipy.stats import wilcoxon

    attr = f"per_sample_{metric}"
    arr_a = getattr(result_a, attr)
    arr_b = getattr(result_b, attr)

    if arr_a.shape != arr_b.shape:
        raise ValueError(
            f"Shape mismatch: {arr_a.shape} vs {arr_b.shape}. "
            "Both models must be evaluated on the same dataset."
        )

    rows: list[dict[str, object]] = []
    for c, cname in enumerate(result_a.class_names):
        diff = arr_a[:, c] - arr_b[:, c]
        # Wilcoxon requires non-zero differences
        nonzero = diff[diff != 0]
        if len(nonzero) < 10:
            logger.warning(
                f"Class '{cname}': fewer than 10 non-zero differences "
                f"({len(nonzero)}), skipping Wilcoxon test."
            )
            rows.append({
                "class": cname,
                "statistic": np.nan,
                "p_value": np.nan,
                "significant_005": False,
            })
            continue

        stat, p_val = wilcoxon(arr_a[:, c], arr_b[:, c])
        rows.append({
            "class": cname,
            "statistic": float(stat),
            "p_value": float(p_val),
            "significant_005": p_val < 0.05,
        })

    return pd.DataFrame(rows)


# ======================================================================== #
#  Report export                                                            #
# ======================================================================== #

def generate_report_table(
    comparison_df: pd.DataFrame,
    format: str = "markdown",
) -> str:
    """Export comparison DataFrame as formatted table.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of :func:`compare_models`.
    format : str
        ``"markdown"`` or ``"latex"``.

    Returns
    -------
    str
        Formatted table string.
    """
    # Pivot to model × (class, metric)
    pivot = comparison_df.pivot_table(
        index=["model"],
        columns=["class", "metric"],
        values="mean",
    )

    if format == "latex":
        return pivot.to_latex(float_format="%.4f")
    return pivot.to_markdown(floatfmt=".4f")
