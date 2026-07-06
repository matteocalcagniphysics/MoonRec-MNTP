"""Evaluation-specific visualisation for segmentation results.

Produces publication-quality plots for:
- 4-panel prediction analysis (Raw / GT / Prediction / Error map)
- Per-class comparison grids
- Metrics summary bar charts
- Cross-model comparison charts
- Threshold sensitivity curves

All methods return ``(fig, axes)`` tuples so callers can customise
further in notebooks before saving.

Typical usage
-------------
>>> from lunar_segmentation.visualization.eval_plotter import SegmentationVisualizer
>>> viz = SegmentationVisualizer(class_names=CLASS_NAMES)
>>> fig, axes = viz.plot_prediction_panel(image, gt_mask, pred_mask)
>>> fig.savefig("panel.pdf")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ======================================================================== #
#  Style defaults                                                           #
# ======================================================================== #

_DEFAULT_STYLE: dict[str, Any] = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
}

# Error-map colours (literature standard)
_FP_COLOR = np.array([0.90, 0.20, 0.20, 0.65])  # Red — False Positive
_FN_COLOR = np.array([0.20, 0.40, 0.90, 0.65])  # Blue — False Negative
_TP_COLOR = np.array([0.20, 0.85, 0.40, 0.40])  # Green — True Positive

# Per-class colour palette (categorical, colourblind-safe inspired)
_CLASS_PALETTE = [
    "#e6194B",  # impact_crater
    "#f58231",  # pit_skylight
    "#ffe119",  # wrinkle_ridge
    "#3cb44b",  # lobate_scarp
    "#42d4f4",  # irregular_mare_patch
    "#4363d8",  # apollo_site
    "#911eb4",  # candidate_rille
]


# ======================================================================== #
#  Helpers                                                                  #
# ======================================================================== #

def _to_2d(arr: np.ndarray) -> np.ndarray:
    """Accept (H, W), (1, H, W), or (C, H, W) and return 2-D grayscale."""
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3):
            return arr[0]
        if arr.shape[2] in (1, 3):
            return arr[:, :, 0]
    raise ValueError(f"Cannot convert shape {arr.shape} to 2-D grayscale.")


def _build_error_rgba(
    gt: np.ndarray,
    pred: np.ndarray,
) -> np.ndarray:
    """Build an RGBA error overlay from binary GT and Pred masks.

    Parameters
    ----------
    gt : np.ndarray
        ``(H, W)`` binary ground-truth.
    pred : np.ndarray
        ``(H, W)`` binary prediction.

    Returns
    -------
    np.ndarray
        ``(H, W, 4)`` RGBA overlay.
    """
    h, w = gt.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)

    tp = (gt > 0) & (pred > 0)
    fp = (gt == 0) & (pred > 0)
    fn = (gt > 0) & (pred == 0)

    overlay[tp] = _TP_COLOR
    overlay[fp] = _FP_COLOR
    overlay[fn] = _FN_COLOR

    return overlay


# ======================================================================== #
#  SegmentationVisualizer                                                   #
# ======================================================================== #

class SegmentationVisualizer:
    """Publication-quality visualiser for segmentation evaluation.

    Parameters
    ----------
    class_names : list[str]
        Ordered names of the segmentation classes.
    style : dict, optional
        Matplotlib rcParams overrides.  Merged with dark defaults.
    dpi : int
        Default DPI for saved figures.
    """

    def __init__(
        self,
        class_names: list[str],
        style: dict[str, Any] | None = None,
        dpi: int = 150,
    ) -> None:
        self.class_names = class_names
        self.dpi = dpi
        self._style = {**_DEFAULT_STYLE, **(style or {})}

    def _apply_style(self) -> None:
        """Push style to matplotlib."""
        plt.rcParams.update(self._style)

    # ------------------------------------------------------------------ #
    #  4-panel: Raw / GT / Pred / Error                                     #
    # ------------------------------------------------------------------ #

    def plot_prediction_panel(
        self,
        image: np.ndarray,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        class_idx: int = 0,
        title: str | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Four-panel view for a single class channel.

        Panels: **Raw image** | **GT overlay** | **Prediction overlay** |
        **Error map** (TP=green, FP=red, FN=blue).

        Parameters
        ----------
        image : np.ndarray
            ``(C, H, W)`` or ``(H, W)`` input image.
        gt_mask : np.ndarray
            ``(C, H, W)`` or ``(H, W)`` ground-truth mask.
        pred_mask : np.ndarray
            ``(C, H, W)`` or ``(H, W)`` predicted mask.
        class_idx : int
            Which class channel to visualise (ignored if masks are 2-D).
        title : str, optional
            Figure suptitle.

        Returns
        -------
        tuple[plt.Figure, np.ndarray]
            Matplotlib figure and axes array.
        """
        self._apply_style()
        gray = _to_2d(image)

        gt_2d = gt_mask[class_idx] if gt_mask.ndim == 3 else gt_mask
        pred_2d = pred_mask[class_idx] if pred_mask.ndim == 3 else pred_mask

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Panel 1 — Raw
        axes[0].imshow(gray, cmap="gray")
        axes[0].set_title("Raw Image")
        axes[0].axis("off")

        # Panel 2 — GT overlay
        axes[1].imshow(gray, cmap="gray")
        gt_rgba = np.zeros((*gray.shape, 4), dtype=np.float32)
        gt_rgba[gt_2d > 0] = [0.0, 0.9, 0.4, 0.55]
        axes[1].imshow(gt_rgba)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        # Panel 3 — Prediction overlay
        axes[2].imshow(gray, cmap="gray")
        pred_rgba = np.zeros((*gray.shape, 4), dtype=np.float32)
        pred_rgba[pred_2d > 0] = [0.2, 0.6, 1.0, 0.55]
        axes[2].imshow(pred_rgba)
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        # Panel 4 — Error map
        axes[3].imshow(gray, cmap="gray")
        error_overlay = _build_error_rgba(
            (gt_2d > 0).astype(np.float32),
            (pred_2d > 0).astype(np.float32),
        )
        axes[3].imshow(error_overlay)
        axes[3].set_title("Error Map (TP/FP/FN)")
        axes[3].axis("off")

        # Legend for error map
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=_TP_COLOR[:3], alpha=0.65, label="TP"),
            Patch(facecolor=_FP_COLOR[:3], alpha=0.65, label="FP"),
            Patch(facecolor=_FN_COLOR[:3], alpha=0.65, label="FN"),
        ]
        axes[3].legend(
            handles=legend_elements,
            loc="lower right",
            fontsize=8,
            framealpha=0.7,
        )

        class_label = (
            self.class_names[class_idx]
            if class_idx < len(self.class_names)
            else f"class_{class_idx}"
        )
        fig.suptitle(
            title or f"Prediction Analysis — {class_label}",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        return fig, axes

    # ------------------------------------------------------------------ #
    #  Per-class comparison grid                                            #
    # ------------------------------------------------------------------ #

    def plot_class_comparison(
        self,
        image: np.ndarray,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        title: str | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Grid showing GT / Pred / Error for each class.

        Parameters
        ----------
        image : np.ndarray
            ``(C, H, W)`` or ``(H, W)`` input image.
        gt_mask : np.ndarray
            ``(N_classes, H, W)`` ground-truth.
        pred_mask : np.ndarray
            ``(N_classes, H, W)`` predictions.
        title : str, optional
            Figure suptitle.

        Returns
        -------
        tuple[plt.Figure, np.ndarray]
            Figure and axes.
        """
        self._apply_style()
        gray = _to_2d(image)
        n_classes = gt_mask.shape[0]

        fig, axes = plt.subplots(n_classes, 3, figsize=(15, 4 * n_classes))
        if n_classes == 1:
            axes = axes[np.newaxis, :]

        col_titles = ["Ground Truth", "Prediction", "Error Map"]
        for col, ct in enumerate(col_titles):
            axes[0, col].set_title(ct, fontsize=12, fontweight="bold")

        for c in range(n_classes):
            cname = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
            gt_2d = gt_mask[c]
            pred_2d = pred_mask[c]

            # GT
            axes[c, 0].imshow(gray, cmap="gray")
            gt_rgba = np.zeros((*gray.shape, 4), dtype=np.float32)
            gt_rgba[gt_2d > 0] = [0.0, 0.9, 0.4, 0.55]
            axes[c, 0].imshow(gt_rgba)
            axes[c, 0].set_ylabel(cname, fontsize=11, rotation=0, labelpad=80, va="center")
            axes[c, 0].set_xticks([])
            axes[c, 0].set_yticks([])

            # Pred
            axes[c, 1].imshow(gray, cmap="gray")
            pred_rgba = np.zeros((*gray.shape, 4), dtype=np.float32)
            pred_rgba[pred_2d > 0] = [0.2, 0.6, 1.0, 0.55]
            axes[c, 1].imshow(pred_rgba)
            axes[c, 1].axis("off")

            # Error
            axes[c, 2].imshow(gray, cmap="gray")
            error_overlay = _build_error_rgba(
                (gt_2d > 0).astype(np.float32),
                (pred_2d > 0).astype(np.float32),
            )
            axes[c, 2].imshow(error_overlay)
            axes[c, 2].axis("off")

        fig.suptitle(
            title or "Per-Class Segmentation Comparison",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        fig.tight_layout()
        return fig, axes

    # ------------------------------------------------------------------ #
    #  Batch predictions                                                    #
    # ------------------------------------------------------------------ #

    def plot_batch_predictions(
        self,
        images: np.ndarray,
        gt_masks: np.ndarray,
        pred_masks: np.ndarray,
        class_idx: int = 0,
        max_samples: int = 8,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot prediction panels for multiple samples in a batch.

        Parameters
        ----------
        images : np.ndarray
            ``(B, C, H, W)`` batch of images.
        gt_masks : np.ndarray
            ``(B, N_classes, H, W)`` batch of GT masks.
        pred_masks : np.ndarray
            ``(B, N_classes, H, W)`` batch of predicted masks.
        class_idx : int
            Which class to visualise.
        max_samples : int
            Maximum number of samples to show.

        Returns
        -------
        tuple[plt.Figure, np.ndarray]
            Figure and axes.
        """
        self._apply_style()
        n = min(images.shape[0], max_samples)
        fig, axes = plt.subplots(n, 4, figsize=(20, 4 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        col_titles = ["Raw", "Ground Truth", "Prediction", "Error Map"]
        for col, ct in enumerate(col_titles):
            axes[0, col].set_title(ct, fontsize=11, fontweight="bold")

        for i in range(n):
            gray = _to_2d(images[i])
            gt_2d = gt_masks[i, class_idx]
            pred_2d = pred_masks[i, class_idx]

            axes[i, 0].imshow(gray, cmap="gray")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(gray, cmap="gray")
            gt_rgba = np.zeros((*gray.shape, 4), dtype=np.float32)
            gt_rgba[gt_2d > 0] = [0.0, 0.9, 0.4, 0.55]
            axes[i, 1].imshow(gt_rgba)
            axes[i, 1].axis("off")

            axes[i, 2].imshow(gray, cmap="gray")
            pred_rgba = np.zeros((*gray.shape, 4), dtype=np.float32)
            pred_rgba[pred_2d > 0] = [0.2, 0.6, 1.0, 0.55]
            axes[i, 2].imshow(pred_rgba)
            axes[i, 2].axis("off")

            axes[i, 3].imshow(gray, cmap="gray")
            error_overlay = _build_error_rgba(
                (gt_2d > 0).astype(np.float32),
                (pred_2d > 0).astype(np.float32),
            )
            axes[i, 3].imshow(error_overlay)
            axes[i, 3].axis("off")

        fig.tight_layout()
        return fig, axes

    # ------------------------------------------------------------------ #
    #  Metrics summary                                                      #
    # ------------------------------------------------------------------ #

    def plot_metrics_summary(
        self,
        metrics_df: pd.DataFrame,
        title: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Grouped bar chart of metrics per class.

        Parameters
        ----------
        metrics_df : pd.DataFrame
            Must contain columns ``class``, ``metric``, ``mean``
            (optionally ``std``).
        title : str, optional
            Figure title.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
        """
        self._apply_style()
        metrics = metrics_df["metric"].unique()
        classes = metrics_df["class"].unique()
        n_metrics = len(metrics)
        n_classes = len(classes)

        fig, ax = plt.subplots(figsize=(max(10, n_classes * 1.5), 6))

        x = np.arange(n_classes)
        width = 0.8 / n_metrics

        colors = plt.cm.viridis(np.linspace(0.2, 0.85, n_metrics))

        for j, metric in enumerate(metrics):
            subset = metrics_df[metrics_df["metric"] == metric]
            values = [
                float(subset[subset["class"] == c]["mean"].values[0])
                if len(subset[subset["class"] == c]) > 0
                else 0.0
                for c in classes
            ]
            yerr = None
            if "std" in metrics_df.columns:
                yerr = [
                    float(subset[subset["class"] == c]["std"].values[0])
                    if len(subset[subset["class"] == c]) > 0
                    else 0.0
                    for c in classes
                ]
            ax.bar(
                x + j * width - (n_metrics - 1) * width / 2,
                values,
                width,
                yerr=yerr,
                label=metric,
                color=colors[j],
                edgecolor="white",
                linewidth=0.5,
                capsize=2,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_title(title or "Metrics Summary", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------ #
    #  Model comparison                                                     #
    # ------------------------------------------------------------------ #

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric: str = "IoU",
        title: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Grouped bar chart comparing models on a single metric.

        Parameters
        ----------
        comparison_df : pd.DataFrame
            Output of :func:`compare_models` — must contain ``model``,
            ``class``, ``metric``, ``mean``, and optionally
            ``ci_lower``, ``ci_upper``.
        metric : str
            Which metric to plot (e.g. ``"IoU"``, ``"Dice"``).
        title : str, optional
            Figure title.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
        """
        self._apply_style()
        subset = comparison_df[comparison_df["metric"] == metric].copy()
        models = subset["model"].unique()
        classes = subset["class"].unique()
        n_models = len(models)
        n_classes = len(classes)

        fig, ax = plt.subplots(figsize=(max(10, n_classes * 1.5), 6))
        x = np.arange(n_classes)
        width = 0.8 / n_models

        colors = plt.cm.plasma(np.linspace(0.15, 0.85, n_models))

        for j, model in enumerate(models):
            model_data = subset[subset["model"] == model]
            values = [
                float(model_data[model_data["class"] == c]["mean"].values[0])
                if len(model_data[model_data["class"] == c]) > 0
                else 0.0
                for c in classes
            ]
            yerr = None
            if "ci_lower" in subset.columns and "ci_upper" in subset.columns:
                yerr_lo = [
                    values[i] - float(model_data[model_data["class"] == c]["ci_lower"].values[0])
                    if len(model_data[model_data["class"] == c]) > 0
                    else 0.0
                    for i, c in enumerate(classes)
                ]
                yerr_hi = [
                    float(model_data[model_data["class"] == c]["ci_upper"].values[0]) - values[i]
                    if len(model_data[model_data["class"] == c]) > 0
                    else 0.0
                    for i, c in enumerate(classes)
                ]
                yerr = [yerr_lo, yerr_hi]

            ax.bar(
                x + j * width - (n_models - 1) * width / 2,
                values,
                width,
                yerr=yerr,
                label=model,
                color=colors[j],
                edgecolor="white",
                linewidth=0.5,
                capsize=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.set_title(
            title or f"Model Comparison — {metric}",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------ #
    #  Threshold sensitivity                                                #
    # ------------------------------------------------------------------ #

    def plot_threshold_sensitivity(
        self,
        sweep_result: dict[str, Any],
        title: str | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot IoU and Dice vs threshold for each class.

        Parameters
        ----------
        sweep_result : dict
            Output of :func:`threshold_sweep` — keys ``"thresholds"``,
            ``"iou"`` ``(T, C)``, ``"dice"`` ``(T, C)``.
        title : str, optional
            Figure suptitle.

        Returns
        -------
        tuple[plt.Figure, np.ndarray]
        """
        self._apply_style()
        thresholds = sweep_result["thresholds"].numpy()
        iou_arr = sweep_result["iou"].numpy()      # (T, C)
        dice_arr = sweep_result["dice"].numpy()     # (T, C)
        n_classes = iou_arr.shape[1]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = [
            _CLASS_PALETTE[i % len(_CLASS_PALETTE)]
            for i in range(n_classes)
        ]

        for c in range(n_classes):
            label = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
            axes[0].plot(thresholds, iou_arr[:, c], marker="o", markersize=4,
                         label=label, color=colors[c], linewidth=1.5)
            axes[1].plot(thresholds, dice_arr[:, c], marker="s", markersize=4,
                         label=label, color=colors[c], linewidth=1.5)

        for ax, ylabel in zip(axes, ["IoU", "Dice"]):
            ax.set_xlabel("Threshold", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3, linestyle="--")
            ax.legend(fontsize=8, loc="lower left")

        fig.suptitle(
            title or "Threshold Sensitivity Analysis",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        return fig, axes

    # ------------------------------------------------------------------ #
    #  Precision-Recall Curve                                               #
    # ------------------------------------------------------------------ #

    def plot_pr_curve(
        self,
        sweep_result: dict[str, Any],
        title: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot Precision-Recall curve for each class.

        Parameters
        ----------
        sweep_result : dict
            Output of threshold sweep. Must contain 'precision', 'recall' and 'thresholds'.
        title : str, optional
            Figure title.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
        """
        self._apply_style()
        precision = sweep_result["precision"].numpy()  # (T, C)
        recall = sweep_result["recall"].numpy()        # (T, C)
        n_classes = precision.shape[1]

        fig, ax = plt.subplots(figsize=(8, 6))

        colors = [
            _CLASS_PALETTE[i % len(_CLASS_PALETTE)]
            for i in range(n_classes)
        ]

        for c in range(n_classes):
            label = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
            
            # Sort by recall for plotting and AUC
            rec = recall[:, c]
            prec = precision[:, c]
            sort_idx = np.argsort(rec)
            rec_sorted = rec[sort_idx]
            prec_sorted = prec[sort_idx]
            
            # Compute AUC using trapezoidal rule
            pr_auc = np.trapz(prec_sorted, rec_sorted)
            
            ax.plot(rec_sorted, prec_sorted, marker="o", markersize=4,
                    label=f"{label} (AUC: {pr_auc:.3f})", color=colors[c], linewidth=1.5)

        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(fontsize=9, loc="lower left")

        fig.suptitle(
            title or "Precision-Recall Curve",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------ #
    #  Metric Distributions (Violin/Box Plot)                               #
    # ------------------------------------------------------------------ #

    def plot_metric_distributions(
        self,
        eval_result: Any, # EvaluationResult
        metric: str = "iou", # 'iou', 'dice', 'precision', 'recall', 'f1'
        title: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot violin plots of metric distributions per class.

        Parameters
        ----------
        eval_result : EvaluationResult
            Contains per-sample metrics.
        metric : str
            Which metric to plot ('iou', 'dice', 'precision', 'recall', 'f1').
        title : str, optional
            Figure title.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
        """
        self._apply_style()
        
        # Extract data
        metric = metric.lower()
        if metric == "iou":
            data = eval_result.per_sample_iou
        elif metric == "dice":
            data = eval_result.per_sample_dice
        elif metric == "precision":
            data = eval_result.per_sample_precision
        elif metric == "recall":
            data = eval_result.per_sample_recall
        elif metric == "f1":
            data = eval_result.per_sample_f1
        else:
            raise ValueError(f"Unknown metric {metric}")
            
        n_classes = data.shape[1] if data.ndim == 2 else 1
        
        fig, ax = plt.subplots(figsize=(max(8, n_classes * 1.5), 6))

        colors = [
            _CLASS_PALETTE[i % len(_CLASS_PALETTE)]
            for i in range(n_classes)
        ]
        
        # Prepare data for matplotlib violinplot
        plot_data = [data[:, c] for c in range(n_classes)] if n_classes > 1 else [data]
        
        # Filter out NaN/Inf if any
        plot_data = [d[np.isfinite(d)] for d in plot_data]
        # Avoid empty sequences
        plot_data = [d if len(d) > 0 else np.array([0.0]) for d in plot_data]
        
        parts = ax.violinplot(plot_data, showmeans=True, showmedians=False)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('white')
            pc.set_alpha(0.7)
            
        parts['cmeans'].set_color('white')
        parts['cbars'].set_color('white')
        parts['cmaxes'].set_color('white')
        parts['cmins'].set_color('white')

        ax.set_xticks(np.arange(1, n_classes + 1))
        ax.set_xticklabels(self.class_names[:n_classes], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(metric.upper() if metric in ['iou', 'f1'] else metric.capitalize(), fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_title(
            title or f"{metric.upper() if metric in ['iou', 'f1'] else metric.capitalize()} Distribution",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------ #
    #  Confidence Panel (Heatmap)                                           #
    # ------------------------------------------------------------------ #

    def plot_confidence_panel(
        self,
        image: np.ndarray,
        gt_mask: np.ndarray,
        prob_mask: np.ndarray,
        class_idx: int = 0,
        title: str | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Four-panel view emphasizing raw probabilities.

        Panels: **Raw image** | **GT overlay** | **Confidence Map** |
        **Uncertainty Map**.

        Parameters
        ----------
        image : np.ndarray
            ``(C, H, W)`` or ``(H, W)`` input image.
        gt_mask : np.ndarray
            ``(C, H, W)`` or ``(H, W)`` ground-truth mask.
        prob_mask : np.ndarray
            ``(C, H, W)`` or ``(H, W)`` raw predicted probabilities (0.0 to 1.0).
        class_idx : int
            Which class channel to visualise (ignored if masks are 2-D).
        title : str, optional
            Figure suptitle.

        Returns
        -------
        tuple[plt.Figure, np.ndarray]
            Matplotlib figure and axes array.
        """
        self._apply_style()
        gray = _to_2d(image)

        gt_2d = gt_mask[class_idx] if gt_mask.ndim == 3 else gt_mask
        prob_2d = prob_mask[class_idx] if prob_mask.ndim == 3 else prob_mask

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Panel 1 — Raw
        axes[0].imshow(gray, cmap="gray")
        axes[0].set_title("Raw Image")
        axes[0].axis("off")

        # Panel 2 — GT overlay
        axes[1].imshow(gray, cmap="gray")
        gt_rgba = np.zeros((*gray.shape, 4), dtype=np.float32)
        gt_rgba[gt_2d > 0] = [0.0, 0.9, 0.4, 0.55]
        axes[1].imshow(gt_rgba)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        # Panel 3 — Confidence Map (Heatmap)
        im3 = axes[2].imshow(prob_2d, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[2].set_title("Confidence Map (Probability)")
        axes[2].axis("off")
        fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        # Panel 4 — Uncertainty Map (closer to 0.5 is more uncertain)
        uncertainty = 1.0 - 2.0 * np.abs(prob_2d - 0.5)
        im4 = axes[3].imshow(uncertainty, cmap="magma", vmin=0.0, vmax=1.0)
        axes[3].set_title("Uncertainty Map")
        axes[3].axis("off")
        fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

        class_label = (
            self.class_names[class_idx]
            if class_idx < len(self.class_names)
            else f"class_{class_idx}"
        )
        fig.suptitle(
            title or f"Confidence Analysis — {class_label}",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        return fig, axes

    # ------------------------------------------------------------------ #
    #  Save utility                                                         #
    # ------------------------------------------------------------------ #

    def save_figure(
        self,
        fig: plt.Figure,
        path: str | Path,
        dpi: int | None = None,
        close: bool = True,
    ) -> Path:
        """Save figure to disk.

        Parameters
        ----------
        fig : plt.Figure
            Figure to save.
        path : str or Path
            Output path (supports .png, .pdf, .svg).
        dpi : int, optional
            Override default DPI.
        close : bool
            Close the figure after saving to free memory.

        Returns
        -------
        Path
            Resolved output path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi or self.dpi, bbox_inches="tight")
        logger.info(f"Saved figure to {path}")
        if close:
            plt.close(fig)
        return path
