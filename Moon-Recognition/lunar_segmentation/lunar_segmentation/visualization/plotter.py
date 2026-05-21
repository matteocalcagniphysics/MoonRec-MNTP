import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class colour palette — one colour per geological class
# ---------------------------------------------------------------------------
CLASS_COLORS = [
    [1.0, 0.2, 0.2],   # impact_crater          red
    [0.2, 0.8, 1.0],   # pit_skylight           cyan
    [0.2, 1.0, 0.4],   # wrinkle_ridge          green
    [1.0, 0.8, 0.2],   # lobate_scarp           yellow
    [0.8, 0.2, 1.0],   # irregular_mare_patch   purple
    [1.0, 0.5, 0.0],   # apollo_site            orange
    [0.2, 0.4, 1.0],   # candidate_rille        blue
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_2d_gray(img: np.ndarray) -> np.ndarray:
    """Accept (H, W) or (C, H, W) and return a 2D grayscale array."""
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[0] in (1, 3):
            return img[0]
        if img.shape[2] in (1, 3):
            return img[:, :, 0]
    raise ValueError(f"Cannot convert shape {img.shape} to 2D grayscale")


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def _make_combined_mask(mask: np.ndarray) -> np.ndarray:
    """
    Blend all class binary maps into a single RGB image.

    Parameters
    ----------
    mask : np.ndarray, shape (C, H, W)

    Returns
    -------
    np.ndarray, shape (H, W, 3), values in [0, 1]
    """
    C, H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(min(C, len(CLASS_COLORS))):
        for ch in range(3):
            rgb[:, :, ch] += mask[c] * CLASS_COLORS[c][ch]
    return np.clip(rgb, 0, 1)


# ---------------------------------------------------------------------------
# Existing function — unchanged
# ---------------------------------------------------------------------------

def generate_pretty_image(gray_img: np.ndarray, prob_cube: np.ndarray,
                          class_names: list, output_path: Path,
                          threshold: float = 0.5):
    """
    Generates a grid of images showing the original WAC crop and overlays of
    each class.  Uses a more aggressive relative thresholding to ensure
    features are visible even if the model is under-confident.
    """
    gray_img = _ensure_2d_gray(gray_img)
    n = len(class_names)
    ncols = 3
    nrows = (n // ncols) + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    axes[0].imshow(gray_img, cmap='gray')
    axes[0].set_title('Original WAC Crop')
    axes[0].axis('off')

    for i, name in enumerate(class_names, start=1):
        axes[i].imshow(gray_img, cmap='gray')
        current_prob = prob_cube[i - 1]
        max_p = np.max(current_prob)
        effective_threshold = max(threshold, max_p * 0.8)
        if max_p < 0.4:
            sorted_probs = np.sort(current_prob.flatten())
            effective_threshold = sorted_probs[-int(current_prob.size * 0.001)]
            logger.info(
                f"Low confidence for {name} (max {max_p:.2f}). "
                f"Using top 0.1% pixels. T={effective_threshold:.4f}"
            )
        else:
            logger.info(f"Visualizing {name} with threshold {effective_threshold:.4f}")
        mask = current_prob > effective_threshold
        if np.any(mask):
            axes[i].imshow(mask, cmap='inferno', alpha=0.6)
        axes[i].set_title(f"{name} (T={effective_threshold:.2f})")
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved visualization to {output_path}")


# ---------------------------------------------------------------------------
# NEW — Data exploration functions 
# ---------------------------------------------------------------------------

def plot_positive_pixel_distribution(index_csv: Path,
                                     output_path: Path | None = None):
    """
    Histogram of positive-pixel counts across all tiles.
    Reproduces the chart from lecture slide 21.

    Parameters
    ----------
    index_csv   : path to the index.csv file
    output_path : if given, saves the figure; otherwise shows interactively
    """
    df = pd.read_csv(index_csv)
    median_val = int(df['positive_pixels'].median())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df['positive_pixels'], bins=100, color='darkred', edgecolor='none')
    ax.axvline(median_val, color='white', linestyle='--', linewidth=1.5,
               label=f'median={median_val:,}')
    ax.set_xlabel('positive_pixels')
    ax.set_ylabel('tile count')
    ax.set_title('Distribution of positive pixels per Marius Hills tile')
    ax.legend()

    stats = (
        f"n={len(df):,}  "
        f"median={median_val:,}  "
        f"min={int(df['positive_pixels'].min()):,}  "
        f"max={int(df['positive_pixels'].max()):,}"
    )
    ax.text(0.02, 0.97, stats, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved distribution plot to {output_path}")
    else:
        plt.show()


def plot_spatial_coverage_heatmap(index_csv: Path,
                                  output_path: Path | None = None):
    """
    2-D heatmap of positive-pixel counts mapped to tile (row, col) coordinates.
    Reproduces the spatial coverage map from lecture slide 24.

    Parameters
    ----------
    index_csv   : path to the index.csv file
    output_path : if given, saves the figure; otherwise shows interactively
    """
    df = pd.read_csv(index_csv)
    row_max = df['row'].max() + 1
    col_max = df['col'].max() + 1
    grid = np.zeros((row_max, col_max), dtype=np.float32)
    for _, r in df.iterrows():
        grid[int(r['row']), int(r['col'])] = r['positive_pixels']

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(grid, aspect='auto', cmap='plasma')
    plt.colorbar(im, ax=ax, label='positive pixels')
    ax.set_xlabel('tile column index')
    ax.set_ylabel('tile row index')
    ax.set_title('Spatial map of positive_pixels in index.csv')

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved spatial heatmap to {output_path}")
    else:
        plt.show()




def plot_combined_mask_overlay(image: np.ndarray, mask: np.ndarray,
                               class_names: list,
                               tile_name: str = '',
                               output_path: Path | None = None):
    """
    Three-panel figure: green-tinted image | combined colour mask | overlay.
   

    Parameters
    ----------
    image       : np.ndarray (C, H, W) — raw tile image
    mask        : np.ndarray (7, H, W) — binary class masks
    class_names : list of 7 class name strings
    tile_name   : optional label shown in the figure title
    output_path : if given, saves the figure; otherwise shows interactively
    """
    gray = _normalize(_ensure_2d_gray(image))

    # Green-tinted image (one channel → G only)
    img_green = np.zeros((*gray.shape, 3), dtype=np.float32)
    img_green[:, :, 1] = gray

    combined = _make_combined_mask(mask)

    # Legend — only classes that have at least one active pixel
    patches = [
        mpatches.Patch(color=CLASS_COLORS[c], label=class_names[c])
        for c in range(len(class_names))
        if mask[c].sum() > 0
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_green)
    axes[0].set_title('Lunar image tile')
    axes[0].axis('off')

    axes[1].imshow(combined)
    axes[1].set_title('Combined mask')
    axes[1].axis('off')

    axes[2].imshow(img_green)
    axes[2].imshow(combined, alpha=0.5)
    axes[2].set_title('Image + mask overlay')
    axes[2].axis('off')

    if patches:
        fig.legend(handles=patches, loc='lower center',
                   ncol=len(patches), fontsize=9,
                   bbox_to_anchor=(0.5, -0.04))

    plt.suptitle(tile_name or 'Combined mask overlay', fontsize=12)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved combined mask overlay to {output_path}")
    else:
        plt.show()


def plot_single_channel_overlay(image: np.ndarray, mask: np.ndarray,
                                channel: int, class_names: list,
                                tile_name: str = '',
                                output_path: Path | None = None):
    """
    Three-panel figure: raw image | single-channel mask | colour overlay.
   

    Parameters
    ----------
    image       : np.ndarray (C, H, W)
    mask        : np.ndarray (7, H, W)
    channel     : which class channel to display (0-6)
    class_names : list of class name strings
    tile_name   : optional label for the figure title
    output_path : if given, saves; otherwise shows interactively
    """
    gray = _normalize(_ensure_2d_gray(image))
    color = CLASS_COLORS[channel]

    # Coloured overlay for the selected channel
    overlay = np.stack([gray] * 3, axis=-1)
    color_layer = np.zeros_like(overlay)
    for ch in range(3):
        color_layer[:, :, ch] = mask[channel] * color[ch]
    blended = np.clip(overlay * 0.7 + color_layer * 0.6, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title('Raw image')
    axes[0].axis('off')

    axes[1].imshow(mask[channel], cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Mask channel {channel}\n({class_names[channel]})')
    axes[1].axis('off')

    axes[2].imshow(blended)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.suptitle(
        f'{tile_name} | mask channel {channel} — {class_names[channel]}',
        fontsize=11
    )
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved single-channel overlay to {output_path}")
    else:
        plt.show()


def plot_real_vs_prediction(image: np.ndarray, mask: np.ndarray,
                            pred: np.ndarray, class_names: list,
                            threshold: float = 0.35,
                            tile_name: str = '',
                            output_path: Path | None = None):
    """
    Side-by-side combined masks: ground truth vs model prediction.

    Parameters
    ----------
    image       : np.ndarray (C, H, W)
    mask        : np.ndarray (7, H, W) binary ground truth
    pred        : np.ndarray (7, H, W) sigmoid probabilities from the model
    class_names : list of class name strings
    threshold   : probability threshold for binarising predictions
    tile_name   : optional label for the figure title
    output_path : if given, saves; otherwise shows interactively
    """
    gray = _normalize(_ensure_2d_gray(image))
    img_green = np.zeros((*gray.shape, 3), dtype=np.float32)
    img_green[:, :, 1] = gray

    real_combined = _make_combined_mask(mask)
    pred_combined = _make_combined_mask((pred > threshold).astype(np.float32))

    patches = [
        mpatches.Patch(color=CLASS_COLORS[c], label=class_names[c])
        for c in range(len(class_names))
        if mask[c].sum() > 0 or (pred[c] > threshold).any()
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_green)
    axes[0].set_title('Input image')
    axes[0].axis('off')

    axes[1].imshow(real_combined)
    axes[1].set_title('Ground truth')
    axes[1].axis('off')

    axes[2].imshow(pred_combined)
    axes[2].set_title(f'Prediction (threshold={threshold})')
    axes[2].axis('off')

    if patches:
        fig.legend(handles=patches, loc='lower center',
                   ncol=len(patches), fontsize=9,
                   bbox_to_anchor=(0.5, -0.04))

    plt.suptitle(tile_name or 'Ground truth vs prediction', fontsize=12)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved real-vs-prediction plot to {output_path}")
    else:
        plt.show()


def plot_iou_per_class(pred: np.ndarray, mask: np.ndarray,
                       class_names: list, threshold: float = 0.35,
                       output_path: Path | None = None):
    """
    Horizontal bar chart of IoU per class.

    Parameters
    ----------
    pred        : np.ndarray (7, H, W) sigmoid probabilities
    mask        : np.ndarray (7, H, W) binary ground truth
    class_names : list of class name strings
    threshold   : probability threshold for binarising predictions
    output_path : if given, saves; otherwise shows interactively
    """
    ious, pixel_counts = [], []
    for c in range(len(class_names)):
        p = (pred[c] > threshold).astype(np.float32)
        t = mask[c].astype(np.float32)
        intersection = (p * t).sum()
        union = p.sum() + t.sum() - intersection
        ious.append(float(intersection / (union + 1e-6)))
        pixel_counts.append(int(t.sum()))

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [CLASS_COLORS[c] for c in range(len(class_names))]
    bars = ax.barh(class_names, ious, color=colors, edgecolor='none')

    for bar, iou, px in zip(bars, ious, pixel_counts):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{iou:.3f}  ({px} px)',
                va='center', fontsize=9)

    ax.set_xlim(0, 1.15)
    ax.set_xlabel('IoU')
    ax.set_title(f'IoU per class (threshold={threshold})')
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved IoU chart to {output_path}")
    else:
        plt.show()


def plot_threshold_baseline(image: np.ndarray, mask: np.ndarray,
                            threshold: float = 0.3,
                            class_names: list | None = None,
                            output_path: Path | None = None):
    """
    Compare a simple dark-pixel threshold against the real impact_crater mask.
    Useful as a sanity check before training — if the baseline IoU is already
    decent, the images are informative; if it is near zero, the mask encodes
    something the brightness alone cannot predict.

    Parameters
    ----------
    image       : np.ndarray (C, H, W)
    mask        : np.ndarray (7, H, W), channel 0 = impact_crater
    threshold   : pixels darker than this (after normalisation) are flagged
    class_names : optional list for labelling
    output_path : if given, saves; otherwise shows interactively
    """
    gray = _normalize(_ensure_2d_gray(image))
    pred_baseline = (gray < threshold).astype(np.float32)
    real_crater = mask[0].astype(np.float32)

    tp = (pred_baseline * real_crater).sum()
    fp = (pred_baseline * (1 - real_crater)).sum()
    fn = ((1 - pred_baseline) * real_crater).sum()
    iou = float(tp / (tp + fp + fn + 1e-6))

    # TP=green, FP=red, FN=blue overlay
    ov = np.zeros((*gray.shape, 3), dtype=np.float32)
    ov[:, :, 1] = pred_baseline * real_crater          # TP green
    ov[:, :, 0] = pred_baseline * (1 - real_crater)    # FP red
    ov[:, :, 2] = (1 - pred_baseline) * real_crater    # FN blue

    class_label = class_names[0] if class_names else 'impact_crater'

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(gray, cmap='gray');         axes[0].set_title('Input image');          axes[0].axis('off')
    axes[1].imshow(real_crater, cmap='gray');  axes[1].set_title(f'Real mask\n({class_label})'); axes[1].axis('off')
    axes[2].imshow(pred_baseline, cmap='gray');axes[2].set_title(f'Threshold < {threshold}');    axes[2].axis('off')
    axes[3].imshow(gray, cmap='gray', alpha=0.5)
    axes[3].imshow(ov, alpha=0.7)
    axes[3].set_title(f'TP=green FP=red FN=blue\nIoU={iou:.3f}')
    axes[3].axis('off')

    plt.suptitle('Threshold baseline vs ground truth', fontsize=12)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved threshold baseline plot to {output_path}")
    else:
        plt.show()

    return {'iou': iou, 'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}


def plot_augmentation_check(image: np.ndarray, mask: np.ndarray,
                            aug_image: np.ndarray, aug_mask: np.ndarray,
                            channel: int = 0, class_names: list | None = None,
                            output_path: Path | None = None):
    """
    Visual sanity check that the same geometric transform was applied to both
    image and mask (as required by the slides).

    Parameters
    ----------
    image     / mask     : original (C, H, W) / (7, H, W)
    aug_image / aug_mask : augmented versions
    channel              : which mask channel to display (default 0 = craters)
    class_names          : optional list for labelling
    output_path          : if given, saves; otherwise shows interactively
    """
    gray     = _normalize(_ensure_2d_gray(image))
    gray_aug = _normalize(_ensure_2d_gray(aug_image))
    class_label = class_names[channel] if class_names else f'channel {channel}'

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(mask[channel], cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Original mask\n({class_label})')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(gray_aug, cmap='gray')
    axes[1, 0].set_title('Augmented image')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(aug_mask[channel], cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Augmented mask\n(must match transform)')
    axes[1, 1].axis('off')

    plt.suptitle('Augmentation check — image and mask must share the same transform',
                 fontsize=11)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved augmentation check to {output_path}")
    else:
        plt.show()
