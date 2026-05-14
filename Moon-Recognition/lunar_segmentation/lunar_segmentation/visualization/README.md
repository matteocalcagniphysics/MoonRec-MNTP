# Visualization Module

This module provides tools for generating high-quality visualizations of lunar segmentation results. It includes both general-purpose plotting utilities and a specialized evaluation plotter for detailed error analysis.

## Key Features

- **Publication-ready Plots**: Default "dark mode" aesthetic designed for presentations and reports.
- **Error Mapping**: Automated generation of TP/FP/FN error maps (Green/Red/Blue) to localize model failures.
- **Relative Thresholding**: Smart visualization that highlights features even when model confidence is low.
- **Analysis Panels**: 4-panel views showing Raw Image, Ground Truth, Prediction, and Error Map side-by-side.
- **Metric Charts**: Comparison bar charts and threshold sensitivity curves.

## File Overview

- `plotter.py`: Basic utilities for overlaying class probabilities on lunar imagery. Uses adaptive thresholding for better feature visibility.
- `eval_plotter.py`: Advanced `SegmentationVisualizer` class for comprehensive model performance analysis and metric visualization.

## Quickstart

### Creating a Prediction Analysis Panel
The `SegmentationVisualizer` is the main entry point for detailed analysis:

```python
from lunar_segmentation.visualization.eval_plotter import SegmentationVisualizer

# Initialize with your class names
viz = SegmentationVisualizer(class_names=["crater", "rille", "ridge"])

# Generate a 4-panel plot (Raw, GT, Prediction, Error Map)
fig, axes = viz.plot_prediction_panel(image, gt_mask, pred_mask, class_idx=0)
viz.save_figure(fig, "results/crater_analysis.png")
```

### Visualizing Class Probabilities
Use `plotter.py` for a quick overview of all predicted classes:

```python
from pathlib import Path
from lunar_segmentation.visualization.plotter import generate_pretty_image

generate_pretty_image(
    gray_img=image,
    prob_cube=predictions,
    class_names=CLASS_NAMES,
    output_path=Path("outputs/prediction_overlay.png")
)
```

## Styling
The module uses a custom dark theme by default. You can override matplotlib settings by passing a `style` dictionary to the `SegmentationVisualizer` constructor.
