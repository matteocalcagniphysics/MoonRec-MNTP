# Evaluation Module

This module provides a comprehensive framework for evaluating lunar feature segmentation models. It supports both pixel-level semantic metrics (IoU, Dice) and instance-level metrics (COCO-style AP), along with statistical significance testing.

## Key Features

- **Pixel-level Metrics**: Differentiable-safe implementation of IoU, Dice, Precision, Recall, and F1-score.
- **Instance-level Metrics**: Average Precision (AP) calculation for Mask R-CNN outputs.
- **Model Comparison**: Automated orchestration to evaluate multiple models on a dataset.
- **Statistical Significance**: Wilcoxon signed-rank tests and bootstrap confidence intervals for robust model comparison.
- **Synthetic Data**: Utilities to generate realistic dummy data for pipeline validation.

## File Overview

- `metrics.py`: Core pixel-level segmentation metrics operating on PyTorch tensors.
- `comparison.py`: High-level orchestration for evaluating models and running statistical tests.
- `instance_metrics.py`: COCO-style evaluation (AP/mAP) for instance segmentation (Mask R-CNN).
- `protocols.py`: Structural contracts (PEP 544 Protocols) defining the model interface.
- `synthetic.py`: Synthetic lunar imagery generators and mock models for testing.

## Quickstart

### Evaluating a Model
To evaluate a model, wrap it in a `SemanticModelAdapter` and use the `evaluate_model` function:

```python
from lunar_segmentation.evaluation.protocols import SemanticModelAdapter
from lunar_segmentation.evaluation.comparison import evaluate_model

# Wrap your PyTorch model
adapter = SemanticModelAdapter(model=my_unet, model_name="UNet-v1")

# Run evaluation on a DataLoader
results = evaluate_model(adapter, val_loader, class_names=["crater", "rille"])

print(f"Mean IoU: {results.mean_iou:.4f}")
```

### Statistical Comparison
Compare two models to see if performance differences are statistically significant:

```python
from lunar_segmentation.evaluation.comparison import significance_test

# result_a and result_b are EvaluationResult objects from evaluate_model
report = significance_test(result_a, result_b, metric="iou")
print(report)
```

## Testing & Demos
You can run a full demonstration of the evaluation pipeline using synthetic data:

```bash
python -m lunar_segmentation.evaluation.synthetic
```
