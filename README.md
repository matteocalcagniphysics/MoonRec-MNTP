# Lunar Segmentation Project: Panoptic FPN

This repository implements a **Panoptic Feature Pyramid Network (Panoptic FPN)** (based on Kirillov et al., [arXiv:1901.02446](https://arxiv.org/abs/1901.02446)) for joint semantic and instance segmentation of lunar features (such as impact craters, pit skylights, wrinkle ridges, and lobate scarps).

---

## 1. Panoptic FPN Architecture & File Structure

The project splits the panoptic segmentation pipeline into separate modules. Below is the file mapping and structural layout:

```
lunar_segmentation/
├── data/
│   └── datasets.py           # Custom dataset generating semantic targets & Mask R-CNN shapes
├── models/
│   ├── PAN1_backbone.py      # ResNet Multi-Scale Feature Extractor
│   ├── PAN2_layers.py        # FPN utility layers (Interpolate, Sum)
│   ├── PAN3_fpn.py           # Semantic Branch, Custom Mask R-CNN, & PanopticFPN wrapper
│   └── PAN4_factory.py       # Builder factory for components initialization
└── training/
    ├── trainer.py            # Unified trainer with PanopticTrainer & evaluation loops
    └── PAN_trainer.py        # Panoptic training script (Adam, BCE+Dice Loss, mAP metrics)
```

### Module Breakdown

#### 📂 [PAN1_backbone.py](file:///home/matteocalcagni/Desktop/MoonRec-MNTP/Moon-Recognition/lunar_segmentation/lunar_segmentation/models/PAN1_backbone.py)
* **`SequentialMultiOutput`**: A subclass of `nn.Sequential` that returns intermediate feature map outputs of all layers as a tuple during the forward pass.
* **`ResNetFeatureMapsExtractor`**: Wraps a standard torchvision ResNet (e.g. ResNet-18). It groups the stem (conv1, bn1, relu, maxpool) and the four subsequent ResNet layer blocks to extract multi-resolution maps. Supports a dictionary output toggle `out_mask_rcnn` specifically for compatibility with torchvision Mask R-CNN heads.

#### 📂 [PAN2_layers.py](file:///home/matteocalcagni/Desktop/MoonRec-MNTP/Moon-Recognition/lunar_segmentation/lunar_segmentation/models/PAN2_layers.py)
* **`ModulizedFunction`**: Wrapper converting standard functions (via `functools.partial`) into PyTorch `nn.Module` objects.
* **`Interpolate`**: Subclass of `ModulizedFunction` performing bilinear interpolation (via `F.interpolate`) to upsample features.
* **`Sum`**: A simple layer that computes the element-wise sum of a list of input tensors.

#### 📂 [PAN3_fpn.py](file:///home/matteocalcagni/Desktop/MoonRec-MNTP/Moon-Recognition/lunar_segmentation/lunar_segmentation/models/PAN3_fpn.py)
* **`Parallel`**: Runs a single input across multiple modules, or maps lists of inputs element-wise across a list of modules in parallel.
* **`SemanticBranch`**: Implements the semantic segmentation head. It projects varying backbone feature maps to a standard depth (`hidden_channels = 256`), upsamples them iteratively to align resolutions, sums them, and outputs predictions for the target number of classes.
* **`CustomMaskRCNNHeads`**: Houses torchvision's `AnchorGenerator`, `RegionProposalNetwork` (RPN), `MultiScaleRoIAlign` pooling, and `RoIHeads` to detect bounding boxes, predict class scores, and construct binary mask outlines.
* **`PanopticFPN`**: The main model wrapper. It uses torchvision's `FeaturePyramidNetwork` as a standardizing FPN bridge to transform raw ResNet outputs (varying channels) into uniform `256`-channel inputs for the instance head, while routing raw outputs directly to the semantic head.

#### 📂 [PAN4_factory.py](file:///home/matteocalcagni/Desktop/MoonRec-MNTP/Moon-Recognition/lunar_segmentation/lunar_segmentation/models/PAN4_factory.py)
* **`build_models`**: Querying ResNet backbone dimensions dynamically via a dry-run pass (`_get_shapes`), it configures and constructs the `ResNetFeatureMapsExtractor`, `SemanticBranch` (with interpolation aligned to input size), and the `CustomMaskRCNNHeads` instance branch.

#### 📂 [PAN_trainer.py](file:///home/matteocalcagni/Desktop/MoonRec-MNTP/Moon-Recognition/lunar_segmentation/lunar_segmentation/training/PAN_trainer.py)
* The entry-point script to load index files, initialize the combined `PanopticFPN` model, configure the target metrics (`MeanAveragePrecision` for instance, `BCE + Dice` loss for semantic), and execute training epochs using `PanopticTrainer`.

---

## 2. Key Data Mappings & Core Fixes

To correctly utilize the data from the local `data` directory, the following data flow modifications were implemented:

1. **Semantic target labeling alignment**:
   Background pixels (all channels empty) are mapped to index `0`. The 7 foreground classes are shifted to indices `1` to `7` (total `num_classes = 8`). This resolves the bug where `np.argmax(mask)` incorrectly labeled empty background as class `0` (`impact_crater`).
2. **Metric evaluation crash resolution**:
   In validation, `PanopticTrainer` converts the single-channel `[Batch, Height, Width]` class map into a one-hot `[Batch, 8, Height, Width]` tensor. It slices off the background channel (index `0`), providing matching 7-channel tensors to `multilabel_metrics` matching the `CLASS_NAMES` length.

---

## 3. Proposed Workload Division

| Member Name | Target Branch | Primary Responsibilities (Package Modules) |
| :--- | :--- | :--- |
| **Tina Gonzati** | `feature/baseline-cv` | **Data & Baseline Pipeline:** Develop `data/dataset.py` to load `.npz` tiles and masks, and `data/preprocessing.py`. Implement the baseline computer vision model using thresholding, morphology, and connected components. |
| **Matteo Calcagni** | `feature/unet-model` | **Semantic Segmentation:** Develop the U-Net architecture in `models/unet.py` and the training loop in `training/train_unet.py`. Handle semantic evaluation metrics (Dice/IoU). |
| **Pasquale Andreacchio** | `feature/mask-rcnn-model` | **Instance Segmentation:** Develop the Mask R-CNN architecture in `models/mask_rcnn.py` and its training loop in `training/train_mask_rcnn.py`. Focus on extracting instances for craters and pits. |
| **Nicola Lavarda** | `feature/visualization-report` | **Inference & Evaluation:** Develop `inference/predictor.py`, `visualization/overlays.py`, and `utils/metrics.py`. Manage dataset QA, plot overlays (raw, mask, prediction), and compile the final model comparison for the report. |
