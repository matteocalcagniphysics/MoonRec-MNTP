# Computational-Physics-2026

### **Proposed Workload Division**

| Member Name | Target Branch | Primary Responsibilities (Package Modules) |
| :--- | :--- | :--- |
| **Tina Gonzati** | `feature/baseline-cv` | **Data & Baseline Pipeline:** Develop `data/dataset.py` to load `.npz` tiles and masks, and `data/preprocessing.py`. Implement the baseline computer vision model using thresholding, morphology, and connected components. |
| **Matteo Calcagni** | `feature/unet-model` | **Semantic Segmentation:** Develop the U-Net architecture in `models/unet.py` and the training loop in `training/train_unet.py`. Handle semantic evaluation metrics (Dice/IoU). |
| **Nicola Lavarda** | `feature/mask-rcnn-model` | **Instance Segmentation:** Develop the Mask R-CNN architecture in `models/mask_rcnn.py` and its training loop in `training/train_mask_rcnn.py`. Focus on extracting instances for craters and pits. |
| **Pasquale Andreacchio** | `feature/visualization-report` | **Inference & Evaluation:** Develop `inference/predictor.py`, `visualization/overlays.py`, and `utils/metrics.py`. Manage dataset QA, plot overlays (raw, mask, prediction), and compile the final model comparison for the report. |