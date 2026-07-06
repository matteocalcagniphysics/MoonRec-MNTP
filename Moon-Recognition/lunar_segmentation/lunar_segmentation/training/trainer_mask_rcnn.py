import torch
import torch.nn.functional as F
from torchmetrics.detection import MeanAveragePrecision
import torchvision.models.detection.roi_heads as _roi_heads
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
from tqdm import tqdm


logger = logging.getLogger(__name__)


# Helpers for monkey-patching torchvision's internal loss functions.
# We keep references to the originals so we can restore them later.
_ORIG_FASTRCNN_LOSS = _roi_heads.fastrcnn_loss
_ORIG_MASKRCNN_LOSS = _roi_heads.maskrcnn_loss


def _make_weighted_fastrcnn_loss(class_weights: torch.Tensor):
    def weighted_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
        # Same as torchvision.models.detection.roi_heads.fastrcnn_loss,
        # but the classification CE is weighted per class.
        labels_cat = torch.cat(labels, dim=0)
        regression_targets_cat = torch.cat(regression_targets, dim=0)

        # weighted classification loss
        weight = class_weights.to(class_logits.device)
        classification_loss = F.cross_entropy(class_logits, labels_cat, weight=weight)

        # box regression (smooth L1) - untouched, same as upstream
        sampled_pos_inds_subset = torch.where(labels_cat > 0)[0]
        labels_pos = labels_cat[sampled_pos_inds_subset]
        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets_cat[sampled_pos_inds_subset],
            beta=1.0 / 9,
            reduction="sum",
        )
        box_loss = box_loss / labels_cat.numel()

        return classification_loss, box_loss
    return weighted_fastrcnn_loss


def _make_weighted_maskrcnn_loss(class_weights: torch.Tensor):
    def weighted_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
        # Same idea as the fastrcnn version above, applied to the mask BCE.
        discretization_size = mask_logits.shape[-1]
        labels = [
            gt_label[idxs]
            for gt_label, idxs in zip(gt_labels, mask_matched_idxs)
        ]
        mask_targets = [
            _roi_heads.project_masks_on_boxes(m, p, i, discretization_size)
            for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
        ]

        labels_cat = torch.cat(labels, dim=0)
        mask_targets_cat = torch.cat(mask_targets, dim=0)

        if mask_targets_cat.numel() == 0:
            return mask_logits.sum() * 0

        # Pick out the logit for the ground-truth class of each RoI
        mask_logits_selected = mask_logits[
            torch.arange(labels_cat.shape[0], device=labels_cat.device),
            labels_cat
        ]

        # Look up the per-class weight for each RoI, then apply it element-wise 
        # it broadcasts over the H x W mask
        w = class_weights.to(labels_cat.device)
        sample_weights = w[labels_cat]
        pixel_loss = F.binary_cross_entropy_with_logits(
            mask_logits_selected, mask_targets_cat, reduction="none"
        )
        weighted_loss = (pixel_loss * sample_weights[:, None, None]).mean()

        return weighted_loss
    return weighted_maskrcnn_loss

class MaskRCNN_Trainer:
    def __init__(
        self,
        model,
        optimizer,
        threshold: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        grad_clip_norm: float = 1.0,
        class_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize trainer.
        Args:
            model: PyTorch model
            optimizer: Optimizer instance
            threshold: Confidence threshold for predictions
            device: Device string (GPU/CPU, autodetected)
            grad_clip_norm: Max norm for gradient clipping (0 = disabled)
            class_weights: Dict {class_name: weight} to rebalance the losses
                against class frequency. Keys must match CLASS_NAMES (no
                background). Background weight is fixed to 1.0.
        """
        from ..data.preprocessing import CLASS_NAMES

        self.model = model
        self.optimizer = optimizer
        self.threshold = threshold
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        self.model.to(device)
        self.metric = MeanAveragePrecision().to(self.device)

        # class-weighted losses
        if class_weights is not None:
            # build the [background, class_0, ..., class_N] tensor.
            # index 0 is reserved for background by torchvision.
            bg_weight = 1.0
            weight_list = [bg_weight] + [
                class_weights.get(name, 1.0) for name in CLASS_NAMES
            ]
            w_tensor = torch.tensor(weight_list, dtype=torch.float32)
            logger.info(
                "Class weights enabled: %s",
                dict(zip(['background'] + CLASS_NAMES, weight_list))
            )

            # patch torchvision's internal loss functions
            _roi_heads.fastrcnn_loss = _make_weighted_fastrcnn_loss(w_tensor)
            _roi_heads.maskrcnn_loss  = _make_weighted_maskrcnn_loss(w_tensor)
            self._patched_roi_heads   = True
        else:
            # restore the originals, in case this trainer is re-created
            # without weights after a previous run patched them
            _roi_heads.fastrcnn_loss = _ORIG_FASTRCNN_LOSS
            _roi_heads.maskrcnn_loss  = _ORIG_MASKRCNN_LOSS
            self._patched_roi_heads   = False

    
    def train_one_epoch(self, loader):
        """
        Train model for one epoch.
        Args:
            loader: DataLoader for training data 
        Returns:
            avg_loss: Average loss for the epoch for logging
        """
        
        self.model.train()
        losses = []

        # progress bar for training batches
        pbar = tqdm(loader, desc="Training Batch")

        for images, targets in pbar:
            self.optimizer.zero_grad()
            # move images and targets to the GPU
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # R-CNN returns a dict with 5 different losses
            loss_dict = self.model(images, targets)
            total_loss = torch.stack(list(loss_dict.values())).sum()   # sum them up
            total_loss.backward()  # backprop

            # clip gradients to avoid exploding-gradient issues
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.optimizer.step()

            # log the loss value and update the progress bar
            current_loss = total_loss.item()
            losses.append(current_loss)
            pbar.set_postfix(loss=f"{current_loss:.4f}")

        return float(np.mean(losses)) if losses else np.nan


    def evaluate(self, loader):
        """
        Evaluate the model using Mean Average Precision (mAP) metric computed by torchmetrics.
        Args:
            loader: DataLoader for training data
        Returns:
            metrics_df: PandasDataFrame with mAP metrics for logging 
        """
        
        self.model.eval()

        with torch.no_grad():
            for images, targets in tqdm(loader, desc="Validation Batch"):
                # move images and targets to the GPU, then run inference
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                outputs = self.model(images)  # get predictions
                
                # filter out low-confidence detections with a threshold mask
                for output in outputs:
                    mask = output['scores'] > self.threshold
                    output['boxes'] = output['boxes'][mask]
                    output['labels'] = output['labels'][mask]
                    output['scores'] = output['scores'][mask]
                    output['masks'] = output['masks'][mask]

                # metric update
                self.metric.update(outputs, targets)
            
            # compute mAP globally once all batches are done, then turn it into a DataFrame for logging
            mAP_dict = self.metric.compute()

            # flatten in case some entries are tensors with one value per
            # class (e.g. map_per_class) rather than plain scalars
            metrics_clean = {}
            for k, v in mAP_dict.items():
                t = v.cpu()
                if t.numel() == 1:
                    metrics_clean[k] = t.item()
                else:
                    for i, val in enumerate(t.tolist()):
                        metrics_clean[f"{k}_{i}"] = val

            metrics_df = pd.Series(metrics_clean).to_frame(name='value')
            
            logger.info("Evaluation Metrics:")
            logger.info(metrics_df)
            

            # reset metric for the next evaluation phase
            self.metric.reset()
            
            return metrics_df
