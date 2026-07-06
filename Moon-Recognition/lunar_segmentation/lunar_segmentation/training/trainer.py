import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from torchmetrics.detection import MeanAveragePrecision
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.models.detection.roi_heads as _roi_heads
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(0, 2, 3))
    den = probs.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
    dice = 1 - (num + eps) / (den + eps)
    return dice.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.bce(logits, targets) + dice_loss(logits, targets)

def panoptic_dice_loss(probs: torch.Tensor, targets_one_hot: torch.Tensor, eps: float = 1e-6, weights=None) -> torch.Tensor:
    # probs and targets are both [B, C, H, W]
    num = 2 * (probs * targets_one_hot).sum(dim=(0, 2, 3))   # shape [C]
    den = probs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))  # shape [C]
    dice = 1 - (num + eps) / (den + eps)                       # shape [C]
    if weights is not None:
        dice = dice * weights
    return dice.mean()


class PanopticBCEDiceLoss(nn.Module):
    def __init__(self, class_weights=None):
        """
        Args:
            class_weights: optional 1-D tensor of length num_classes (including
                           background at index 0). Applied to both BCE and Dice.
        """
        super().__init__()
        # Store as a buffer so it moves to the right device automatically
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        """
        logits:  [B, C, H, W] float predictions (raw, before sigmoid)
        targets: [B, H, W]    long integer class indices (0 = background)
        """
        num_classes = logits.shape[1]
        
        # 1. Convert [B, H, W] index map to [B, C, H, W] one-hot float map
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # 2. Weighted BCE
        if self.class_weights is not None:
            # pos_weight for BCEWithLogitsLoss must be broadcastable to [B, C, H, W]
            # We reshape to [1, C, 1, 1]
            pw = self.class_weights.view(1, -1, 1, 1)
            loss_bce = F.binary_cross_entropy_with_logits(logits, targets_one_hot, pos_weight=pw)
        else:
            loss_bce = F.binary_cross_entropy_with_logits(logits, targets_one_hot)
        
        # 3. Weighted Dice
        probs = torch.sigmoid(logits)
        loss_dice = panoptic_dice_loss(probs, targets_one_hot, weights=self.class_weights)
        
        return loss_bce + loss_dice


def multilabel_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6):
    from ..data.preprocessing import CLASS_NAMES
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    per_class = []
    for i, name in enumerate(CLASS_NAMES):
        p = preds[:, i]
        t = targets[:, i]
        tp = (p * t).sum().item()
        fp = (p * (1 - t)).sum().item()
        fn = ((1 - p) * t).sum().item()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        per_class.append({'class': name, 'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou})
    return pd.DataFrame(per_class)

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

class Trainer:
    def __init__(self, model, optimizer, criterion, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize trainer.

        Args:
            model: PyTorch model
            optimizer: Optimizer instance
            criterion: Loss function
            device: Device string ('cuda' or 'cpu', auto-detected)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(self.device)

    def train_one_epoch(self, loader):
        self.model.train()
        losses = []
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return float(np.mean(losses)) if losses else np.nan

    def evaluate(self, loader, criterion=None):
        self.model.eval()
        metrics_list = []
        criterion = criterion or self.criterion
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                if criterion:
                    metrics_list.append(multilabel_metrics(logits, y))
                else:
                    # If no criterion provided, we just need logits
                    pass

        if not metrics_list:
            return None

        # Combine metrics from all batches
        return pd.concat(metrics_list).groupby('class').mean()

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

class PanopticTrainer:
    def __init__(self, model, optimizer, criterion, metric, threshold=0.5,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 class_weights=None):
        """Initialize trainer.

        Args:
            model: PyTorch model
            optimizer: Optimizer instance
            criterion: Loss function (PanopticBCEDiceLoss or similar)
            metric: Evaluation metric
            threshold: Confidence threshold for instance predictions
            device: Device string ('cuda' or 'cpu', auto-detected)
            class_weights: dict or tensor of per-class weights; if dict,
                           maps CLASS_NAMES strings to float values.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.threshold = threshold
        self.device = device
        self.model.to(self.device)
        self.metric = MeanAveragePrecision().to(self.device)

        if class_weights is not None:
            if isinstance(class_weights, dict):
                from ..data.preprocessing import CLASS_NAMES
                weights_list = [1.0]  # background weight
                for name in CLASS_NAMES:
                    weights_list.append(class_weights.get(name, 1.0))
                self.class_weights = torch.tensor(weights_list, dtype=torch.float32, device=self.device)
            elif isinstance(class_weights, (list, tuple)):
                self.class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            else:
                self.class_weights = class_weights.to(self.device)

            # Push weights into the criterion (semantic branch)
            if hasattr(criterion, 'class_weights'):
                criterion.class_weights = self.class_weights

            # Patch torchvision's instance-branch loss functions
            import torchvision.models.detection.roi_heads as roi_heads
            import torch.nn.functional as F
            from torchvision.models.detection.roi_heads import project_masks_on_boxes

            weights_tensor = self.class_weights

            def weighted_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
                labels = torch.cat(labels, dim=0)
                regression_targets = torch.cat(regression_targets, dim=0)
                classification_loss = F.cross_entropy(class_logits, labels, weight=weights_tensor)
                sampled_pos_inds_subset = torch.where(labels > 0)[0]
                labels_pos = labels[sampled_pos_inds_subset]
                N, num_classes = class_logits.shape
                box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
                box_loss = F.smooth_l1_loss(
                    box_regression[sampled_pos_inds_subset, labels_pos],
                    regression_targets[sampled_pos_inds_subset],
                    beta=1 / 9,
                    reduction="sum",
                )
                box_loss = box_loss / labels.numel()
                return classification_loss, box_loss

            def weighted_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
                discretization_size = mask_logits.shape[-1]
                labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
                mask_targets = [
                    project_masks_on_boxes(m, p, i, discretization_size)
                    for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
                ]
                labels = torch.cat(labels, dim=0)
                mask_targets = torch.cat(mask_targets, dim=0)
                if mask_targets.numel() == 0:
                    return mask_logits.sum() * 0
                unreduced = F.binary_cross_entropy_with_logits(
                    mask_logits[torch.arange(labels.shape[0], device=labels.device), labels],
                    mask_targets,
                    reduction="none",
                )
                loss_per_instance = unreduced.mean(dim=(1, 2))
                inst_weights = weights_tensor[labels]
                return (loss_per_instance * inst_weights).mean()

            roi_heads.fastrcnn_loss = weighted_fastrcnn_loss
            roi_heads.maskrcnn_loss = weighted_maskrcnn_loss
        else:
            self.class_weights = None


    def train_one_epoch(self, loader):
        self.model.train()
        losses = []

        pbar = tqdm(loader, desc="Training Batch")

        for images, semantic_targets, instance_targets in pbar:
            # Move list of image tensors to device
            images = [img.to(self.device) for img in images]
            
            # Move single stacked semantic tensor to device
            semantic_targets = semantic_targets.to(self.device)
            
            # Move list of dictionaries (and all tensors inside them) to device
            instance_targets = [
                {k: v.to(self.device) for k, v in target_dict.items()} 
                for target_dict in instance_targets
            ]

            self.optimizer.zero_grad(set_to_none=True)
            logits, _, rpn_losses, roi_losses = self.model(images, instance_targets)
            semantic_loss = self.criterion(logits, semantic_targets)
            instance_loss = sum(rpn_losses.values()) + sum(roi_losses.values())
            total_loss = semantic_loss + instance_loss
            total_loss.backward()
            self.optimizer.step()
            current_loss = total_loss.item() 
            losses.append(current_loss)

            pbar.set_postfix(loss=f"{current_loss:.4f}")

        return float(np.mean(losses)) if losses else np.nan

    def evaluate(self, loader, criterion=None):
        self.model.eval()
        semantic_metrics_list = []
        criterion = criterion or self.criterion
        with torch.no_grad():
            for images, semantic_targets, instance_targets in loader:
                # Move list of image tensors to device
                images = [img.to(self.device) for img in images]
                
                # Move single stacked semantic tensor to device
                semantic_targets = semantic_targets.to(self.device)
                
                # Move list of dictionaries (and all tensors inside them) to device
                instance_targets = [
                    {k: v.to(self.device) for k, v in target_dict.items()} 
                    for target_dict in instance_targets
                ]     

                logits, detections, _, _ = self.model(images, instance_targets)
                if criterion:
                    # Convert integer target [B, H, W] to one-hot float mask [B, C, H, W]
                    targets_one_hot = F.one_hot(semantic_targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
                    # Skip the background channel (index 0) for metrics matching CLASS_NAMES
                    semantic_metrics_list.append(multilabel_metrics(logits[:, 1:], targets_one_hot[:, 1:]))
                else:
                    # If no criterion provided, we just need logits
                    pass
                
                # Apply threshold to predictions using a mask to filter out low-confidence detections
                for output in detections:
                    mask = output['scores'] > self.threshold
                    output['boxes'] = output['boxes'][mask]
                    output['labels'] = output['labels'][mask]
                    output['scores'] = output['scores'][mask]
                    output['masks'] = output['masks'][mask]

                # Metric update
                self.metric.update(detections, instance_targets)
            
            # Global mAP computation after processing all batches, then transform results to a DataFrame for logging
            mAP_dict = self.metric.compute()
            # Some keys (e.g. map_per_class) are multi-element tensors: convert scalars with .item(), vectors with .tolist()
            metrics_clean = {
                k: v.item() if v.numel() == 1 else v.tolist()
                for k, v in mAP_dict.items()
            }
            instance_metrics_df = pd.Series(metrics_clean).to_frame(name='value')
            
            logger.info("Evaluation Metrics:")
            logger.info(instance_metrics_df)
            
            # Reset metric for the next evaluation phase
            self.metric.reset()

        if not semantic_metrics_list:
            return instance_metrics_df
        else:
            # Combine metrics from all batches
            semantic_metrics_df = pd.concat(semantic_metrics_list).groupby('class').mean()
            return semantic_metrics_df, instance_metrics_df

