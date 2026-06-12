import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from torchmetrics.detection import MeanAveragePrecision
import torch.nn.functional as F
from tqdm import tqdm

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

def panoptic_dice_loss(probs: torch.Tensor, targets_one_hot: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # probs and targets are both [B, C, H, W]
    num = 2 * (probs * targets_one_hot).sum(dim=(0, 2, 3))
    den = probs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))
    dice = 1 - (num + eps) / (den + eps)
    return dice.mean()

class PanopticBCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] float predictions
        targets: [B, H, W] long integer class indices
        """
        num_classes = logits.shape[1]
        
        # 1. Convert [B, H, W] index map to [B, C, H, W] one-hot float map
        # one_hot outputs [B, H, W, C], so we permute to get [B, C, H, W]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # 2. Calculate BCE
        loss_bce = self.bce(logits, targets_one_hot)
        
        # 3. Calculate Dice (using sigmoid probabilities to match BCE logic)
        probs = torch.sigmoid(logits)
        loss_dice = panoptic_dice_loss(probs, targets_one_hot)
        
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


class PanopticTrainer:
    def __init__(self, model, optimizer, criterion, metric, threshold=0.5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize trainer.

        Args:
            model: PyTorch model
            optimizer: Optimizer instance
            criterion: Loss function
            metric: Evaluation metric
            threshold: Confidence threshold for instance predictions
            device: Device string ('cuda' or 'cpu', auto-detected)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.threshold = threshold
        self.device = device
        self.model.to(self.device)
        self.metric = MeanAveragePrecision().to(self.device)

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
                    semantic_metrics_list.append(multilabel_metrics(logits, semantic_targets))
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
            metrics_clean = {k: v.item() for k, v in mAP_dict.items()}
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

