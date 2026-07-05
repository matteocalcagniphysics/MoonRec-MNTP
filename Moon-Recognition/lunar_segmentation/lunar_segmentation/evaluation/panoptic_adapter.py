"""Panoptic FPN → semantic-mask adapter for unified evaluation.

Converts the dual output of the Panoptic FPN model (semantic logits + instance detections)
into dense semantic segmentation masks (B, C, H, W). In `fused` mode, instance 
predictions with high confidence overwrite the base semantic background to produce
a unified result.

The adapter is registered under the ``"panoptic"`` key.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn

from lunar_segmentation.evaluation.protocols import register_adapter
from lunar_segmentation.models.PAN4_factory import build_models
from lunar_segmentation.models.PAN3_fpn import PanopticFPN

logger = logging.getLogger(__name__)


class PanopticModelWrapper(PanopticFPN):
    """A wrapper for PanopticFPN to facilitate instantiation from YAML config.
    
    The standard PanopticFPN requires instantiated modules (backbone, semantic_branch,
    instance_branch) which is difficult to pass from a simple YAML config. 
    This wrapper takes base parameters and builds the components internally.
    """
    def __init__(self, name: str = 'resnet18', num_classes: int = 8, pretrained: bool = False):
        # Build the three branches using the factory
        backbone, semantic_branch, instance_branch = build_models(
            name=name, 
            num_classes=num_classes, 
            pretrained=pretrained
        )
        
        # Initialize the parent PanopticFPN with the instantiated branches
        super().__init__(
            backbone=backbone, 
            semantic_branch=semantic_branch, 
            instance_branch=instance_branch
        )


@register_adapter("panoptic")
class PanopticModelAdapter:
    """Wraps a PanopticFPN model to satisfy the SegmentationModel protocol.
    
    Can evaluate the semantic branch only, the instance branch only, or a fused 
    combination of both.
    
    Parameters
    ----------
    model : nn.Module
        The PanopticFPN (or PanopticModelWrapper) instance.
    model_name : str
        Human-readable name for reports.
    num_classes : int
        Number of semantic classes (default 7).
    eval_mode : str
        One of "semantic", "instance", or "fused".
    score_threshold : float
        Minimum confidence score for an instance prediction.
    mask_threshold : float
        Threshold to binarize instance soft masks.
    device : str or None
        Override device.
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        num_classes: int = 7,
        eval_mode: str = "fused",
        score_threshold: float = 0.70,
        mask_threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        if eval_mode not in ["semantic", "instance", "fused"]:
            raise ValueError(f"Unknown eval_mode '{eval_mode}' for Panoptic adapter.")
            
        self._model = model
        self._model_name = model_name
        self._num_classes = num_classes
        self._eval_mode = eval_mode
        self._score_threshold = score_threshold
        self._mask_threshold = mask_threshold
        
        if device is not None:
            self._device = torch.device(device)
        else:
            try:
                self._device = next(model.parameters()).device
            except StopIteration:
                self._device = torch.device("cpu")
        self._model.to(self._device)
        
        logger.info(
            f"PanopticModelAdapter created: name={model_name}, mode={eval_mode}, "
            f"num_classes={num_classes}, score_thresh={score_threshold}, "
            f"mask_thresh={mask_threshold}, device={self._device}"
        )

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def output_is_logits(self) -> bool:
        """
        - semantic: returns raw logits (evaluator will apply sigmoid).
        - instance / fused: returns float probabilities [0,1] (sigmoid already applied).
        """
        return self._eval_mode == "semantic"

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Run inference on a batch of images and return (B, C, H, W) mask."""
        src_device = images.device
        B, _, H, W = images.shape
        self._model.eval()

        # The panoptic model expects a list of image tensors
        image_list = [img.to(self._device) for img in images]
        
        # Forward pass returns: semantic_output, detections, rpn_losses, roi_losses
        semantic_output, detections, _, _ = self._model(image_list)
        
        if self._eval_mode == "semantic":
            out = semantic_output.to(src_device)  # (B, C_model, H, W)
            if out.shape[1] > self._num_classes:
                out = out[:, -self._num_classes:]  # drop background channel (index 0)
            return out

        # For 'instance' and 'fused' modes, we assemble the final semantic mask
        semantic_batch = torch.zeros(B, self._num_classes, H, W, dtype=torch.float32, device=src_device)
        
        for b in range(B):
            base_mask = None
            if self._eval_mode == "fused":
                # semantic_output[b] may have an extra background channel (e.g. 8 channels
                # for 7 semantic classes). Strip it so the tensor matches self._num_classes.
                sem_b = semantic_output[b]  # (C_model, H, W)
                if sem_b.shape[0] > self._num_classes:
                    # Assume background is at index 0; keep only the semantic class channels.
                    sem_b = sem_b[-self._num_classes:]  # last N channels are semantic classes
                base_mask = torch.sigmoid(sem_b)
                
            semantic_batch[b] = self._fuse_instances_to_semantic(
                detections[b], H, W, src_device, base_mask=base_mask
            )
            
        return semantic_batch

    def _fuse_instances_to_semantic(
        self, 
        detection: dict, 
        H: int, 
        W: int, 
        target_device: torch.device,
        base_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Converts detections to a dense semantic mask, optionally fused with a base mask."""
        if base_mask is not None:
            # Fused mode: start with the semantic probabilities
            semantic = base_mask.to(target_device).clone()
        else:
            # Instance mode: start with an empty canvas
            semantic = torch.zeros(self._num_classes, H, W, dtype=torch.float32, device=target_device)
            
        scores = detection.get("scores", torch.tensor([]))
        if scores.numel() == 0:
            return semantic
            
        # 1. Filter by confidence score
        keep = scores > self._score_threshold
        if not keep.any():
            return semantic
            
        masks = detection["masks"][keep]     # (N, 1, H, W)
        labels = detection["labels"][keep]   # (N,) 1-indexed
        
        # 2. Binarize
        binary = (masks.squeeze(1) > self._mask_threshold).float()
        
        # Resize if necessary
        if binary.shape[-2:] != (H, W):
            binary = torch.nn.functional.interpolate(
                binary.unsqueeze(1),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            binary = (binary > self._mask_threshold).float()
            
        binary = binary.to(target_device)
        labels = labels.to(target_device)
        
        # 3. Paint instances onto the semantic map
        for i in range(binary.shape[0]):
            ch = labels[i].item() - 1  # 1-indexed to 0-indexed
            if 0 <= ch < self._num_classes:
                # Where the instance mask is 1, push the probability for that class to 1.0
                semantic[ch] = torch.maximum(semantic[ch], binary[i])
                
                # In a strict panoptic fusion, we might also want to zero-out other classes 
                # at these pixels to resolve overlap, but torch.maximum is robust and matches 
                # the semantic evaluator's multi-label expectation.
            else:
                logger.warning(
                    f"Instance label {labels[i].item()} out of range [1, {self._num_classes}]; skipping."
                )
                
        return semantic

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name={self._model_name!r}, "
            f"mode={self._eval_mode}, "
            f"num_classes={self._num_classes})"
        )
