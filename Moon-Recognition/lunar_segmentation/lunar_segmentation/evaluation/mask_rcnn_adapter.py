"""Mask R-CNN → semantic-mask adapter for unified evaluation.

Converts the instance-level output of torchvision Mask R-CNN models
(list of dicts with ``boxes``, ``masks``, ``labels``, ``scores``) into
dense semantic segmentation masks ``(B, C, H, W)`` so that the standard
pixel-level evaluation pipeline can compute IoU, Dice, Precision, Recall
and F1 identically to any semantic model (e.g. U-Net).

The adapter is registered under the ``"instance"`` key and is
automatically selected when the YAML config specifies ``type: instance``.

Typical usage
-------------
>>> from lunar_segmentation.evaluation.protocols import create_adapter
>>> adapter = create_adapter(
...     model=mask_rcnn, model_name="MaskRCNN-v1",
...     model_type="instance", score_threshold=0.5, mask_threshold=0.5,
... )
>>> semantic = adapter.predict(images)  # (B, 7, H, W) float probabilities
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .protocols import register_adapter

logger = logging.getLogger(__name__)


@register_adapter("instance")
class InstanceModelAdapter:
    """Wraps a Mask R-CNN model to satisfy the :class:`SegmentationModel`
    protocol by converting instance predictions into dense semantic masks.

    Parameters
    ----------
    model : nn.Module
        A ``torchvision``-style Mask R-CNN whose ``forward`` (in eval
        mode) returns ``list[dict]`` with keys ``masks``, ``labels``,
        ``scores`` (and optionally ``boxes``).
    model_name : str
        Human-readable name for reports / plots.
    num_classes : int
        Number of **semantic** classes (excluding background).
        Default 7 for the lunar segmentation task.
    score_threshold : float
        Minimum confidence score to keep a predicted instance.
        Predictions below this value are discarded as noise.
    mask_threshold : float
        Threshold to binarize the soft instance masks (which are
        float probabilities in [0, 1] after the internal sigmoid of
        the Mask R-CNN head).
    device : str or None
        Override device.  If *None*, inferred from the model's first
        parameter.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        num_classes: int = 7,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        self._model = model
        self._model_name = model_name
        self._num_classes = num_classes
        self._score_threshold = score_threshold
        self._mask_threshold = mask_threshold

        # Resolve device
        if device is not None:
            self._device = torch.device(device)
        else:
            try:
                self._device = next(model.parameters()).device
            except StopIteration:
                self._device = torch.device("cpu")
        self._model.to(self._device)

        logger.info(
            f"InstanceModelAdapter created: name={model_name}, "
            f"num_classes={num_classes}, score_thresh={score_threshold}, "
            f"mask_thresh={mask_threshold}, device={self._device}"
        )

    # -- Protocol properties ------------------------------------------------

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def output_is_logits(self) -> bool:
        """Mask R-CNN outputs are already probabilities (post-sigmoid),
        so the evaluation pipeline must NOT apply sigmoid again."""
        return False

    # -- Protocol method ----------------------------------------------------

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Run Mask R-CNN inference and convert to dense semantic masks.

        Parameters
        ----------
        images : torch.Tensor
            Batch of images ``(B, C_in, H, W)`` on any device.  Values
            should be in ``[0, 1]`` as expected by torchvision detection
            models.

        Returns
        -------
        torch.Tensor
            Dense semantic mask ``(B, num_classes, H, W)`` as **float
            probabilities** (binary 0/1 after thresholding).  Lives on
            the same device as *images*.
        """
        src_device = images.device
        B, _, H, W = images.shape

        self._model.eval()

        # Mask R-CNN expects a list of tensors, not a batched tensor
        image_list = [img.to(self._device) for img in images]

        # Forward pass → list[dict] with 'boxes', 'labels', 'scores', 'masks'
        outputs = self._model(image_list)

        # Convert each per-image instance output to a semantic mask
        semantic_batch = torch.zeros(
            B, self._num_classes, H, W,
            dtype=torch.float32, device=src_device,
        )
        for b, output in enumerate(outputs):
            semantic_batch[b] = self._instances_to_semantic(
                output, H, W, target_device=src_device,
            )

        return semantic_batch

    # -- Internals ----------------------------------------------------------

    def _instances_to_semantic(
        self,
        output: dict[str, torch.Tensor],
        H: int,
        W: int,
        target_device: torch.device,
    ) -> torch.Tensor:
        """Convert a single image's instance predictions to a dense
        semantic mask ``(C, H, W)``.

        Algorithm
        ---------
        1. Filter predictions by ``score_threshold``.
        2. For each surviving prediction:
           a. Binarize the soft mask with ``mask_threshold``.
           b. Map the 1-indexed R-CNN label to a 0-indexed channel
              (``channel = label - 1``).
           c. Paint the binary mask onto the corresponding channel
              using logical OR (``torch.maximum``), so overlapping
              instances of the same class merge correctly.
        3. Return the assembled ``(C, H, W)`` tensor.

        Parameters
        ----------
        output : dict
            Single-image output from the Mask R-CNN, containing at
            minimum ``"masks"`` ``(N, 1, H, W)``, ``"labels"`` ``(N,)``
            and ``"scores"`` ``(N,)``.
        H, W : int
            Spatial dimensions of the output mask.
        target_device : torch.device
            Device for the returned tensor.

        Returns
        -------
        torch.Tensor
            ``(num_classes, H, W)`` float tensor with binary values.
        """
        semantic = torch.zeros(
            self._num_classes, H, W,
            dtype=torch.float32, device=target_device,
        )

        scores = output.get("scores", torch.tensor([]))
        if scores.numel() == 0:
            return semantic

        # 1. Filter by confidence score
        keep = scores > self._score_threshold
        if not keep.any():
            return semantic

        masks = output["masks"][keep]    # (N_keep, 1, H, W) float [0, 1]
        labels = output["labels"][keep]  # (N_keep,) int64, 1-indexed

        # 2. Binarize soft masks → squeeze the singleton channel dim
        # masks shape: (N, 1, H, W) → (N, H, W)
        binary = (masks.squeeze(1) > self._mask_threshold).float()

        # Handle possible spatial size mismatch (R-CNN may resize masks)
        if binary.shape[-2:] != (H, W):
            binary = torch.nn.functional.interpolate(
                binary.unsqueeze(1),             # (N, 1, h, w)
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)                          # (N, H, W)
            binary = (binary > self._mask_threshold).float()

        # Move to target device
        binary = binary.to(target_device)
        labels = labels.to(target_device)

        # 3. Paint each instance into its semantic channel
        for i in range(binary.shape[0]):
            ch = labels[i].item() - 1  # 1-indexed → 0-indexed
            if 0 <= ch < self._num_classes:
                semantic[ch] = torch.maximum(semantic[ch], binary[i])
            else:
                logger.warning(
                    f"Instance label {labels[i].item()} out of range "
                    f"[1, {self._num_classes}]; skipping."
                )

        return semantic

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name={self._model_name!r}, "
            f"num_classes={self._num_classes}, "
            f"score_threshold={self._score_threshold}, "
            f"mask_threshold={self._mask_threshold}, "
            f"device={self._device})"
        )
