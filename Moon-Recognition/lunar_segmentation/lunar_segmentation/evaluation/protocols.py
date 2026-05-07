"""Model protocols for evaluation framework.

Defines structural contracts (PEP 544 Protocols) that any segmentation
model must satisfy to be evaluated by this framework.  Using Protocols
instead of abstract base classes means collaborators do **not** need to
inherit from anything — their models just need the right method signatures
(structural subtyping / typed duck-typing).

Typical usage
-------------
>>> from lunar_segmentation.evaluation.protocols import SemanticModelAdapter
>>> adapter = SemanticModelAdapter(model=my_unet, model_name="SmallUNet-v1")
>>> logits = adapter.predict(batch_of_images)
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class SegmentationModel(Protocol):
    """Structural protocol for any segmentation model.

    Any object that exposes ``predict``, ``num_classes`` and
    ``model_name`` satisfies this protocol without explicit inheritance.

    Attributes
    ----------
    num_classes : int
        Number of output channels (classes).
    model_name : str
        Human-readable identifier used in reports and plots.
    """

    @property
    def num_classes(self) -> int:
        ...

    @property
    def model_name(self) -> str:
        ...

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Run inference on a batch of images.

        Parameters
        ----------
        images : torch.Tensor
            Input tensor of shape ``(B, C_in, H, W)`` on any device.

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``(B, C_out, H, W)`` on the **same**
            device as ``images``.
        """
        ...


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

class SemanticModelAdapter:
    """Wraps a vanilla ``nn.Module`` to satisfy :class:`SegmentationModel`.

    This adapter handles device placement transparently: images are
    moved to the model's device before forward, and logits are returned
    on the **input** device so that downstream code stays device-agnostic.

    Parameters
    ----------
    model : nn.Module
        A PyTorch module whose ``forward`` returns logits ``(B, C, H, W)``.
    model_name : str
        Human-readable name for reports.
    num_classes : int, optional
        If not provided, inferred from the last Conv2d layer.
    device : str, optional
        Override device.  If *None*, uses the device of the first
        parameter found in *model*.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        num_classes: int | None = None,
        device: str | None = None,
    ) -> None:
        self._model = model
        self._model_name = model_name

        # Resolve device
        if device is not None:
            self._device = torch.device(device)
        else:
            try:
                self._device = next(model.parameters()).device
            except StopIteration:
                self._device = torch.device("cpu")
        self._model.to(self._device)

        # Resolve num_classes
        if num_classes is not None:
            self._num_classes = num_classes
        else:
            self._num_classes = self._infer_num_classes()

    # -- Protocol properties ------------------------------------------------

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def model_name(self) -> str:
        return self._model_name

    # -- Protocol method ----------------------------------------------------

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits on the *input* device.

        Parameters
        ----------
        images : torch.Tensor
            ``(B, C_in, H, W)`` tensor on any device.

        Returns
        -------
        torch.Tensor
            ``(B, C_out, H, W)`` logits, on the same device as *images*.
        """
        src_device = images.device
        self._model.eval()
        logits = self._model(images.to(self._device))
        return logits.to(src_device)

    # -- Internals ----------------------------------------------------------

    def _infer_num_classes(self) -> int:
        """Walk the module tree backwards to find the last Conv2d."""
        last_conv: nn.Conv2d | None = None
        for module in self._model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            return last_conv.out_channels
        raise ValueError(
            "Cannot infer num_classes: no Conv2d found. "
            "Pass num_classes explicitly."
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name={self._model_name!r}, "
            f"num_classes={self._num_classes}, "
            f"device={self._device})"
        )
