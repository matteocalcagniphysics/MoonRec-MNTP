"""Model protocols and adapter registry for the evaluation framework.

Defines structural contracts (PEP 544 Protocols) that models must satisfy,
allowing evaluation of various model types (semantic, instance, panoptic)
via registered adapters.

Typical usage
-------------
>>> from lunar_segmentation.evaluation.protocols import create_adapter
>>> adapter = create_adapter(
...     model=my_model, model_name="MaskRCNN-v1",
...     model_type="instance", score_threshold=0.5,
... )
>>> semantic_mask = adapter.predict(batch_of_images)   # (B, C, H, W)
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ======================================================================== #
#  Adapter Registry                                                         #
# ======================================================================== #

_ADAPTER_REGISTRY: dict[str, type] = {}


def register_adapter(model_type: str):
    """Class decorator that registers an adapter in the global registry.

    Parameters
    ----------
    model_type : str
        Key used in the YAML config ``type`` field (e.g. ``"semantic"``,
        ``"instance"``, ``"panoptic"``).

    Example
    -------
    >>> @register_adapter("instance")
    ... class InstanceModelAdapter:
    ...     ...
    """
    def decorator(cls):
        if model_type in _ADAPTER_REGISTRY:
            logger.warning(
                f"Adapter type '{model_type}' already registered "
                f"({_ADAPTER_REGISTRY[model_type].__name__}); "
                f"overwriting with {cls.__name__}."
            )
        _ADAPTER_REGISTRY[model_type] = cls
        return cls
    return decorator


def create_adapter(
    model: nn.Module,
    model_name: str,
    model_type: str = "semantic",
    **kwargs,
):
    """Factory: instantiate the correct adapter based on *model_type*.

    Parameters
    ----------
    model : nn.Module
        The underlying PyTorch model.
    model_name : str
        Human-readable name for reports / plots.
    model_type : str
        Must match a key previously registered via
        :func:`register_adapter`.
    **kwargs
        Extra arguments forwarded to the adapter constructor
        (e.g. ``score_threshold``, ``mask_threshold``).

    Returns
    -------
    object
        An adapter instance satisfying :class:`SegmentationModel`.

    Raises
    ------
    ValueError
        If *model_type* is not found in the registry.
    """
    adapter_cls = _ADAPTER_REGISTRY.get(model_type)
    if adapter_cls is None:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Registered types: {sorted(_ADAPTER_REGISTRY.keys())}"
        )
    return adapter_cls(model=model, model_name=model_name, **kwargs)


# ======================================================================== #
#  Protocols                                                                #
# ======================================================================== #

@runtime_checkable
class SegmentationModel(Protocol):
    """Structural protocol for any segmentation model.

    Any object that exposes ``predict``, ``num_classes``,
    ``model_name``, and ``output_is_logits`` satisfies this protocol
    without explicit inheritance.

    Attributes
    ----------
    num_classes : int
        Number of output channels (classes).
    model_name : str
        Human-readable identifier used in reports and plots.
    output_is_logits : bool
        ``True`` if ``predict()`` returns raw logits (sigmoid not yet
        applied), ``False`` if it returns probabilities or binary masks.
    """

    @property
    def num_classes(self) -> int:
        ...

    @property
    def model_name(self) -> str:
        ...

    @property
    def output_is_logits(self) -> bool:
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
            Dense mask of shape ``(B, C_out, H, W)`` on the **same**
            device as ``images``.  Whether values are logits or
            probabilities is indicated by :attr:`output_is_logits`.
        """
        ...


# ======================================================================== #
#  Adapters                                                                 #
# ======================================================================== #

@register_adapter("semantic")
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

    @property
    def output_is_logits(self) -> bool:
        """Semantic models return raw logits — sigmoid is applied downstream."""
        return True

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
