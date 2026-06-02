from typing import Tuple

import torch
from torch import nn
import torchvision as tv

from layers import Interpolate
from fpn import PanopticFPN
from utils import _get_shapes
from backbone import ResNetFeatureMapsExtractor


def make_fpn_resnet(name: str = 'resnet18',
                    fpn_type: str = 'fpn',
                    out_size: Tuple[int, int] = (256, 256), # Input dimentions 
                    fpn_channels: int = 256,
                    num_classes: int = 7, # Classes to predict
                    pretrained: bool = True,
                    in_channels: int = 3) -> nn.Module:
    """Create an FPN model with a ResNet backbone.

    If `in_channels > 3`, uses the fusion technique described in the paper,
    *FuseNet*, by Hazirbas et al.
    (https://vision.in.tum.de/_media/spezial/bib/hazirbasma2016fusenet.pdf)
    that adds a parallel resnet backbone for the new channels. All the
    pretrained weights are retained.

    Args:
        name (str, optional): Name of the resnet backbone. Only those available
            in torchvision are supported. Defaults to 'resnet18'.
        fpn_type (str, optional): Type of FPN. 'fpn' | 'panoptic' | 'panet'.
            Defaults to 'fpn'.
        out_size (Tuple[int, int], optional): Size of segmentation output.
            Defaults to (224, 224).
        fpn_channels (int, optional): Number of hidden channels to use in the
            FPN. Defaults to 256.
        num_classes (int, optional): Number of classes for which to make
            predictions. Determines the channel width of the output.
            Defaults to 1000.
        pretrained (bool, optional): Whether to use pretrained backbone.
            Defaults to True.
        in_channels (int, optional): Channel width of the input. If less than
            3, conv1 is replaced with a smaller one.  Defaults to 3.

    Raises:
        NotImplementedError: On unknown fpn_style.

    Returns:
        nn.Module: the FPN model
    """

    # Checks if the needed inputs are present or not
    assert in_channels > 0
    assert num_classes > 0
    assert out_size[0] > 0 and out_size[1] > 0

    # Checks if it was requested a pretrained network or not
    weights_arg = "DEFAULT" if pretrained else None
    resnet = tv.models.resnet.__dict__[name](weights=weights_arg)
    
    # I will always have 3 channels in input: image shape is (3, 256, 256)
    if in_channels == 3:
        backbone = ResNetFeatureMapsExtractor(model=resnet)

    # Extracts feature maps shape
    feat_shapes = _get_shapes(backbone, channels=in_channels, size=out_size)
    
    if fpn_type == 'panoptic':
        fpn = PanopticFPN(
            feat_shapes,
            hidden_channels=fpn_channels,
            out_channels=num_classes)
    else:
        raise NotImplementedError()
 
    # BUILD THE MODEL
    # yapf: disable
    model = nn.Sequential(
        backbone,
        fpn,
        Interpolate(size=out_size))
    # yapf: enable
    return model
