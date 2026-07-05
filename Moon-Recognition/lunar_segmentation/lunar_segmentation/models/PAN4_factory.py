from typing import Tuple, List

import torch
from torch import nn
import torchvision as tv

from .PAN2_layers import Interpolate
from .PAN3_fpn import SemanticBranch, CustomMaskRCNNHeads
from .PAN1_backbone import ResNetFeatureMapsExtractor



def _get_shapes(model: nn.Module,
                channels: int = 3,
                size: Tuple[int, int] = (224, 224)) -> List[Tuple[int, ...]]:
    """Extract shapes of feature maps computed by the model.

    The model must be an nn.Module whose __call__ method returns all feature
    maps when called with an input.
    """
    # Saves the current state
    # boolean variable, if true -> I was on training mode
    # Else -> I was in evaluating mode
    state = model.training 

    # Creates a dummy input to test the output of the model
    model.eval()
    with torch.no_grad():
        x = torch.empty(1, channels, *size)
        feats = model(x)

    # restore state. If false the model is kept in evaluation mode
    model.train(state)

    # If the output is a normal tensor, I transform it into a list
    # In my case the output is a tuple
    if isinstance(feats, torch.Tensor):
        feats = [feats]
    
    # Saves the outputs shape of each FPN level
    feat_shapes = [f.shape for f in feats]
    return feat_shapes # (B, C, H, W)



def build_models(name: str = 'resnet18',
                    out_size: Tuple[int, int] = (256, 256), # Input dimentions 
                    fpn_channels: int = 256,
                    num_classes: int = 8, # Classes to predict
                    pretrained: bool = True,
                    in_channels: int = 3) -> nn.Module:
    """Create an FPN model with a ResNet backbone.

    Args:
        name (str, optional): Name of the resnet backbone. Only those available
            in torchvision are supported. Defaults to 'resnet18'.
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
        nn.Module: the three fundamental models for the panoptic architecture
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
    Semantic_head = SemanticBranch(
        feat_shapes,
        hidden_channels=fpn_channels,
        out_channels=num_classes)
    
    Semantic_branch = nn.Sequential(
        Semantic_head,  #output of the Semantic head has size (8, 64, 64)
        Interpolate(size=out_size)  # I change the size of the output to the original input size (8, 256, 256)
    )

    Instance_branch = CustomMaskRCNNHeads(num_classes=num_classes)

    # I return the three fundamental components of the architecture: 
    # the backbone, the semantic branch and the instance branch
    return backbone, Semantic_branch, Instance_branch
