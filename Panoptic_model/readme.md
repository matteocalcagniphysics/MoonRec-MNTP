# Panoptic Model Documentation

## Overview
This module provides a PyTorch implementation of a Panoptic Feature Pyramid Network (FPN). The architecture is designed to perform dense pixel-wise predictions (like semantic or panoptic segmentation) by leveraging a standard image classification backbone (e.g., ResNet) and a feature pyramid structure. The implementation is heavily inspired by the paper "Panoptic Feature Pyramid Networks" by Kirilov et al..

Additionally, if the input requires more than 3 channels, the model incorporates a fusion technique inspired by *FuseNet* (Hazirbas et al.) to process the additional channels using a parallel backbone while retaining pretrained weights.

---

## Architecture Design

The overall model is built using a sequential pipeline consisting of three main stages:

1. **Backbone (`ResNetFeatureMapsExtractor`)**: Extracts hierarchical feature maps.
2. **Panoptic FPN (`PanopticFPN`)**: Merges and upsamples the features into a single, unified representation.
3. **Interpolate**: Scales the final output to match the desired `out_size`.

### 1. Multi-Output Backbone
The backbone wraps a standard `torchvision` ResNet model. Instead of just returning the final classification vector, the `ResNetFeatureMapsExtractor` extracts and returns a tuple of intermediate feature maps from different layers (`stem`, `layer1`, `layer2`, `layer3`, `layer4`). 

To handle this flow, the module uses a custom `SequentialMultiOutput` container. Here is the conceptual flow:

```text
  input
    │
    V
[1st layer]───────> 1st out (e.g., stem features)
    │
    V
[2nd layer]───────> 2nd out (e.g., layer1 features)
    │
    V
    .
    .
[nth layer]───────> nth out (e.g., layer4 features)

```
import torch
from Panoptic_model.factory import make_fpn_resnet

# 1. Instantiate the model
model = make_fpn_resnet(
    name='resnet18',           # Standard torchvision ResNet backbone
    fpn_type='panoptic',       # The Panoptic FPN style
    out_size=(224, 224),       # Expected spatial size of the final output
    fpn_channels=256,          # Hidden channels in the FPN
    num_classes=21,            # Number of segmentation classes (output channels)
    pretrained=True,           # Use ImageNet pretrained weights
    in_channels=3              # Standard RGB input
)

# 2. Perform a forward pass
batch_size = 2
dummy_input = torch.randn(batch_size, 3, 224, 224)

# Set to eval mode if not training
model.eval()

with torch.no_grad():
    predictions = model(dummy_input)
    
print(predictions.shape) 
# Expected output: torch.Size([2, 21, 224, 224])

# How to use the module

```python
import torch
from Panoptic_model.factory import make_fpn_resnet

# 1. Instantiate the model
model = make_fpn_resnet(
    name='resnet18',           # Standard torchvision ResNet backbone
    fpn_type='panoptic',       # The Panoptic FPN style
    out_size=(224, 224),       # Expected spatial size of the final output
    fpn_channels=256,          # Hidden channels in the FPN
    num_classes=21,            # Number of segmentation classes (output channels)
    pretrained=True,           # Use ImageNet pretrained weights
    in_channels=3              # Standard RGB input
)

# 2. Perform a forward pass
batch_size = 2
dummy_input = torch.randn(batch_size, 3, 224, 224)

# Set to eval mode if not training
model.eval()

with torch.no_grad():
    predictions = model(dummy_input)
    
print(predictions.shape) 
# Expected output: torch.Size([2, 21, 224, 224])

```