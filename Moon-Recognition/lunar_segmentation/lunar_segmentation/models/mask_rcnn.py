import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights


class MaskRCNN(nn.Module):
    '''
    Mask R-CNN model for instance segmentation. We will use the "maskrcnn_resnet50_fpn" from torchvision.
    Parameters:
     - num_classes (int): Number of classes (including background).
     - pretrained_backbone (bool): Whether to use a pre-trained backbone. If False, the backbone will be randomly initialized.
    '''

    def __init__(self, num_classes: int = 2, pretrained_backbone: bool = True):
        super().__init__()
        weights_backbone = ResNet50_Weights.DEFAULT if pretrained_backbone else None
        self.model = maskrcnn_resnet50_fpn(weights=None, num_classes=num_classes, weights_backbone=weights_backbone)

    def forward(self, images, targets=None):
        return self.model(images, targets)

        
    