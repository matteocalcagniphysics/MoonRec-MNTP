import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights


class MaskRCNN(nn.Module):
    '''
    Mask R-CNN model for instance segmentation. We will use the "maskrcnn_resnet50_fpn" from torchvision.
    Parameters:
     - num_classes (int): Number of classes, default is 8 (7 lunar classes + 1 background).
     - pretrained (bool): Whether to use a fully pre-trained COCO model and fine-tune.
    '''

    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        if pretrained:
            # 1. Load the complete pre-trained model on the COCO dataset
            weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = maskrcnn_resnet50_fpn(weights=weights)
            
            # 2. Replace the Box Predictor (to classify bounding boxes)
            # Get the number of input features for the current classifier
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            # Replace the pre-trained head with a new, untrained one with the correct number of classes
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
            # 3. Replace the Mask Predictor (to generate pixel masks)
            # Get the number of input features for the mask predictor
            in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            # Replace the mask head
            self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
                in_features_mask, hidden_layer, num_classes
            )
        else:
            # Fallback option: initialize only with the base architecture (as originally done)
            self.model = maskrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    

    def forward(self, images, targets=None):
        '''
        During training, it returns a dictionary with the losses.
        During inference, it returns a list of dictionaries with 'boxes', 'labels', 'scores', and 'masks'.
        '''
        return self.model(images, targets)

        
    