import torch
from factory import make_fpn_resnet

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
