from torch import nn
from .containers import SequentialMultiOutput

class ResNetFeatureMapsExtractor(nn.Module):
    def __init__(self, model: nn.Module, out_mask_rcnn: bool = False):
        super().__init__()
        # yapf: disable
        stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        )
        layers = [
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        ]
        
        # Checks whether the backbone will be used for Mask RCNN heads or not. 
        # If it is the case, I need to return the output in a dictionary format
        # with keys '0', '1', '2', '3' corresponding to the 4 FPN levels.
        self.out_mask_rcnn = out_mask_rcnn

        self.m = SequentialMultiOutput(stem, *layers)
        # yapf: enable

    def forward(self, x):
        
        out = self.m(x)
        
        # This adds the dictionary output format needed for Mask RCNN heads
        if(self.out_mask_rcnn):
            correct_out = out[1:] # I don't need the output of the stem for Mask RCNN heads
            return {str(i): correct_out[i] for i in range(len(correct_out))}
        else:
            return out



