from torch import nn
from typing import Any

#############################################################################
# PYRAMID OUTPUT STRUCTURE 
#############################################################################

class SequentialMultiOutput(nn.Sequential):
    """
    Like nn.Squential but returns all intermediate outputs as a tuple.

      input
        │
        │
        V
    [1st layer]───────> 1st out
        │
        │
        V
    [2nd layer]───────> 2nd out
        │
        │
        V
        .
        .
        .
        │
        │
        V
    [nth layer]───────> nth out

    """

    def forward(self, x: Any) -> tuple:
        outs = [None] * len(self)           # In my case the backbone has 5 layers
        last_out = x                        # Here last out is the actual input 

        # Now I feed the input to layer 0, then its output to layer 1 and so on
        # At every layer I save the output to the corresponding place in the
        # previously created list 
        for i, module in enumerate(self):   
            last_out = module(last_out)
            outs[i] = last_out
        return tuple(outs)                  # At the end I return a tuple


#############################################################################
# RESNET BACKBONE FOR THE FPN
#############################################################################

class ResNetFeatureMapsExtractor(nn.Module):
    def __init__(self, model: nn.Module, out_mask_rcnn: bool = False):
        super().__init__()
        # This is the layer 0, which preprocess the input to extract the most 
        # general features. This helps the network in the actual FPN levels
        # It will be then discarded in the Mask R-CNN heads
        stem = nn.Sequential(       
            model.conv1,    # stride 2
            model.bn1,
            model.relu,
            model.maxpool   # stride 2
        )
        layers = [
            model.layer1,   # stride 1 padding 1
            model.layer2,   # stride 2 padding 0
            model.layer3,   # stride 2 padding 0
            model.layer4,   # stride 2 padding 0
        ]
        
        # Checks whether the backbone will be used for Mask RCNN heads or not. 
        # If it is the case, I need to return the output in a dictionary format
        # with keys '0', '1', '2', '3' corresponding to the 4 FPN levels.
        self.out_mask_rcnn = out_mask_rcnn

        self.m = SequentialMultiOutput(stem, *layers)
        # yapf: enable

    def forward(self, x):
        
        out = self.m(x) # This gives the multi-layer output tuple
        
        # This adds the dictionary output format needed for Mask RCNN heads
        if(self.out_mask_rcnn):
            # I don't need the output of the stem for Mask RCNN heads
            correct_out = out[1:] 
            # Returns a dictionary {level_index : level output} (For Mask R-CNN heads)
            return {str(i): correct_out[i] for i in range(len(correct_out))}
        else:
            return out # Output for the Semantic branch



