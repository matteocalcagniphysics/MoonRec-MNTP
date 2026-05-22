from torch import nn
from containers import SequentialMultiOutput

class ResNetFeatureMapsExtractor(nn.Module):
    def __init__(self, model: nn.Module):
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
        
        self.m = SequentialMultiOutput(stem, *layers)
        # yapf: enable

    def forward(self, x):
        return self.m(x)



