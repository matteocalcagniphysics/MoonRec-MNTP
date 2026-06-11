from functools import partial
from torch import nn
from torch.nn import functional as F



class ModulizedFunction(nn.Module):
    """Convert a function to an nn.Module."""

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = partial(fn, *args, **kwargs)

    def forward(self, x):
        return self.fn(x)

class Interpolate(ModulizedFunction):
    def __init__(self, mode='bilinear', align_corners=False, **kwargs):
        super().__init__(
            F.interpolate, mode='bilinear', align_corners=False, **kwargs)

class Sum(nn.Module):
    def forward(self, inps):
        return sum(inps)


