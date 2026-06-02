from typing import Any
from torch import nn

######################################################################################
# KEEP  
######################################################################################
class Parallel(nn.ModuleList):
    ''' Passes inputs through multiple `nn.Module`s in parallel.
    Returns a tuple of outputs.
    '''

    def forward(self, xs: Any) -> tuple:
        # if multiple inputs, pass the 1st input through the 1st module,
        # the 2nd input through the 2nd module, and so on.
        if isinstance(xs, (list, tuple)):
            return tuple(m(x) for m, x in zip(self, xs))
        # if single input, pass it through all modules
        return tuple(m(xs) for m in self)

######################################################################################
# KEEP  
######################################################################################
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
        outs = [None] * len(self)
        last_out = x
        for i, module in enumerate(self):
            last_out = module(last_out)
            outs[i] = last_out
        return tuple(outs)
