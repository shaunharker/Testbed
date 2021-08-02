from torch.nn import Module, ModuleList
from torch.cuda.amp import autocast
from types import GeneratorType

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        layers = sum([list(layer) if type(layer)==GeneratorType else [layer] for layer in layers],[])
        self.layers = ModuleList(layers)

    @autocast()
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
