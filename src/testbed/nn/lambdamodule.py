from torch.nn import Module
import dill

class Lambda(Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['f'] = dill.dumps(self.f)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.f = dill.loads(self.f)
