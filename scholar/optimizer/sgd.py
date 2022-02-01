import torch
from torch.optim import Optimizer


class SGD:
    """
    parameters: may be model.parameters() or model.named_parameters()
    """
    def __init__(self,
                 parameters,
                 lr=lambda n: 0.0 if n < 1000 else 1e-6,
                 weight_decay=lambda n: 0.01,
                 update=lambda n: True,
                 n=0):
        try:
            self.parameters = dict(parameters)
        except:
            self.parameters = dict(enumerate(parameters))
        self.n = 0
        self.last_update = -1
        self.state = {
            name: {
                'lr': lr,
                'weight_decay': weight_decay,
                'update': update,
                }
            for (name, p) in self.parameters.items()}
        for (name, p) in self.parameters.items():
            try:
                with torch.no_grad():
                    p.grad.data *= 0.0
            except:
                pass

    @torch.no_grad()
    def step(self, closure):
        with torch.enable_grad():
            result = closure()
        for (name, p) in self.parameters.items():
            n = self.n
            state = self.state[name]
            if state["update"](n):
                g = torch.tanh(torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0))
                dp = g.add_(p.data, alpha=state["weight_decay"](n))
                p.data.sub_(dp, alpha=state["lr"](n))
                p.grad.data *= 0.0
        self.n += 1
        return result
