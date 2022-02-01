import torch
from torch.optim import Optimizer


class EMAFilter:
    def __init__(self, param, init="zeros"):
        self.param = param
        self.n = 0
        self.x = None
        self.init = init
    def __call__(self, x):
        if self.x is None:
            if self.init == "zeros":
                self.x = torch.zeros_like(x, memory_format=torch.preserve_format)
            elif self.init == "ones":
                self.x = torch.ones_like(x, memory_format=torch.preserve_format)
            else:
                raise ValueError(f"EMAFilter: unrecognized choice init = {self.init}")
        else:
            beta = self.param(self.n)
            self.x.mul_(beta).add_(x, alpha=1-beta)
        self.n += 1
        return self.x


class AdamWLS:
    """
    parameters: may be model.parameters() or model.named_parameters()
    """
    def __init__(self,
                 parameters,
                 lr=lambda n: 0.0 if n < 1000 else 1e-6,
                 beta1=lambda n: 0.9,
                 beta2=lambda n: 0.999,
                 weight_decay=lambda n: 0.01,
                 update=lambda n: True,
                 n=0):
        try:
            self.parameters = dict(parameters)
        except:
            self.parameters = dict(enumerate(parameters))
        self.n = 0
        self.rejects = 0
        self.state = {
            name: {
                'lr': lr,
                'G': EMAFilter(beta1, init="zeros"),
                'G2': EMAFilter(beta2, init="zeros"),
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
        #print("enter")
        with torch.enable_grad():
            result, batch = closure()
        n = self.n
        for (name, p) in self.parameters.items():
            state = self.state[name]
            if state["update"](n):
                g = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)  # scaling?
                G = state['G'](g)
                G2 = state['G2'](torch.square(g))
                state['dp'] = G/torch.sqrt(G2)
                state['dp'].nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0).add_(p.data, alpha=state["weight_decay"](n))
                p.data.sub_(state['dp'], alpha=state["lr"](n))
                p.grad.data *= 0.0
        with torch.no_grad():
            new_result, _ = closure(batch=batch, backward=False)
            if new_result > result:
                self.rejects += 1
                for (name, p) in self.parameters.items():
                    state = self.state[name]
                    if state["update"](n):
                        p.data.add_(state['dp'], alpha=state["lr"](n))
                        del state['dp']
        self.n += 1
        #print("exit")
        return result, batch
