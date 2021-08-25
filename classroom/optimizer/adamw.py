import torch
from torch.optim import Optimizer


class EMAFilter:
    def __init__(self, param):
        self.param = param
        self.n = 0
        self.x = None

    def __call__(self, x):
        if self.x is None:
            self.x = torch.zeros_like(x, memory_format=torch.preserve_format)
        else:
            beta = self.param(self.n)
            self.x.mul_(beta).add_(x, alpha=1-beta)
        self.n += 1
        return self.x


class AdamW:
    """
    parameters: may be model.parameters() or model.named_parameters()
    """
    def __init__(self,
                 parameters,
                 lr=lambda n: 0.0 if n < 1000 else 1e-5,
                 alpha=lambda n: 0.0,
                 beta1=lambda n: 0.9,
                 beta2=lambda n: 0.999,
                 weight_decay=lambda n: 0.01,
                 n=0):
        try:
            self.parameters = dict(parameters)
        except:
            self.parameters = dict(enumerate(parameters))
        self.lr = {name: lr for (name, p) in self.parameters.items()}
        self.alpha = {name: alpha for (name, p) in self.parameters.items()}
        self.beta1 = {name: beta1 for (name, p) in self.parameters.items()}
        self.beta2 = {name: beta2 for (name, p) in self.parameters.items()}
        self.weight_decay = {name: weight_decay for (name, p) in self.parameters.items()}
        self.n = 0
        self.state = {}

    @torch.no_grad()
    def step(self, closure):
        for (name, p) in self.parameters.items():
            alpha = self.alpha[name]
            try:
                p.grad.data *= alpha(n)
            except:
                pass
        with torch.enable_grad():
            result = closure()
        for (name, p) in self.parameters.items():
            lr = self.lr[name]
            beta1 = self.beta1[name]
            beta2 = self.beta2[name]
            weight_decay = self.weight_decay[name]
            n = self.n
            if n == 0:
                self.state[name] = {'\hat{g}': EMAFilter(beta1), '\hat{g^2}': EMAFilter(beta2)}
            state = self.state[name]
            bias_correction1 = 1 - beta1(n) ** (n+1)
            bias_correction2 = 1 - beta2(n) ** (n+1)
            assert bias_correction1 > 0, f"bias_correction1 <= 0, at {bias_correction1}"
            assert bias_correction2 > 0, f"bias_correction2 <= 0, at {bias_correction2}"
            g = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
            G = state['\hat{g}'](g)/bias_correction1
            g.square_()
            g = state['\hat{g^2}'](g)/bias_correction2
            torch.sqrt_(g)
            G.div_(g).nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0).add_(p.data,alpha=weight_decay(n))
            p.data.sub_(G,alpha=lr(n))
        self.n += 1
        return result
