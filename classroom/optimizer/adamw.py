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



class AdamW(Optimizer):
    r"""

    Reimplemented AdamW so we pass hyperparameters as callables that take step as an argument
    and return the value of the hyperparameter appropriate for that step.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Args:
        parameters (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): initial learning rate (default: .001)
        beta1 (float, optional): EMA parameter for grad/sqrgrad (default: 0.9)
        beta2 (float, optional): EMA parameter for grad/sqrgrad (default: 0.999)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: .01)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101

    Returns:

        For convenience, returns the results of the closure.

    Note:

        WARNING: side effect: grad data is squared in-place
    """

    def __init__(self,
                 parameters,
                 lr=lambda n: 0.0 if n < 1000 else 1e-5,
                 alpha=lambda n: 0.0 if n == 0 else 1.0,
                 beta1=lambda n: 0.9,
                 beta2=lambda n: 0.999,
                 weight_decay=lambda n: 0.01,
                 n=0):
        super().__init__(parameters, {"lr": lr, "alpha": alpha, "beta1": beta1, "beta2": beta2, "weight_decay": weight_decay, "n": n})
        self.lr = {p:lr for p in parameters}

    @torch.no_grad()
    def step(self, closure):
        for group in self.param_groups:
            alpha = group["alpha"]
            for p in group["params"]:
                if p not in self.lr:
                    self.lr[p] = group["lr"]
                try:
                    p.grad.data *= alpha(n)
                except:
                    pass
        with torch.enable_grad():
            result = closure()
        for group in self.param_groups:
            alpha = group["alpha"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            n = group["n"]
            if n == 0:
                for p in group["params"]:
                    self.state[p]={'ema_grad': EMAFilter(beta1), 'ema_sqr_grad': EMAFilter(beta2)}
            n += 1
            group["n"] = n
            bias_correction1 = 1 - beta1(n) ** n
            bias_correction2 = 1 - beta2(n) ** n
            assert bias_correction1 > 0
            assert bias_correction2 > 0
            for p in group["params"]:
                state = self.state[p]
                lr = self.lr[p]
                g = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
                G = state['ema_grad'](g)/bias_correction1
                g.square_()
                g = state['ema_sqr_grad'](g)/bias_correction2
                torch.sqrt_(g)
                G.div_(g).nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0).add_(p.data,alpha=weight_decay(n))
                p.data.sub_(G,alpha=lr(n))
        return result
