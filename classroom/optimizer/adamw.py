import torch
from torch.optim import Optimizer


class EMA:
    def __init__(self, param, x=None):
        self.param = param
        self.x = x

    def step(self, x):
        if self.x is None:
            self.x = x
        else:
            self.x.mul_(self.param).add_(x, alpha=1-self.param)
        return self.x

    def __call__(self):
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
                 lr=lambda step: .001,
                 beta1=lambda step: 0.9,
                 beta2=lambda step: 0.999,
                 eps=lambda step: 1e-8,
                 weight_decay=lambda step: 0.01,
                 initial_step=0):
        super().__init__(parameters, {})
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.initial_step = initial_step
        self.n = initial_step

    def params(self):
        return [p for group in self.param_groups for p in group['params']
                if p.grad is not None and not p.grad.is_sparse]

    def _setup(self):
        for (idx, p) in enumerate(self.params()):
            zeros = lambda : torch.zeros_like(p, memory_format=torch.preserve_format)
            self.state[idx]={'ema_grad': EMA(self.beta1(self.n), zeros()),
                             'ema_sqr_grad': EMA(self.beta2(self.n), zeros())}

    @torch.no_grad()
    def step(self, closure):
        with torch.enable_grad():
            result = closure()
        if self.n == self.initial_step:
            self._setup()
        self.n += 1
        for (idx, p) in enumerate(self.params()):
            p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
            self.state[idx]['ema_grad'].step(p.grad.data)
            p.grad.data.square_()
            self.state[idx]['ema_sqr_grad'].step(p.grad.data)
        bias_correction1 = 1 - self.beta1(self.n) ** self.n
        bias_correction2 = 1 - self.beta2(self.n) ** self.n
        for (idx, p) in enumerate(self.params()):
            G = self.state[idx]['ema_grad']()/bias_correction1
            G2 = self.state[idx]['ema_sqr_grad']()/bias_correction2
            torch.sqrt_(G2).add_(self.eps(self.n))
            G.div_(G2).add_(p.data,alpha=self.weight_decay(self.n))
            p.data.sub_(G,alpha=self.lr(self.n))
        return result
