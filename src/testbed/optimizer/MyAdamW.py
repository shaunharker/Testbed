import torch
from torch.optim import Optimizer
from torch.distributions.log_normal import LogNormal
from .tracker import Accumulator, EMA, MedianTracker
import math

class Sonny(Optimizer):
    r"""Currently = AdamW. Template for experiments.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.
    The Sonny variant is being proposed now. It is a significant and horrible regression
    from the state of the art.
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
    """

    def __init__(self,
                 parameters,
                 lr=.001,
                 beta1=0.9,
                 beta2=0.999,
                 eps=1e-8,
                 weight_decay=0.01):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr value: {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 value: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 value: {beta2}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        super().__init__(parameters, {})
        self.state['lr'] = lr
        self.state['beta1'] = beta1
        self.state['beta2'] = beta2
        self.state['eps'] = eps
        self.state['weight_decay'] = weight_decay
        self.state['step'] = 0
        self.closure = None

    def params(self):
        return [p for group in self.param_groups for p in group['params']
                if p.grad is not None and not p.grad.is_sparse]

    def setup(self):
        self.state['stats']['loss'] = Accumulator()
        for (idx, p) in enumerate(self.params()):
            zeros = lambda : torch.zeros_like(p, memory_format=torch.preserve_format)
            self.state[idx]={'ema_grad': EMA(self.state['beta1'], zeros()),
                             'ema_sqr_grad': EMA(self.state['beta2'], zeros())}

    @torch.no_grad()
    def step(self, closure):
        with torch.enable_grad():
            (mean_loss, mean_sqr_loss, examples) = closure()
        if self.state['step'] == 0:
            self.setup()
        self.state['step'] += 1
        self.state['stats']['loss'].step([mean_loss*examples, mean_sqr_loss*examples, examples])
        for (idx, p) in enumerate(self.params()):
            self.state[idx]['ema_grad'].step(p.grad.data)
            self.state[idx]['ema_sqr_grad'].step(p.grad.data**2)
        bias_correction1 = 1 - self.state['beta1'] ** self.state['step']
        bias_correction2 = 1 - self.state['beta2'] ** self.state['step']
        for (idx, p) in enumerate(self.params()):
            v = ((self.state[idx]['ema_grad']()/bias_correction1)/
                 (torch.sqrt(self.state[idx]['ema_sqr_grad']()/bias_correction2)+self.state['eps']))
            p.data -= self.state['lr']*(v + self.state['weight_decay'] * p.data)
        return (mean_loss, mean_sqr_loss, examples)
