import torch
from torch.optim import Optimizer
from torch.distributions.log_normal import LogNormal
import math

class Sonny(Optimizer):
    r"""Implements Sonny algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.
    The Sonny variant is being proposed now. It is a significant and horrible regression
    from the state of the art.
    Args:
        parameters (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): initial learning rate (default: .01)
        alpha (float, optional): EMA parameter for loss/sqrloss (default: 0.5)
        beta1 (float, optional): EMA parameter for grad/sqrgrad (default: 0.01)
        beta2 (float, optional): EMA parameter for grad/sqrgrad (default: 0.0001)
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
                 beta1=0.01,
                 beta2=0.0001,
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
        """
        return parameters in a list. parameter specific information
        shall be stored in self.state[p][key]
        """
        return [p for group in self.param_groups for p in group['params']
                if p.grad is not None and not p.grad.is_sparse]

    def move(self, h):
        """
        Step along search direction by factor h
        p <- p + h * v
        """
        for p in self.params():
            p.data += h*self.state[p]['v']

    def update(self, substep=None):
        """
        Recomputes loss, grad at original point and updates moving averages.
        """
        if substep is None:
            self.state['substep'] = 0
            self.state['examples'] = 0

        with torch.enable_grad():
            (mean_loss, mean_sqr_loss, new_examples) = self.closure()

        if self.state['step'] == 0 and self.state['substep'] == 0:
            self.state['stats']['zscores'] = [0.0]

        if self.state['step'] > 0:
            zscore = (mean_loss - self.state['mean_loss'])/math.sqrt(self.state['var_loss'])
            self.state['stats']['zscores'].append(zscore)

        if self.state['step'] == 0 and self.state['substep'] == 0:
            for p in self.params():
                self.state[p]['ema_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                self.state[p]['ema_sqr_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)

        if self.state['substep'] == 0:
            self.state['examples'] = 0
            self.state['mean_loss'] = mean_loss
            self.state['mean_sqr_loss'] = mean_sqr_loss

        avg = lambda x,y : (self.state['examples']*x + new_examples*y)/(self.state['examples']+new_examples)
        self.state['mean_loss'] = avg(self.state['mean_loss'], mean_loss)
        self.state['mean_sqr_loss'] = avg(self.state['mean_sqr_loss'], mean_sqr_loss)
        self.state['var_loss'] = self.state['mean_sqr_loss'] - self.state['mean_loss']**2

        for p in self.params():
            self.state[p]['ema_grad'].mul_(1-self.state['beta1']).add_(p.grad.data, alpha=self.state['beta1'])
            self.state[p]['ema_sqr_grad'].mul_(1-self.state['beta2']).addcmul_(p.grad.data, p.grad.data, value=self.state['beta2'])
            #self.state[p]['ema_grad'] += self.state['beta1'] * (p.grad.data - self.state[p]['ema_grad'])
            #self.state[p]['ema_sqr_grad'] += self.state['beta2'] * ((p.grad.data ** 2) - self.state[p]['ema_sqr_grad'])
        self.state['examples'] += new_examples
        return (mean_loss, mean_sqr_loss, new_examples)

    def compute_search_direction(self):
        bias_correction1 = 1 - (1-self.state['beta1']) ** (1+self.state['step'])
        bias_correction2 = 1 - (1-self.state['beta2']) ** (1+self.state['step'])
        for p in self.params():
            v = ((self.state[p]['ema_grad']/bias_correction1)/
                 (torch.sqrt(self.state[p]['ema_sqr_grad']/bias_correction2)+self.state['eps']))
            self.state[p]['v'] = -(v + self.state['weight_decay'] * p.data)
            if torch.any(torch.isnan(v)):
                raise RuntimeError("nan")

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self.closure = closure
        # print(f"Step {self.state['step']}. Starting optimization.")
        h = self.state['lr']
        (base_mean_loss, base_mean_sqr_loss, base_examples) = self.update() # computes loss and grad and updates statistics at current param
        self.compute_search_direction()
        self.move(h)
        self.state['step'] += 1
        return (base_mean_loss, base_mean_sqr_loss, base_examples)

    def tune(self, closure):
        pass
