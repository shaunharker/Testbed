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
        llr (float, optional): initial log learning rate (default: -1)
        alpha (float, optional): EMA parameter for loss/sqrloss (default: 0.5)
        beta (float, optional): EMA parameter for grad/sqrgrad (default: 0.5)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    """

    def __init__(self,
                 parameters,
                 llr=math.log(.0001), # log learning rate. So much cooler!
                 alpha=0.5,
                 beta=0.1,
                 eps=1e-8,
                 weight_decay=0.0):
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        super().__init__(parameters, {})
        self.state['llr'] = llr
        self.state['alpha'] = alpha
        self.state['beta'] = beta
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
            self.state['stats']['histogram'] = [1 if i == 4 else 0 for i in range(9)]

        if self.state['step'] > 0:
            zscore = (mean_loss - self.state['mean_loss'])/math.sqrt(self.state['var_loss'])
            self.state['stats']['zscores'].append(zscore)
            # also make a simplistic histogram for convenience
            bin = round(zscore)
            if bin >= -4 and bin <= 4:
                self.state['stats']['histogram'][bin+4] += 1
            print(f"Histogram: {self.state['stats']['histogram']}")
        #print(f"Step {self.state['step']}. Substep {self.state['substep']}. {(mean_loss.item(), mean_sqr_loss.item(), examples)}")

        if self.state['step'] == 0 and self.state['substep'] == 0:
            self.state['ema_mean_loss'] = mean_loss
            self.state['ema_mean_sqr_loss'] = mean_sqr_loss
            for p in self.params():
                self.state[p]['ema_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                self.state[p]['ema_sqr_grad'] = torch.ones_like(p, memory_format=torch.preserve_format)
        if self.state['substep'] == 0:
            self.state['examples'] = 0
            self.state['mean_loss'] = mean_loss
            self.state['mean_sqr_loss'] = mean_sqr_loss

        avg = lambda x,y : (self.state['examples']*x + new_examples*y)/(self.state['examples']+new_examples)
        self.state['mean_loss'] = avg(self.state['mean_loss'], mean_loss)
        self.state['mean_sqr_loss'] = avg(self.state['mean_sqr_loss'], mean_sqr_loss)
        self.state['var_loss'] = self.state['mean_sqr_loss'] - self.state['mean_loss']**2
        self.state['ema_mean_loss'] += self.state['alpha'] * (mean_loss - self.state['ema_mean_loss'])
        self.state['ema_mean_sqr_loss'] += self.state['alpha'] * (mean_sqr_loss - self.state['ema_mean_sqr_loss'])
        self.state['var_ema_loss'] = self.state['ema_mean_sqr_loss'] - self.state['ema_mean_loss']**2
        for p in self.params():
            self.state[p]['ema_grad'] += self.state['beta'] * (p.grad.data - self.state[p]['ema_grad'])
            self.state[p]['ema_sqr_grad'] += self.state['beta'] * ((p.grad.data ** 2) - self.state[p]['ema_sqr_grad'])
        self.state['examples'] += new_examples
        return (mean_loss, mean_sqr_loss, new_examples)

    def compute_search_direction(self):
        #torch.set_printoptions(profile="full")
        for p in self.params():
            var_ema_grad = self.state[p]['ema_sqr_grad'] - self.state[p]['ema_grad']**2
            v = torch.erf(self.state[p]['ema_grad'] /
                          torch.sqrt(torch.clamp(var_ema_grad, min=self.state['eps'])))
            self.state[p]['v'] = -(v + self.state['weight_decay'] * p.data)
            #print(f"compute_search_direction. var_ema_grad = {var_ema_grad}")
            #print(f"compute_search_direction. v = {v}")
            if torch.any(torch.isnan(v)):
                raise RuntimeError("nan")
        #torch.set_printoptions(profile="default")

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self.closure = closure
        # print(f"Step {self.state['step']}. Starting optimization.")
        h = LogNormal(self.state['llr'], 1).sample()
        (base_mean_loss, base_mean_sqr_loss, base_examples) = self.update() # computes loss and grad and updates statistics at current param
        self.compute_search_direction()
        self.move(h)
        self.state['step'] += 1
        return (base_mean_loss, base_mean_sqr_loss, base_examples)
