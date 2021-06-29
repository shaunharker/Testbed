import torch
from torch.optim import Optimizer
from torch import sqrt
from torch.distributions.log_normal import LogNormal

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
                 llr=-1, # log learning rate. So much cooler!
                 alpha=0.5,
                 beta=0.5,
                 eps=1e-8,
                 weight_decay=1e-3):
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        super().__init__(parameters)
        self.state['llr'] = llr
        self.state['alpha'] = alpha
        self.state['beta'] = beta
        self.state['eps'] = eps
        self.state['weight_decay'] = weight_decay
        self.state['step'] = 0

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

    def compute_loss_and_grad():
        """
        computes loss at current parameter and also populates grad
        returns loss.item()
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                loss.backward()
        else:
            raise RuntimeError("Sonny needs a closure to function.")
        return loss.item()

    def update(self, alpha=None, beta=None):
        """
        Recomputes loss, grad at original point and updates moving averages.
        Computes a new search direction v.
        All outputs are stored in self.state
        """
        if alpha is None:
            alpha = self.state['alpha']
        if beta is None:
            beta = self.state['beta']

        f = self.state['loss'] = compute_loss_and_grad()
        f2 = f ** 2
        if 'mean_loss' not in self.state:
            self.state['mean_loss'] = f
            self.state['sqr_mean_loss'] = f2
        else:
            self.state['mean_loss'] += alpha * (f - self.state['mean_loss'])
            self.state['sqr_mean_loss'] += alpha * (f2 - self.state['sqr_mean_loss'])
        self.state['var_loss'] = self.state['sqr_mean_loss'] - self.state['mean_loss']**2

        for p in self.params():
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError('Sonny does not support sparse gradients')
            state = self.state[p]
            df = p.grad.data
            df2 = p.grad.data ** 2
            if len(state) == 0:
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.ones_like(p, memory_format=torch.preserve_format)
            g = state['exp_avg']
            g2 = state['exp_avg_sq']
            g += beta * (df - g)
            g2 += beta * ((df2) - g2)
        return f

    def compute_search_direction(self):
        k = self.state['weight_decay']
        for p in self.params():
            state = self.state[p]
            x = p.data
            g = state['exp_avg']
            g2 = state['exp_avg_sq']
            var = torch.clamp(g2 - g**2, min=0)
            v = torch.erf(g / sqrt(var)) # v = g / sqrt(var)
            state['v'] = -(v + k * x)

    def trial(self, h, closure):
        self.move(h)
        with torch.no_grad():
            loss = closure()
        self.move(-h)
        return loss.item()

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if closure is None:
            raise RuntimeError("Sonny requires a closure.")

        self.state['step'] += 1
        # We draw the step size from a log-normal distribution.
        # We slowly adjust the parameters of this distribution
        # to track the step-sizes that worked after a look-ahead.
        while True:
            h = LogNormal(self.state['llr'], 1.0).sample()
            self.update() # computes loss and grad and updates statistics at current param
            self.compute_search_direction()
            trial_loss = self.trial(h, closure)
            zscore = (trial_loss - self.state['mean_loss'])/math.sqrt(self.state['var_loss'])
            erf = torch.erf(torch.tensor(zscore)).item() # not very elegant
            self.state['llr'] -= .01*erf*(math.log(h) - self.state['llr'])
            if zscore <= -1.0:
                move(h)
                return self.state['loss']
