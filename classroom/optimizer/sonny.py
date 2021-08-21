import torch
from torch.optim import Optimizer

class Sonny(Optimizer):
    def __init__(self,
                 parameters,
                 lr=lambda n: .001,
                 alpha=lambda n: 0.9,
                 n=0):
        super().__init__(parameters, {"lr": lr, "alpha": alpha, "n": n})

    @torch.no_grad()
    def step(self, closure):
        for group in self.param_groups:
            alpha = group["alpha"]
            for p in group["params"]:
                try:
                    p.grad.data *= alpha(n)
                except:
                    pass
        with torch.enable_grad():
            result = closure()
        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            n = group["n"]
            # if n == 0:
            #     for p in group["params"]:
            #         self.state[p] = ... some internal state ...
            group["n"] = n + 1
            for p in group["params"]:
                p.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0) # TODO: initializers?
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                search_direction = -(p.grad-torch.mean(p.grad))/torch.sqrt(torch.var(p.grad))
                search_direction.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                step_size = lr(n)*torch.sqrt(torch.var(p))
                p += step_size * search_direction
        return result
