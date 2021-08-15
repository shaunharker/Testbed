import torch
from torch.optim import Optimizer
import time
import numpy as np

class FloydFilter:
    def __init__(self):
        self.reg = None

    def __call__(self, x):
        fresh = lambda: [0, torch.zeros_like(x, memory_format=torch.preserve_format), torch.ones_like(x, memory_format=torch.preserve_format)]
        if self.reg is None:
            self.reg = fresh()
        x2 = x**2
        self.reg[0] += 1
        self.reg[1] += x
        self.reg[2] += x2
        n = self.reg[0]
        ex = self.reg[1] / n
        ex2 = self.reg[2] / n
        return (ex, ex2)

class Floyd(Optimizer):
    def __init__(self,
                 parameters,
                 lr=lambda n: .001,
                 n=0):
        super().__init__(parameters, {"lr": lr, "n": n})

    @torch.no_grad()
    def step(self, closure):
        with torch.enable_grad():
            result = closure()
        for group in self.param_groups:
            lr = group["lr"]
            n = group["n"]
            if n == 0:
                for p in group["params"]:
                    self.state[p] = FloydFilter()
            n += 1
            group["n"] = n
            for p in group["params"]:
                g = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
                (EX, EX2) = self.state[p](g)
                dp = torch.nan_to_num(EX/torch.sqrt(EX2), nan=0.0, posinf=0.0, neginf=0.0)
                dp = lr(n)*dp
                p.data -= dp
        return result
