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


class FloydFilter2:
    def __init__(self, lag=64):
        self.lag = lag
        self.reg1 = [0, None, None]
        self.reg2 = [0, None, None]
        self.count = 0
        assert self.lag % 2 == 0

    def __call__(self, x):
        self.reg1[0] += 1
        if self.reg1[1] == None:
            self.reg1[1] = torch.zeros_like(x, memory_format=torch.preserve_format)
        self.reg1[1] += x
        if self.reg1[2] == None:
            self.reg1[2] = torch.zeros_like(x, memory_format=torch.preserve_format)
        self.reg1[2] += x*x
        self.reg2[0] += 1
        if self.reg2[1] == None:
            self.reg2[1] = torch.zeros_like(x, memory_format=torch.preserve_format)
        self.reg2[1] += x
        if self.reg2[2] == None:
            self.reg2[2] = torch.zeros_like(x, memory_format=torch.preserve_format)
        self.reg2[2] += x*x
        i = self.count % self.lag
        j = self.lag // 2
        self.count = self.count + 1
        def meanvar(reg):
            n = reg[0]
            EX = reg[1]/n
            EX2 = reg[2]/n
            mean = EX
            if n > 1:
                variance = (n/(n-1)) * (EX2 - EX**2)
            else:
                variance = torch.ones_like(x, memory_format=torch.preserve_format)
            return (mean, variance)
        (m1, v1) = meanvar(self.reg1)
        (m2, v2) = meanvar(self.reg2)
        if i == 0:
            self.reg2 = [0, None, None]
            (mean, variance) = (m1, v1)
        elif i == j:
            self.reg1 = [0, None, None]
            (mean, variance) = (m2, v2)
        elif i < j:
            t = i/j
            mean = (1-t)*m1 + t*m2
            variance = (1-t)*v1 + t*v2
        elif i > j:
            t = (i-j)/j
            mean = t*m1 + (1.0-t)*m2
            variance = t*v1 + (1.0-t)*v2
        return (mean, variance)

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
