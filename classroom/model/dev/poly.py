import torch
from torch.nn import Module, Embedding, Linear, CrossEntropyLoss, Softmax
from torch.cuda.amp import autocast
from torch.autograd import Function
import math


class poly(Function):
    @staticmethod
    def forward(ctx, x, coefs):
        """
        Uses Horner's method to compute \sum_{i=0}^{degree} params[i] x^(degree-i)
        """
        ctx.save_for_backward(x, coefs)
        degree = coefs.shape[0]-1
        y = torch.zeros_like(x, memory_format=torch.preserve_format)
        for idx in range(degree+1):
            y.mul_(x)
            y.add_(coefs[degree-idx])
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        (x, coefs) = ctx.saved_tensors
        #print(f"backward. x.shape={x.shape} grad_output.shape={grad_output.shape} coefs.shape={coefs.shape}")
        degree = coefs.shape[0]-1
        y = torch.zeros_like(x, memory_format=torch.preserve_format)
        z = torch.ones_like(x, memory_format=torch.preserve_format)
        for idx in range(degree):
            if idx > 0:
                y.mul_(x)
            y.add_(degree*coefs[degree-idx])
        grad_input = y * grad_output
        grad_coef = []
        for idx in range(degree+1):
            #print(f"idx = {idx} z.shape = {z.shape}, computed {torch.sum(z * grad_output)}")
            grad_coef.append(torch.sum(z * grad_output).view(1))
            if idx < degree:
                z.mul_(x)
        print(grad_coef)
        grad_coef = torch.cat(grad_coef)

        #print(f"grad_input.shape={grad_input.shape}, grad_coef.shape={grad_coef.shape}")
        return (grad_input, grad_coef)


class Poly(Module):
    def __init__(self, degree):
        super().__init__()
        self.degree = degree
        self.coefs = torch.nn.Parameter(torch.zeros(degree+1))
        self.fun = poly.apply

    def forward(self, x):
        return self.fun(x, self.coefs)
