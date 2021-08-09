import torch
from torch.nn import Module, Embedding, Linear, CrossEntropyLoss, Softmax
from torch.cuda.amp import autocast
from torch.autograd import Function
import math


class multipoly(Function):
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
        degree = coefs.shape[0]-1
        y = torch.zeros_like(x, memory_format=torch.preserve_format)
        for idx in range(degree):
            y.mul_(x)
            y.add_(degree*coefs[degree-idx])
        z = torch.pow(x.view([1] + list(x.shape)),
                      torch.arange(0, degree+1,
                                   device=x.device).view([degree+1] + [1]*len(x.shape)))
        #print(f"y{y.shape} z{z.shape} go{grad_output.shape}")
        return (y * grad_output, (z * grad_output.view([1] + list(grad_output.shape))).transpose(0,1))


class MultiPoly(Module):
    def __init__(self, shape, degree):
        super().__init__()
        self.degree = degree
        if type(shape) == int:
            shape = [shape]
        self.coefs = torch.nn.Parameter(torch.zeros([degree+1] + shape))
        self.fun = multipoly.apply

    def forward(self, x):
        return self.fun(torch.sigmoid(x), self.coefs)
