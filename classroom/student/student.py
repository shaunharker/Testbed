import torch
from ..util import memory_allocated, memory_free
import numpy as np
import copy
import random
from random import randrange
from time import time

class Student:
    def __init__(self, path=None, model=None, optimizer=None, dataset=None, batch_size=None, example_length=None):
        if path is not None:
            self.load(path)
        if model is not None:
            self.model = model
        if optimizer is not None:
            self.optimizer = optimizer
        if dataset is not None:
            self.dataset = dataset
        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = 1
        if example_length is not None:
            self.example_length = example_length
        else:
            self.example_length = self.model.n_ctx + 1
        self.call_minibatches_cache = {} # cache to remember how to split up memory constrained step calls, keyed by (batch_size, example_length)
        self.times = []
        self.grades = []
        self.time = 0.0
        self.exps = []

    def __del__(self):
        del self.model
        del self.optimizer

    def clone(self):
        return copy.deepcopy(self)

    def set_bs(self, lr):
        self.batch_size = lambda _: bs
        return self

    def set_lr(self, lr):
        self.optimizer.param_groups[0]["lr"] = lambda _: lr
        return self

    def mutate(self):
        r = random.choice([0.5, 0.75, 1.0/0.75, 2.0])
        self.batch_size = int(r*self.batch_size)
        r = random.choice([0.5, 0.75, 1.0/0.75, 2.0])
        lr = self.optimizer.param_groups[0]["lr"](0)
        lr = lr*r
        if lr == 0.0:
            lr = 1e-6 # minimum learning rate, maybe should lower
        self.optimizer.param_groups[0]["lr"] = lambda _: lr

    def save(self, path):
        checkpoint = {
            "model": self.model,
            "optimizer": self.optimizer,
            "dataset": self.dataset,
            "batch_size": self.batch_size,
            "example_length": self.example_length}
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model = checkpoint["model"]
        self.optimizer = checkpoint["optimizer"]
        self.dataset = checkpoint["dataset"]
        self.batch_size = checkpoint["batch_size"]
        self.example_length = checkpoint["example_length"]

    def study(self):
        @autocast()
        def closure():
            batch_size = self.batch_size
            example_length = self.example_length
            self.exps.append(batch_size * example_length)
            if torch.is_grad_enabled():
                self.optimizer.zero_grad()
            X = self.dataset.batch(batch_size=batch_size, example_length=example_length)
            try:
                batch_losses = self.model(X)
            except Exception as e:
                if self.batch_size == 1:
                    raise e
                print(f"Due to {e}, setting batch_size to {batch_size//2}")
                self.batch_size = max(1, self.batch_size//2)
            if torch.is_grad_enabled():
                loss = torch.sum(batch_losses)/torch.numel(batch_losses)
                loss.backward()
            return batch_losses.detach().cpu().numpy()
        start = time()
        losses = self.optimizer.step(closure)
        elapsed = time() - start
        grade = 1.0 - np.sum(losses)/self.batch_size
        self.grades.append(grade)
        self.times.append(elapsed)
        self.time += elapsed




    @torch.no_grad()
    def autocomplete(self, prompt=None, n_generate=128, n_ctx=None, output=None):
        Categorical = torch.distributions.Categorical
        decode = self.dataset.decode
        encode = self.dataset.encode
        batch = self.dataset.batch
        if n_ctx is None:
            n_ctx = self.model.n_ctx
        if prompt is None:
            prompt = decode(batch(1, 2*n_ctx).tolist()[0]) # kludge
        x = encode(prompt)
        x = x[-n_ctx:]
        def sampler(x):
            x = list(x)
            for _ in range(n_generate):
                y = Categorical(self.model(torch.tensor(x, dtype=torch.long,device='cuda').unsqueeze(0)).view(-1)[-self.model.n_vocab_out:]).sample().item()
                x = (x + [y])[-n_ctx:]
                if output is not None:
                    output.append(y)
                yield y
        return decode(list(sampler(x)))
