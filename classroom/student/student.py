import torch
from torch.cuda.amp import autocast
from ..util import TwoWindowFilter
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
        self.time = 0.0
        self.times = []
        self.grades = []
        self.loss_shaping = None
        self.baseline_model = None
        self.baseline_grades = []
        self.shaped_losses = []
        self.relative_grades = []
        self.example_losses = []

    def set_baseline(self, baseline_model):
        self.baseline_model = baseline_model

    def __del__(self):
        del self.model
        del self.optimizer

    def clone(self):
        return copy.deepcopy(self)

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

    @autocast()
    def study(self):
        def closure():
            batch_size = self.batch_size
            example_length = self.example_length
            X = self.dataset.batch(batch_size=batch_size, example_length=example_length)
            losses = self.model(X)
            if self.baseline_model is not None:
                with torch.no_grad():
                    denom_losses = self.baseline_model(X)
                    self.baseline_grades.append(1.0 - torch.mean(denom_losses).item())
                try:
                    shaped_losses = self.loss_shaping(losses, denom_losses)
                except:
                    shaped_losses = losses
            else:
                try:
                    shaped_losses = self.loss_shaping(losses)
                except:
                    shaped_losses = losses
            shaped_losses = torch.nan_to_num(shaped_losses, nan=0.0, posinf=0.0, neginf=0.0)
            torch.mean(shaped_losses).backward()
            return losses.detach().cpu().numpy(), shaped_losses.detach().cpu().numpy()
        start = time()
        losses, shaped_losses = self.optimizer.step(closure)
        elapsed = time() - start
        self.time += elapsed
        self.grades.append(1.0 - np.sum(losses)/self.batch_size)
        self.example_losses.extend(losses)
        self.times.append(elapsed)
        self.shaped_losses.append(np.sum(shaped_losses)/self.batch_size)
        if len(self.baseline_grades) > 0:
            self.relative_grades.append(self.grades[-1]/self.baseline_grades[-1])

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
                y = Categorical(self.model.inference(torch.tensor(x, dtype=torch.long,device='cuda').unsqueeze(0)).view(-1)[-self.model.n_vocab_out:]).sample().item()
                x = (x + [y])[-n_ctx:]
                if output is not None:
                    output.append(y)
                yield y
        return decode(list(sampler(x)))
