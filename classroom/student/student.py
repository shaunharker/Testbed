import torch
from ..util import memory_allocated, memory_free
import numpy as np
import copy
from random import randrange

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

    def clone(self):
        return copy.deepcopy(self)

    def mutate(self):
        c = random.choice(["batch_size", "lr"])
        if c == "batch_size":
            if self.batch_size == 1:
                self.batch_size = 2
            else:
                c = random.choice(["half","double"])
                if c == "half":
                    self.batch_size = self.batch_size // 2
                elif c == "double":
                    self.batch_size = self.batch_size * 2
        elif c == "lr":
            c = random.choice(["half","double"])
            if c == "half":
                self.optimizer.lr = self.optimizer.lr // 2
            elif c == "double":
                self.optimizer.lr = self.optimizer.lr * 2

    def save(self, path):
        checkpoint = {"model": self.model, "optimizer": self.optimizer, "dataset": self.dataset,
            "batch_size": self.batch_size, "example_length": example_length}
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model = checkpoint["model"]
        self.optimizer = checkpoint["optimizer"]
        self.dataset = checkpoint["dataset"]
        self.batch_size = checkpoint["batch_size"]
        self.example_length = checkpoint["example_length"]

    def study(self):
        closure = lambda: self.grad_computation()
        return np.sum(self.optimizer.step(closure))/self.batch_size

    def grad_computation(self):
        batch_size = self.batch_size
        example_length = self.example_length
        minibatches = self.call_minibatches_cache.get((batch_size, example_length), 1)
        minibatch_size = batch_size // minibatches
        while True:
            try:
                if torch.is_grad_enabled():
                    self.optimizer.zero_grad()
                Y = []
                for _ in range(minibatches):
                    X = self.dataset.batch(batch_size=minibatch_size, example_length=example_length)
                    batch_losses = self.model(X)
                    if torch.is_grad_enabled():
                        loss = torch.sum(batch_losses)/torch.numel(batch_losses)
                        loss.backward()
                    Y.append(batch_losses.detach())
                Y = torch.cat(Y)
                return Y.cpu().numpy()
            except RuntimeError as e:
                if "CUDA" in str(e):
                    torch.cuda.empty_cache()
                    minibatches *= 2
                    minibatch_size = batch_size // minibatches
                    if minibatch_size == 0:
                        raise RuntimeError("Cannot compute gradient even with minibatch_size=1.")
                    self.call_minibatches_cache[(batch_size, example_length)] = minibatches
                    f = memory_free()
                    a = memory_allocated()
                    print(f"Splitting batch of {batch_size} examples into "
                             f"{minibatches} minibatches of size {minibatch_size} "
                             f"due to memory constraints.\n")
                else:
                    raise e

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
                y = Categorical(self.model(torch.tensor(x, dtype=torch.long,device='cuda').unsqueeze(0)).view(-1)[-self.model.n_vocab_out:]).sample().item() # and it's as simple as that! :P
                x = (x + [y])[-n_ctx:]
                if output is not None:
                    output.append(y)
                yield y
        return decode(list(sampler(x)))
