import torch
from ..util import memory_allocated, memory_free
import numpy as np

class Learner:
    def __init__(self, path=None, config=None):
        if path is not None:
            self.load(path)
        if config is not None:
            if "model" in config:
                try:
                    self.model = config["model"]["type"](**config["model"]["kwargs"]).to('cuda')
                except Exception as e:
                    self.model = config["model"]
            if "optimizer" in config:
                try:
                    self.optimizer = config["optimizer"]["type"](parameters=self.model.parameters(), **config["optimizer"]["kwargs"])
                except Exception as e:
                    self.optimizer = config["optimizer"]
            if "dataset" in config:
                try:
                    self.dataset = config["dataset"]["type"](**config["dataset"]["kwargs"])
                except Exception as e:
                    self.dataset = config["dataset"]
        self.call_minibatches_cache = {} # cache to remember how to split up memory constrained step calls, keyed by (batch_size, example_length)

    def save(self, path):
        checkpoint = {"model": self.model, "optimizer": self.optimizer, "dataset": self.dataset}
        torch.save(checkpoint, path)
        return "saved"

    def load(self, path):
        checkpoint = torch.load(path)
        self.model = checkpoint["model"]
        self.optimizer = checkpoint["optimizer"]
        self.dataset = checkpoint["dataset"]

    def step(self, batch_size, example_length):
        closure = lambda: grad_computation(batch_size, example_length)
        return self.optimizer.step(closure)

    def grad_computation(self, batch_size, example_length):
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
            except RuntimeError as e: # CUDA OOM
                if "CUDA" in str(e): # false positives?
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
                             f"due to memory constraints.\n",
                             batch_size=batch_size,
                             example_length=example_length,
                             cuda_memory={"free": f"{f//2**20}MiB",
                                          "allocated": f"{a//2**20}MiB",
                                          "total": f"{(f+a)//2**20}MiB"})
                else:
                    raise e

    def autocomplete(self, prompt=None, n_generate=128, min_ctx=32, max_ctx=32, callback=None):
        Categorical = torch.distributions.Categorical
        decode = self.dataset.decode
        encode = self.dataset.encode
        batch = self.dataset.batch
        if max_ctx is None:
            max_ctx = model.max_ctx
        if prompt is None:
            prompt = decode(batch(1, max_ctx).tolist()[0])
        else:
            if prompt == "":
                prompt = " "
            prompt = decode(encode(prompt))
        x = encode(prompt)
        x = x[-max_ctx:]
        assert len(x) <= max_ctx, f"x={x} max_ctx={max_ctx}"
        def sampler(x):
            x = list(x)
            for _ in range(n_generate):
                y = Categorical(self.model.probs(torch.tensor(x, dtype=torch.long,device='cuda').unsqueeze(0)).view(-1)[-self.model.n_vocab_out:]).sample().item()
                x = (x + [y])[-max_ctx:]
                if callback is not None:
                    callback(y)
                yield y
        return decode(list(sampler(x)))
