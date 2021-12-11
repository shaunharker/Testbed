import torch
import numpy as np
import copy
import random
from time import time


class Trainer:
    """
    Encapsulates `model`, `optimizer`, `dataset`, `batch_size`, `example_length` for the purposes of training.
    """
    def __init__(self, model=None, optimizer=None, dataset=None, batch_size=None, example_length=None, device=None):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.batch_size = batch_size
        self.example_length = example_length
        self.n = 0
        if device is None:
            device = 'cuda'
        self.device = device

    @staticmethod
    def load_from_path(path):
        """
        Load the `Trainer` object stored at `path` and return it.
        """
        trainer = Trainer()
        trainer.load(path)
        return trainer

    def load(self, path):
        """
        Load the `Trainer` object stored at `path` into `self`.
        """
        checkpoint = torch.load(path)
        self.model = checkpoint.get("model", None)
        self.optimizer = checkpoint.get("optimizer", None)
        self.dataset = checkpoint.get("dataset", None)
        self.batch_size = checkpoint.get("batch_size", None)
        self.example_length = checkpoint.get("example_length", None)
        self.n = checkpoint.get("n", 0)

    def save(self, path):
        """
        Save `self` to `path`.
        """
        checkpoint = {
            "model": self.model,
            "optimizer": self.optimizer,
            "dataset": self.dataset,
            "batch_size": self.batch_size,
            "example_length": self.example_length,
            "n": self.n}
        torch.save(checkpoint, path)

    def clone(self):
        """
        Create a clone of `self` and return it.
        """
        try:
            clone = copy.deepcopy(self)
        except Exception as e:
            print('clone exception', e)
        return clone

    def clone_model(self):
        return copy.deepcopy(self.model)

    def step(self):
        """
        Use `self.optimizer` to train `self.model` for one step using a batch obtained from `self.dataset` using training hyperparameters `self.batch_size` and `self.example_length`.
        """
        def closure():
            batch = self.dataset.batch(batch_size=self.batch_size, example_length=self.example_length, offset=None)
            losses = self.model(batch)
            losses = torch.nan_to_num(losses, nan=0.0, posinf=0.0, neginf=0.0)
            loss = torch.mean(losses.detach()).item()
            return loss

        loss = self.optimizer.step(closure)
        self.n += 1
        return loss, baseline_loss

    @torch.no_grad()
    def autocomplete(self, prompt=None, n_generate=128, n_ctx=None, temp=1.0, encode=None, decode=None, output=None):
        """
        Autocomplete using the model

        ## Args
        * `prompt: str` an optional prompt to begin with
        * `n_generate: int` the number of bytes/tokens to generate
        * `n_ctx: int` the number of bytes/tokens in the context window
        * `encode` the function that can turn an str into a sequence of bytes/tokens suitable for the model.
        defaults to utf8encode
        * `decode` the function that can turn the sequences of bytes/tokens used by the model to a str
        defaults to utf8decode
        * `output: Optional[List[int]]` a list to stream the output bytes/tokens to (as `int`s; they will not be decoded to `str`).

        ## TODO
        * make streaming autocomplete with streamed characters (i.e. length 1 strings) using asyncio
        """
        Categorical = torch.distributions.Categorical
        if n_ctx is None:
            n_ctx = self.model.n_ctx
        if encode is None:
            encode = self.dataset.encode # utf8encode, gpt2encode
        if decode is None:
            decode = self.dataset.decode # utf8decode, gpt2decode
        if prompt is None:
            prompt = decode(self.dataset.batch(1, 2*n_ctx, offset=None).tolist()[0])  # kludge
        x = encode(prompt)
        x = x[-n_ctx:]
        prompt = decode(x)
        print(f"=== Prompt ===\n{prompt}\n=== Autocompletion ===\n")

        def sampler(x):
            x = list(x)
            for _ in range(n_generate):
                probs = self.model.inference(torch.tensor(x, dtype=torch.long, device=self.device).unsqueeze(0)).view(-1)[-self.model.n_vocab_out:]
                if temp > 0:
                    y = Categorical(probs=probs**(1.0/temp)).sample().item()
                else:
                    y = torch.argmax(probs).item()
                x = (x + [y])[-n_ctx:]
                if output is not None:
                    output.append(y)
                yield y
        return decode(list(sampler(x)))
