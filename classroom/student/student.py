import torch
from torch.cuda.amp import autocast
import numpy as np
import copy
import random
from time import time
from ..dataset.utf8 import utf8encode
from ..dataset.utf8 import utf8decode
from ..dataset import BytesDataset

class Student:
    """
    Encapsulates `model`, `optimizer`, `dataset`, `batch_size`, `example_length` for the purposes of training.
    Stores training metrics (`time`, `times`, `grades`) associated with the calls of `study`
    ### Notes:
    * `save` and `load` serialize to and from disk
    * `push` and `pop` serialize to and from a stack in memory (implemented through the `self.parent` reference)
    * `clone` creates a clone which is a deepcopy except for `self.parent`, which is not a copy.
    * `mutate` is a coarse proof-of-principle function at this point
    The methods `push`, `pop`, `mutate`, and `clone` may be used by genetic algorithms.

    """
    def __init__(self, model=None, optimizer=None, dataset=None, batch_size=None, example_length=None):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.batch_size = batch_size
        self.example_length = example_length

        self.time = 0.0
        self.times = []
        self.grades = []

        self.parent = None

    @staticmethod
    def load_from_path(path):
        student = Student()
        student.load(path)
        return student

    def load(self, path):
        checkpoint = torch.load(path)
        self.model = checkpoint["model"]
        self.optimizer = checkpoint["optimizer"]
        self.dataset = checkpoint["dataset"]
        self.batch_size = checkpoint["batch_size"]
        self.example_length = checkpoint["example_length"]
        self.time = checkpoint["time"]
        self.times = checkpoint["times"]
        self.grades = checkpoint["grades"]
        self.parent = checkpoint["parent"]

    def save(self, path):
        checkpoint = {
            "model": self.model,
            "optimizer": self.optimizer,
            "dataset": self.dataset,
            "batch_size": self.batch_size,
            "example_length": self.example_length,
            "time": self.time,
            "times": self.times,
            "grades": self.grades,
            "parent": self.parent}
        torch.save(checkpoint, path)

    def clone(self):
        """
        Create a clone of `self` and return it.
        The clone's `parent` attribute (if present)
        will be the same reference as the original.
        Everything else will be a deep copy.
        """
        tmp = self.parent
        self.parent = None
        clone = copy.deepcopy(self)
        self.parent = tmp
        clone.parent = tmp
        return clone

    def push(self):
        """
        Create a clone of `self` and store it in `self.parent`.
        Note that this has a Russian doll effect if called repeatedly,
        with backups having older backups.
        """
        self.parent = self.clone()

    def pop(self):
        """
        Revert to the state stored in `self.parent` on the previous `backup` call.
        If no such call took place, then do nothing.
        """
        if self.parent is None:
            return  # silly rabbit
        clone = self.parent.clone()
        self.model = clone.model
        self.optimizer = clone.optimizer
        self.dataset = clone.dataset
        self.batch_size = clone.batch_size
        self.example_length = clone.example_length
        self.time = clone.time
        self.times.clear()
        self.times.extend(clone.times)
        self.grades.clear()
        self.grades.extend(clone.grades)
        self.parent = clone.parent

    @autocast()
    def study(self, batch):
        """
        Train the model a step using the supplied `batch` of examples.

        ## Args
        ### `batch`:
        ```python
        assert type(batch) == torch.Tensor
        assert batch.dtype == torch.long
        (batch_size, example_length) = batch.shape
        assert example_length == self.model.n_ctx + 1
        ```
        ## Returns
        ```python
        None
        ```
        """
        def closure():
            batch = self.dataset.batch(batch_size=self.batch_size, example_length=self.example_length)
            losses = self.model(batch)
            losses = torch.nan_to_num(losses, nan=0.0, posinf=0.0, neginf=0.0)
            torch.mean(losses).backward()
            return losses.detach().cpu().numpy()
        start = time()
        losses = self.optimizer.step(closure)
        elapsed = time() - start
        self.time += elapsed
        self.times.append(elapsed)
        self.grades.append(1.0 - np.mean(losses))

    def mutate(self):
        """
        Mutate `self` by randomly altering `self.batch_size` and `self.optimizer.param_groups[0]["lr"]`
        """
        r = random.choice([0.5, 0.75, 1.0/0.75, 2.0])
        self.batch_size = int(r*self.batch_size)
        r = random.choice([0.5, 0.75, 1.0/0.75, 2.0])
        lr = self.optimizer.param_groups[0]["lr"](0)
        lr = lr*r
        if lr == 0.0:
            lr = 1e-6  # minimum learning rate, maybe should lower
        self.optimizer.param_groups[0]["lr"] = lambda n: lr

    @torch.no_grad()
    def autocomplete(self, prompt=None, n_generate=128, n_ctx=None, dataset=None, encode=None, decode=None, output=None):
        """
        Autocomplete using the model

        ## Args
        ### `prompt`
        the
        ### `n_generate`
        ### `n_ctx`
        ### `dataset`
        ### `encode`
        ### `decode`
        ### `output`
        """
        Categorical = torch.distributions.Categorical
        if n_ctx is None:
            n_ctx = self.model.n_ctx
        if encode is None:
            encode = utf8encode
        if decode is None:
            decode = utf8decode
        if prompt is None:
            if dataset is None:
                dataset = BytesDataset()
            if dataset is not None:
                prompt = decode(dataset.batch(1, 2*n_ctx).tolist()[0])  # kludge
            else:
                prompt = " Every dog goes to heaven." * 128
        x = encode(prompt)
        x = x[-n_ctx:]
        def sampler(x):
            x = list(x)
            for _ in range(n_generate):
                y = Categorical(self.model.inference(torch.tensor(x,dtype=torch.long,device='cuda').unsqueeze(0)).view(-1)[-self.model.n_vocab_out:]).sample().item()
                x = (x + [y])[-n_ctx:]
                if output is not None:
                    output.append(y)
                yield y
        return decode(list(sampler(x)))
