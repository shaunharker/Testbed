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
    * `mutate` is experimental in value at this point
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
    def study(self):
        """
        Use `self.optimizer` to train `self.model` for one step using a batch obtained from `self.dataset` using `self.batch_size` and `self.example_length`.
        Then add/append training data to `self.time`, `self.times`, and `self.grades`.
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

    def parameter_histograms(self):
        """
        Return a dictionary the keys of which are the names of parameters
        as returned by `self.model.named_parameters()` and the values of
        which are pairs (X, Y) which give the pdf of the distribution of
        individual parameter values.
        ### Example
        ```python
        H = student.parameter_histograms()
        plots = [Plot(x="value",y=f"pdf",**{key: H[key]}) for key in H]
        plots[0]
        ```
        """
        pd = {name: p for (name, p) in self.model.named_parameters()}
        H = {}
        for (name, p) in pd.items():
            n = torch.numel(p)
            bins = math.floor(math.sqrt(n))
            data = p.detach().cpu().numpy().reshape(-1)
            Y, X = np.histogram(data, bins=int(len(data)**(1/2)), density=True)
            H[name] = (X, Y)
        return H

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
    def autocomplete(self, prompt=None, n_generate=128, n_ctx=None, encode=None, decode=None, output=None):
        """
        Autocomplete using the model

        ## Args
        * `prompt: str` an optional prompt to begin with
        * `n_generate: int` the number of bytes/tokens to generate
        * `n_ctx: int` the number of bytes/tokens in the context window
        * `encode: TODO` the function that can turn an str into a sequence of bytes/tokens suitable for the model.
        defaults to utf8encode
        * `decode: TODO` the function that can turn the sequences of bytes/tokens used by the model to a str
        defaults to utf8decode
        * `output: Optional[List[int]]` a list to stream the output bytes/tokens to (as `int`s; they will not be decoded to `str`).

        ## TODO
        * make streaming autocomplete with streamed characters (i.e. length 1 strings) using asyncio
        """
        Categorical = torch.distributions.Categorical
        if n_ctx is None:
            n_ctx = self.model.n_ctx
        if encode is None:
            encode = utf8encode
        if decode is None:
            decode = utf8decode
        if prompt is None:
            prompt = decode(self.dataset.batch(1, 2*n_ctx).tolist()[0])  # kludge
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
