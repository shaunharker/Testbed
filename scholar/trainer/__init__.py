import torch
import asyncio
import numpy as np
import copy
import random
import time
from datetime import datetime

class Trainer:
    """
    Encapsulates `model`, `optimizer`, `dataset`, `batch_size`, `example_length` for the purposes of training.
    """
    def __init__(self, model=None, optimizer=None, dataset=None, batch_size=None, example_length=None):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.batch_size = batch_size
        self.example_length = example_length
        self.n = 0
        self.losses = []

    def status(self):
        dn = self.n
        dt = time.time() - self.t0
        N = dn//2
        L = 8*np.mean(np.array(self.losses[dn-N:dn]))
        now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        message = '\n'.join([
            f"time            = {now[:-2]}",
            f"L               ~ {int(L*1e6)/1e6} bpc",
            f"batch_size      = {self.batch_size}",
            f"example_length  = {self.example_length}",
            f"dn              = {dn} steps",
            f"dt              = {int(dt)} seconds",
            f"dn/dt           = {int(dn/dt*10)/10} steps per second",
            f"feeding rate    = {int(self.batch_size*self.example_length*dn/dt/1024)} KiBps",
        ])
        return message

    async def train(self):
        self.t0 = time.time()
        while True:
            batch, lossitem = self.step()
            self.n += 1
            self.losses.append(lossitem)
            await asyncio.sleep(1e-4)

    def step(self, batch=None):
        """
        Use `self.optimizer` to train `self.model` for one step using a batch obtained from `self.dataset` using training hyperparameters `self.batch_size` and `self.example_length`.
        Return the batch and evaluated lossitem
        """
        forward = self.model
        batch = batch or self.fetch()
        closure = lambda: self.F(forward=forward, batch=batch)
        lossitem = self.optimizer.step(closure)
        return batch, lossitem

    @torch.no_grad()
    def eval(self, batch=None):
        """
        Evaluate the model on a dataset batch.
        Return the evaluated loss (as a float)
        """
        forward = self.model
        batch = batch or self.fetch()
        closure = lambda: self.F(forward=forward, batch=batch)
        lossitem = closure()
        return batch, lossitem

    def fetch(self):
        """
        Return a batch from the dataset
        """
        batch = self.dataset.batch(batch_size=self.batch_size, example_length=self.example_length)
        return batch

    def shaping(self, batch, losses):
        """
        Given a batch and losses, return `loss` such that
        calling `loss.backward()` accumulates gradients
        """
        losses = torch.nan_to_num(losses, nan=0.0, posinf=0.0, neginf=0.0)
        loss = torch.mean(losses)
        return loss

    def F(self, forward, batch):
        losses = forward(batch)
        loss = self.shaping(batch, losses)
        if loss.requires_grad:
            loss.backward()
        lossitem = loss.item()
        return lossitem
