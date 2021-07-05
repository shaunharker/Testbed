import numpy as np
import torch
from torch.optim import AdamW
from ..optim import Sonny
from ..data import Loader
import torch.multiprocessing
ctx = torch.multiprocessing.get_context("spawn")
from queue import Empty
from ..util import IgnoreKeyboardInterrupt, Stopwatch, default_device
import time

class Worker(ctx.Process):
    def __init__(self, inbox, outbox, loss_outbox):
        super().__init__(daemon=True)
        self.inbox = inbox
        self.outbox = outbox
        self.loss_outbox = loss_outbox
        self.step = 0
        self.compute_time = 0
        self.compute_energy = 0

    def closure(self):
        while True:
            try:
                return self._closure()
            except RuntimeError:
                self.minibatches *= 2
                self.minibatch_size = self.batch_size // self.minibatches
                if self.minibatch_size == 0:
                    raise RuntimeError("Cannot compute gradient even with minibatch_size=1.")
                print(f"{self.step}. Splitting batch of {self.batch_size} examples into {self.minibatches} of {self.minibatch_size} examples due to memory constraints. The computation will not be affected.")

    def _closure(self):
        num_examples = 0
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
        Y = []
        for _ in range(self.minibatches):
            try:
                self.instruction = self.inbox.get(False)
                if self.instruction != "start":
                    print(f"{self.step}. Interrupted by instruction '{self.instruction}'.")
                    raise KeyboardInterrupt
            except Empty:
                pass
            X = self.loader.batch(batch_size=self.minibatch_size).to(device='cuda',dtype=torch.long, non_blocking=True)
            batch_losses = self.model(X)
            if torch.is_grad_enabled():
                loss = torch.sum(batch_losses)/torch.numel(batch_losses)
                loss.backward()
            num_examples += torch.numel(batch_losses)
            Y.append(batch_losses.detach())
        Y = torch.cat(Y)
        if torch.any(torch.isnan(Y)):
            raise RuntimeError("nan")
        mean_loss = torch.mean(Y).item()
        mean_sqr_loss = torch.mean(Y*Y).item()
        return (mean_loss, mean_sqr_loss, num_examples)

    def train(self):
        print(f"{self.step}. Entering training loop.")
        print(f"{self.step}. self.compute_time = {self.compute_time}")
        print(f"{self.step}. self.batch_size = {self.batch_size}")
        print(f"{self.step}. self.dataset.example_length = {self.dataset.example_length}")
        print(f"{self.step}. There are {len(self.dataset)} examples in self.dataset.")
        print(f"{self.step}. They will be looped through repeatedly until interrupted.")
        with Stopwatch() as stopwatch:
            # Try to tune the optimizer if it has this feature
            try:
                self.optimizer.tune(self.closure)
            except:
                pass
            while True:
                if not self.parent.is_alive():
                    raise RuntimeError("Parent process is not alive.")
                try:
                    (mean_loss, var_loss, num_examples) = self.optimizer.step(self.closure)
                except KeyboardInterrupt:
                    self.instruction = self.instruction
                    break
                self.step += 1
                try:
                    self.compute_energy += self.model.compute_energy() * self.batch_size
                except:
                    self.compute_energy += 0 # not implemented for model, so return 0.
                step_info = {'step': self.step,
                             'compute_time': self.compute_time + stopwatch.time_elapsed,
                             'compute_energy': self.compute_energy,
                             'mean_loss': mean_loss,
                             'var_loss': var_loss,
                             'num_examples': num_examples}
                self.losses.append(step_info)
                self.loss_outbox.put(step_info)
        self.compute_time += stopwatch.total_run_time
        print(f"{self.step}. Exiting training loop at time {self.compute_time}.")
        return self.instruction # for clarity: this function only returns because it gets interrupted by an self.instruction

    def run(self):
        with IgnoreKeyboardInterrupt(): # Jupyter sends these in error sometimes when cells are interrupted.
            self.outbox.put("ready")
            self.model = self.model = self.inbox.get()
            self.example_length = self.inbox.get()
            self.batch_size = self.inbox.get()
            self.OptimizerType = self.inbox.get()
            self.dataset = self.inbox.get()
            self.compute_time = self.inbox.get()
            self.compute_energy = self.inbox.get()
            self.step = self.inbox.get()
            self.losses = self.inbox.get()
            self.dataset.set_example_length(self.example_length)
            self.dataset.cache_data()
            self.loader = Loader(self.dataset, batch_size=self.batch_size)
            self.optimizer = self.OptimizerType(self.model.parameters())
            self.minibatches = 1
            self.minibatch_size = self.batch_size
            print(f"{self.step}. This is the worker process.")
            if self.step > 0:
                print(f"Model has been trained for {self.compute_time}s so far.")
            self.parent = torch.multiprocessing.parent_process()
            self.instruction = "pause"
            while self.instruction != "stop":
                print(f"{self.step}. Waiting for instruction.")
                try:
                    self.instruction = self.inbox.get(True,1.0)
                    self.dispatch()
                except Empty:
                    if not self.parent.is_alive():
                        print(f"{self.step}. Orphaned, exiting.")
                        self.instruction = "stop"
        print(f"{self.step}. Exiting process.")

    def dispatch(self):
        print(f"{self.step}. Received instruction '{self.instruction}' at time {self.compute_time}.")
        if self.instruction == "start":
            self.instruction = self.train()
            # ... and then case fall-through for whatever it returns.
        if self.instruction == "set_optimizer_settings":
            settings = self.inbox.get()
            self.set_optimizer_settings(settings)
        if self.instruction == "get_optimizer_stats":
            print(f"{self.step}. Getting optimizer settings.")
            self.outbox.put(self.optimizer.state['stats'])
        if self.instruction == "set_batch_size":
            self.batch_size = self.inbox.get()
            self.set_batch_size(self.batch_size)
        if self.instruction == "set_example_length":
            self.example_length = self.inbox.get()
            self.set_example_length(self.example_length)
        if self.instruction == "save":
            savefile = self.inbox.get()
            self.save(savefile)
            self.outbox.put("saved")
        if self.instruction == "pause":
            self.outbox.put("paused")

    def set_optimizer_settings(self, settings):
        print(f"{self.step}. Setting optimizer settings.")
        for (k,v) in settings.items():
            self.optimizer.state[k] = v
            if k == 'batch_size':
                self.set_batch_size(v)
            if k == 'example_length':
                self.set_example_length(v)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        print(f"{self.step}. Setting new batch size.")
        print(f"{self.step}. The batch size is now {self.batch_size}.")
        self.minibatch_size = self.batch_size
        self.minibatches = 1

    def set_example_length(self, example_length):
        self.example_length = example_length
        print(f"{self.step}. Setting example length.")
        self.example_length = self.inbox.get()
        self.dataset.set_example_length(self.example_length)
        print(f"{self.step}. The example length is now {self.example_length}.")

    def save(self, savefile):
        checkpoint = {
            'compute_time': self.compute_time,
            'step': self.step,
            'losses': self.losses,
            'model': self.model,
            'example_length': self.example_length,
            'batch_size': self.batch_size,
            'OptimizerType': self.OptimizerType,
            'dataset.filename': self.dataset.filename}
        torch.save(checkpoint, savefile)
