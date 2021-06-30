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
        self.step_info = (0,0,0,0)

    def closure(self):
        crit = 1024
        if self.batch_size > crit:
            self.outer_batch = self.batch_size // crit
            self.inner_batch = crit
        else:
            self.outer_batch = 1
            self.inner_batch = self.batch_size
        num_examples = self.outer_batch * self.inner_batch # <= self.batch_size
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
        Y = []
        for _ in range(self.outer_batch):
            try:
                self.instruction = self.inbox.get(False)
                if self.instruction != "start":
                    print(f"{self.step}. Interrupted at time {self.step_info[1]} by instruction '{self.instruction}'.")
                    raise KeyboardInterrupt
            except Empty:
                pass
            X = self.loader.batch(batch_size=self.inner_batch).to(device='cuda',dtype=torch.long, non_blocking=True)
            batch_losses = self.model(X)
            if torch.is_grad_enabled():
                loss = torch.sum(batch_losses)/num_examples
                loss.backward()
            Y.append(batch_losses.detach())
        Y = torch.cat(Y)
        if torch.any(torch.isnan(Y)):
            raise RuntimeError("nan")
        mean_loss = torch.mean(Y).item()
        mean_sqr_loss = torch.mean(Y*Y).item()
        return (mean_loss, mean_sqr_loss, num_examples)

    def run(self):
        with IgnoreKeyboardInterrupt():
            self.outbox.put("ready")
            self.model = self.model = self.inbox.get()
            self.example_length = self.inbox.get()
            self.batch_size = self.inbox.get()
            self.OptimizerType = self.inbox.get()
            self.dataset = self.inbox.get()
            self.compute_time = self.inbox.get()
            self.step = self.inbox.get()
            self.losses = self.inbox.get()
            self.dataset.cache_data()
            self.loader = Loader(self.dataset, batch_size=self.batch_size)
            self.optimizer = self.OptimizerType(self.model.parameters())
            self.outer_batch = 1 # if batches can't fit, we'll divide it up and loop before calling self.optimizer
            print(f"{self.step}. This is the worker process.")
            if self.step > 0:
                print(f"Model has been trained for {self.compute_time}s so far.")
            parent = torch.multiprocessing.parent_process()
            waiting = False
            while True:
                if not waiting:
                    print(f"{self.step}. Waiting for instruction.")
                    waiting = True
                try:
                    instruction = self.inbox.get(True,1.0)
                    waiting = False
                except:
                    if not parent.is_alive():
                        print(f"{self.step}. Orphaned, exiting.")
                        instruction = "stop"
                        situation = "break"
                        break
                    continue
                print(f"{self.step}. Received instruction '{instruction}' at time {self.compute_time}.")
                if instruction == "start":
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
                            if not parent.is_alive():
                                raise RuntimeError("Parent process is not alive.")
                            try:
                                (mean_loss, var_loss, num_examples) = self.optimizer.step(self.closure)
                            except KeyboardInterrupt:
                                instruction = self.instruction
                                break
                            self.step += 1
                            self.step_info = [self.step, self.compute_time + stopwatch.time_elapsed, mean_loss, var_loss, num_examples]
                            self.losses.append(self.step_info)
                            self.loss_outbox.put(self.step_info)
                    self.compute_time += stopwatch.total_run_time
                    print(f"{self.step}. Exiting training loop at time {self.compute_time}.")
                if instruction == "pause":
                    self.outbox.put("paused")
                    continue
                if instruction == "stop":
                    break
                if instruction == "set_optimizer_settings":
                    print(f"{self.step}. Setting optimizer settings.")
                    settings = self.inbox.get()
                    for (k,v) in settings.items():
                        self.optimizer.state[k] = v
                        if k == 'batch_size':
                            self.batch_size = v
                            print(f"{self.step}. The batch size is now {self.batch_size}.")
                        if k == 'example_length':
                            self.example_length = v
                            self.dataset.set_example_length(self.example_length)
                            print(f"{self.step}. The example length is now {example_length} tokens.")
                    continue
                if instruction == "get_optimizer_stats":
                    print(f"{self.step}. Getting optimizer settings.")
                    self.outbox.put(self.optimizer.state['stats'])
                    continue
                if instruction == "set_batch_size":
                    print(f"{self.step}. Setting new batch size.")
                    self.batch_size = self.inbox.get()
                    print(f"{self.step}. The batch size is now {self.batch_size}.")
                    continue
                if instruction == "set_example_length":
                    print(f"{self.step}. Setting example length.")
                    self.example_length = self.inbox.get()
                    self.dataset.set_example_length(self.example_length)
                    print(f"{self.step}. The example length is now {example_length} tokens.")
                    continue
                if instruction == "save":
                    savefile = self.inbox.get()
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
                    self.outbox.put("saved")
        print(f"{self.step}. Exiting process.")
