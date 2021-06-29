import numpy as np
import torch
from torch.optim import AdamW
from ..optim import Sonny
from torch.utils.data import DataLoader, RandomSampler
import torch.multiprocessing
ctx = torch.multiprocessing.get_context("spawn")
from queue import Empty
from ..util import IgnoreKeyboardInterrupt, Reporter, Stopwatch, default_device
import time

class Worker(ctx.Process):
    def __init__(self, inbox, outbox, loss_outbox):
        super().__init__(daemon=True)
        self.inbox = inbox
        self.outbox = outbox
        self.loss_outbox = loss_outbox

    def closure(self):
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
        mean_loss = 0.0
        var_loss = 0.0
        for outer_batch_idx in range(self.outer_batch):
            try:
                self.instruction = self.inbox.get(False)
                if self.instruction != "start":
                    print(f"{step}. Interrupted at time {step_info[1]} by instruction '{instruction}'.")
                    raise KeyboardInterrupt
            except Empty:
                pass
            X = self.loader.batch(batch_size=self.inner_batch).to(device='cuda',dtype=torch.long, non_blocking=True)
            losses = self.model(X)
            loss = torch.sum(losses)/self.inner_batch
            mean_loss += loss.item()
            if self.inner_batch > 1:
                var_loss += torch.sum((losses.detach() - loss.detach())**2).item()
            if torch.is_grad_enabled():
                loss.backward()
        mean_loss /= self.outer_batch
        var_loss /= (self.inner_batch*self.outer_batch - 1) # bessel correction
        return (mean_loss, var_loss)

    def run(self):
        inbox = self.inbox
        outbox = self.outbox
        loss_outbox = self.loss_outbox
        with IgnoreKeyboardInterrupt():
            outbox.put("ready")
            model = self.model = inbox.get()
            N = inbox.get()
            B = inbox.get()
            OptimizerType = inbox.get()
            dataset = self.dataset = inbox.get()
            compute_time = inbox.get()
            step = inbox.get()
            losses = inbox.get()
            dataset.cache_data()
            self.loader = Loader(dataset, batch_size=B)
            optimizer = self.optimizer = OptimizerType(model.parameters())
            self.outer_batch = 1 # if batches can't fit, we'll divide it up and loop before calling optimizer
            print(f"{step}. This is the worker process.")
            if step > 0:
                print(f"Model has been trained for {compute_time}s so far.")
            # reporter = Reporter()
            parent = torch.multiprocessing.parent_process()
            waiting = False
            while True:
                if not waiting:
                    print(f"{step}. Waiting for instruction.")
                    waiting = True
                try:
                    instruction = inbox.get(True,1.0)
                    waiting = False
                except:
                    if not parent.is_alive():
                        print(f"{step}. Orphaned, exiting.")
                        instruction = "stop"
                        situation = "break"
                        break
                    continue
                print(f"{step}. Received instruction '{instruction}' at time {compute_time}.")
                if instruction == "start":
                    print(f"{step}. Entering training loop at time {compute_time + stopwatch.time_elapsed}. B={B} N={dataset.N}")
                    print(f"{step}. There are {len(dataset)} examples in the dataset.")
                    with Stopwatch() as stopwatch:
                        while True:
                            if not parent.is_alive():
                                raise RuntimeError("Parent process is not alive.")
                            (self.outer_batch, self.inner_batch) = (
                                (B // 8192, 8192) if B > 8192 else
                                (1, B) )        # if B <= 8192
                            self.loader.set_batch_size(self.inner_batch)
                            try:
                                (loss, var) = optimizer.step(self.closure)
                            except KeyboardInterrupt:
                                instruction = self.instruction
                                break
                            step += 1
                            step_info = [step, compute_time + stopwatch.time_elapsed, loss, var]
                            losses.append(step_info)
                            loss_outbox.put(step_info)
                    compute_time += stopwatch.total_run_time
                    print(f"{step}. Exiting compute loop at time {compute_time}.")
                if instruction == "pause":
                    outbox.put("paused")
                    continue
                if instruction == "stop":
                    break
                if instruction == "set_batch_size":
                    print(f"{step}. Setting new batch size.")
                    B = inbox.get()
                    print(f"{step}. There are {len(dataset)//B} batches in the dataset.")
                    continue
                if instruction == "set_example_length":
                    print(f"{step}. Setting new example length.")
                    example_length = inbox.get()
                    dataset.set_example_length(example_length)
                    print(f"{step}. The example length is {example_length} tokens.")
                    continue
                if instruction == "save":
                    savefile = inbox.get()
                    checkpoint = {
                        'time': compute_time,
                        'step': step,
                        'loss': losses,
                        'model': model,
                        'sequence_length': N,
                        'batch_size': B,
                        'optimizer': optimizer,
                        'datafile': dataset.filename}
                    torch.save(checkpoint, savefile)
                    outbox.put("saved")
        print(f"{step}. Exiting process.")
